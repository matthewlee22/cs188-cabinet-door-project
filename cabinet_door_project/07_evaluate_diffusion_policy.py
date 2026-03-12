"""
Step 7 (Diffusion): Evaluate a Diffusion Policy
================================================
Runs a checkpoint from 06_train_diffusion_policy.py against the live
OpenCabinet environment and reports success rate.

Key differences from the simple BC evaluator:
  - Maintains a rolling observation buffer (n_obs_steps history)
  - Runs DDIM denoising (10 steps) to generate an action chunk
  - Executes n_action_steps actions before replanning

Usage:
    python 07_evaluate_diffusion_policy.py \\
        --checkpoint /tmp/diffusion_policy_ckpt/best_diffusion_policy.pt

    # Evaluate on held-out kitchens
    python 07_evaluate_diffusion_policy.py \\
        --checkpoint best_diffusion_policy.pt \\
        --split target --num_rollouts 50
"""

import argparse
import collections
import math
import os
import sys

# Disable MuJoCo rendering — state-only policy, no cameras needed
os.environ["MUJOCO_GL"] = "disabled"
os.environ.pop("PYOPENGL_PLATFORM", None)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# Patch robosuite.make to strip all rendering flags before robocasa is imported.
# The gym wrapper calls robosuite.make internally; without this patch it tries
# to create an EGL/offscreen context which fails in WSL2.
import robosuite as _rs
_orig_rs_make = _rs.make
def _headless_make(env_name, **kwargs):
    kwargs["has_renderer"] = False
    kwargs["has_offscreen_renderer"] = False
    kwargs["use_camera_obs"] = False
    return _orig_rs_make(env_name, **kwargs)
_rs.make = _headless_make

import robocasa  # noqa: F401 — registers gym envs (uses patched robosuite.make)

# ---------------------------------------------------------------------------
# Action layout (from modality.json — must match 06_train_diffusion_policy.py)
# ---------------------------------------------------------------------------
ACTION_KEYS = [
    "action.base_motion",          # [0:4]
    "action.control_mode",         # [4:5]
    "action.end_effector_position", # [5:8]
    "action.end_effector_rotation", # [8:11]
    "action.gripper_close",         # [11:12]
]
ACTION_SPLITS = [
    ("action.base_motion", 0, 4),
    ("action.control_mode", 4, 5),
    ("action.end_effector_position", 5, 8),
    ("action.end_effector_rotation", 8, 11),
    ("action.gripper_close", 11, 12),
]

# Observation layout (from modality.json)
OBS_STATE_KEYS_ORDERED = [
    "state.base_position",                  # [0:3]
    "state.base_rotation",                  # [3:7]
    "state.end_effector_position_relative", # [7:10]
    "state.end_effector_rotation_relative", # [10:14]
    "state.gripper_qpos",                   # [14:16]
]
# When checkpoint obs_dim==19, the raw env's door_obj_to_robot0_eef_pos (3-dim)
# is appended to OBS_STATE_KEYS_ORDERED to form the full 19-dim state vector.


# ---------------------------------------------------------------------------
# Model (copy of architecture from 06_train_diffusion_policy.py)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=x.device) * -freq)
        emb = x.float()[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        obs_dim, action_dim, n_obs_steps, n_action_steps,
        d_model=256, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.0,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.action_head = nn.Linear(d_model, action_dim)

        n_tokens = 1 + n_obs_steps + n_action_steps
        self.pos_emb = nn.Embedding(n_tokens, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

    def forward(self, noisy_action, timestep, obs):
        t_tok = self.time_emb(timestep).unsqueeze(1)
        obs_tok = self.obs_proj(obs)
        act_tok = self.action_proj(noisy_action)
        tokens = torch.cat([t_tok, obs_tok, act_tok], dim=1)
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.pos_emb(pos)
        out = self.transformer(tokens)
        return self.action_head(out[:, 1 + self.n_obs_steps:])


class DDPMScheduler:
    def __init__(self, num_train_steps=100, beta_start=1e-4, beta_end=0.02):
        self.num_train_steps = num_train_steps
        betas = torch.linspace(beta_start, beta_end, num_train_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    def to(self, device):
        for attr in ["betas", "alphas", "alphas_cumprod",
                     "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    @torch.no_grad()
    def ddim_sample(self, model, obs, n_action_steps, action_dim, device,
                    num_inference_steps=10):
        B = obs.shape[0]
        step_ratio = self.num_train_steps // num_inference_steps
        timesteps = list(range(self.num_train_steps - 1, -1, -step_ratio))[:num_inference_steps]

        x = torch.randn(B, n_action_steps, action_dim, device=device)
        for i, t_val in enumerate(timesteps):
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
            pred_noise = model(x, t_tensor, obs)
            alpha_bar_t = self.alphas_cumprod[t_val]
            # No clamp — z-score normalized actions can legitimately exceed ±1
            x0_pred = (x - (1.0 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alphas_cumprod[t_prev]
                x = alpha_bar_prev.sqrt() * x0_pred + (1.0 - alpha_bar_prev).sqrt() * pred_noise
            else:
                x = x0_pred
        return x


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    kwargs = ckpt["model_kwargs"]
    model = DiffusionTransformer(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scheduler = DDPMScheduler(num_train_steps=ckpt.get("num_diffusion_steps", 100)).to(device)
    act_names = [
        "base_motion_x","base_motion_y","base_motion_z","base_motion_w",
        "control_mode",
        "eef_pos_x","eef_pos_y","eef_pos_z",
        "eef_rot_x","eef_rot_y","eef_rot_z",
        "gripper_close",
    ]
    print(
        f"Loaded checkpoint: epoch={ckpt['epoch']}  loss={ckpt['loss']:.6f}\n"
        f"  obs_dim={ckpt['obs_dim']}  action_dim={ckpt['action_dim']}  "
        f"n_obs_steps={ckpt['n_obs_steps']}  n_action_steps={ckpt['n_action_steps']}\n"
        f"  act_mean: {dict(zip(act_names, ckpt['act_mean'].round(4)))}\n"
        f"  act_std:  {dict(zip(act_names, ckpt['act_std'].round(4)))}"
    )
    return model, scheduler, ckpt


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def gym_obs_to_state(gym_obs: dict, raw_env=None) -> np.ndarray:
    """Convert gym wrapper obs dict → state vector matching training data.

    If raw_env is provided (non-None), appends door_obj_to_robot0_eef_pos (3-dim)
    to produce a 19-dim vector for door-augmented checkpoints.
    """
    parts = []
    for key in OBS_STATE_KEYS_ORDERED:
        parts.append(np.atleast_1d(gym_obs[key]).astype(np.float32))
    state = np.concatenate(parts)  # (16,)

    if raw_env is not None:
        raw_inner_obs = raw_env._get_observations(force_update=False)
        door_to_eef = raw_inner_obs.get(
            "door_obj_to_robot0_eef_pos", np.zeros(3, dtype=np.float32)
        ).astype(np.float32)
        state = np.concatenate([state, door_to_eef])  # (19,)

    return state


def flat_action_to_gym_dict(action_flat: np.ndarray) -> dict:
    """Split 12-dim action into the dict format expected by the gym wrapper step()."""
    return {k: action_flat[s:e] for k, s, e in ACTION_SPLITS}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args, model, scheduler, ckpt, device):
    n_obs_steps = ckpt["n_obs_steps"]
    n_action_steps = ckpt["n_action_steps"]        # prediction horizon
    n_action_exec = ckpt.get("n_action_exec", n_action_steps)  # steps to execute (≤ prediction)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    obs_mean = torch.from_numpy(ckpt["obs_mean"]).to(device)
    obs_std = torch.from_numpy(ckpt["obs_std"]).to(device)
    act_mean = ckpt["act_mean"]
    act_std = ckpt["act_std"]

    env = gym.make(f"robocasa/OpenCabinet", split=args.split, enable_render=False)

    # If the checkpoint has obs_dim==19 (door-augmented), we need the raw env
    # to query door_obj_to_robot0_eef_pos at every step.
    use_door_aug = (obs_dim == 19)
    raw_env = env.unwrapped.env if use_door_aug else None
    if use_door_aug:
        print("  [door-aug mode] appending door_obj_to_robot0_eef_pos to every obs")

    successes = []
    ep_lengths = []

    for ep in range(args.num_rollouts):
        raw_obs, info = env.reset()

        # For door-aug mode, force a fresh observations snapshot after reset
        if use_door_aug:
            raw_env._get_observations(force_update=True)

        # Fill obs buffer with the initial observation repeated
        obs_buffer = collections.deque(maxlen=n_obs_steps)
        init_state = gym_obs_to_state(raw_obs, raw_env)
        for _ in range(n_obs_steps):
            obs_buffer.append(init_state)

        success = False
        action_queue = collections.deque()  # pre-computed action chunk

        init_state_raw = gym_obs_to_state(raw_obs, raw_env)
        init_state_norm = (init_state_raw - ckpt["obs_mean"]) / ckpt["obs_std"]
        if ep == 0:
            print(f"  Initial obs (raw):  {init_state_raw.round(3)}")
            print(f"  Initial obs (norm): {init_state_norm.round(3)}")
        prev_state = init_state_raw

        for step in range(args.max_steps):
            # Replan when the action queue is empty
            if not action_queue:
                obs_seq = np.stack(list(obs_buffer), axis=0)  # (T_o, obs_dim)
                obs_t = torch.from_numpy(obs_seq).unsqueeze(0).to(device)  # (1, T_o, obs_dim)
                obs_t = (obs_t - obs_mean) / obs_std

                action_chunk = scheduler.ddim_sample(
                    model, obs_t, n_action_steps, action_dim, device,
                    num_inference_steps=10,
                )  # (1, T_a, action_dim)
                action_chunk = action_chunk.squeeze(0).cpu().numpy()  # (T_a, action_dim)
                action_chunk = action_chunk * act_std + act_mean  # unnormalize

                # Debug: print action stats for ep 0, first 3 replans
                if ep == 0 and step < n_action_steps * 3:
                    a = action_chunk[0]
                    print(
                        f"    [dbg step={step:3d}] "
                        f"base_motion={a[0:4].round(3)}  "
                        f"ctrl={a[4]:.2f}  "
                        f"eef_pos={a[5:8].round(3)}  "
                        f"eef_rot={a[8:11].round(3)}  "
                        f"grip={a[11]:.2f}"
                    )

                # Only enqueue the first n_action_exec steps (temporal aggregation)
                for a in action_chunk[:n_action_exec]:
                    action_queue.append(a)

            action_flat = action_queue.popleft()
            gym_action = flat_action_to_gym_dict(action_flat)

            raw_obs, reward, terminated, truncated, info = env.step(gym_action)
            new_state = gym_obs_to_state(raw_obs, raw_env)

            # Debug: obs change for ep 0, first 24 steps
            if ep == 0 and step < 24:
                delta = np.abs(new_state - prev_state)
                door_info = ""
                if use_door_aug:
                    door_to_eef = new_state[16:19]
                    door_info = f"  door_dist={np.linalg.norm(door_to_eef):.3f}"
                print(
                    f"    [dbg step={step:3d}] "
                    f"base_pos_delta={delta[0:3].round(4)}  "
                    f"eef_pos_delta={delta[7:10].round(4)}"
                    f"{door_info}"
                )
            prev_state = new_state

            obs_buffer.append(new_state)

            if info.get("success", False) or terminated:
                success = True
                break

        successes.append(success)
        ep_lengths.append(step + 1)
        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep + 1:3d}/{args.num_rollouts}: {status}  (steps={step + 1})")

    env.close()

    success_rate = sum(successes) / len(successes)
    print(f"\nSuccess rate: {sum(successes)}/{len(successes)} = {success_rate:.1%}")
    print(f"Avg episode length: {np.mean(ep_lengths):.1f}")
    return success_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion policy on OpenCabinet")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pt checkpoint from 06_train_diffusion_policy.py",
    )
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument(
        "--split", type=str, default="pretrain", choices=["pretrain", "target"],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, scheduler, ckpt = load_checkpoint(args.checkpoint, device)
    run_evaluation(args, model, scheduler, ckpt, device)


if __name__ == "__main__":
    main()
