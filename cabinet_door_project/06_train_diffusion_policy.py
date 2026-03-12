"""
Step 6 (Diffusion): Train a Diffusion Policy
=============================================
Ports the Diffusion Policy transformer model into this project's
self-contained training structure.  No Hydra / wandb required.

Architecture:
  - Transformer-based denoising network (DiT-style)
  - DDPM training (100 noise steps), DDIM inference (10 steps)
  - Low-dim state observations only

Observation modes
  default (16-dim):  [base_pos(3), base_rot(4), eef_pos(3), eef_rot(4), gripper(2)]
  --use_door_aug (19-dim): above + door_obj_to_robot0_eef_pos(3)
    Requires running augment_door_obs.py first to produce data/door_aug/*.npy

Usage:
    python 06_train_diffusion_policy.py
    python 06_train_diffusion_policy.py --use_door_aug
    python 06_train_diffusion_policy.py --epochs 500 --batch_size 256
    python 06_train_diffusion_policy.py --checkpoint_dir /tmp/my_ckpts
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Observation / action layout (from modality.json)
# ---------------------------------------------------------------------------
OBS_DIM_BASE = 16   # base proprioceptive: [base_pos(3), base_rot(4), eef_pos(3), eef_rot(4), gripper(2)]
OBS_DIM_DOOR = 3    # door augmentation:   door_obj_to_robot0_eef_pos(3)
OBS_DIM      = OBS_DIM_BASE  # updated to 19 when --use_door_aug is set
ACTION_DIM   = 12   # action: [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper_close(1)]
N_OBS_STEPS = 2
N_ACTION_STEPS = 16   # prediction horizon (paper: predict 16, execute 8 = temporal lookahead)
N_ACTION_EXEC = 8     # how many of the 16 predicted steps are actually executed before replanning


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for diffusion timestep."""

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
    """
    Transformer denoising network.

    Tokens: [timestep | obs_0 | obs_1 | action_0 ... action_{T_a-1}]
    Full self-attention over all tokens; output action tokens predict noise.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        n_obs_steps: int = N_OBS_STEPS,
        n_action_steps: int = N_ACTION_STEPS,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        # Token projections
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.action_head = nn.Linear(d_model, action_dim)

        # Learnable positional embeddings
        n_tokens = 1 + n_obs_steps + n_action_steps
        self.pos_emb = nn.Embedding(n_tokens, d_model)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        noisy_action: torch.Tensor,   # (B, T_a, action_dim)
        timestep: torch.Tensor,        # (B,) int
        obs: torch.Tensor,             # (B, T_o, obs_dim)
    ) -> torch.Tensor:                 # (B, T_a, action_dim)
        # Timestep token
        t_tok = self.time_emb(timestep).unsqueeze(1)          # (B, 1, d)
        obs_tok = self.obs_proj(obs)                           # (B, T_o, d)
        act_tok = self.action_proj(noisy_action)               # (B, T_a, d)

        tokens = torch.cat([t_tok, obs_tok, act_tok], dim=1)  # (B, 1+T_o+T_a, d)

        pos = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.pos_emb(pos)

        out = self.transformer(tokens)                         # (B, 1+T_o+T_a, d)
        action_out = out[:, 1 + self.n_obs_steps:]            # (B, T_a, d)
        return self.action_head(action_out)                    # (B, T_a, action_dim)


# ---------------------------------------------------------------------------
# DDPM noise scheduler
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """
    Linear-beta DDPM scheduler.
    Training: add_noise()  — forward process q(x_t | x_0)
    Inference: ddim_sample() — fast deterministic reverse with ~10 steps
    """

    def __init__(self, num_train_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
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
        for attr in [
            "betas", "alphas", "alphas_cumprod",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def add_noise(
        self,
        x0: torch.Tensor,      # (B, T_a, action_dim)
        noise: torch.Tensor,   # same shape
        t: torch.Tensor,       # (B,) int
    ) -> torch.Tensor:
        s_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        s_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return s_ab * x0 + s_1mab * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        obs: torch.Tensor,             # (B, T_o, obs_dim)
        n_action_steps: int,
        action_dim: int,
        device,
        num_inference_steps: int = 10,
    ) -> torch.Tensor:                 # (B, T_a, action_dim)
        B = obs.shape[0]

        # Evenly-spaced timesteps from T-1 → 0
        step_ratio = self.num_train_steps // num_inference_steps
        timesteps = list(range(self.num_train_steps - 1, -1, -step_ratio))[:num_inference_steps]

        x = torch.randn(B, n_action_steps, action_dim, device=device)

        for i, t_val in enumerate(timesteps):
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
            pred_noise = model(x, t_tensor, obs)

            alpha_bar_t = self.alphas_cumprod[t_val]
            # Predict clean action (no clamp — z-score normalized actions can exceed ±1)
            x0_pred = (x - (1.0 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alphas_cumprod[t_prev]
                # DDIM deterministic update
                x = alpha_bar_prev.sqrt() * x0_pred + (1.0 - alpha_bar_prev).sqrt() * pred_noise
            else:
                x = x0_pred

        return x


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Loads obs-action sequences from the LeRobot parquet dataset.

    Each sample: obs_seq (n_obs_steps, obs_dim) + action_seq (n_action_steps, 12)
    obs_dim = 16 (base) or 19 (base + door_to_eef) when door_aug_dir is given.
    Actions and observations are z-score normalised.
    """

    def __init__(
        self,
        dataset_path: str,
        n_obs_steps: int = N_OBS_STEPS,
        n_action_steps: int = N_ACTION_STEPS,
        max_episodes: int | None = None,
        door_aug_dir: str | None = None,
    ):
        import pyarrow.parquet as pq  # lazy import — not needed at import time

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # Locate the parquet data directory
        data_dir = os.path.join(dataset_path, "data")
        if not os.path.exists(data_dir):
            # Try one level deeper (e.g. path points to parent of lerobot/)
            data_dir = os.path.join(dataset_path, "lerobot", "data")
        chunk_dir = os.path.join(data_dir, "chunk-000")
        if not os.path.exists(chunk_dir):
            raise FileNotFoundError(f"Expected parquet directory: {chunk_dir}")

        files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".parquet"))
        if max_episodes:
            files = files[:max_episodes]

        obs_list, act_list = [], []
        for ep_idx, fn in enumerate(files):
            df = pq.read_table(os.path.join(chunk_dir, fn)).to_pandas()
            if "observation.state" not in df.columns or "action" not in df.columns:
                continue
            obs_ep = np.stack(df["observation.state"].values).astype(np.float32)  # (T, 16)
            act_ep = np.stack(df["action"].values).astype(np.float32)             # (T, 12)

            # Optionally append door_obj_to_robot0_eef_pos (3-dim)
            if door_aug_dir is not None:
                aug_path = os.path.join(door_aug_dir, f"ep_{ep_idx:04d}.npy")
                if os.path.exists(aug_path):
                    door_aug = np.load(aug_path).astype(np.float32)  # (T, 3)
                    # Truncate or pad to match episode length
                    T = min(len(obs_ep), len(door_aug))
                    obs_ep = np.concatenate([obs_ep[:T], door_aug[:T]], axis=1)  # (T, 19)
                    act_ep = act_ep[:T]
                else:
                    print(f"  WARNING: no door aug for ep {ep_idx}, skipping")
                    continue

            obs_list.append(obs_ep)
            act_list.append(act_ep)

        if not obs_list:
            raise RuntimeError("No valid episodes found in dataset.")

        actual_obs_dim = obs_list[0].shape[1]

        # Build sequences
        seqs_obs, seqs_act = [], []
        T_o, T_a = n_obs_steps, n_action_steps
        for obs_ep, act_ep in zip(obs_list, act_list):
            T = len(obs_ep)
            for t in range(T_o - 1, T - T_a + 1):
                seqs_obs.append(obs_ep[t - T_o + 1 : t + 1])   # (T_o, obs_dim)
                seqs_act.append(act_ep[t : t + T_a])             # (T_a, 12)

        all_obs = np.stack(seqs_obs)   # (N, T_o, obs_dim)
        all_act = np.stack(seqs_act)   # (N, T_a, 12)

        # Z-score statistics (computed over entire dataset)
        flat_obs = all_obs.reshape(-1, actual_obs_dim)
        flat_act = all_act.reshape(-1, ACTION_DIM)
        self.obs_mean = flat_obs.mean(0).astype(np.float32)
        self.obs_std = (flat_obs.std(0) + 1e-6).astype(np.float32)
        self.act_mean = flat_act.mean(0).astype(np.float32)
        self.act_std = (flat_act.std(0) + 1e-6).astype(np.float32)
        self.obs_dim = actual_obs_dim

        self.obs = (all_obs - self.obs_mean) / self.obs_std
        self.act = (all_act - self.act_mean) / self.act_std

        print(
            f"Dataset: {len(self.obs):,} sequences from {len(obs_list)} episodes  "
            f"| obs_dim={actual_obs_dim}  action_dim={ACTION_DIM}"
        )

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "action": torch.from_numpy(self.act[idx]),
        }

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.act_std + self.act_mean

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_dataset_path() -> str:
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    # get_ds_path returns the HDF5 path; the lerobot dataset is a sibling dir
    hdf5_path = get_ds_path("OpenCabinet", source="human")
    if hdf5_path is None:
        raise FileNotFoundError("OpenCabinet dataset not found. Run 04_download_dataset.py.")
    # Typical layout: .../OpenCabinet/<date>/human/demo.hdf5
    #                 .../OpenCabinet/<date>/lerobot/
    lerobot_path = os.path.join(os.path.dirname(hdf5_path), "..", "lerobot")
    lerobot_path = os.path.normpath(lerobot_path)
    if not os.path.exists(lerobot_path):
        # Try the path from the known absolute location
        lerobot_path = (
            "/mnt/c/Users/mlee2/projects/cs188-cabinet-door-project"
            "/robocasa/datasets/v1.0/pretrain/atomic/OpenCabinet/20250819/lerobot"
        )
    return lerobot_path


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    door_aug_dir = None
    if getattr(args, "door_aug_dir", None):
        door_aug_dir = args.door_aug_dir
        if not os.path.isabs(door_aug_dir):
            door_aug_dir = os.path.join(os.path.dirname(__file__), door_aug_dir)
        if not os.path.isdir(door_aug_dir):
            raise FileNotFoundError(f"Door aug directory not found: {door_aug_dir}")
        print(f"Using door augmentation from: {door_aug_dir}")
    elif getattr(args, "use_door_aug", False):
        door_aug_dir = os.path.join(
            os.path.dirname(__file__), "data", "door_aug"
        )
        if not os.path.isdir(door_aug_dir):
            raise FileNotFoundError(
                f"Door aug directory not found: {door_aug_dir}\n"
                "Run augment_door_obs.py first."
            )
        print(f"Using door augmentation from: {door_aug_dir}")

    dataset = SequenceDataset(dataset_path, N_OBS_STEPS, N_ACTION_STEPS,
                              door_aug_dir=door_aug_dir)
    obs_dim = dataset.obs_dim

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    model = DiffusionTransformer(
        obs_dim=obs_dim, action_dim=ACTION_DIM,
        n_obs_steps=N_OBS_STEPS, n_action_steps=N_ACTION_STEPS,
        d_model=256, nhead=8, num_layers=8, dim_feedforward=1024,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = DDPMScheduler(num_train_steps=100).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader)
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            obs = batch["obs"].to(device)     # (B, T_o, 16)
            action = batch["action"].to(device)  # (B, T_a, 12)
            B = obs.shape[0]

            t = torch.randint(0, scheduler.num_train_steps, (B,), device=device)
            noise = torch.randn_like(action)
            noisy = scheduler.add_noise(action, noise, t)

            pred_noise = model(noisy, t, obs)
            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_sched.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:4d}/{args.epochs}  Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "obs_mean": dataset.obs_mean,
                    "obs_std": dataset.obs_std,
                    "act_mean": dataset.act_mean,
                    "act_std": dataset.act_std,
                    "obs_dim": obs_dim,
                    "action_dim": ACTION_DIM,
                    "n_obs_steps": N_OBS_STEPS,
                    "n_action_steps": N_ACTION_STEPS,
                    "num_diffusion_steps": 100,
                    "loss": best_loss,
                    "n_action_exec": N_ACTION_EXEC,
                    # Model hyperparams for re-instantiation
                    "model_kwargs": dict(
                        obs_dim=obs_dim, action_dim=ACTION_DIM,
                        n_obs_steps=N_OBS_STEPS, n_action_steps=N_ACTION_STEPS,
                        d_model=256, nhead=8, num_layers=8, dim_feedforward=1024,
                    ),
                },
                os.path.join(args.checkpoint_dir, "best_diffusion_policy.pt"),
            )

    print(f"\nTraining complete.  Best loss: {best_loss:.6f}")
    print(f"Checkpoint: {args.checkpoint_dir}/best_diffusion_policy.pt")


def main():
    parser = argparse.ArgumentParser(description="Train diffusion policy for OpenCabinet")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/diffusion_policy_ckpt",
    )
    parser.add_argument(
        "--use_door_aug",
        action="store_true",
        help="Augment obs with door_obj_to_robot0_eef_pos (19-dim instead of 16-dim). "
             "Requires running augment_door_obs.py first.",
    )
    parser.add_argument(
        "--door_aug_dir",
        type=str,
        default=None,
        help="Path to door augmentation directory (overrides --use_door_aug default path). "
             "Can be absolute or relative to this script's directory.",
    )
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
