# Cabinet Door Opening Robot - CS 188 Starter Project

### Disclaimer

This project was designed for CS 188 - Intro to Robotics as a template starter project. If you have any issues with the codebase, please email me at holdengs @ cs.ucla.edu!

## Overview

In this project you will build a robot that learns to open kitchen cabinet doors
using **RoboCasa365**, a large-scale simulation benchmark for everyday robot
tasks. You will progress from understanding the simulation environment, to
collecting demonstrations, to training a neural-network policy that controls the
robot autonomously.

### What you will learn

1. How robotic manipulation environments are structured (MuJoCo + robosuite + RoboCasa)
2. How the `OpenCabinet` task works -- sensors, actions, success criteria
3. How to collect and use demonstration datasets (human + MimicGen)
4. How to train a behavior-cloning policy from demonstrations
5. How to evaluate your trained policy in simulation

### The robot

We use the **PandaOmron** mobile manipulator -- a Franka Panda 7-DOF arm
mounted on an Omron wheeled base with a torso lift joint. This is the default
and best-supported robot in RoboCasa.

---

## Installation

Run the install script (works on **macOS** and **WSL/Linux**):

```bash
./install.sh
```

This will:
- Create a Python virtual environment (`.venv`)
- Clone and install robosuite and robocasa
- Install all Python dependencies (PyTorch, numpy, matplotlib, etc.)
- Download RoboCasa kitchen assets (~10 GB)

After installation, activate the environment:

```bash
source .venv/bin/activate
```

Then verify everything works:

```bash
cd cabinet_door_project
python 00_verify_installation.py
```

> **macOS note:** Scripts that open a rendering window (03, 05) require
> `mjpython` instead of `python`. The install script will remind you of this.

---

## Project Structure

```
cabinet_door_project/
  00_verify_installation.py      # Check that everything is installed correctly
  01_explore_environment.py      # Create the OpenCabinet env, inspect observations/actions
  02_random_rollouts.py          # Run random actions, save video, understand the task
  03_teleop_collect_demos.py     # Teleoperate the robot to collect your own demonstrations
  04_download_dataset.py         # Download the pre-collected OpenCabinet dataset
  05_playback_demonstrations.py  # Play back demonstrations to see expert behavior
  06_train_policy.py             # Train a simple MLP behavior-cloning policy
  07_evaluate_policy.py          # Evaluate your trained policy in simulation
  08_visualize_policy_rollout.py # Visualize a rollout of your policy in RoboCasa
  configs/
    diffusion_policy.yaml        # Training hyperparameters
  notebook.ipynb                 # Interactive Jupyter notebook companion
install.sh                       # Installation script (macOS + WSL/Linux)
README.md                        # This file
```

---

## Step-by-Step Guide

### Step 0: Verify Installation

```bash
python 00_verify_installation.py
```

This checks that MuJoCo, robosuite, RoboCasa, and all dependencies are
correctly installed and that the `OpenCabinet` environment can be created.

### Step 1: Explore the Environment

```bash
python 01_explore_environment.py
```

This script creates the `OpenCabinet` environment and prints detailed
information about:
- **Observation space**: what the robot sees (camera images, joint positions,
  gripper state, base pose)
- **Action space**: what the robot can do (arm movement, gripper open/close,
  base motion, control mode)
- **Task description**: the natural language instruction for the episode
- **Success criteria**: how the environment determines task completion

### Step 2: Random Rollouts

```bash
python 02_random_rollouts.py
```

Runs the robot with random actions to see what happens (spoiler: nothing
useful, but it helps you understand the action space). Saves a video to
`/tmp/cabinet_random_rollouts.mp4`.

### Step 3: Teleoperate and Collect Demonstrations

```bash
# Mac users: use mjpython instead of python
python 03_teleop_collect_demos.py
```

Control the robot yourself using the keyboard to open cabinet doors. This
gives you intuition for the task difficulty and generates demonstration data.

**Keyboard controls:**
| Key | Action |
|-----|--------|
| Ctrl+q | Reset simulation |
| spacebar | Toggle gripper (open/close) |
| up-right-down-left | Move horizontally in x-y plane |
| .-; | Move vertically |
| o-p | Rotate (yaw) |
| y-h | Rotate (pitch) |
| e-r | Rotate (roll) |
| b | Toggle arm/base mode (if applicable) |
| s | Switch active arm (if multi-armed robot) |
| = | Switch active robot (if multi-robot environment) |              

### Step 4: Download Pre-collected Dataset

```bash
python 04_download_dataset.py
```

Downloads the official OpenCabinet demonstration dataset from the RoboCasa
servers. This includes both human demonstrations and MimicGen-expanded data
across diverse kitchen scenes.

### Step 5: Play Back Demonstrations

```bash
python 05_playback_demonstrations.py
```

Visualize the downloaded demonstrations to see how an expert opens cabinet
doors. This is the data your policy will learn from.

### Step 6: Train a Policy

```bash
python 06_train_policy.py
```

Trains a simple MLP behavior-cloning policy on low-dimensional state-action
pairs from the demonstration data. This is meant to illustrate the
data-loading → training → checkpoint pipeline, not to produce a policy that
can reliably solve the task.


#### [Note: The instructions below in this step don't work since they refer to a non-existent file. I cloned the Diffusion Policy repo into the cabinet_door_project directory and added the missing yaml. However, I recommend just ignoring the instructions here since you'll still probably run into a bunch of dependency errors when trying to run the code, and I couldn't get meaningful results after finally getting it to run. Instead, you can try to implement the model yourself over the starter code, which is what I'm attempting as well, or you can maybe look into the other suggested solutions.]

For a policy that actually works, use one of the official training repos:

```bash
# Diffusion Policy (recommended for single-task)
git clone https://github.com/robocasa-benchmark/diffusion_policy
cd diffusion_policy && pip install -e .
python train.py --config-name=train_diffusion_transformer_bs192 task=robocasa/OpenCabinet
```

You can also print setup instructions for Diffusion Policy, pi-0, and GR00T
directly from the script:

```bash
python 06_train_policy.py --use_diffusion_policy
```

### Step 7: Evaluate Your Policy

```bash
python 07_evaluate_policy.py --checkpoint path/to/checkpoint.pt
```

Runs your trained policy in the simulation environment and reports success
rate across multiple episodes and kitchen scenes.

---

## Key Concepts

### The OpenCabinet Task

- **Goal**: Open a kitchen cabinet door
- **Fixture**: `HingeCabinet` (a cabinet with hinged doors)
- **Initial state**: Cabinet door is closed; robot is positioned nearby
- **Success**: `fixture.is_open(env)` returns `True`
- **Horizon**: 500 timesteps at 20 Hz control frequency (25 seconds)
- **Scene variety**: 2,500+ kitchen layouts/styles for generalization

### Observation Space (PandaOmron)

| Key | Shape | Description |
|-----|-------|-------------|
| `robot0_agentview_left_image` | (256, 256, 3) | Left shoulder camera |
| `robot0_agentview_right_image` | (256, 256, 3) | Right shoulder camera |
| `robot0_eye_in_hand_image` | (256, 256, 3) | Wrist-mounted camera |
| `robot0_gripper_qpos` | (2,) | Gripper finger positions |
| `robot0_base_pos` | (3,) | Base position (x, y, z) |
| `robot0_base_quat` | (4,) | Base orientation quaternion |
| `robot0_base_to_eef_pos` | (3,) | End-effector pos relative to base |
| `robot0_base_to_eef_quat` | (4,) | End-effector orientation relative to base |

### Action Space (PandaOmron)

| Key | Dim | Description |
|-----|-----|-------------|
| `end_effector_position` | 3 | Delta (dx, dy, dz) for the end-effector |
| `end_effector_rotation` | 3 | Delta rotation (axis-angle) |
| `gripper_close` | 1 | 0 = open, 1 = close |
| `base_motion` | 4 | (forward, side, yaw, torso) |
| `control_mode` | 1 | 0 = arm control, 1 = base control |

### Dataset Format (LeRobot)

Datasets are stored in LeRobot format:
```
dataset/
  meta/           # Episode metadata (task descriptions, camera info)
  videos/         # MP4 videos from each camera
  data/           # Parquet files with actions, states, rewards
  extras/         # Per-episode metadata
```

---

## Architecture Diagram

```
                    RoboCasa Stack
                    ==============

  +-------------------+     +-------------------+
  |   Kitchen Scene   |     |   OpenCabinet     |
  |  (2500+ layouts)  |     |   (Task Logic)    |
  +--------+----------+     +--------+----------+
           |                         |
           v                         v
  +------------------------------------------------+
  |              Kitchen Base Class                 |
  |  - Fixture management (cabinets, fridges, etc)  |
  |  - Object placement (bowls, cups, etc)          |
  |  - Robot positioning                            |
  +------------------------+-----------------------+
                           |
                           v
  +------------------------------------------------+
  |              robosuite (Backend)                |
  |  - MuJoCo physics simulation                   |
  |  - Robot models (PandaOmron, GR1, Spot, ...)   |
  |  - Controller framework                        |
  +------------------------+-----------------------+
                           |
                           v
  +------------------------------------------------+
  |              MuJoCo 3.3.1 (Physics)            |
  |  - Contact dynamics, rendering, sensors        |
  +------------------------------------------------+
```

---

## Research Directions

The MLP baseline in `06_train_policy.py` is intentionally simple — it
demonstrates the pipeline but will basically always fail. Here are three
fun directions to improve the model:

### Minimal Diffusion Policy

Replace the direct-regression MLP with a diffusion-based action generator.
The core loop is to corrupt ground-truth actions with Gaussian noise,
train the network to predict that noise conditioned on the current state, and
at inference iteratively denoise from pure noise to produce an action. This
properly handles multi-modal demonstrations (e.g., approaching the handle from
the left vs. right) that MSE loss averages into useless mean actions.
See [Chi et al., 2023](https://diffusion-policy.cs.columbia.edu/) for the
full approach — a minimal version can be built in ~100 lines on top of the
existing MLP backbone.

### DAgger (Online Correction)

Script 03 already provides keyboard teleoperation. I have it set up with a DAgger mode that may or may not be kinda buggy. Use it to close the loop:
train a policy, roll it out, then have a human take over and correct the robot
whenever it fails. Aggregate these corrections into the training set and
retrain. This directly attacks distribution shift — the fundamental reason
offline BC degrades at test time — by collecting data in the states the policy
actually visits. Even one or two rounds of DAgger can dramatically improve
robustness. See [Ross et al., 2011](https://arxiv.org/abs/1011.0686).

### Action Chunking

Instead of predicting one action per timestep, predict the next *K* actions at
once and execute them open-loop before re-planning. This is the key idea behind
ACT ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)) and directly fixes
the jerky, temporally incoherent behavior of single-step BC. Fair warning, though, this will probably require a more sophisticated model (Transformer, Diffusion or other) to provide real benefits. Implementation is
straightforward: widen the output head to `K * action_dim`, train with the same
MSE loss over the full chunk, and add a small FIFO buffer at inference. Try
sweeping K = 4, 8, 16 and compare smoothness and success rate.

### Other Ideas
- Gaussian Mixture Model for output logits. Can ameliorate the MSE multimodality issue.
- Vision Transformer. Will need a beefier computer to see benefits but definitely can improve policy at scale.
- Hooking in an existing VLM and experimenting with zero-shot inference.

---

## Troubleshooting

I'll continually update this section as students find bugs in the system. Please, let me know if you encounter issues!

| Problem | Solution |
|---------|----------|
| `MuJoCo version must be 3.3.1` | `pip install mujoco==3.3.1` |
| `numpy version must be 2.2.5` | `pip install numpy==2.2.5` |
| Rendering crashes on Mac | Use `mjpython` instead of `python` |
| `GLFW error` on headless server | Set `export MUJOCO_GL=egl` or `osmesa` |
| Out of GPU memory during training | Reduce batch size in `configs/diffusion_policy.yaml` |
| Kitchen assets not found | Run `python -m robocasa.scripts.download_kitchen_assets` |

---

## References

- [RoboCasa Paper & Website](https://robocasa.ai/)
- [RoboCasa GitHub](https://github.com/robocasa/robocasa)
- [robosuite Documentation](https://robosuite.ai/)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
