# Unitree G1 VLA Simulation

MuJoCo manipulation tasks for Unitree G1, including model-free control, demonstration collection, and GR00T N1.7 evaluation.

## Tasks

| Environment | Task |
| --- | --- |
| 1 | Pick up a mug and place it on a coaster |
| 2 | Sort an apple and soda can into matching targets |
| 3 | Grasp a brush and sweep a block into a target |

All tasks support fixed and randomized layouts. Objects move only through MuJoCo contact.

## Pipeline and Files

```text
Task -> Control -> NPZ collection -> REAL_G1 export -> GR00T -> Results
```

| Step | Files | Purpose |
| --- | --- | --- |
| Task | `envs/`, `unitree_robots/g1/` | Scenes, randomization, and success criteria |
| Control | `verify_model_free_control.py` | Jacobian IK and PD torque control |
| Collect | `collect_demonstrations.py` | Record 20 Hz multimodal NPZ episodes |
| Export | `export_groot_dataset.py`, `groot_real_g1_config.py` | Convert to LeRobot `REAL_G1` format |
| Infer | `prepare_groot_model.py`, `run_groot_closed_loop.py` | Run GR00T with MuJoCo feedback |
| Check | `test_envs.py`, `replay_groot_episode.py` | Test tasks and render videos |

Successful source episodes are in `data/groot_source/`; final media is in `visualizations/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Test environments
MUJOCO_GL=egl python -m unittest test_envs

# Collect one fixed episode per task
MUJOCO_GL=egl python collect_demonstrations.py \
  --output-dir data/groot_source \
  --environment all --episodes 1 --seed 0 --variant fixed

# Export successful episodes
python export_groot_dataset.py \
  --input-dir data/groot_source \
  --output-dir /path/to/groot_g1_dataset
```

## GR00T Evaluation

Requires the official Isaac-GR00T repository and `nvidia/GR00T-N1.7-3B` checkpoint.

```bash
MUJOCO_GL=egl python run_groot_closed_loop.py \
  --groot-repo /path/to/Isaac-GR00T \
  --model-path /path/to/GR00T-N1.7-3B \
  --output visualizations/groot_environment1_pick_and_place
```

Submitted zero-shot result: 25 policy queries and 200 action targets. The integration ran successfully, but the policy did not contact the mug.

## Submission Videos

- [Environment 1: Pick and place](visualizations/submission_environment1_pick_and_place.mp4)
- [Environment 2: Object sorting](visualizations/submission_environment2_sort_and_grasp.mp4)
- [Environment 3: Tool use](visualizations/submission_environment3_grasp_and_sweep.mp4)
- [GR00T: Environment 1 policy evaluation](visualizations/groot_environment1_pick_and_place.mp4)
