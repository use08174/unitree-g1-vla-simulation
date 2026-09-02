# Unitree G1 VLA Simulation

Three MuJoCo manipulation tasks with model-free verification, multimodal rollout collection, and GR00T N1.7 evaluation.

## Tasks

1. Pick up a red mug and place it on a black coaster.
2. Place an apple and soda can in their matching targets.
3. Grasp a brush and sweep a block into the green target.

Each environment supports fixed and randomized layouts. Objects remain free MuJoCo bodies; the controllers do not teleport or attach them.

## Pipeline and Files

```text
Task scenes -> Model-free control -> Raw NPZ data -> REAL_G1 export
    -> GR00T inference -> MuJoCo feedback -> Videos and results
```

1. **Define the tasks**
  - `unitree_robots/g1/g1_task*_scene.xml` defines the G1, table, objects, cameras, and collisions.
  - `envs/base_env.py` provides the MuJoCo reset, observation, and step interface.
  - `envs/tasks.py` adds instructions, randomized layouts, metrics, and success criteria.

2. **Generate model-free actions**
  - `verify_model_free_control.py` converts Cartesian hand targets to joint targets with Jacobian IK and tracks them with PD torque control.
  - `collect_demonstrations.py` defines each task's waypoint sequence and executes it in MuJoCo.

3. **Collect raw rollout data**
  - `collect_demonstrations.py` records synchronized RGB, `qpos`, `qvel`, torque actions, wrist poses, interaction labels, language, and metadata at 20 Hz.
  - `data/groot_source/` contains one successful fixed-layout NPZ episode per task.

4. **Convert data for GR00T**
  - `groot_real_g1_config.py` defines GR00T's 49D state and 53D action structure.
  - `export_groot_dataset.py` maps the raw NPZ episodes to the `REAL_G1` LeRobot dataset format.

5. **Run policy inference**
  - `prepare_groot_model.py` downloads the GR00T N1.7 checkpoint and applies the Tesla T4 setting.
  - `run_groot_closed_loop.py` sends images, language, and G1 state to GR00T, applies predicted targets through the PD controller, and returns the new MuJoCo observations to the policy.

6. **Validate and visualize**
  - `test_envs.py` checks environment behavior and task success conditions.
  - `replay_groot_episode.py` renders saved NPZ episodes as videos.
  - `render_environment_preview.py` renders randomized setup variants.
  - `visualizations/` contains the report figures, rule-based videos, and GR00T evaluation video.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Headless rendering uses `MUJOCO_GL=egl`.

## Verify and Collect

```bash
MUJOCO_GL=egl python -m unittest test_envs

MUJOCO_GL=egl python collect_demonstrations.py \
  --output-dir data/groot_source \
  --environment all --episodes 1 --seed 0 --variant fixed
```

The raw NPZ files preserve RGB images, full MuJoCo state, torque actions, wrist poses, interaction labels, language, and metadata. They are converted to GR00T format during export:

```bash
python export_groot_dataset.py \
  --input-dir data/groot_source \
  --output-dir /path/to/groot_g1_dataset
```

## GR00T Evaluation

GR00T evaluation requires NVIDIA's official Isaac-GR00T repository, its dependencies, and the `nvidia/GR00T-N1.7-3B` checkpoint.

```bash
MUJOCO_GL=egl python run_groot_closed_loop.py \
  --groot-repo /path/to/Isaac-GR00T \
  --model-path /path/to/GR00T-N1.7-3B \
  --output visualizations/groot_environment1_pick_and_place
```

The submitted zero-shot rollout executed 25 policy queries and 200 action targets. The integration worked, but the policy did not contact the mug.

## Submission Videos

- [Environment 1: Pick and place](visualizations/submission_environment1_pick_and_place.mp4)
- [Environment 2: Object sorting](visualizations/submission_environment2_sort_and_grasp.mp4)
- [Environment 3: Tool use](visualizations/submission_environment3_grasp_and_sweep.mp4)
- [GR00T: Environment 1 policy evaluation](visualizations/groot_environment1_pick_and_place.mp4)
