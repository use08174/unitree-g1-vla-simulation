# Demonstration Data

`groot_source/` contains one successful fixed-layout MuJoCo episode for each task.

Each NPZ episode stores 20 Hz RGB observations, robot state (`qpos`, `qvel`, wrist poses), torque actions, interaction labels, language instructions, task metrics, and metadata. Objects remain free MuJoCo bodies and are moved only through physical contact.

## Collect

```bash
MUJOCO_GL=egl python -m scripts.data_pipeline.collect_demonstrations \
  --output-dir data/groot_source \
  --environment all --episodes 1 --seed 0 --variant fixed
```

## Export to GR00T

```bash
python -m scripts.data_pipeline.export_groot_dataset \
  --input-dir data/groot_source \
  --output-dir /path/to/groot_g1_dataset
```

The exporter converts the raw episodes to LeRobot format using GR00T's `REAL_G1` 49D state and 53D action layout. Only episodes that satisfy the full task success criteria are exported.

See the project-level `README.md` for the complete collection, inference, replay, and visualization pipeline.