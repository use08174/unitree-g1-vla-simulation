"""Convert collected G1 MuJoCo episodes to the GR00T LeRobot v2 format."""

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCENES = {
    1: ROOT / "unitree_robots/g1/g1_task1_scene.xml",
    2: ROOT / "unitree_robots/g1/g1_task2_scene.xml",
    3: ROOT / "unitree_robots/g1/g1_task3_scene.xml",
}
STATE_LAYOUT = {
    "left_wrist_eef_9d": (0, 9),
    "right_wrist_eef_9d": (9, 18),
    "left_hand": (18, 25),
    "right_hand": (25, 32),
    "left_arm": (32, 39),
    "right_arm": (39, 46),
    "waist": (46, 49),
}
ACTION_LAYOUT = {
    **STATE_LAYOUT,
    "base_height_command": (49, 50),
    "navigate_command": (50, 53),
}
MINIMUM_FPS = 20.0
ACTION_HORIZON = 40
LEFT_HAND_JOINTS = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)
RIGHT_HAND_JOINTS = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
LEFT_ARM_JOINTS = tuple(
    f"left_{name}_joint"
    for name in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
)
RIGHT_ARM_JOINTS = tuple(name.replace("left_", "right_") for name in LEFT_ARM_JOINTS)
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def write_jsonl(path, values):
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="ascii"
    )


def environment_id(metadata):
    return int(metadata["environment"].rsplit(" ", 1)[-1])


def episode_succeeded(input_path):
    with np.load(input_path) as episode:
        return bool(episode["task_success"])


def eef_pose(data, body_name):
    body = data.body(body_name)
    rotation_6d = body.xmat.reshape(3, 3)[:2].reshape(6)
    return np.concatenate((body.xpos, rotation_6d)).astype(np.float32)


def joint_positions(model, qpos, joint_names):
    return np.asarray(
        [qpos[model.jnt_qposadr[model.joint(name).id]] for name in joint_names],
        dtype=np.float32,
    )


def has_joint(model, joint_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0


def episode_states(model, qpos_samples):
    data = mujoco.MjData(model)
    states = []
    base_heights = []
    for qpos in qpos_samples:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        left_hand = (
            joint_positions(model, qpos, LEFT_HAND_JOINTS)
            if has_joint(model, LEFT_HAND_JOINTS[0])
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            joint_positions(model, qpos, RIGHT_HAND_JOINTS)
            if has_joint(model, RIGHT_HAND_JOINTS[0])
            else np.zeros(7, dtype=np.float32)
        )
        states.append(
            np.concatenate(
                (
                    eef_pose(data, "left_wrist_yaw_link"),
                    eef_pose(data, "right_wrist_yaw_link"),
                    left_hand,
                    right_hand,
                    joint_positions(model, qpos, LEFT_ARM_JOINTS),
                    joint_positions(model, qpos, RIGHT_ARM_JOINTS),
                    joint_positions(model, qpos, WAIST_JOINTS),
                )
            ).astype(np.float32)
        )
        base_heights.append(qpos[2])
    return np.stack(states), np.asarray(base_heights, dtype=np.float32)


def target_actions(states, base_heights):
    next_states = np.concatenate((states[1:], states[-1:]))
    next_heights = np.concatenate((base_heights[1:], base_heights[-1:]))[:, None]
    navigation = np.zeros((len(states), 3), dtype=np.float32)
    return np.concatenate((next_states, next_heights, navigation), axis=1)


def feature_statistics(values):
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def encode_video(path, images, fps):
    height, width = images.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        output_params=["-pix_fmt", "yuv420p"],
    )
    writer.send(None)
    try:
        for image in images:
            writer.send(np.ascontiguousarray(image, dtype=np.uint8))
    finally:
        writer.close()


def convert_episode(input_path, output_directory, episode_index, global_offset):
    with np.load(input_path) as episode:
        metadata = json.loads(str(episode["metadata"]))
        images = episode["rgb_images"].copy()
        qpos = episode["qpos"].copy()
        successful = bool(episode["task_success"])

    scene_path = SCENES[environment_id(metadata)]
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    states, base_heights = episode_states(model, qpos)
    actions = target_actions(states, base_heights)
    frame_count = len(states)
    fps = float(metadata["control_frequency_hz"])
    if fps < MINIMUM_FPS:
        raise ValueError(
            f"{input_path} is sampled at {fps:g} Hz; REAL_G1 requires at least "
            f"{MINIMUM_FPS:g} Hz for its 40-step action horizon. Recollect the episode."
        )
    if frame_count <= ACTION_HORIZON:
        raise ValueError(
            f"{input_path} has {frame_count} frames; more than {ACTION_HORIZON} are required."
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"{input_path} contains non-finite state or action values")

    data_path = output_directory / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
    video_path = (
        output_directory
        / "videos/chunk-000/observation.images.ego_view"
        / f"episode_{episode_index:06d}.mp4"
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(
        {
            "observation.state": list(states),
            "action": list(actions),
            "timestamp": np.arange(frame_count, dtype=np.float32) / fps,
            "frame_index": np.arange(frame_count, dtype=np.int64),
            "episode_index": np.full(frame_count, episode_index, dtype=np.int64),
            "index": np.arange(global_offset, global_offset + frame_count, dtype=np.int64),
            "task_index": np.full(frame_count, environment_id(metadata) - 1, dtype=np.int64),
            "annotation.human.task_description": np.full(
                frame_count, environment_id(metadata) - 1, dtype=np.int64
            ),
            "next.reward": np.zeros(frame_count, dtype=np.float32),
            "next.done": np.arange(frame_count) == frame_count - 1,
        }
    )
    dataframe.to_parquet(data_path, index=False)
    encode_video(video_path, images, fps)
    return metadata, frame_count, images.shape[1:], states, actions, successful


def main(args):
    input_directory = Path(args.input_dir)
    output_directory = Path(args.output_dir)
    episode_paths = sorted(input_directory.glob("environment*/episode_*.npz"))
    if not episode_paths:
        raise SystemExit(f"No episodes found under {input_directory}")
    if not args.include_failed:
        failed_paths = [path for path in episode_paths if not episode_succeeded(path)]
        for failed_path in failed_paths:
            print(f"skipping failed demonstration: {failed_path}")
        episode_paths = [path for path in episode_paths if path not in failed_paths]
        if not episode_paths:
            raise SystemExit("No successful demonstrations found; recollect before exporting.")

    records = []
    tasks = {}
    total_frames = 0
    image_shape = None
    fps = None
    all_states = []
    all_actions = []
    for episode_index, input_path in enumerate(episode_paths):
        metadata, frame_count, image_shape, states, actions, successful = convert_episode(
            input_path, output_directory, episode_index, total_frames
        )
        fps = float(metadata["control_frequency_hz"])
        task_index = environment_id(metadata) - 1
        tasks[task_index] = metadata["instruction"]
        records.append(
            {
                "episode_index": episode_index,
                "tasks": [metadata["instruction"]],
                "length": frame_count,
                "successful": successful,
                "source_seed": metadata["seed"],
            }
        )
        all_states.append(states)
        all_actions.append(actions)
        total_frames += frame_count
        print(
            f"converted {input_path} -> episode {episode_index} "
            f"({frame_count} frames, successful={successful})"
        )

    meta_directory = output_directory / "meta"
    meta_directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(meta_directory / "episodes.jsonl", records)
    write_jsonl(
        meta_directory / "tasks.jsonl",
        [{"task_index": key, "task": value} for key, value in sorted(tasks.items())],
    )
    write_json(
        meta_directory / "modality.json",
        {
            "state": {
                key: {"start": start, "end": end}
                for key, (start, end) in STATE_LAYOUT.items()
            },
            "action": {
                key: {"start": start, "end": end}
                for key, (start, end) in ACTION_LAYOUT.items()
            },
            "video": {
                "ego_view": {"original_key": "observation.images.ego_view"}
            },
            "annotation": {
                "human.task_description": {"original_key": "task_index"}
            },
        },
    )
    height, width, channels = image_shape
    write_json(
        meta_directory / "info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "unitree_g1",
            "total_episodes": len(records),
            "total_frames": total_frames,
            "total_tasks": len(tasks),
            "fps": fps,
            "chunks_size": 1000,
            "splits": {"train": f"0:{len(records)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.ego_view": {
                    "dtype": "video",
                    "shape": [height, width, channels],
                },
                "observation.state": {"dtype": "float32", "shape": [49]},
                "action": {"dtype": "float32", "shape": [53]},
                "task_index": {"dtype": "int64", "shape": [1]},
            },
        },
    )
    write_json(
        meta_directory / "stats.json",
        {
            "observation.state": feature_statistics(np.concatenate(all_states)),
            "action": feature_statistics(np.concatenate(all_actions)),
        },
    )
    print(f"GR00T LeRobot v2 dataset written to {output_directory}")
    print("Generated dataset statistics in meta/stats.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/groot_source")
    parser.add_argument("--output-dir", default=ROOT / "data/groot_lerobot")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include episodes that fail the full task criteria (diagnostics only).",
    )
    main(parser.parse_args())