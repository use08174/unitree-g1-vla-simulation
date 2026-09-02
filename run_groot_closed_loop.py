"""Run GR00T N1.7 in a receding-horizon MuJoCo control loop."""

import argparse
import json
from collections import deque
from pathlib import Path
import sys
import time

import imageio_ffmpeg
import mujoco
import numpy as np

from envs.tasks import G1Task1Env
from verify_model_free_control import JointPdController, RIGHT_HAND_JOINTS


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "visualizations/groot_environment1_closed_loop"
LEFT_HAND_JOINTS = tuple(name.replace("right_", "left_") for name in RIGHT_HAND_JOINTS)
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
STATE_KEYS = (
    "left_wrist_eef_9d",
    "right_wrist_eef_9d",
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "waist",
)


def joint_positions(model, data, joint_names):
    return np.asarray(
        [data.qpos[model.jnt_qposadr[model.joint(name).id]] for name in joint_names],
        dtype=np.float32,
    )


def wrist_pose(data, body_name):
    body = data.body(body_name)
    return np.concatenate((body.xpos, body.xmat.reshape(3, 3)[:2].reshape(6))).astype(
        np.float32
    )


def current_state(env):
    return {
        "left_wrist_eef_9d": wrist_pose(env.data, "left_wrist_yaw_link"),
        "right_wrist_eef_9d": wrist_pose(env.data, "right_wrist_yaw_link"),
        "left_hand": joint_positions(env.model, env.data, LEFT_HAND_JOINTS),
        "right_hand": joint_positions(env.model, env.data, RIGHT_HAND_JOINTS),
        "left_arm": joint_positions(env.model, env.data, LEFT_ARM_JOINTS),
        "right_arm": joint_positions(env.model, env.data, RIGHT_ARM_JOINTS),
        "waist": joint_positions(env.model, env.data, WAIST_JOINTS),
    }


def policy_observation(env, image_history):
    state = current_state(env)
    return {
        "video": {
            "ego_view": np.stack((image_history[0], image_history[-1]))[None]
        },
        "state": {key: state[key][None, None] for key in STATE_KEYS},
        "language": {
            "annotation.human.task_description": [[env.instruction]],
        },
    }


class MockPolicy:
    """Shape-compatible hold policy used to validate the physical rollout path."""

    def get_action(self, observation):
        state = observation["state"]
        action = {
            key: np.repeat(state[key], 40, axis=1).astype(np.float32)
            for key in STATE_KEYS
        }
        action["base_height_command"] = np.zeros((1, 40, 1), dtype=np.float32)
        action["navigate_command"] = np.zeros((1, 40, 3), dtype=np.float32)
        return action, {}

    def reset(self):
        return {}


def load_policy(args):
    if args.mock:
        return MockPolicy()
    if args.groot_repo:
        sys.path.insert(0, str(Path(args.groot_repo).resolve()))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GR00T closed-loop inference requires a CUDA GPU")
    free_bytes, _ = torch.cuda.mem_get_info(args.device)
    free_vram_gb = free_bytes / 1024**3
    if free_vram_gb < args.minimum_free_vram_gb:
        raise RuntimeError(
            f"GR00T needs an idle GPU; only {free_vram_gb:.1f} GiB is free "
            f"({args.minimum_free_vram_gb:.1f} GiB required)."
        )
    from gr00t.policy import Gr00tPolicy

    return Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag="REAL_G1",
        device=args.device,
        strict=True,
    )


def actuator_qpos_map(model):
    return {
        model.joint(model.actuator(index).trnid[0]).name: index
        for index in range(model.nu)
    }


def validate_action(action, execution_horizon):
    required = {
        "right_arm": 7,
        "right_hand": 7,
    }
    for key, dimension in required.items():
        if key not in action:
            raise KeyError(f"Policy action is missing {key!r}")
        value = np.asarray(action[key])
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != dimension:
            raise ValueError(f"Expected {key} shape (1, T, {dimension}), got {value.shape}")
        if value.shape[1] < execution_horizon:
            raise ValueError(
                f"{key} horizon {value.shape[1]} is shorter than {execution_horizon}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"Policy action {key!r} contains non-finite values")


def apply_policy_target(env, controller, base_target, action, chunk_index, max_delta):
    target = base_target.copy()
    joint_to_actuator = actuator_qpos_map(env.model)
    groups = (
        (RIGHT_ARM_JOINTS, np.asarray(action["right_arm"])[0, chunk_index]),
        (RIGHT_HAND_JOINTS, np.asarray(action["right_hand"])[0, chunk_index]),
    )
    for joint_names, values in groups:
        for joint_name, value in zip(joint_names, values):
            actuator_index = joint_to_actuator[joint_name]
            qpos_address = controller.all_qpos_addresses[actuator_index]
            joint_id = env.model.joint(joint_name).id
            current = env.data.qpos[qpos_address]
            lower, upper = env.model.jnt_range[joint_id]
            target[actuator_index] = np.clip(value, max(lower, current - max_delta), min(upper, current + max_delta))
    return target


def step_physics(env, torque):
    env.data.ctrl[:] = torque
    mujoco.mj_step(env.model, env.data)
    env.step_count += 1
    metrics = env._task_metrics()
    success = env._check_success(metrics)
    return success or env.step_count >= env.max_episode_steps, {
        "success": success,
        "time_limit_reached": env.step_count >= env.max_episode_steps,
        "metrics": metrics,
    }


def encode_video(path, images, fps=20):
    height, width = images[0].shape[:2]
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


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run(args):
    if not 1 <= args.execution_horizon <= 40:
        raise ValueError("--execution-horizon must be between 1 and 40")
    policy = load_policy(args)
    env = G1Task1Env()
    observation = env.reset(seed=args.seed, options={"variant": "fixed"})
    controller = JointPdController(env.model)
    base_target = env.data.qpos[controller.all_qpos_addresses].copy()
    image_history = deque([observation["rgb_image"]] * 21, maxlen=21)
    frames = [observation["rgb_image"]]
    state_samples = [np.concatenate(tuple(current_state(env).values()))]
    right_arm_actions = []
    right_hand_actions = []
    policy.reset()
    inference_times = []
    final_info = {"success": False, "metrics": env._task_metrics()}
    physics_steps_per_action = round((1.0 / 20.0) / env.model.opt.timestep)

    try:
        for query_index in range(args.max_queries):
            policy_input = policy_observation(env, image_history)
            start = time.perf_counter()
            action, _ = policy.get_action(policy_input)
            inference_times.append(time.perf_counter() - start)
            validate_action(action, args.execution_horizon)

            for chunk_index in range(args.execution_horizon):
                right_arm_actions.append(np.asarray(action["right_arm"])[0, chunk_index])
                right_hand_actions.append(np.asarray(action["right_hand"])[0, chunk_index])
                target = apply_policy_target(
                    env, controller, base_target, action, chunk_index, args.max_joint_delta
                )
                for _ in range(physics_steps_per_action):
                    torque = controller.torque_action(env.data, target)
                    done, final_info = step_physics(env, torque)
                    if done:
                        break
                observation = env._get_obs()
                image_history.append(observation["rgb_image"])
                frames.append(observation["rgb_image"])
                state_samples.append(np.concatenate(tuple(current_state(env).values())))
                if done:
                    break

            metrics = final_info["metrics"]
            print(
                f"query={query_index + 1} success={final_info['success']} "
                f"distance={metrics['mug_to_coaster_xy']:.3f} "
                f"displacement={metrics['mug_displacement']:.3f} "
                f"contacted={metrics['has_contacted']}"
            )
            if final_info["success"] or final_info.get("time_limit_reached"):
                break
    finally:
        env.close()

    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    encode_video(output_prefix.with_suffix(".mp4"), frames)
    result = {
        "success": bool(final_info["success"]),
        "metrics": final_info["metrics"],
        "queries": len(inference_times),
        "action_steps": len(frames) - 1,
        "mean_inference_seconds": float(np.mean(inference_times)),
        "mock_policy": args.mock,
    }
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        states=np.asarray(state_samples, dtype=np.float32),
        right_arm_actions=np.asarray(right_arm_actions, dtype=np.float32),
        right_hand_actions=np.asarray(right_hand_actions, dtype=np.float32),
        result=json.dumps(result, default=json_default),
    )
    print(json.dumps(result, indent=2, default=json_default))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--groot-repo", default="/scratch/sumins/Isaac-GR00T")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--max-queries", type=int, default=28)
    parser.add_argument("--max-joint-delta", type=float, default=0.12)
    parser.add_argument("--minimum-free-vram-gb", type=float, default=13.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--mock", action="store_true")
    run(parser.parse_args())