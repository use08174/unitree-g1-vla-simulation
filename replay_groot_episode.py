"""Replay a collected G1 episode in MuJoCo or render it to MP4."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
SCENES = {
    "Environment 1": ROOT / "unitree_robots/g1/g1_task1_scene.xml",
    "Environment 2": ROOT / "unitree_robots/g1/g1_task2_scene.xml",
    "Environment 3": ROOT / "unitree_robots/g1/g1_task3_scene.xml",
}


def load_episode(path):
    with np.load(path) as episode:
        return (
            json.loads(str(episode["metadata"])),
            episode["qpos"].copy(),
            episode["qvel"].copy(),
        )


def set_frame(model, data, qpos, qvel):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)


def render_video(
    model,
    data,
    qpos,
    qvel,
    output_path,
    fps,
    width,
    height,
    camera,
    predicted_right_arm=None,
    predicted_right_hand=None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    has_predictions = predicted_right_arm is not None or predicted_right_hand is not None
    output_width = width * 2 if has_predictions else width
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{output_width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        rendered_frames = len(qpos)
        if not has_predictions:
            for frame_qpos, frame_qvel in zip(qpos, qvel):
                set_frame(model, data, frame_qpos, frame_qvel)
                renderer.update_scene(data, camera=camera)
                process.stdin.write(renderer.render().tobytes())
        else:
            predicted_data = mujoco.MjData(model)
            predicted_renderer = mujoco.Renderer(model, height=height, width=width)
            right_arm_addresses = np.array(
                [
                    model.jnt_qposadr[model.joint(f"right_{name}_joint").id]
                    for name in (
                        "shoulder_pitch",
                        "shoulder_roll",
                        "shoulder_yaw",
                        "elbow",
                        "wrist_roll",
                        "wrist_pitch",
                        "wrist_yaw",
                    )
                ]
            )
            right_hand_addresses = np.array(
                [
                    model.jnt_qposadr[model.joint(f"right_hand_{name}_joint").id]
                    for name in (
                        "thumb_0",
                        "thumb_1",
                        "thumb_2",
                        "index_0",
                        "index_1",
                        "middle_0",
                        "middle_1",
                    )
                ]
            )
            target_qpos = np.concatenate((qpos[1:], qpos[-1:]))
            target_qvel = np.concatenate((qvel[1:], qvel[-1:]))
            rendered_frames = min(
                len(target_qpos),
                len(predicted_right_arm) if predicted_right_arm is not None else len(target_qpos),
                len(predicted_right_hand) if predicted_right_hand is not None else len(target_qpos),
            )
            for frame_index in range(rendered_frames):
                frame_qpos = target_qpos[frame_index]
                frame_qvel = target_qvel[frame_index]
                set_frame(model, data, frame_qpos, frame_qvel)
                renderer.update_scene(data, camera=camera)
                ground_truth_image = renderer.render()

                predicted_qpos = frame_qpos.copy()
                if predicted_right_arm is not None:
                    predicted_qpos[right_arm_addresses] = predicted_right_arm[frame_index]
                if predicted_right_hand is not None:
                    predicted_qpos[right_hand_addresses] = predicted_right_hand[frame_index]
                set_frame(model, predicted_data, predicted_qpos, frame_qvel)
                predicted_renderer.update_scene(predicted_data, camera=camera)
                predicted_image = predicted_renderer.render()

                comparison = Image.fromarray(
                    np.concatenate((ground_truth_image, predicted_image), axis=1)
                )
                labels = ImageDraw.Draw(comparison)
                labels.text((12, 12), "GROUND TRUTH", fill=(50, 220, 80))
                labels.text((width + 12, 12), "GR00T PREDICTED RIGHT ARM", fill=(255, 90, 60))
                process.stdin.write(np.asarray(comparison).tobytes())
            predicted_renderer.close()
    finally:
        if process.stdin:
            process.stdin.close()
        return_code = process.wait()
        renderer.close()
    if return_code:
        raise SystemExit(f"FFmpeg exited with status {return_code}")
    print(f"Rendered {rendered_frames} frames to {output_path}")


def launch_viewer(model, data, qpos, qvel, fps):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for frame_qpos, frame_qvel in zip(qpos, qvel):
                if not viewer.is_running():
                    return
                set_frame(model, data, frame_qpos, frame_qvel)
                viewer.sync()
                time.sleep(1.0 / fps)


def main(args):
    metadata, qpos, qvel = load_episode(args.episode)
    scene_path = SCENES[metadata["environment"]]
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    for body_name, position in metadata.get("fixed_body_positions", {}).items():
        model.body_pos[model.body(body_name).id] = position
    data = mujoco.MjData(model)
    fps = float(metadata["control_frequency_hz"])
    if qpos.shape[1] != model.nq or qvel.shape[1] != model.nv:
        raise ValueError(
            f"Episode/model mismatch: qpos={qpos.shape[1]}/{model.nq}, "
            f"qvel={qvel.shape[1]}/{model.nv}"
        )
    if args.viewer:
        if args.predictions:
            raise ValueError("--predictions is supported by MP4 rendering, not --viewer")
        launch_viewer(model, data, qpos, qvel, fps)
    else:
        predicted_right_arm = None
        predicted_right_hand = None
        if args.predictions:
            with np.load(args.predictions) as predictions:
                predicted_right_arm = predictions["pred_right_arm"].copy()
                if "pred_right_hand" in predictions:
                    predicted_right_hand = predictions["pred_right_hand"].copy()
            if predicted_right_arm.ndim != 2 or predicted_right_arm.shape[1] != 7:
                raise ValueError(
                    "Expected predicted right arm shape (frames, 7), "
                    f"got {predicted_right_arm.shape}"
                )
            if predicted_right_hand is not None and predicted_right_hand.shape != predicted_right_arm.shape:
                raise ValueError(
                    "Predicted right hand and arm must have the same shape; "
                    f"got {predicted_right_hand.shape} and {predicted_right_arm.shape}"
                )
        render_video(
            model,
            data,
            qpos,
            qvel,
            args.output,
            fps,
            args.width,
            args.height,
            args.camera,
            predicted_right_arm,
            predicted_right_hand,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        type=Path,
        default=ROOT / "data/groot_source/environment1/episode_0000.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/scratch/sumins/groot_visualizations/task1_demonstration.mp4"),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--viewer", action="store_true")
    main(parser.parse_args())