"""Render seeded previews of all three G1 task environments to MP4."""

import argparse
from pathlib import Path
import subprocess
import textwrap

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.tasks import G1Task1Env, G1Task2Env, G1Task3Env


FFMPEG = Path("/scratch/sumins/ffmpeg-runtime/bin/ffmpeg")
DEFAULT_OUTPUT = Path("/scratch/sumins/groot_visualizations/g1_environment_overview.mp4")
TASKS = (
    ("ENVIRONMENT 1 / PICK AND PLACE", G1Task1Env),
    ("ENVIRONMENT 2 / OBJECT SORTING", G1Task2Env),
    ("ENVIRONMENT 3 / BILATERAL TOOL USE", G1Task3Env),
)


def load_font(size, bold=False):
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    font_path = Path("/usr/share/fonts/truetype/dejavu") / font_name
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def add_overlay(frame, title, instruction, seed, view_name):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    wrapped_instruction = "\n".join(textwrap.wrap(instruction, width=64))
    draw.rectangle((0, 0, image.width, 112), fill=(10, 13, 17, 210))
    draw.rectangle((0, image.height - 34, image.width, image.height), fill=(10, 13, 17, 190))
    draw.text((20, 13), title, font=load_font(24, bold=True), fill=(255, 255, 255, 255))
    draw.multiline_text(
        (20, 49), wrapped_instruction, font=load_font(17), fill=(225, 231, 238, 255), spacing=3
    )
    draw.text(
        (20, image.height - 27),
        f"ENVIRONMENT PREVIEW  |  RESET VARIANT: SEED {seed}  |  {view_name}",
        font=load_font(14, bold=True),
        fill=(120, 220, 180, 255),
    )
    return np.asarray(image)


def render_task_frames(title, env_class, seeds, fps, seconds_per_seed, width, height):
    env = env_class()
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    frames_per_seed = max(2, round(fps * seconds_per_seed))
    head_frames = max(1, frames_per_seed // 3)
    orbit_camera = mujoco.MjvCamera()
    orbit_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    orbit_camera.lookat[:] = [0.42, 0.0, 0.82]
    orbit_camera.distance = 1.15
    orbit_camera.elevation = -24.0

    try:
        for seed in seeds:
            env.reset(seed=seed)
            for _ in range(25):
                mujoco.mj_step(env.model, env.data)

            for frame_index in range(frames_per_seed):
                if frame_index < head_frames:
                    view_name = "HEAD CAMERA"
                    renderer.update_scene(env.data, camera="head_camera")
                else:
                    progress = (frame_index - head_frames) / max(1, frames_per_seed - head_frames - 1)
                    orbit_camera.azimuth = 135.0 + 90.0 * progress
                    view_name = "ORBIT VIEW"
                    renderer.update_scene(env.data, camera=orbit_camera)
                yield add_overlay(
                    renderer.render(), title, env.instruction, seed, view_name
                )
    finally:
        renderer.close()
        env.close()


def render_video(output, seeds, fps, seconds_per_seed, width, height):
    if not FFMPEG.is_file():
        raise FileNotFoundError(f"FFmpeg runtime not found: {FFMPEG}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(FFMPEG),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    frame_count = 0
    try:
        for title, env_class in TASKS:
            for frame in render_task_frames(
                title, env_class, seeds, fps, seconds_per_seed, width, height
            ):
                process.stdin.write(frame.tobytes())
                frame_count += 1
    finally:
        if process.stdin:
            process.stdin.close()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg exited with status {return_code}")
    return frame_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 17])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds-per-seed", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    frame_count = render_video(
        args.output,
        args.seeds,
        args.fps,
        args.seconds_per_seed,
        args.width,
        args.height,
    )
    duration = frame_count / args.fps
    print(f"Rendered {frame_count} frames ({duration:.1f}s) to {args.output}")


if __name__ == "__main__":
    main()