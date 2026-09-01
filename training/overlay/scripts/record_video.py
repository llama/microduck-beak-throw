#!/usr/bin/env python3
"""Record a clean, overlay-free video of a trained policy.

Bypasses the interactive viewers entirely (the native viewer hardcodes the
world-axes overlay; viser has a version incompatibility): builds the play env
with render_mode="rgb_array", steps the policy, and writes frames straight to
an mp4 via the offscreen renderer — no GUI, no axes, full camera control.

Usage:
    uv run python scripts/record_video.py Mjlab-BeakThrow-Flat-MicroDuck \
        --checkpoint-file <model.pt> --steps 300 --out throw.mp4 \
        --width 1920 --height 1080 --azimuth 130 --elevation -12 --distance 1.1
"""
import argparse
from dataclasses import asdict
from pathlib import Path

import imageio
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.rl.runner import MjlabOnPolicyRunner

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument("--checkpoint-file", required=True)
parser.add_argument("--steps", type=int, default=300, help="control steps at 50Hz (300 = 6s)")
parser.add_argument("--episodes", type=int, default=1,
                    help="record this many separately reset episodes")
parser.add_argument("--steps-per-episode", type=int, default=None,
                    help="steps per episode (default: the task time limit)")
parser.add_argument("--out", default="duck_video.mp4")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--azimuth", type=float, default=None)
parser.add_argument("--orbit-degrees", type=float, default=0.0,
                    help="smoothly add this many azimuth degrees over the complete video")
parser.add_argument(
    "--heading-relative-azimuths",
    default=None,
    help=(
        "comma-separated fixed camera azimuth offsets from each episode's reset "
        "heading; 180 looks into the throw, +90/-90 are opposing side views"
    ),
)
parser.add_argument("--elevation", type=float, default=None)
parser.add_argument("--distance", type=float, default=None)
parser.add_argument("--fps", type=int, default=50, help="output fps (50 = realtime, 25 = half speed)")
parser.add_argument("--device", default="cpu")
parser.add_argument("--seed", type=int, default=42,
                    help="fixed reset/domain-randomization seed for comparable takes")
parser.add_argument("--episode-length", type=float, default=None,
                    help="override episode length in seconds (one long take)")
parser.add_argument("--no-early-termination", action="store_true",
                    help="drop all terminations except time_out — nothing cuts the take")
args = parser.parse_args()

heading_relative_azimuths = None
if args.heading_relative_azimuths:
    heading_relative_azimuths = [
        float(value.strip())
        for value in args.heading_relative_azimuths.split(",")
        if value.strip()
    ]
    if len(heading_relative_azimuths) != args.episodes:
        parser.error(
            "--heading-relative-azimuths must contain one value per episode"
        )

env_cfg = load_env_cfg(args.task, play=True) if "play" in load_env_cfg.__code__.co_varnames else load_env_cfg(args.task)
env_cfg.seed = args.seed
agent_cfg = load_rl_cfg(args.task)
env_cfg.scene.num_envs = 1
if args.episodes < 1:
    parser.error("--episodes must be at least 1")
if args.episodes > 1:
    # Preserve each episode's terminal frame; reset explicitly between takes.
    env_cfg.auto_reset = False
if args.episode_length is not None:
    env_cfg.episode_length_s = args.episode_length
if args.no_early_termination:
    for name in list(env_cfg.terminations.keys()):
        term = env_cfg.terminations[name]
        if not getattr(term, "time_out", False) or name != "time_out":
            if name != "time_out":
                del env_cfg.terminations[name]
env_cfg.viewer.height = args.height
env_cfg.viewer.width = args.width
for name in ("azimuth", "elevation", "distance"):
    v = getattr(args, name)
    if v is not None and hasattr(env_cfg.viewer, name):
        setattr(env_cfg.viewer, name, v)

env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
runner.load(args.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=args.device)
policy = runner.get_inference_policy(device=args.device)

obs, _ = wrapped.reset()
frames = []
steps_per_episode = args.steps
if args.episodes > 1:
    steps_per_episode = args.steps_per_episode or wrapped.max_episode_length
total_steps = args.episodes * steps_per_episode
start_azimuth = env_cfg.viewer.azimuth


def camera_azimuth_for_episode(episode: int) -> float | None:
    if heading_relative_azimuths is None:
        return None
    robot = env.scene["robot"]
    root = env.sim.data.qpos[0, robot.indexing.free_joint_q_adr]
    qw, qx, qy, qz = root[3], root[4], root[5], root[6]
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    heading_deg = float(torch.rad2deg(yaw).item())
    return heading_deg + heading_relative_azimuths[episode]


with torch.no_grad():
    for episode in range(args.episodes):
        if episode:
            wrapped.seed(args.seed + episode)
            obs, _ = wrapped.reset()
        fixed_azimuth = camera_azimuth_for_episode(episode)
        if fixed_azimuth is not None:
            print(
                f"  episode {episode + 1}: fixed camera azimuth "
                f"{fixed_azimuth:.1f} degrees"
            )
        for step in range(steps_per_episode):
            actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions)
            global_step = episode * steps_per_episode + step
            if env._offline_renderer is not None:
                progress = global_step / max(total_steps - 1, 1)
                env._offline_renderer._cam.azimuth = (
                    fixed_azimuth
                    if fixed_azimuth is not None
                    else start_azimuth + args.orbit_degrees * progress
                )
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            if (global_step + 1) % 50 == 0:
                print(
                    f"  {global_step + 1}/{total_steps} steps, "
                    f"episode {episode + 1}/{args.episodes}, {len(frames)} frames"
                )

out = Path(args.out)
imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=1)
print(f"wrote {out} ({len(frames)} frames @ {args.fps}fps, {args.width}x{args.height})")
