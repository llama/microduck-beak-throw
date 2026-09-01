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
parser.add_argument("--out", default="duck_video.mp4")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--azimuth", type=float, default=None)
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

env_cfg = load_env_cfg(args.task, play=True) if "play" in load_env_cfg.__code__.co_varnames else load_env_cfg(args.task)
env_cfg.seed = args.seed
agent_cfg = load_rl_cfg(args.task)
env_cfg.scene.num_envs = 1
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
with torch.no_grad():
    for i in range(args.steps):
        actions = policy(obs)
        obs, _, _, _ = wrapped.step(actions)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.steps} steps, {len(frames)} frames")

out = Path(args.out)
imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=1)
print(f"wrote {out} ({len(frames)} frames @ {args.fps}fps, {args.width}x{args.height})")
