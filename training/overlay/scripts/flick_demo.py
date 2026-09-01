#!/usr/bin/env python3
"""Hand-scripted header-flick demo in the interactive viewer (mjpython).

Auto-loops forever: ball drops into the head-top tray → settles → scripted
neck/head snap launches it → pause to watch the flight → ball teleports back.

Physics honesty: the four head servos run the REALISTIC actuator model
(kp 0.55, ±0.6405 Nm = the 1.75 A current limit) so the flick speed is what
the real robot could do. The BASE IS PINNED (and the legs stiff) — the head
whip's recoil (38% of body mass!) otherwise topples the robot, which is
exactly the part the trained policy learns to manage. This demo showcases
the throw mechanics only.

Usage:
    uv run mjpython scripts/flick_demo.py [--slow 2] [--snap 1.6]
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from mjlab_microduck.robot.microduck_constants import (
    TOSS_TRAY_OFFSET_FROM_ROOT,
    build_toss_scene_spec,
)

DEFAULT = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530,
                    0.3491, 0.3491, 0.0, 0.0,
                    0.0, 0.0873, 0.4579, 0.0049, -0.4530])

parser = argparse.ArgumentParser()
parser.add_argument("--slow", type=float, default=1.0, help="wall-clock slowdown factor")
parser.add_argument("--snap", type=float, default=1.6, help="flick amplitude (rad added to neck+head pitch)")
args = parser.parse_args()

model = build_toss_scene_spec().compile()
model.opt.timestep = 0.005
for i in range(model.nu):
    head = 5 <= i <= 8
    kp, tau, kv = (0.55, 0.6405, -0.05) if head else (15.0, 20.0, -0.5)
    model.actuator_gainprm[i, 0] = kp
    model.actuator_biasprm[i, 1] = -kp
    model.actuator_biasprm[i, 2] = kv
    model.actuator_forcerange[i] = [-tau, tau]

data = mujoco.MjData(model)
fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
adr = model.jnt_qposadr[fj]
jq = [model.jnt_qposadr[model.actuator(i).trnid[0]] for i in range(model.nu)]
bj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
badr = model.jnt_qposadr[bj]
bvadr = model.jnt_dofadr[bj]
ball_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toss_ball")

def reset_all():
    data.qpos[adr:adr + 3] = [0.0, 0.0, 0.117]
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    data.qvel[:6] = 0
    for i, q in enumerate(jq):
        data.qpos[q] = DEFAULT[i]
        data.qvel[model.jnt_dofadr[model.actuator(i).trnid[0]]] = 0.0
    data.ctrl[:] = DEFAULT
    off = TOSS_TRAY_OFFSET_FROM_ROOT
    data.qpos[badr:badr + 3] = [off[0], off[1], 0.117 + off[2] + 0.018]
    data.qpos[badr + 3:badr + 7] = [1, 0, 0, 0]
    data.qvel[bvadr:bvadr + 6] = 0
    mujoco.mj_forward(model, data)

reset_all()
print("Flick demo: settle (1.5s) -> flick -> watch (2.5s) -> repeat. Ctrl-C to quit.")

SETTLE, FLICK, WATCH = 1.5, 1.2, 2.5
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    phase, t = "settle", 0.0
    while viewer.is_running():
        step_start = time.time()
        if phase == "settle" and t >= SETTLE:
            data.ctrl[5] = DEFAULT[5] + args.snap
            data.ctrl[6] = DEFAULT[6] + args.snap
            phase, t = "flick", 0.0
            print("FLICK!")
        elif phase == "flick" and t >= FLICK:
            phase, t = "watch", 0.0
        elif phase == "watch" and t >= WATCH:
            bp = data.xpos[ball_b]
            print(f"ball landed at x={bp[0]:+.2f}m — resetting")
            reset_all()
            phase, t = "settle", 0.0
        for _ in range(4):
            data.qpos[adr:adr + 7] = [0.0, 0.0, 0.117, 1, 0, 0, 0]
            data.qvel[:6] = 0
            mujoco.mj_step(model, data)
        t += 0.02
        viewer.sync()
        elapsed = time.time() - step_start
        sleep = 0.02 * args.slow - elapsed
        if sleep > 0:
            time.sleep(sleep)
