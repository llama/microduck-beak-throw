#!/usr/bin/env python3
"""Headless, physics-level evaluation for Header toss policies.

Unlike reward logs, this script measures the thing the task is meant to do:
the ball's velocity at physical tray separation and its forward distance to
the first floor contact.  It uses the same plain-MuJoCo rehearsal model and
61D observation builder as ``infer_policy.py``.

Examples::

    uv run scripts/eval_header.py header_throw.onnx --episodes 10
    uv run scripts/eval_header.py header_throw.onnx --hold header_hold.onnx
    uv run scripts/eval_header.py header_throw_v9.onnx header_throw.onnx
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from infer_policy import DEFAULT_POSE, PolicyInference
from mjlab_microduck.robot.microduck_constants import (
    TOSS_BALL_RADIUS,
    TOSS_TRAY_OFFSET_FROM_ROOT,
    build_toss_scene_spec,
)


CONTROL_DT = 0.02
PHYSICS_DT = 0.005
DECIMATION = round(CONTROL_DT / PHYSICS_DT)


@dataclass
class TossResult:
    policy: str
    episode: int
    released: bool
    landed: bool
    release_time_s: float
    release_forward_mps: float
    release_up_mps: float
    release_speed_mps: float
    apex_m: float
    carry_forward_m: float
    lateral_m: float
    trunk_min_z_m: float
    trunk_max_tilt_deg: float


def _pair_in_contacts(model: mujoco.MjModel, data: mujoco.MjData, a: int, bs: set[int]) -> bool:
    for i in range(data.ncon):
        contact = data.contact[i]
        if (contact.geom1 == a and contact.geom2 in bs) or (
            contact.geom2 == a and contact.geom1 in bs
        ):
            return True
    return False


def _set_current_limit(model: mujoco.MjModel, current_limit: float) -> None:
    if current_limit <= 0.0:
        return
    from bam.model import load_model

    torque_limit = load_model(motor_name="xl330", model="m6").kt.value * current_limit
    model.actuator_forcerange[:, 0] = -torque_limit
    model.actuator_forcerange[:, 1] = torque_limit
    model.actuator_forcelimited[:] = 1


def _tilt_deg(data: mujoco.MjData, trunk_body_id: int) -> float:
    # Body-frame +z expressed in world coordinates is the third rotation-matrix column.
    cos_tilt = float(np.clip(data.xmat[trunk_body_id].reshape(3, 3)[2, 2], -1.0, 1.0))
    return math.degrees(math.acos(cos_tilt))


class HeaderEvaluator:
    @classmethod
    def create(cls, first_policy: Path, current_limit: float) -> "HeaderEvaluator":
        self = cls.__new__(cls)
        self.model = build_toss_scene_spec().compile()
        self.model.opt.timestep = PHYSICS_DT
        _set_current_limit(self.model, current_limit)
        self.data = mujoco.MjData(self.model)
        self.policy = PolicyInference(
            self.model,
            self.data,
            walking_onnx_path=str(first_policy),
            action_scale=1.0,
            new_cmd_obs=True,
            use_projected_gravity=True,
        )

        self.trunk_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
        )
        self.trunk_qpos_adr = int(self.model.jnt_qposadr[self.trunk_joint_id])
        self.trunk_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"
        )
        ball_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
        )
        self.ball_qpos_adr = int(self.model.jnt_qposadr[ball_joint_id])
        self.ball_qvel_adr = int(self.model.jnt_dofadr[ball_joint_id])
        self.ball_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "toss_ball"
        )
        self.ball_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "toss_ball_geom"
        )
        self.tray_geom_ids = {
            i
            for i in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
                "tray_"
            )
        }
        self.floor_geom_ids = {
            i
            for i in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
            in {"floor", "terrain"}
        }
        if not self.floor_geom_ids:
            # The rehearsal scene's plane is normally named floor.  A body-based
            # fallback keeps the evaluator useful if the scene exporter renames it.
            self.floor_geom_ids = {
                i
                for i in range(self.model.ngeom)
                if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
            }
        return self

    def set_session(self, session: ort.InferenceSession) -> None:
        self.policy.ort_session = session
        self.policy.input_name = session.get_inputs()[0].name
        self.policy.output_name = session.get_outputs()[0].name

    def reset(self, rng: np.random.Generator, joint_noise: float, ball_noise: float) -> None:
        mujoco.mj_resetData(self.model, self.data)
        qadr = self.trunk_qpos_adr
        self.data.qpos[qadr : qadr + 3] = [0.0, 0.0, rng.uniform(0.11, 0.12)]
        self.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        noise = rng.uniform(-joint_noise, joint_noise, len(self.policy.joint_qpos_indices))
        for i, qpos_idx in enumerate(self.policy.joint_qpos_indices):
            self.data.qpos[qpos_idx] = DEFAULT_POSE[i] + noise[i]
        self.data.ctrl[:] = DEFAULT_POSE

        off = np.asarray(TOSS_TRAY_OFFSET_FROM_ROOT, dtype=float).copy()
        off += rng.uniform(-ball_noise, ball_noise, 3)
        # Match BALL_DROP_HEIGHT in the training config, not infer_policy's old 18mm reset.
        off[2] += 0.008
        self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3] = (
            self.data.qpos[qadr : qadr + 3] + off
        )
        self.data.qpos[self.ball_qpos_adr + 3 : self.ball_qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[:] = 0.0
        self.policy.last_action[:] = 0.0
        self.policy.command[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def control_step(self) -> None:
        action = self.policy.infer()
        self.policy.apply_action(action)
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

    def run(
        self,
        policy_path: Path,
        session: ort.InferenceSession,
        episode: int,
        duration_s: float,
        hold_session: ort.InferenceSession | None,
        hold_s: float,
        reset_action_on_switch: bool,
    ) -> TossResult:
        if hold_session is not None:
            self.set_session(hold_session)
            for _ in range(round(hold_s / CONTROL_DT)):
                self.control_step()
        if reset_action_on_switch:
            self.policy.last_action[:] = 0.0
        self.set_session(session)

        saw_tray = False
        separated_steps = 0
        released = False
        landed = False
        release_t = math.nan
        release_xy = np.full(2, np.nan)
        release_v = np.full(3, np.nan)
        landing_xy = np.full(2, np.nan)
        apex = float(self.data.xpos[self.ball_body_id, 2])
        trunk_min_z = float(self.data.xpos[self.trunk_body_id, 2])
        trunk_max_tilt = _tilt_deg(self.data, self.trunk_body_id)

        # Physical release = sustained separation after the ball has genuinely
        # contacted the tray. Three 5ms steps reject solver contact flicker.
        for control_step in range(round(duration_s / CONTROL_DT)):
            self.control_step()
            for _ in range(1):
                on_tray = _pair_in_contacts(
                    self.model, self.data, self.ball_geom_id, self.tray_geom_ids
                )
                saw_tray |= on_tray
                separated_steps = 0 if on_tray else separated_steps + 1
                if saw_tray and not released and separated_steps >= 3:
                    released = True
                    release_t = (control_step + 1) * CONTROL_DT
                    release_xy = self.data.xpos[self.ball_body_id, :2].copy()
                    release_v = self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 3].copy()

                on_floor = _pair_in_contacts(
                    self.model, self.data, self.ball_geom_id, self.floor_geom_ids
                )
                if released and on_floor:
                    landed = True
                    landing_xy = self.data.xpos[self.ball_body_id, :2].copy()

            apex = max(apex, float(self.data.xpos[self.ball_body_id, 2]))
            trunk_min_z = min(trunk_min_z, float(self.data.xpos[self.trunk_body_id, 2]))
            trunk_max_tilt = max(trunk_max_tilt, _tilt_deg(self.data, self.trunk_body_id))
            if landed:
                break

        carry = landing_xy - release_xy if landed else np.full(2, np.nan)
        return TossResult(
            policy=policy_path.name,
            episode=episode,
            released=released,
            landed=landed,
            release_time_s=float(release_t),
            release_forward_mps=float(release_v[0]),
            release_up_mps=float(release_v[2]),
            release_speed_mps=float(np.linalg.norm(release_v)),
            apex_m=apex,
            carry_forward_m=float(carry[0]),
            lateral_m=float(carry[1]),
            trunk_min_z_m=trunk_min_z,
            trunk_max_tilt_deg=trunk_max_tilt,
        )


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _summary(path: Path, results: list[TossResult]) -> None:
    good = [r for r in results if r.landed]
    released = sum(r.released for r in results)
    print(f"\n{path}: released {released}/{len(results)}, landed {len(good)}/{len(results)}")
    if not good:
        return
    for field, label, unit in (
        ("release_forward_mps", "release forward", "m/s"),
        ("release_up_mps", "release upward", "m/s"),
        ("carry_forward_m", "forward carry", "m"),
        ("lateral_m", "lateral", "m"),
        ("apex_m", "apex", "m"),
        ("trunk_max_tilt_deg", "max trunk tilt", "deg"),
    ):
        values = np.asarray([getattr(r, field) for r in good], dtype=float)
        print(
            f"  {label:18s} mean={np.mean(values):7.3f}  "
            f"p10={np.percentile(values, 10):7.3f}  min={np.min(values):7.3f} {unit}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies", type=Path, nargs="+")
    parser.add_argument("--hold", type=Path, default=None, help="optional hold policy before throw")
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--joint-noise", type=float, default=0.0, help="uniform radians")
    parser.add_argument("--ball-noise", type=float, default=0.0, help="uniform metres")
    parser.add_argument("--current-limit", type=float, default=1.75)
    parser.add_argument(
        "--reset-action-on-switch",
        action="store_true",
        help="zero previous-action obs when switching hold -> throw",
    )
    args = parser.parse_args()

    for path in args.policies:
        if not path.exists():
            parser.error(f"policy not found: {path}")
    if args.hold is not None and not args.hold.exists():
        parser.error(f"hold policy not found: {args.hold}")

    evaluator = HeaderEvaluator.create(args.policies[0], args.current_limit)
    sessions = {path: ort.InferenceSession(str(path)) for path in args.policies}
    hold_session = ort.InferenceSession(str(args.hold)) if args.hold else None
    all_results: dict[Path, list[TossResult]] = {}
    for path in args.policies:
        results = []
        for episode in range(args.episodes):
            # Identical reset samples across policies make A/B comparisons paired.
            rng = np.random.default_rng(args.seed + episode)
            evaluator.reset(rng, args.joint_noise, args.ball_noise)
            result = evaluator.run(
                path,
                sessions[path],
                episode,
                args.duration_s,
                hold_session,
                args.hold_s,
                args.reset_action_on_switch,
            )
            results.append(result)
            print(
                f"{path.name:28s} ep={episode:02d} release={result.released!s:5s} "
                f"vf={_fmt(result.release_forward_mps)} vz={_fmt(result.release_up_mps)} "
                f"carry={_fmt(result.carry_forward_m)}m apex={_fmt(result.apex_m)}m "
                f"tilt={_fmt(result.trunk_max_tilt_deg, 1)}deg"
            )
        all_results[path] = results
        _summary(path, results)


if __name__ == "__main__":
    main()
