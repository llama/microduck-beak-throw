#!/usr/bin/env python3
"""Headless ONNX evaluation for the collision-resolved beak throw.

The ball is a free body from reset onward. A separately actuated lower bill
holds it in the shallow contact pocket and opens at the same phase used by
training/runtime. Release is recorded only after the ball physically separates
from the pocket; results are measured in the robot's reset heading, so sideways
motion cannot count as throw distance.

Example::

    uv run python scripts/eval_beak.py beak_throw.onnx --episodes 20
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
    BEAK_BALL_SITE_OFFSET,
    BEAK_CLOSED_ANGLE_RAD,
    BEAK_HOLD_ANGLE_RAD,
    BEAK_MOUTH_JOINT,
    BEAK_OPEN_ANGLE_RAD,
    TOSS_BALL_RADIUS,
    build_beak_scene_spec,
)
from mjlab_microduck.tasks.microduck_beak_throw_env_cfg import (
    MOUTH_OPEN_DURATION_PHASE,
    RELEASE_PHASE,
    THROW_PERIOD_S,
)


CONTROL_DT = 0.02
PHYSICS_DT = 0.005
DECIMATION = round(CONTROL_DT / PHYSICS_DT)
LOCAL_GRIP_OFFSET = np.asarray(BEAK_BALL_SITE_OFFSET)
RELEASE_DISTANCE_M = 0.020


@dataclass
class ThrowResult:
    policy: str
    episode: int
    released: bool
    valid_release: bool
    landed: bool
    release_time_s: float
    release_forward_mps: float
    release_up_mps: float
    release_lateral_mps: float
    predicted_carry_m: float
    carry_forward_m: float
    lateral_m: float
    apex_m: float
    trunk_min_z_m: float
    trunk_max_tilt_deg: float
    trunk_handoff_tilt_deg: float
    trunk_post_handoff_max_tilt_deg: float
    trunk_final_tilt_deg: float
    max_abs_action_rad: float
    max_target_limit_excess_rad: float


def _set_current_limit(model: mujoco.MjModel, current_limit: float) -> None:
    if current_limit <= 0.0:
        return
    from bam.model import load_model

    torque_limit = load_model(motor_name="xl330", model="m6").kt.value * current_limit
    model.actuator_forcerange[:, 0] = -torque_limit
    model.actuator_forcerange[:, 1] = torque_limit
    model.actuator_forcelimited[:] = 1


def _tilt_deg(data: mujoco.MjData, body_id: int) -> float:
    cos_tilt = float(np.clip(data.xmat[body_id].reshape(3, 3)[2, 2], -1.0, 1.0))
    return math.degrees(math.acos(cos_tilt))


def _pair_in_contacts(
    data: mujoco.MjData, geom_id: int, other_ids: set[int]
) -> bool:
    for i in range(data.ncon):
        contact = data.contact[i]
        if (contact.geom1 == geom_id and contact.geom2 in other_ids) or (
            contact.geom2 == geom_id and contact.geom1 in other_ids
        ):
            return True
    return False


def _ballistic_carry(height: float, forward: float, upward: float) -> float:
    height = max(height, 0.0)
    flight = (upward + math.sqrt(max(upward * upward + 2.0 * 9.81 * height, 0.0))) / 9.81
    return max(forward, 0.0) * max(flight, 0.0)


class BeakEvaluator:
    def __init__(
        self,
        first_policy: Path,
        current_limit: float,
        action_scale: float,
        clip_targets_to_joint_limits: bool,
    ):
        self.model = build_beak_scene_spec().compile()
        self.model.opt.timestep = PHYSICS_DT
        _set_current_limit(self.model, current_limit)
        self.data = mujoco.MjData(self.model)
        self.policy = PolicyInference(
            self.model,
            self.data,
            walking_onnx_path=str(first_policy),
            action_scale=action_scale,
            new_cmd_obs=True,
            use_projected_gravity=True,
        )

        trunk_joint = self.model.joint("trunk_base_freejoint")
        ball_joint = self.model.joint("ball_free")
        self.trunk_qpos_adr = int(self.model.jnt_qposadr[trunk_joint.id])
        self.ball_qpos_adr = int(self.model.jnt_qposadr[ball_joint.id])
        self.ball_qvel_adr = int(self.model.jnt_dofadr[ball_joint.id])
        mouth_joint = self.model.joint(BEAK_MOUTH_JOINT)
        self.mouth_qpos_adr = int(self.model.jnt_qposadr[mouth_joint.id])
        self.mouth_qvel_adr = int(self.model.jnt_dofadr[mouth_joint.id])
        self.mouth_ctrl_id = next(
            i
            for i in range(self.model.nu)
            if self.model.actuator_trnid[i, 0] == mouth_joint.id
        )
        self.trunk_body_id = self.model.body("trunk_base").id
        self.ball_body_id = self.model.body("toss_ball").id
        self.ball_geom_id = self.model.geom("toss_ball_geom").id
        self.mouth_site_id = self.model.site("mouth_tip").id
        self.floor_geom_ids = {
            i
            for i in range(self.model.ngeom)
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        }
        policy_joint_ids = self.model.actuator_trnid[
            self.policy.policy_actuator_ids, 0
        ]
        self.policy_joint_ranges = self.model.jnt_range[policy_joint_ids].copy()
        self.action_scale = action_scale
        self.clip_targets_to_joint_limits = clip_targets_to_joint_limits

    def set_session(self, session: ort.InferenceSession) -> None:
        self.policy.ort_session = session
        self.policy.input_name = session.get_inputs()[0].name
        self.policy.output_name = session.get_outputs()[0].name

    def _grip_position(self) -> np.ndarray:
        rotation = self.data.site_xmat[self.mouth_site_id].reshape(3, 3)
        return self.data.site_xpos[self.mouth_site_id] + rotation @ LOCAL_GRIP_OFFSET

    def _write_ball(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3] = position
        self.data.qpos[self.ball_qpos_adr + 3 : self.ball_qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 3] = velocity
        self.data.qvel[self.ball_qvel_adr + 3 : self.ball_qvel_adr + 6] = 0.0

    def reset(
        self,
        rng: np.random.Generator,
        joint_noise: float,
        spawn_z_noise: float,
        yaw_noise_deg: float,
    ) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        qadr = self.trunk_qpos_adr
        yaw = math.radians(rng.uniform(-yaw_noise_deg, yaw_noise_deg))
        self.data.qpos[qadr : qadr + 3] = [
            0.0,
            0.0,
            0.115 + rng.uniform(-spawn_z_noise, spawn_z_noise),
        ]
        self.data.qpos[qadr + 3 : qadr + 7] = [
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ]
        noise = rng.uniform(-joint_noise, joint_noise, len(self.policy.joint_qpos_indices))
        for i, qpos_idx in enumerate(self.policy.joint_qpos_indices):
            self.data.qpos[qpos_idx] = DEFAULT_POSE[i] + noise[i]
        self.data.qpos[self.mouth_qpos_adr] = BEAK_HOLD_ANGLE_RAD
        self.data.qvel[self.mouth_qvel_adr] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.policy.policy_actuator_ids] = DEFAULT_POSE
        self.data.ctrl[self.mouth_ctrl_id] = BEAK_HOLD_ANGLE_RAD
        self.data.qvel[:] = 0.0
        self.policy.last_action[:] = 0.0
        self.policy.command[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        grip = self._grip_position()
        self._write_ball(grip, np.zeros(3))
        mujoco.mj_forward(self.model, self.data)
        return np.array([math.cos(yaw), math.sin(yaw)])

    def run(
        self,
        path: Path,
        session: ort.InferenceSession,
        episode: int,
        heading: np.ndarray,
        duration_s: float,
        period_s: float,
        release_phase: float,
        stand_session: ort.InferenceSession | None = None,
    ) -> ThrowResult:
        self.set_session(session)
        phase = 0.0
        released = False
        valid_release = False
        landed = False
        release_time = math.nan
        release_position = np.full(3, np.nan)
        release_velocity = np.full(3, np.nan)
        landing_position = np.full(3, np.nan)
        apex = float(self._grip_position()[2])
        trunk_min_z = float(self.data.xpos[self.trunk_body_id, 2])
        trunk_max_tilt = _tilt_deg(self.data, self.trunk_body_id)
        handoff_tilt = math.nan
        post_handoff_max_tilt = math.nan
        max_abs_action = 0.0
        max_target_limit_excess = 0.0
        previous_targets = self.data.ctrl[self.policy.policy_actuator_ids].copy()

        for control_idx in range(round(duration_s / CONTROL_DT)):
            elapsed_s = control_idx * CONTROL_DT
            in_stand = stand_session is not None and elapsed_s >= period_s
            self.policy.command[:] = 0.0
            if in_stand:
                self.set_session(stand_session)
                if not math.isfinite(handoff_tilt):
                    handoff_tilt = _tilt_deg(self.data, self.trunk_body_id)
                    post_handoff_max_tilt = handoff_tilt
            else:
                self.set_session(session)
                self.policy.command[0] = math.cos(2.0 * math.pi * phase)
                self.policy.command[1] = math.sin(2.0 * math.pi * phase)
            action = self.policy.infer()
            max_abs_action = max(max_abs_action, float(np.max(np.abs(action))))
            target = DEFAULT_POSE + self.action_scale * action
            excess = np.maximum(
                self.policy_joint_ranges[:, 0] - target,
                target - self.policy_joint_ranges[:, 1],
            )
            max_target_limit_excess = max(
                max_target_limit_excess, float(np.max(np.maximum(excess, 0.0)))
            )
            self.policy.apply_action(action)
            if self.clip_targets_to_joint_limits and not in_stand:
                ids = self.policy.policy_actuator_ids
                self.data.ctrl[ids] = np.clip(
                    self.data.ctrl[ids],
                    self.policy_joint_ranges[:, 0],
                    self.policy_joint_ranges[:, 1],
                )
            if in_stand:
                # robotd's first ordinary-policy tick low-passes from the final
                # throw target: alpha=.5 for head and .7 for legs.
                ids = self.policy.policy_actuator_ids
                alpha = np.full(len(ids), 0.7)
                alpha[5:9] = 0.5
                self.data.ctrl[ids] = (
                    alpha * self.data.ctrl[ids]
                    + (1.0 - alpha) * previous_targets
                )
                self.data.ctrl[ids] = np.clip(self.data.ctrl[ids], -math.pi, math.pi)
                self.data.ctrl[self.mouth_ctrl_id] = BEAK_CLOSED_ANGLE_RAD
            else:
                opening = np.clip(
                    (phase - release_phase) / MOUTH_OPEN_DURATION_PHASE,
                    0.0,
                    1.0,
                )
                opening = opening * opening * (3.0 - 2.0 * opening)
                self.data.ctrl[self.mouth_ctrl_id] = (
                    BEAK_HOLD_ANGLE_RAD
                    + opening * (BEAK_OPEN_ANGLE_RAD - BEAK_HOLD_ANGLE_RAD)
                )
            previous_targets = self.data.ctrl[self.policy.policy_actuator_ids].copy()

            for _ in range(DECIMATION):
                mujoco.mj_step(self.model, self.data)
                apex = max(apex, float(self.data.xpos[self.ball_body_id, 2]))
                trunk_min_z = min(
                    trunk_min_z, float(self.data.xpos[self.trunk_body_id, 2])
                )
                trunk_max_tilt = max(
                    trunk_max_tilt, _tilt_deg(self.data, self.trunk_body_id)
                )
                if in_stand:
                    post_handoff_max_tilt = max(
                        post_handoff_max_tilt,
                        _tilt_deg(self.data, self.trunk_body_id),
                    )
                if released and not landed and _pair_in_contacts(
                    self.data, self.ball_geom_id, self.floor_geom_ids
                ):
                    landed = True
                    landing_position = self.data.xpos[self.ball_body_id].copy()
            if not released and phase >= release_phase:
                grip = self._grip_position()
                ball_position = self.data.xpos[self.ball_body_id].copy()
                if np.linalg.norm(ball_position - grip) > RELEASE_DISTANCE_M:
                    released = True
                    release_time = (control_idx + 1) * CONTROL_DT
                    release_position = ball_position
                    release_velocity = self.data.qvel[
                        self.ball_qvel_adr : self.ball_qvel_adr + 3
                    ].copy()
                    valid_release = (
                        _tilt_deg(self.data, self.trunk_body_id) < 50.0
                        and ball_position[2] > 0.15
                    )
            if not in_stand:
                phase = (phase + CONTROL_DT / period_s) % 1.0

        if released:
            forward_vel = float(np.dot(release_velocity[:2], heading))
            lateral_axis = np.array([-heading[1], heading[0]])
            lateral_vel = float(np.dot(release_velocity[:2], lateral_axis))
            predicted = _ballistic_carry(
                release_position[2] - TOSS_BALL_RADIUS,
                forward_vel,
                float(release_velocity[2]),
            )
        else:
            forward_vel = lateral_vel = predicted = math.nan

        if landed:
            delta = landing_position[:2] - release_position[:2]
            carry = float(np.dot(delta, heading))
            lateral = float(np.dot(delta, np.array([-heading[1], heading[0]])))
        else:
            carry = lateral = math.nan

        final_tilt = _tilt_deg(self.data, self.trunk_body_id)
        if not math.isfinite(handoff_tilt):
            handoff_tilt = final_tilt
        if not math.isfinite(post_handoff_max_tilt):
            post_handoff_max_tilt = final_tilt
        return ThrowResult(
            policy=path.name,
            episode=episode,
            released=released,
            valid_release=valid_release,
            landed=landed,
            release_time_s=float(release_time),
            release_forward_mps=forward_vel,
            release_up_mps=float(release_velocity[2]) if released else math.nan,
            release_lateral_mps=lateral_vel,
            predicted_carry_m=predicted,
            carry_forward_m=carry,
            lateral_m=lateral,
            apex_m=apex,
            trunk_min_z_m=trunk_min_z,
            trunk_max_tilt_deg=trunk_max_tilt,
            trunk_handoff_tilt_deg=handoff_tilt,
            trunk_post_handoff_max_tilt_deg=post_handoff_max_tilt,
            trunk_final_tilt_deg=final_tilt,
            max_abs_action_rad=max_abs_action,
            max_target_limit_excess_rad=max_target_limit_excess,
        )


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _summary(
    path: Path,
    results: list[ThrowResult],
    min_carry: float,
    min_success_rate: float,
    max_target_limit_excess: float,
    max_abs_lateral: float,
) -> bool:
    good = [result for result in results if result.valid_release and result.landed]
    passed = [
        result
        for result in good
        if result.carry_forward_m >= min_carry
        and abs(result.lateral_m) <= max_abs_lateral
        and result.release_up_mps > 0.0
        and result.trunk_max_tilt_deg < 75.0
        and result.trunk_handoff_tilt_deg < 30.0
        and result.trunk_final_tilt_deg < 30.0
        and result.max_target_limit_excess_rad <= max_target_limit_excess
    ]
    print(
        f"\n{path}: valid+landed {len(good)}/{len(results)}, "
        f"acceptance {len(passed)}/{len(results)} (carry >= {min_carry:.2f}m, "
        f"|lateral| <= {max_abs_lateral:.2f}m, "
        f"raw target excess <= {max_target_limit_excess:.2f}rad)"
    )
    if not good:
        return False
    for field, label, unit in (
        ("release_forward_mps", "release forward", "m/s"),
        ("release_up_mps", "release upward", "m/s"),
        ("release_lateral_mps", "release lateral", "m/s"),
        ("predicted_carry_m", "predicted carry", "m"),
        ("carry_forward_m", "actual carry", "m"),
        ("lateral_m", "lateral", "m"),
        ("trunk_max_tilt_deg", "max trunk tilt", "deg"),
        ("trunk_handoff_tilt_deg", "handoff tilt", "deg"),
        ("trunk_post_handoff_max_tilt_deg", "post-handoff max", "deg"),
        ("trunk_final_tilt_deg", "final tilt", "deg"),
        ("max_abs_action_rad", "max abs action", "rad"),
        ("max_target_limit_excess_rad", "target limit excess", "rad"),
    ):
        values = np.asarray([getattr(result, field) for result in good])
        print(
            f"  {label:18s} median={np.median(values):7.3f}  "
            f"p10={np.percentile(values, 10):7.3f}  min={np.min(values):7.3f} {unit}"
        )
    return len(passed) / len(results) >= min_success_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies", type=Path, nargs="+")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="total evaluation duration (default: one throw cycle, or cycle + post-handoff)",
    )
    parser.add_argument(
        "--stand-policy",
        type=Path,
        default=None,
        help="standing ONNX to activate at the exact robotd handoff",
    )
    parser.add_argument(
        "--post-handoff-s",
        type=float,
        default=2.0,
        help="standing-policy evaluation after the 2.4 s throw (default: 2.0)",
    )
    parser.add_argument("--period-s", type=float, default=THROW_PERIOD_S)
    parser.add_argument("--release-phase", type=float, default=RELEASE_PHASE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--joint-noise", type=float, default=0.01)
    parser.add_argument("--spawn-z-noise", type=float, default=0.005)
    parser.add_argument("--yaw-noise-deg", type=float, default=180.0)
    parser.add_argument("--current-limit", type=float, default=1.75)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument(
        "--clip-targets-to-joint-limits",
        action="store_true",
        help="match robotd's beak-throw anatomical target clamp",
    )
    parser.add_argument("--min-carry", type=float, default=0.50)
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    parser.add_argument(
        "--max-abs-lateral",
        type=float,
        default=0.20,
        help="largest permitted absolute sideways carry",
    )
    parser.add_argument(
        "--max-target-limit-excess",
        type=float,
        default=0.15,
        help="largest permitted raw target overshoot past a joint limit",
    )
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help="exit non-zero unless every episode clears the acceptance gate",
    )
    args = parser.parse_args()
    for path in args.policies:
        if not path.is_file():
            parser.error(f"policy not found: {path}")
    if args.stand_policy is not None and not args.stand_policy.is_file():
        parser.error(f"standing policy not found: {args.stand_policy}")
    if args.episodes <= 0 or args.period_s <= 0.0:
        parser.error("episodes and period must be positive")
    if not 0.0 < args.release_phase < 1.0:
        parser.error("release phase must be in (0, 1)")
    if not 0.0 <= args.min_success_rate <= 1.0:
        parser.error("minimum success rate must be in [0, 1]")
    if args.max_target_limit_excess < 0.0:
        parser.error("maximum target limit excess must be non-negative")
    if args.max_abs_lateral < 0.0:
        parser.error("maximum absolute lateral carry must be non-negative")

    if args.action_scale <= 0.0:
        parser.error("action scale must be positive")
    if args.post_handoff_s < 0.0:
        parser.error("post-handoff duration must be non-negative")
    duration_s = args.duration_s
    if duration_s is None:
        duration_s = args.period_s + (
            args.post_handoff_s if args.stand_policy is not None else 0.0
        )
    if duration_s <= 0.0:
        parser.error("duration must be positive")
    if args.stand_policy is not None and duration_s <= args.period_s:
        parser.error("duration must extend past the handoff when --stand-policy is used")
    evaluator = BeakEvaluator(
        args.policies[0],
        args.current_limit,
        args.action_scale,
        args.clip_targets_to_joint_limits,
    )
    sessions = {path: ort.InferenceSession(str(path)) for path in args.policies}
    stand_session = (
        ort.InferenceSession(str(args.stand_policy))
        if args.stand_policy is not None
        else None
    )
    all_passed = True
    for path in args.policies:
        results = []
        for episode in range(args.episodes):
            rng = np.random.default_rng(args.seed + episode)
            heading = evaluator.reset(
                rng, args.joint_noise, args.spawn_z_noise, args.yaw_noise_deg
            )
            result = evaluator.run(
                path,
                sessions[path],
                episode,
                heading,
                duration_s,
                args.period_s,
                args.release_phase,
                stand_session,
            )
            results.append(result)
            print(
                f"{path.name:28s} ep={episode:02d} valid={result.valid_release!s:5s} "
                f"vf={_fmt(result.release_forward_mps)} "
                f"vz={_fmt(result.release_up_mps)} "
                f"carry={_fmt(result.carry_forward_m)}m "
                f"lateral={_fmt(result.lateral_m)}m "
                f"peak={_fmt(result.trunk_max_tilt_deg, 1)}deg "
                f"handoff={_fmt(result.trunk_handoff_tilt_deg, 1)}deg "
                f"final={_fmt(result.trunk_final_tilt_deg, 1)}deg"
            )
        all_passed &= _summary(
            path,
            results,
            args.min_carry,
            args.min_success_rate,
            args.max_target_limit_excess,
            args.max_abs_lateral,
        )
    if args.require_gate and not all_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
