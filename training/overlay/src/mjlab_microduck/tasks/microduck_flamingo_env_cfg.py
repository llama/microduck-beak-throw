"""Microduck *flamingo nap* task (v1) — stand on the LEFT leg, tuck the right
foot up, and hold; head-nap styling curriculums in once the balance exists.

Nobody had taught a robot duck to sleep like a flamingo before 2026-08-29.

Physics measured BEFORE reward design (see the FLAMINGO section header in
mdp.py for the full numbers):
  - One-foot balance requires a 10-25° whole-body lean toward the stance
    foot (CoM is ~41 mm off-foot at STAND; joints alone recover only ~7 mm,
    the sole patch is ~38 mm wide). So NO verticality demand in the main
    reward — a loose ~50° fallen-ceiling only.
  - No passively-holdable configuration exists (5 mm static margins, low-kp
    servos): this is an ACTIVE balance skill, like walking. The head (38% of
    body mass, fast yaw) is the balance arm, so the nap-head styling reward
    starts at weight 0 and ramps in late — pinning the head during skill
    discovery would confiscate the balance tool.

Reward architecture (per the AGENTS.md lessons):
  - What counts is a hard state gate: stance contact AND lifted-foot
    no-contact AND not-fallen. Toe-touch cheats pay zero.
  - The main term is a gated PRODUCT (clearance × tuck × height): no
    compromise basin, bounded per-step, no jackpots.
  - A bridge term (lifted_foot_clearance) pays for the weight shift while
    both feet are still down, so there is no exploration cliff between
    "standing" and the first rewarded single-support step.
  - Stillness (the "nap" calm) and head styling are late curricula:
    attempt-taxes and style constraints only after the skill exists.

Reset: mix of standing HOME spawns (learn the transition) and pre-balanced
flamingo spawns in the measured lean band (reverse curriculum: on-policy hold
data from step 0). The mix ramps toward standing-heavy as training advances.
"""

import math
from copy import deepcopy

# Symmetry mirror-loss must stay OFF: the task is intrinsically asymmetric
# (left-stance) — a mirrored transition would teach the wrong-leg version.
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to the velocity env for sim2real parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (velocity-matched) ─────────────────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003     # ramped to 0.015 via com_range
HEAD_COM_RANDOMIZATION_RANGE        = 0.003     # ramped to 0.01
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
# One-legged: pushes ramp 0 → ±0.08 → ±0.15 (NOT velocity's ±0.3 — a shove at
# ±0.3 on single support is a guaranteed fall; balance-robustness comes from
# the spawn tilt noise + CoM DR first, pushes polish late).
VELOCITY_PUSH_RANGE                 = (-0.15, 0.15)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

EPISODE_LENGTH_S = 10.0

# ── Flamingo pose targets (servo joint indices, 14-joint layout) ──────────────
# Stance = LEFT leg (indices 0-4 stay near HOME); lifted = RIGHT leg (9-13).
# Tuck: thigh flexed forward, shank folded back, toe relaxed — the foot ends up
# ~3 cm off the ground beside the stance shin, clear of self-collision.
FLAMINGO_TUCK_OVERRIDES = {
    11: 0.9,    # right hip_pitch  (HOME +0.4579; further flexion raises thigh)
    12: -1.2,   # right knee       (HOME +0.0049; negative folds the shank)
    13: -0.3,   # right ankle      (HOME -0.4530; relaxed toe)
}
_LIFTED_LEG_JOINTS = [11, 12, 13]     # priced by the tuck factor
_STANCE_LEG_JOINTS = [0, 2, 3, 4]     # hip_yaw, hip_pitch, knee, ankle — NOT
                                      # hip_roll: the lean needs it free.

# Nap head (style only, late curriculum): neck pulled back, bill dipped into
# the chest, head yawed over the STANCE-side shoulder — which is also the
# CoM-helpful counterweight direction (+3.7 mm toward the stance foot), so
# style and physics pull the same way.
NAP_HEAD_OVERRIDES = {
    5: 0.10,    # neck_pitch (HOME 0.3491; smaller = pulled back)
    6: 0.60,    # head_pitch (bill down)
    7: 1.00,    # head_yaw   (over the left shoulder)
    8: 0.0,     # head_roll
}
_NECK_JOINTS = [5, 6, 7, 8]

# Trunk z at one-leg stance: STAND_Z (0.115, measured) × cos(lean ≈ 15°).
FLAMINGO_Z = 0.110
# Lifted-foot clearance target (m): sole sits at ~0.009 at STAND; 0.03 is a
# clear, unambiguous lift that still keeps the tuck compact.
CLEARANCE_TARGET = 0.03
# Fallen ceiling for all gates: cos(50°). Loose on purpose — see module docstring.
MAX_TILT_COS = math.cos(math.radians(50.0))
MIN_HEIGHT = 0.07

# Measured equilibrium lean band (rad, negative = toward the left/stance foot
# in the ZYX quat convention used by the spawn event).
FLAMINGO_SPAWN_ROLL_RANGE = (-0.35, -0.10)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_flamingo_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck flamingo-nap environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Per-foot sensors — the single-support gate needs each foot separately.
    stance_foot_cfg = ContactSensorCfg(
        name="stance_foot_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^left_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    lifted_foot_cfg = ContactSensorCfg(
        name="lifted_foot_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^right_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (
        feet_ground_cfg,
        stance_foot_cfg,
        lifted_foot_cfg,
        self_collision_cfg,
    )
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Task rewards ──────────────────────────────────────────────────────────
    # Total positive task mass ≈ 6 + 1.5 + 1 + 0.5 (+ late ramps ≈ 2.1) ≈ 11 —
    # matched to velocity's, so the shared sim2real regularisers act at the
    # proven relative strength (the standup ÷4 lesson).

    _lifted_site = SceneEntityCfg("robot", site_names=["right_foot"])

    cfg.rewards["flamingo_composite"] = RewardTermCfg(
        func=microduck_mdp.flamingo_composite,
        weight=6.0,
        params={
            "stance_sensor_name": stance_foot_cfg.name,
            "lifted_sensor_name": lifted_foot_cfg.name,
            "lifted_joint_indices": _LIFTED_LEG_JOINTS,
            "tuck_overrides": FLAMINGO_TUCK_OVERRIDES,
            "target_height": FLAMINGO_Z,
            "height_std": 0.04,
            "tuck_std": 0.5,
            "clearance_target": CLEARANCE_TARGET,
            "max_tilt_cos": MAX_TILT_COS,
            "min_height": MIN_HEIGHT,
            "asset_cfg": _lifted_site,
        },
    )

    cfg.rewards["lift_progress"] = RewardTermCfg(
        func=microduck_mdp.lifted_foot_clearance,
        weight=1.5,
        params={
            "stance_sensor_name": stance_foot_cfg.name,
            "clearance_target": CLEARANCE_TARGET,
            "max_tilt_cos": MAX_TILT_COS,
            "min_height": MIN_HEIGHT,
            "asset_cfg": _lifted_site,
        },
    )

    # Gentle global uprightness pull: cos(tilt). At the required 15-25° lean
    # this costs only 3-9% of its weight — an acceptable, ESCAPABLE tax that
    # provides anti-fall gradient from any orientation. Deliberately light;
    # do not raise it, it directly taxes the mandatory lean.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Keep the stance leg near HOME (self-negating L1 → POSITIVE weight).
    # hip_roll is excluded — the lean rides on it.
    cfg.rewards["pose_stance_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=0.5,
        params={
            "joint_indices": _STANCE_LEG_JOINTS,
            "target_overrides": None,
        },
    )

    # Nap head styling — weight 0, ramped by nap_weight AFTER balance exists
    # (the head is the balance arm during discovery; see module docstring).
    cfg.rewards["head_nap"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=0.0,
        params={
            "std": 0.5,
            "joint_indices": _NECK_JOINTS,
            "target_overrides": NAP_HEAD_OVERRIDES,
        },
    )

    # Nap calm — weight 0, ramped by stillness_weight (attempt-tax lesson).
    cfg.rewards["flamingo_stillness"] = RewardTermCfg(
        func=microduck_mdp.flamingo_stillness,
        weight=0.0,
        params={
            "stance_sensor_name": stance_foot_cfg.name,
            "lifted_sensor_name": lifted_foot_cfg.name,
            "vel_std": 2.0,
            "max_tilt_cos": MAX_TILT_COS,
            "min_height": MIN_HEIGHT,
        },
    )

    # ── Sim2real regularisers (velocity-matched set + weights) ────────────────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards.pop("soft_landing", None)

    # ⚠️ POSITIVE weight: trunk_vertical_accel_penalty returns -|a_z| already
    # (self-negating) — a negative weight would double-negate into a reward
    # for vertical shocks (the sign bug that bit sitstand/standup).
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # The lean demand WILL park hip_roll on its ±0.38 stop without a qpos-side
    # deterrent (command-side costs don't work with low-kp overshoot control).
    # Cost function (≥0) → NEGATIVE weight.
    cfg.rewards["hip_roll_limit"] = RewardTermCfg(
        func=microduck_mdp.joint_pos_limit_proximity,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r"^(left|right)_hip_roll$",)
            ),
            "margin": 0.06,
        },
    )

    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (61D layout, identical to the standup env) ───────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    # NaN-safe wrappers on sensor-derived critic terms (single NaN kills a run
    # via rsl_rl's check_nan; this env falls constantly early on).
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Commands: 61D contract, all slots alive but untracked ────────────────
    # The head steers itself (balance arm + nap styling) — head_pose stays at
    # a permanent tiny alive range (dead-weights rule: a command input that is
    # never non-zero has dead weights forever).
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),
            (-0.05, 0.05),
            (-0.07, 0.07),
            (-0.015, 0.015),
        ),
    )

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    # fell_over kept from the base template (bad_orientation at 70°): the
    # 10-25° working lean sits far inside it, and past ~70° there is no
    # recovery on the flamingo reward stack anyway.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    cfg.events["set_flamingo_spawn"] = EventTermCfg(
        func=microduck_mdp.set_flamingo_spawn,
        mode="reset",
        params={
            # 50/50 at stage 0; the spawn_mix curriculum ramps toward
            # standing-heavy so the stand→flamingo transition dominates late.
            "flamingo_prob": 0.5,
            "flamingo_joint_overrides": FLAMINGO_TUCK_OVERRIDES,
            "flamingo_joint_noise_std": 0.06,
            "roll_range": FLAMINGO_SPAWN_ROLL_RANGE,
            "pitch_max": 0.08,
            "standing_tilt_max": 0.10,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
            # One-leg trunk equilibrium ≈ 0.110 (STAND_Z × cos lean).
            "flamingo_z_min": 0.10,
            "flamingo_z_max": 0.115,
        },
    )

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Spawn mix: reverse-curriculum flamingo spawns keep the hold phase fed
    # early; standing spawns take over as the transition becomes learnable.
    cfg.curriculum["spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_flamingo_spawn",
            "param_stages": [
                {"step": 0,          "params": {"flamingo_prob": 0.50}},
                {"step": 1500 * 24,  "params": {"flamingo_prob": 0.35}},
                {"step": 3000 * 24,  "params": {"flamingo_prob": 0.25}},
            ],
        },
    )

    # action_rate: velocity's exact ramp (discovery under light smoothing).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 500 * 24,   "weight": -0.2},
                {"step": 750 * 24,   "weight": -0.4},
                {"step": 1000 * 24,  "weight": -0.6},
                {"step": 1250 * 24,  "weight": -0.8},
                {"step": 1500 * 24,  "weight": -1.0},
            ],
        },
    )

    # Nap calm: introduced only once single-support holds exist.
    cfg.curriculum["stillness_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "flamingo_stillness",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2000 * 24,  "weight": 0.3},
                {"step": 3500 * 24,  "weight": 0.6},
            ],
        },
    )

    # Head styling: latest of all — the head is the balance arm.
    cfg.curriculum["nap_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "head_nap",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": 0.75},
                {"step": 4000 * 24,  "weight": 1.5},
            ],
        },
    )

    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 3000 * 24,  "weight": -1e-3},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 1500 * 24, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {"step": 3000 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckFlamingoRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_flamingo",
    run_name="microduck_flamingo",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8_000,
)
