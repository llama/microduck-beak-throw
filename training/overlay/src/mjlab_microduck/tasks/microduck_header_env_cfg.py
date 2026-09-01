"""Microduck Header task — flick a ball out of the bill tray, forward.

Episodic policy: the robot starts STANDING (HOME + noise) with a 24mm / 3g
ball resting in the bill tray (toss model variant — see get_toss_spec in
microduck_constants.py). The goal is to toss the ball forward (robot's heading
at reset) at BALL_TARGET_SPEED with a neck/head flick while keeping balance,
then settle back into a clean stand. "Header" as in soccer: the ball leaves
from the head.

Blueprint: the BallKick env (blind actor + critic-sees-ball + placement DR +
reward live from t=0), with three deltas:
  - The ball starts IN the tray, not on the floor: reset_ball_in_tray places
    it above the tray floor (offset measured at HOME), and a small always-on
    ball_held reward makes carrying it worth protecting during exploration.
  - The flick is a HEAD motion: pose_stand_neck is looser (std 0.5) and
    lighter than kick's so the swing stays affordable; both feet carry
    anti-hop grounded rewards (no kicking leg here).
  - Anti-cheat is structural: the toss model's contact masks mean the ball
    only touches the tray/floor/trunk/legs — the policy physically cannot
    accelerate it except through the bill. No engagement gate needed.

Feasibility measured before design (2026-08-30 scratchpad): ball rests HELD in
the tray across head poses; a hand-scripted neck flick under realistic servo
limits (kp 0.55, ±0.6405 Nm) launches at ~1.6 m/s. BALL_TARGET_SPEED=1.2 is
comfortably inside the envelope; the policy's job is a controlled, repeatable
flick plus balance, not raw power.

DR / noise / regularization: velocity-parity, copied from the kick/standup
recipe. Task reward mass ≈ 9 ≈ velocity's ~11.
"""

import math
from copy import deepcopy

# Symmetry — OFF (the flick is sagittal but spawn/DR asymmetries make mirror
# augmentation useless here; matches kick).
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to velocity / standup / kick) ───────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# ── Task constants ────────────────────────────────────────────────────────────
# Toss + ball flight/roll reward window + settle-back.
EPISODE_LENGTH_S = 5.0

# Ball spawn: nominal tray-floor center relative to the trunk root in the yaw
# frame at HOME (measured; see TOSS_TRAY_OFFSET_FROM_ROOT), small placement DR.
BALL_DROP_HEIGHT = 0.008   # v3 lesson: 18mm free-drop fed the landing bounce
BALL_POS_NOISE   = 0.004

# Target toss speed (m/s). MAX-POWER MODE (Doug's call, 2026-08-30): set
# ABOVE the mechanism's measured ceiling (~2.1-2.5 m/s scripted) so the
# forward reward's gradient extends through every achievable speed — "throw
# as hard as you can while staying on your feet" (the recoil/fall cost is the
# natural limiter). The overshoot penalty stays as a dormant safety rail.
# For a gentle fetch-toss later, just lower this back toward ~1.2.
# Weight scaling follows kick's rule: at-target payoff ≈ +3/step → weight = 3/target.
BALL_TARGET_SPEED = 3.0

STAND_Z = 0.115
_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

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

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_TOSS_BALL_CFG,
    MICRODUCK_TOSS_ROBOT_CFG,
    TOSS_TRAY_OFFSET_FROM_ROOT,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_header_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Microduck Header (bill-tray toss) environment configuration."""

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
    left_foot_cfg = ContactSensorCfg(
        name="left_foot_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^left_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    right_foot_cfg = ContactSensorCfg(
        name="right_foot_ground_contact",
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
    # Diagnostic sensor: does the ball EVER touch the tray in the GPU sim?
    # (header v2-v4: held-fraction frozen at ~2% across tray-physics changes —
    # the free-fall signature.) Logged via the ball_tray_contact reward below.
    ball_tray_contact_cfg = ContactSensorCfg(
        name="ball_tray_contact",
        primary=ContactMatch(mode="geom", pattern=r"^ball_geom$", entity="ball"),
        secondary=ContactMatch(mode="body", pattern="bill_tray", entity="robot"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {
        # Robot MUST stay the first entity (reset events write robot root state
        # at qpos[:, 0:7]).
        "robot": MICRODUCK_TOSS_ROBOT_CFG,
        "ball":  MICRODUCK_TOSS_BALL_CFG,
    }
    cfg.scene.sensors = (
        feet_ground_cfg,
        left_foot_cfg,
        right_foot_cfg,
        self_collision_cfg,
        ball_tray_contact_cfg,
    )
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S
    # Contact headroom: full-collision robot + 5 tray boxes + ball + terrain.
    # 50 (kick's value) may silently drop the highest-index pairs — which are
    # exactly ball-tray. Overflow drops are invisible; budget generously.
    cfg.sim.nconmax = 200

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
        "soft_landing",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: toss objective — TARGET speed behind a LAUNCH LATCH ─────────
    # Kick's two-sided recipe, but the forward reward pays ONLY after a
    # legitimate launch (ball exits the held state moving up + forward). The
    # ungated version bred the dump-hack: tip the ball off the tray at t≈0.3s
    # and push it with the body (run nelovxk5 — ball_held ≈ 0, all "toss"
    # reward from shoving). See ball_forward_velocity_tossed in mdp.py.
    # ORDER MATTERS: the forward term owns the latch update; the overshoot
    # term reads it and must be registered after.
    cfg.rewards["ball_forward_velocity"] = RewardTermCfg(
        func=microduck_mdp.ball_forward_velocity_tossed,
        weight=1.0,   # = 3 / BALL_TARGET_SPEED → at-target payoff ≈ +3/step
        params={
            "asset_name": "ball",
            "max_speed": BALL_TARGET_SPEED,
            "min_rel_height": 0.05,
            # LATCH POSTMORTEM (throw_v4, the bug behind 15 variants): the
            # held->not-held transition is detected ~10cm BELOW the head-top
            # tray, i.e. mid-free-fall, where vz is always ~-1.2 m/s — the old
            # min_launch_vz=0.05 was physically unsatisfiable for any real
            # throw. Forward velocity survives the fall (ballistics), so it
            # alone separates throws (fwd >= 0.3 at exit) from dumps
            # (backward/slow). vz requirement disabled.
            "min_launch_vz": -10.0,
            "min_launch_fwd": 0.3,
        },
    )
    # Dormant in max-power mode (target sits above the mechanism's ceiling);
    # kept as the safety rail with the canonical below-forward-slope weight.
    cfg.rewards["ball_speed_overshoot"] = RewardTermCfg(
        func=microduck_mdp.ball_speed_overshoot_tossed,
        weight=-0.5,
        params={"asset_name": "ball", "target_speed": BALL_TARGET_SPEED},
    )

    # Hold the ball until the flick. GENEROUS EARLY, DECAYED BY CURRICULUM
    # (v2 lesson, run wibozuwz: at weight 0.2 the hold gradient was ~0.06% of
    # the episode return — invisible; the policy converged to "stand nicely,
    # ignore the ball" and never explored a flick). At 1.5 the hold is worth
    # ~20% of the return, a gradient PPO can see; the held_weight curriculum
    # decays it to 0.3 by iter 1800 so hold-forever loses to tossing once the
    # skill exists.
    cfg.rewards["ball_held"] = RewardTermCfg(
        func=microduck_mdp.ball_held_reward,
        weight=1.5,
        params={"asset_name": "ball", "min_rel_height": 0.05},
    )

    # Discovery bridge: forward velocity of the HELD ball (v5 lesson — the
    # converged hold suppresses flick exploration; this pays for the head
    # sweep that becomes the launch). Inert after release and on backswing.
    # v6 lesson: max_speed=0.3 killed the gradient at exactly the speeds a
    # launch needs (~1+ m/s) — the policy settled into safe sub-launch rocking.
    # Cap now past launch speed; weight rebalanced so peak payoff is similar.
    cfg.rewards["held_fwd_shaping"] = RewardTermCfg(
        func=microduck_mdp.held_ball_forward_shaping,
        weight=0.8,
        params={"asset_name": "ball", "min_rel_height": 0.05, "max_speed": 1.2},
    )

    # Diagnostics (tiny weights — logging channels, not incentives):
    cfg.rewards["ball_tray_contact"] = RewardTermCfg(
        func=microduck_mdp.single_foot_grounded_reward,
        weight=0.02,
        params={"sensor_name": "ball_tray_contact"},
    )
    cfg.rewards["ball_rel_z_diag"] = RewardTermCfg(
        func=microduck_mdp.ball_rel_height_diag,
        weight=0.02,
        params={"asset_name": "ball"},
    )

    # One-time discovery bonus on the step the launch latch first sets. Prices
    # the flick's DISCOVERY (the per-step roll reward alone was too brief to
    # stand out against the standing stack). Once per episode, unfarmable.
    # Register AFTER ball_forward_velocity (latch state owner).
    cfg.rewards["launch_bonus"] = RewardTermCfg(
        func=microduck_mdp.ball_toss_launch_bonus,
        weight=30.0,
    )

    # Anti-hop: both feet stay planted (the flick is head-only).
    cfg.rewards["left_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.single_foot_grounded_reward,
        weight=1.0,
        params={"sensor_name": left_foot_cfg.name},
    )
    cfg.rewards["right_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.single_foot_grounded_reward,
        weight=1.0,
        params={"sensor_name": right_foot_cfg.name},
    )

    # ── Rewards: stand cleanly before/after the toss ──────────────────────────
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )
    # Neck near HOME between flicks — LOOSE (std 0.5, weight 0.75; kick pins at
    # 0.3/1.0): the flick is a large transient neck deviation and must stay
    # affordable, but the policy should return the head to neutral after.
    cfg.rewards["pose_stand_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=0.75,
        params={
            "std": 0.5,
            "joint_indices": _NECK_JOINTS,
            "target_overrides": None,
        },
    )
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std":           0.04,
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # ── Sim2real regularisers — velocity parity ──────────────────────────────
    cfg.rewards["action_rate_l2"].weight = -0.1  # curriculum ramps to -1.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (unified 61D actor layout, ball-blind) ───────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
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

    # Command obs slots — unified layout parity, head/body zero-padded (the
    # flick is task-driven, not commanded).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # CRITIC-ONLY ball state (asymmetric actor-critic, kick's recipe).
    cfg.observations["critic"].terms["ball_position"] = ObservationTermCfg(
        func=microduck_mdp.ball_pos_in_base, params={"asset_name": "ball"},
    )
    cfg.observations["critic"].terms["ball_velocity"] = ObservationTermCfg(
        func=microduck_mdp.ball_vel_in_base, params={"asset_name": "ball"},
    )

    # ── Command: tiny noise around zero (obs-shape parity only) ───────────────
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
    # fell_over KEPT (starts standing, must stay up through the flick).
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

    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    # Standing-only start (standup env's noisy upright spawn machinery).
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob":   0.0,
            "face_up_prob":     0.0,
            "sitting_prob":     0.0,
            "standing_prob":    1.0,
            "sitting_tilt_max": math.radians(5),
            "standing_z_min":   0.11,
            "standing_z_max":   0.12,
        },
    )

    # Ball placement — MUST come after set_ground_state (events run in dict
    # insertion order; the tray position derives from the final robot pose).
    cfg.events["reset_ball"] = EventTermCfg(
        func=microduck_mdp.reset_ball_in_tray,
        mode="reset",
        params={
            "offset":      TOSS_TRAY_OFFSET_FROM_ROOT,
            "drop_height": BALL_DROP_HEIGHT,
            "noise_xyz":   BALL_POS_NOISE,
            # Reverse curriculum (v7 lesson): 30% of episodes start mid-launch
            # so the critic learns the post-launch payoff the hold-attractor's
            # risk valley otherwise hides.
            "launched_prob": 0.30,
            "asset_name":  "ball",
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

    # ── Terrain: flat only ────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Hold-reward handover: generous while the protect-skill forms, then decay
    # so a completed toss (bonus 30 + roll ≈ 3/step) strictly beats
    # hold-forever (0.3 × 250 = 75 ... vs toss ≈ 30 + ~200 + held-before-toss).
    cfg.curriculum["held_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "ball_held",
            "weight_stages": [
                {"step": 0,          "weight": 1.5},
                {"step": 1000 * 24,  "weight": 0.8},
                {"step": 1800 * 24,  "weight": 0.3},
            ],
        },
    )

    # action_rate ramp — velocity's exact stages. The flick is a fast one-shot
    # swing: if the converged flick is too weak, soften the ramp end first.
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
                    {"step": 500 * 24,  "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {"step": 1000 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckHeaderRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
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
    experiment_name="microduck_header",
    run_name="microduck_header",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=4_000,
)
