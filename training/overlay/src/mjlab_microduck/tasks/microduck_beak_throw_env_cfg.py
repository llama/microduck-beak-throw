"""Phase-controlled, collision-resolved beak throw for maximum forward carry.

The real robot's mouth is the release mechanism and is intentionally outside
the 14-action locomotion policy contract. In simulation a separately scripted
passive mouth joint starts partially closed around the free ball and opens at
the release phase. Explicit beak contacts retain and accelerate the ball; it is
never welded or teleported after reset. Runtime opens the real mouth at the
same phase.

This formulation fixes the central Header failure: release discovery is no
longer sparse. Every episode contains a release, and the dense pre-release
objective is the predicted ballistic forward range. The terminal objective is
measured forward carry to first touchdown (not Euclidean/sideways distance).
"""

from copy import deepcopy
from dataclasses import fields
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.robot.microduck_constants import (
    BEAK_BALL_SITE_OFFSET,
    BEAK_HOLD_ANGLE_RAD,
    BEAK_MOUTH_JOINT,
    BEAK_OPEN_ANGLE_RAD,
    MICRODUCK_BEAK_ROBOT_CFG,
    MICRODUCK_TOSS_BALL_CFG,
    TOSS_BALL_RADIUS,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_header_env_cfg import (
    MicroduckHeaderRlCfg,
    make_microduck_header_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG


THROW_PERIOD_S = 2.4
EPISODE_LENGTH_S = 3.0
WINDUP_PHASE = 0.25
# Command the delayed mouth actuator before the desired physical separation.
# A sweep against the known-good model_1250 body motion measured 0.59 m carry
# at 0.30; commanding at the old virtual-release phase 0.38 made the ball clear
# only after the head reversed (backward/downward release).
RELEASE_PHASE = 0.30
MOUTH_OPEN_DURATION_PHASE = 0.04
# Maximum time after the open command to observe real ball/pocket separation.
# At 2.4 s per cycle this is 384 ms, comfortably beyond the modeled XL330
# delay and 10-degree travel. A ball still captive at the deadline is invalid.
RELEASE_TIMEOUT_PHASE = 0.16
SNAP_END_PHASE = 0.50
RECOVERY_PHASE = 0.78

# Absolute position limits shared by the physical MJCF and robotd's final
# command clamp.  Training with this clip is deliberate: the original throw
# learned to request positions several radians past a linkage stop, using the
# stop as a high-torque brace.  That is effective in MuJoCo and unsafe on a
# real XL330.  The action-over-limit reward below still sees the *raw* request,
# so PPO learns to remain inside these bounds instead of relying on the clip.
HARDWARE_ACTION_CLIP_RAD = {
    r"^left_hip_yaw$": (math.radians(-25.0), math.radians(30.0)),
    r"^right_hip_yaw$": (math.radians(-30.0), math.radians(25.0)),
    r"^.*hip_roll$": (math.radians(-22.0), math.radians(22.0)),
    r"^.*hip_pitch$": (math.radians(-90.0), math.radians(90.0)),
    r"^.*knee$": (math.radians(-90.0), math.radians(90.0)),
    r"^.*ankle$": (math.radians(-90.0), math.radians(90.0)),
    r"^neck_pitch$": (math.radians(-90.0), math.radians(60.0)),
    r"^head_pitch$": (math.radians(-90.0), math.radians(90.0)),
    r"^head_yaw$": (math.radians(-170.0), math.radians(170.0)),
    r"^head_roll$": (math.radians(-25.0), math.radians(25.0)),
}

# A deliberately loose seed motion. Microduck's pitch convention puts the
# mouth forward/up when moving from positive to negative pitch (the opposite
# sign from the old Header seed). The mouth opens mid-snap at RELEASE_PHASE,
# before vertical tip velocity turns downward; SNAP_END_PHASE is merely the
# reference trajectory's follow-through endpoint. Legs, yaw and roll remain
# free so PPO can discover whole-body power/recoil.
WINDUP_POSE = {"neck_pitch": 1.00, "head_pitch": 1.35}
RELEASE_POSE = {"neck_pitch": -1.20, "head_pitch": -1.20}


def make_microduck_beak_throw_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_header_env_cfg(play)
    cfg.scene.entities = {
        "robot": MICRODUCK_BEAK_ROBOT_CFG,
        "ball": MICRODUCK_TOSS_BALL_CFG,
    }
    cfg.scene.sensors = tuple(
        sensor for sensor in cfg.scene.sensors if sensor.name != "ball_tray_contact"
    )
    cfg.scene.num_envs = 16 if play else cfg.scene.num_envs
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.sim.nconmax = 100

    # Execute the seed snap from iteration zero, then fade it entirely. The
    # policy action remains a residual throughout, so it learns balance and
    # eventually reproduces/improves the motion without an export-time helper.
    base_action = cfg.actions["joint_pos"]
    action_cfg_fields = {
        field.name
        for field in fields(microduck_mdp.PhaseReferenceJointPositionActionCfg)
        if field.init
    }
    action_base = {
        key: value for key, value in vars(base_action).items() if key in action_cfg_fields
    }
    cfg.actions["joint_pos"] = microduck_mdp.PhaseReferenceJointPositionActionCfg(
        **{
            **action_base,
            # The restored mouth has its own hardware-faithful actuator but is
            # commanded by phase, not by the body's 14 policy outputs.
            "actuator_names": (r"^(?!passive_).*",),
        },
        command_name="twist",
        windup_phase=WINDUP_PHASE,
        release_phase=SNAP_END_PHASE,
        recovery_phase=RECOVERY_PHASE,
        windup_pose=WINDUP_POSE,
        release_pose=RELEASE_POSE,
        max_bias=0.0 if play else 1.0,
        bias_hold_steps=250 * 24,
        bias_fade_steps=1250 * 24,
        mouth_joint_name=BEAK_MOUTH_JOINT,
        mouth_release_phase=RELEASE_PHASE,
        mouth_open_duration_phase=MOUTH_OPEN_DURATION_PHASE,
        mouth_hold_position=BEAK_HOLD_ANGLE_RAD,
        mouth_open_position=BEAK_OPEN_ANGLE_RAD,
    )

    # Keep only physics/sim2real regularizers from Header. No holding annuity,
    # tray contact, standing jackpot, or latch economics survive into this task.
    keep = {
        "upright",
        "body_ang_vel",
        "angular_momentum",
        "dof_pos_limits",
        "action_rate_l2",
        "self_collisions",
    }
    for name in list(cfg.rewards):
        if name not in keep:
            del cfg.rewards[name]
    cfg.rewards["upright"].weight = 0.30
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards["angular_momentum"].weight = -0.01
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_).*",)
    )

    # ORDER IS STATEFUL. The first term owns attachment and release. The next
    # terms read its one-step flags; landing updates touchdown for termination.
    cfg.rewards["beak_predicted_range"] = RewardTermCfg(
        func=microduck_mdp.beak_grip_predicted_range,
        weight=4.0,
        params={
            "asset_name": "ball",
            "command_name": "twist",
            "release_phase": RELEASE_PHASE,
            "release_settle_phase": RELEASE_TIMEOUT_PHASE,
            "shaping_start_phase": 0.28,
            "local_grip_offset": BEAK_BALL_SITE_OFFSET,
            "ball_radius": TOSS_BALL_RADIUS,
            "max_range": 3.0,
            "asset_cfg": SceneEntityCfg("robot", site_names=("mouth_tip",)),
        },
    )
    cfg.rewards["beak_release_range"] = RewardTermCfg(
        func=microduck_mdp.beak_release_range_bonus,
        weight=40.0,
    )
    # Small metrics with enough non-zero weight to appear in episode logs.
    cfg.rewards["beak_release_forward"] = RewardTermCfg(
        func=microduck_mdp.beak_release_forward_velocity,
        weight=0.05,
        params={"max_speed": 5.0},
    )
    cfg.rewards["beak_release_upward"] = RewardTermCfg(
        func=microduck_mdp.beak_release_upward_velocity,
        weight=0.05,
        params={"max_speed": 5.0},
    )
    cfg.rewards["beak_forward_carry"] = RewardTermCfg(
        func=microduck_mdp.beak_landing_forward_distance,
        weight=150.0,
        params={
            "asset_name": "ball",
            "ball_radius": TOSS_BALL_RADIUS,
            "max_distance": 3.0,
        },
    )
    cfg.rewards["beak_reference_pose"] = RewardTermCfg(
        func=microduck_mdp.beak_reference_pose_track,
        weight=3.0,
        params={
            "command_name": "twist",
            "windup_phase": WINDUP_PHASE,
            "release_phase": SNAP_END_PHASE,
            "recovery_phase": RECOVERY_PHASE,
            "windup_pose": WINDUP_POSE,
            "release_pose": RELEASE_POSE,
            "joint_indices": (5, 6),
            "std": 0.40,
        },
    )

    # Phase is part of the 61D observation. Half of training episodes exactly
    # match deployment (phase 0); the rest start within the pre-release slice,
    # a reverse curriculum that still guarantees a real release.
    base_command = cfg.commands["twist"]
    phase_cfg_fields = {
        field.name
        for field in fields(microduck_mdp.GroundPickPhaseCommandCfg)
        if field.init
    }
    phase_base = {key: value for key, value in vars(base_command).items() if key in phase_cfg_fields}
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **phase_base,
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": THROW_PERIOD_S,
            "randomize_phase": not play,
            "initial_phase_range": (0.0, 0.05),
            "zero_phase_prob": 0.50 if not play else 0.0,
        }
    )

    # Place after set_ground_state; event dict order is the reset order.
    cfg.events["reset_ball"] = EventTermCfg(
        func=microduck_mdp.reset_ball_at_beak,
        mode="reset",
        params={
            "offset": (0.0597, 0.0, 0.0971),
            "noise_xyz": 0.001,
            "asset_name": "ball",
        },
    )
    cfg.events.pop("push_robot", None)

    cfg.terminations["beak_throw_complete"] = TerminationTermCfg(
        func=microduck_mdp.beak_throw_complete_termination,
        time_out=True,
        params={"landing_grace_steps": 2, "post_release_steps": 80},
    )
    # Early residual policies fall before the 1.2s release. Ending those
    # episodes would recreate the no-attempt attractor; invalid fallen releases
    # are reward-gated in the MDP instead and simply earn no carry.
    cfg.terminations.pop("fell_over", None)

    # Remove Header's clock (wrong task), then taper the demonstration and add
    # smoothness only after the release skill has formed. The first physical-
    # beak continuation dropped the remaining 0.5 reference weight abruptly at
    # iteration 1500; carry then drifted from about 1.00 m to 0.84 m. Hold 0.5
    # through the branch point, withdraw it over 750 iterations, and retain a
    # small 0.1 floor so the learned wind-up cannot catastrophically disappear.
    # DR stays at the small initial Header ranges; a later robustness run can
    # widen it after carry is above the acceptance gate.
    cfg.curriculum.clear()
    cfg.curriculum["reference_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "beak_reference_pose",
            "weight_stages": [
                {"step": 0, "weight": 3.0},
                {"step": 500 * 24, "weight": 1.5},
                {"step": 1000 * 24, "weight": 0.5},
                {"step": 1500 * 24, "weight": 0.5},
                {"step": 1875 * 24, "weight": 0.25},
                {"step": 2250 * 24, "weight": 0.10},
            ],
        },
    )
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.05},
                {"step": 750 * 24, "weight": -0.10},
                {"step": 1250 * 24, "weight": -0.20},
            ],
        },
    )
    return cfg


def make_microduck_beak_throw_hardware_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Fine-tune a physical-beak throw under the real robot safety envelope.

    This is a continuation task, not a from-scratch discovery task.  It keeps
    the physical ball/beak contacts and 61-observation/14-action ABI, but makes
    simulator actuation match deployment: absolute anatomical target clamps,
    no reference-motion helper, and an explicit cost on raw over-commands.
    Episodes continue through recovery so landing upright matters.
    """
    cfg = make_microduck_beak_throw_env_cfg(play=play)
    action = cfg.actions["joint_pos"]
    action.clip = dict(HARDWARE_ACTION_CLIP_RAD)
    action.max_bias = 0.0
    # robotd hands control to the standing policy at exactly one 2.4 s cycle.
    # Training beyond that point wrapped the phase and began a second wind-up
    # which the real runtime never executes.
    cfg.episode_length_s = THROW_PERIOD_S

    cfg.rewards["action_over_limit"] = RewardTermCfg(
        func=microduck_mdp.action_over_limit_penalty,
        weight=-12.0,
        params={"action_name": "joint_pos", "overshoot": 0.0},
    )
    cfg.rewards["beak_predicted_lateral"] = RewardTermCfg(
        func=microduck_mdp.beak_grip_predicted_lateral_range,
        weight=-10.0,
        params={
            "asset_name": "ball",
            "command_name": "twist",
            "release_phase": RELEASE_PHASE,
            "release_settle_phase": RELEASE_TIMEOUT_PHASE,
            "shaping_start_phase": 0.24,
            "ball_radius": TOSS_BALL_RADIUS,
            "max_range": 1.0,
        },
    )
    cfg.rewards["beak_release_lateral"] = RewardTermCfg(
        func=microduck_mdp.beak_release_lateral_velocity,
        weight=-20.0,
        params={"max_speed": 5.0},
    )
    cfg.rewards["beak_lateral_carry"] = RewardTermCfg(
        func=microduck_mdp.beak_landing_lateral_distance,
        weight=-150.0,
        params={"asset_name": "ball", "max_distance": 1.0},
    )
    cfg.rewards["sagittal_joint_deviation"] = RewardTermCfg(
        func=microduck_mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    r"^.*hip_yaw$",
                    r"^.*hip_roll$",
                    r"^head_yaw$",
                    r"^head_roll$",
                ),
            )
        },
    )
    cfg.rewards["beak_recovery_upright"] = RewardTermCfg(
        func=microduck_mdp.beak_recovery_upright_linear,
        weight=5.0,
        params={
            "command_name": "twist",
            "start_phase": 0.45,
            "full_phase": 0.70,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["beak_recovery_pose"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=1.0,
        params={
            "command_name": "twist",
            "std": 0.50,
            "hold_end": 0.50,
            "rise_end": 0.78,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["upright"].weight = 0.30
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["action_rate_l2"].weight = -0.10
    cfg.rewards["beak_reference_pose"].weight = 0.10
    cfg.rewards["beak_predicted_range"].weight = 2.0
    cfg.rewards["beak_release_range"].weight = 20.0
    cfg.rewards["beak_forward_carry"].weight = 100.0

    # The former termination stopped the episode just after touchdown.  A real
    # tester needs the duck to absorb the recoil and stay upright, so run the
    # full three-second episode and let the dense upright term train recovery.
    cfg.terminations.pop("beak_throw_complete", None)
    cfg.curriculum.clear()
    return cfg


_BEAK_ACTOR_CFG = deepcopy(MicroduckHeaderRlCfg.actor)
_BEAK_ACTOR_CFG.distribution_cfg = {
    **_BEAK_ACTOR_CFG.distribution_cfg,
    # The reference bias supplies exploration; unit-std residual noise mostly
    # knocks a 25cm biped down before release.
    "init_std": 0.35,
}


MicroduckBeakThrowRlCfg = RslRlOnPolicyRunnerCfg(
    actor=_BEAK_ACTOR_CFG,
    critic=deepcopy(MicroduckHeaderRlCfg.critic),
    algorithm=deepcopy(MicroduckHeaderRlCfg.algorithm),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_beak_throw",
    run_name="microduck_beak_throw",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=3_000,
)


_HARDWARE_ACTOR_CFG = deepcopy(MicroduckBeakThrowRlCfg.actor)
_HARDWARE_ACTOR_CFG.distribution_cfg = {
    **_HARDWARE_ACTOR_CFG.distribution_cfg,
    "init_std": 0.12,
}

_HARDWARE_ALGORITHM_CFG = deepcopy(MicroduckBeakThrowRlCfg.algorithm)
_HARDWARE_ALGORITHM_CFG.learning_rate = 1.0e-4
_HARDWARE_ALGORITHM_CFG.entropy_coef = 0.001
_HARDWARE_ALGORITHM_CFG.desired_kl = 0.003
_HARDWARE_ALGORITHM_CFG.symmetry_cfg = {
    **SYMMETRY_CFG,
    "mirror_loss_coeff": 0.5,
}


MicroduckBeakThrowHardwareRlCfg = RslRlOnPolicyRunnerCfg(
    actor=_HARDWARE_ACTOR_CFG,
    critic=deepcopy(MicroduckBeakThrowRlCfg.critic),
    algorithm=_HARDWARE_ALGORITHM_CFG,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_beak_throw_hardware",
    run_name="microduck_beak_throw_hardware",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=750,
)
