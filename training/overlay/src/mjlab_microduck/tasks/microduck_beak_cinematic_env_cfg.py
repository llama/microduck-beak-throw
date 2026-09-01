"""Cinematic beak-throw tasks built around the original model_1250 motion.

These tasks intentionally optimize appearance and repeatability, not sim-to-real
fidelity.  The visible beak is partially closed around the ball during the
wind-up.  An invisible latch follows the beak pocket until the mouth begins to
open, then releases the ball with the pocket's measured velocity.  Flight and
touchdown remain ordinary MuJoCo physics.

``Cinematic`` is the stable showcase task for the existing checkpoint.
``Cinematic-Train`` has the same observation/action ABI and mechanism, plus a
conservative reward/optimizer recipe for improving distance and presentation
without rapidly erasing the learned wind-up.
"""

from copy import deepcopy
from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.robot.microduck_constants import (
    BEAK_BALL_SITE_OFFSET,
    BEAK_HOLD_ANGLE_RAD,
    BEAK_MOUTH_JOINT,
    BEAK_OPEN_ANGLE_RAD,
    MICRODUCK_CINEMATIC_BEAK_ROBOT_CFG,
    TOSS_BALL_RADIUS,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_beak_throw_env_cfg import (
    MOUTH_OPEN_DURATION_PHASE,
    RECOVERY_PHASE,
    RELEASE_POSE,
    SNAP_END_PHASE,
    THROW_PERIOD_S,
    WINDUP_PHASE,
    WINDUP_POSE,
    MicroduckBeakThrowRlCfg,
    make_microduck_beak_throw_env_cfg,
)


# This is the timing used by the original virtual-release training run that
# produced model_1250.  Opening begins 72 ms earlier (0.03 of a 2.4 s cycle),
# so the beak visibly lets go instead of the ball simply detaching from it.
CINEMATIC_RELEASE_PHASE = 0.38
CINEMATIC_MOUTH_OPEN_PHASE = 0.35
CINEMATIC_MOUTH_OPEN_DURATION_PHASE = MOUTH_OPEN_DURATION_PHASE


def _install_cinematic_action(cfg: ManagerBasedRlEnvCfg) -> None:
    base_action = cfg.actions["joint_pos"]
    action_fields = {
        field.name
        for field in fields(microduck_mdp.CinematicBeakJointPositionActionCfg)
        if field.init
    }
    action_base = {
        key: value for key, value in vars(base_action).items() if key in action_fields
    }
    cfg.actions["joint_pos"] = microduck_mdp.CinematicBeakJointPositionActionCfg(
        **{
            **action_base,
            "class_type": microduck_mdp.CinematicBeakJointPositionAction,
            "max_bias": 0.0,
            "mouth_joint_name": BEAK_MOUTH_JOINT,
            "mouth_release_phase": CINEMATIC_MOUTH_OPEN_PHASE,
            "mouth_open_duration_phase": CINEMATIC_MOUTH_OPEN_DURATION_PHASE,
            "mouth_hold_position": BEAK_HOLD_ANGLE_RAD,
            "mouth_open_position": BEAK_OPEN_ANGLE_RAD,
            "ball_asset_name": "ball",
            "mouth_site_name": "mouth_tip",
            "local_grip_offset": BEAK_BALL_SITE_OFFSET,
            "latch_release_phase": CINEMATIC_RELEASE_PHASE,
        }
    )


def make_microduck_beak_cinematic_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stable showcase environment for model_1250 and compatible policies."""
    cfg = make_microduck_beak_throw_env_cfg(play=play)
    cfg.scene.entities["robot"] = MICRODUCK_CINEMATIC_BEAK_ROBOT_CFG
    _install_cinematic_action(cfg)

    # A throw always starts at the beginning of the learned wind-up.  Random
    # mid-cycle starts are incompatible with a pre-release cinematic latch and
    # make a showcase visually inconsistent.
    command = cfg.commands["twist"]
    command.period = THROW_PERIOD_S
    command.randomize_phase = False
    command.initial_phase_range = (0.0, 0.0)
    command.zero_phase_prob = 0.0

    owner = cfg.rewards["beak_predicted_range"]
    owner.func = microduck_mdp.cinematic_beak_predicted_range
    owner.params = {
        "command_name": "twist",
        "release_phase": CINEMATIC_RELEASE_PHASE,
        "shaping_start_phase": 0.28,
        "ball_radius": TOSS_BALL_RADIUS,
        "max_range": 3.0,
    }

    # The fixed task is deliberately free of training-time weight schedules.
    # Loading model_1250 therefore cannot pick up a hidden reference motion.
    cfg.curriculum.clear()
    return cfg


def make_microduck_beak_cinematic_train_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Conservative continuation environment for polishing model_1250."""
    cfg = make_microduck_beak_cinematic_env_cfg(play=play)

    # Preserve the successful wind-up while rewarding a longer, straighter
    # throw.  The lateral term is terminal, just like forward carry, so PPO
    # cannot farm it while the ball is still latched.
    cfg.rewards["beak_predicted_range"].weight = 6.0
    cfg.rewards["beak_release_range"].weight = 50.0
    cfg.rewards["beak_forward_carry"].weight = 180.0
    cfg.rewards["beak_lateral_carry"] = RewardTermCfg(
        func=microduck_mdp.beak_landing_lateral_distance,
        weight=-60.0,
        params={"asset_name": "ball", "max_distance": 1.0},
    )
    cfg.rewards["beak_reference_pose"].weight = 0.25
    cfg.rewards["upright"].weight = 0.45
    cfg.rewards["action_rate_l2"].weight = -0.10
    return cfg


_CINEMATIC_ACTOR_CFG = deepcopy(MicroduckBeakThrowRlCfg.actor)
_CINEMATIC_ACTOR_CFG.distribution_cfg = {
    **_CINEMATIC_ACTOR_CFG.distribution_cfg,
    "init_std": 0.18,
}

_CINEMATIC_ALGORITHM_CFG = deepcopy(MicroduckBeakThrowRlCfg.algorithm)
_CINEMATIC_ALGORITHM_CFG.learning_rate = 2.0e-4
_CINEMATIC_ALGORITHM_CFG.entropy_coef = 0.002
_CINEMATIC_ALGORITHM_CFG.desired_kl = 0.004


MicroduckBeakCinematicRlCfg = RslRlOnPolicyRunnerCfg(
    actor=deepcopy(MicroduckBeakThrowRlCfg.actor),
    critic=deepcopy(MicroduckBeakThrowRlCfg.critic),
    algorithm=deepcopy(MicroduckBeakThrowRlCfg.algorithm),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_beak_cinematic",
    run_name="microduck_beak_cinematic",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=500,
)


MicroduckBeakCinematicTrainRlCfg = RslRlOnPolicyRunnerCfg(
    actor=_CINEMATIC_ACTOR_CFG,
    critic=deepcopy(MicroduckBeakThrowRlCfg.critic),
    algorithm=_CINEMATIC_ALGORITHM_CFG,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_beak_cinematic_train",
    run_name="microduck_beak_cinematic_train",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=750,
)
