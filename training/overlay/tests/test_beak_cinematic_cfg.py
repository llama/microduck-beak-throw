"""Contracts for the simulator-first cinematic beak throw pair."""

from mjlab_microduck.robot.microduck_constants import (
    BEAK_BALL_SITE_OFFSET,
    MICRODUCK_BEAK_ROBOT_CFG,
    MICRODUCK_CINEMATIC_BEAK_COLLISION,
    MICRODUCK_CINEMATIC_BEAK_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_beak_cinematic_env_cfg import (
    CINEMATIC_MOUTH_OPEN_PHASE,
    CINEMATIC_RELEASE_PHASE,
    MicroduckBeakCinematicTrainRlCfg,
    make_microduck_beak_cinematic_env_cfg,
    make_microduck_beak_cinematic_train_env_cfg,
)
from mjlab_microduck.tasks.microduck_beak_throw_env_cfg import (
    make_microduck_beak_throw_env_cfg,
)


def test_showcase_uses_an_isolated_latch_and_opens_before_release():
    physical = make_microduck_beak_throw_env_cfg(play=True)
    cinematic = make_microduck_beak_cinematic_env_cfg(play=True)

    assert isinstance(
        physical.actions["joint_pos"],
        microduck_mdp.PhaseReferenceJointPositionActionCfg,
    )
    assert not isinstance(
        physical.actions["joint_pos"],
        microduck_mdp.CinematicBeakJointPositionActionCfg,
    )
    action = cinematic.actions["joint_pos"]
    assert isinstance(action, microduck_mdp.CinematicBeakJointPositionActionCfg)
    assert action.class_type is microduck_mdp.CinematicBeakJointPositionAction
    assert action.local_grip_offset == BEAK_BALL_SITE_OFFSET
    assert action.mouth_release_phase == CINEMATIC_MOUTH_OPEN_PHASE
    assert action.latch_release_phase == CINEMATIC_RELEASE_PHASE
    assert action.mouth_release_phase < action.latch_release_phase
    assert action.max_bias == 0.0
    assert physical.scene.entities["robot"] is MICRODUCK_BEAK_ROBOT_CFG
    assert cinematic.scene.entities["robot"] is MICRODUCK_CINEMATIC_BEAK_ROBOT_CFG
    assert cinematic.scene.entities["robot"].collisions == (
        MICRODUCK_CINEMATIC_BEAK_COLLISION,
    )
    assert (
        MICRODUCK_CINEMATIC_BEAK_COLLISION.contype[r"^beak_.*_collision$"]
        == 4
    )


def test_cinematic_checkpoint_abi_and_phase_are_model_1250_compatible():
    physical = make_microduck_beak_throw_env_cfg(play=True)
    cinematic = make_microduck_beak_cinematic_env_cfg(play=True)

    assert list(cinematic.observations["actor"].terms) == list(
        physical.observations["actor"].terms
    )
    assert cinematic.actions["joint_pos"].actuator_names == (r"^(?!passive_).*",)
    command = cinematic.commands["twist"]
    assert command.randomize_phase is False
    assert command.initial_phase_range == (0.0, 0.0)
    assert cinematic.curriculum == {}


def test_cinematic_reward_owner_does_not_reuse_physical_release_detection():
    cfg = make_microduck_beak_cinematic_env_cfg()
    rewards = cfg.rewards
    assert (
        rewards["beak_predicted_range"].func
        is microduck_mdp.cinematic_beak_predicted_range
    )
    assert rewards["beak_predicted_range"].params["release_phase"] == (
        CINEMATIC_RELEASE_PHASE
    )
    assert "release_settle_phase" not in rewards["beak_predicted_range"].params
    assert "local_grip_offset" not in rewards["beak_predicted_range"].params


def test_train_companion_rewards_distance_straightness_and_smoothness():
    cfg = make_microduck_beak_cinematic_train_env_cfg()
    rewards = cfg.rewards
    assert rewards["beak_forward_carry"].weight > 0.0
    assert rewards["beak_lateral_carry"].weight < 0.0
    assert rewards["beak_lateral_carry"].func is (
        microduck_mdp.beak_landing_lateral_distance
    )
    assert rewards["action_rate_l2"].weight < 0.0
    assert rewards["upright"].weight > 0.0
    assert rewards["beak_reference_pose"].weight > 0.0

    alg = MicroduckBeakCinematicTrainRlCfg.algorithm
    assert alg.learning_rate < 1.0e-3
    assert alg.entropy_coef < 0.01
    assert alg.desired_kl < 0.01
    assert MicroduckBeakCinematicTrainRlCfg.actor.distribution_cfg["init_std"] < 0.35
