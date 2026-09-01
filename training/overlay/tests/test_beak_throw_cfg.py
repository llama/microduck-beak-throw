"""Config and physics-contract tests for the phase-controlled beak throw."""

import mujoco
import numpy as np
import torch

from mjlab_microduck.robot.microduck_constants import (
    BEAK_BALL_SITE_OFFSET,
    BEAK_CLOSED_ANGLE_RAD,
    BEAK_HOLD_ANGLE_RAD,
    BEAK_MOUTH_JOINT,
    BEAK_OPEN_ANGLE_RAD,
    MICRODUCK_BEAK_COLLISION,
    MICRODUCK_BEAK_ROBOT_CFG,
    build_beak_scene_spec,
    get_beak_spec,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_beak_throw_env_cfg import (
    HARDWARE_ACTION_CLIP_RAD,
    RELEASE_PHASE,
    SNAP_END_PHASE,
    THROW_PERIOD_S,
    make_microduck_beak_throw_env_cfg,
    make_microduck_beak_throw_hardware_env_cfg,
)


def _collides(g1, g2):
    return bool(
        (int(g1.contype[0]) & int(g2.conaffinity[0]))
        | (int(g2.contype[0]) & int(g1.conaffinity[0]))
    )


def test_beak_variant_has_no_tray_and_ball_ignores_head_hulls():
    spec = get_beak_spec()
    assert all(body.name != "bill_tray" for body in spec.bodies)
    floor = spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1]
    )
    ball_body = spec.worldbody.add_body(name="test_ball", pos=[0.1, 0.0, 0.2])
    ball_body.add_freejoint()
    ball = ball_body.add_geom(
        name="test_ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0]
    )
    ball.contype = 2
    ball.conaffinity = 1
    model = spec.compile()
    bg = model.geom("test_ball_geom")
    fg = model.geom("floor")
    assert _collides(bg, fg)
    jaw_id = model.body("jaw_soft").id
    hulls = [
        model.geom(i)
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == jaw_id and int(model.geom(i).contype[0]) == 4
    ]
    assert len(hulls) == 3
    assert all(not _collides(bg, hull) for hull in hulls)
    assert all(_collides(fg, hull) for hull in hulls)


def test_beak_collision_cfg_survives_mjlab_import_masks():
    assert MICRODUCK_BEAK_ROBOT_CFG.collisions == (MICRODUCK_BEAK_COLLISION,)
    assert MICRODUCK_BEAK_COLLISION.contype[r"^head_hull_.*_collision$"] == 4
    assert not any("tray" in expr for expr in MICRODUCK_BEAK_COLLISION.contype)


def test_plain_mujoco_beak_rehearsal_scene_builds():
    model = build_beak_scene_spec().compile()
    assert model.nu == 15
    assert model.nkey >= 2
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bill_tray") < 0


def test_mouth_uses_hardware_range_and_physical_pocket_retains_then_releases():
    model = build_beak_scene_spec().compile()
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    mouth = model.joint(BEAK_MOUTH_JOINT)
    mouth_q = int(model.jnt_qposadr[mouth.id])
    mouth_ctrl = next(
        i for i in range(model.nu) if model.actuator_trnid[i, 0] == mouth.id
    )
    assert np.allclose(
        model.jnt_range[mouth.id],
        [BEAK_CLOSED_ANGLE_RAD, BEAK_OPEN_ANGLE_RAD],
    )
    assert BEAK_CLOSED_ANGLE_RAD < BEAK_HOLD_ANGLE_RAD < BEAK_OPEN_ANGLE_RAD

    ball = model.joint("ball_free")
    ball_q = int(model.jnt_qposadr[ball.id])
    ball_v = int(model.jnt_dofadr[ball.id])
    site_id = model.site("mouth_tip").id
    jaw_id = model.body("jaw_soft").id
    ball_body_id = model.body("toss_ball").id

    data.qpos[mouth_q] = BEAK_HOLD_ANGLE_RAD
    data.ctrl[mouth_ctrl] = BEAK_HOLD_ANGLE_RAD
    mujoco.mj_forward(model, data)
    site_rot = data.site_xmat[site_id].reshape(3, 3)
    spawn = data.site_xpos[site_id] + site_rot @ np.asarray(BEAK_BALL_SITE_OFFSET)
    data.qpos[ball_q : ball_q + 3] = spawn
    data.qpos[ball_q + 3 : ball_q + 7] = [1, 0, 0, 0]
    data.qvel[ball_v : ball_v + 6] = 0
    robot_qpos = data.qpos[:ball_q].copy()
    mujoco.mj_forward(model, data)

    initial_local = data.xmat[jaw_id].reshape(3, 3).T @ (
        data.xpos[ball_body_id] - data.xpos[jaw_id]
    )
    for _ in range(500):
        data.qpos[:ball_q] = robot_qpos
        data.qpos[mouth_q] = BEAK_HOLD_ANGLE_RAD
        data.ctrl[mouth_ctrl] = BEAK_HOLD_ANGLE_RAD
        data.qvel[:ball_v] = 0
        mujoco.mj_step(model, data)
    held_local = data.xmat[jaw_id].reshape(3, 3).T @ (
        data.xpos[ball_body_id] - data.xpos[jaw_id]
    )
    assert np.linalg.norm(held_local - initial_local) < 0.003

    # At full open, ordinary outward throw velocity clears the smaller throat;
    # no weld, equality, or position rewrite carries the ball forward.
    data.qpos[:ball_q] = robot_qpos
    data.qpos[mouth_q] = BEAK_OPEN_ANGLE_RAD
    data.ctrl[mouth_ctrl] = BEAK_OPEN_ANGLE_RAD
    data.qpos[ball_q : ball_q + 3] = spawn
    data.qpos[ball_q + 3 : ball_q + 7] = [1, 0, 0, 0]
    data.qvel[:ball_v] = 0
    data.qvel[ball_v : ball_v + 6] = [1.5, 0, 0.2, 0, 0, 0]
    mujoco.mj_forward(model, data)
    for _ in range(120):
        data.qpos[:ball_q] = robot_qpos
        data.qpos[mouth_q] = BEAK_OPEN_ANGLE_RAD
        data.ctrl[mouth_ctrl] = BEAK_OPEN_ANGLE_RAD
        data.qvel[:ball_v] = 0
        mujoco.mj_step(model, data)
    assert data.xpos[ball_body_id, 0] > spawn[0] + 0.08
    assert all(
        not (
            model.geom(data.contact[i].geom1).name.startswith("beak_")
            or model.geom(data.contact[i].geom2).name.startswith("beak_")
        )
        for i in range(data.ncon)
        if "toss_ball_geom"
        in (
            model.geom(data.contact[i].geom1).name,
            model.geom(data.contact[i].geom2).name,
        )
    )


def test_beak_env_has_phase_release_and_no_header_economics():
    cfg = make_microduck_beak_throw_env_cfg()
    command = cfg.commands["twist"]
    assert command.class_type is microduck_mdp.GroundPickPhaseCommand
    assert command.period == THROW_PERIOD_S
    assert command.initial_phase_range[1] < RELEASE_PHASE
    assert command.zero_phase_prob >= 0.5

    for retired in (
        "ball_held",
        "held_fwd_shaping",
        "ball_forward_velocity",
        "launch_bonus",
        "ball_distance",
        "ball_tray_contact",
    ):
        assert retired not in cfg.rewards
    assert "push_robot" not in cfg.events
    assert "fell_over" not in cfg.terminations
    assert isinstance(
        cfg.actions["joint_pos"], microduck_mdp.PhaseReferenceJointPositionActionCfg
    )
    assert cfg.actions["joint_pos"].max_bias == 1.0
    assert cfg.actions["joint_pos"].release_phase == SNAP_END_PHASE
    assert cfg.actions["joint_pos"].actuator_names == (r"^(?!passive_).*",)
    assert cfg.actions["joint_pos"].mouth_hold_position == BEAK_HOLD_ANGLE_RAD
    assert cfg.actions["joint_pos"].mouth_open_position == BEAK_OPEN_ANGLE_RAD
    assert RELEASE_PHASE < SNAP_END_PHASE
    # MuJoCo pitch convention: the snap goes from positive to negative pitch.
    assert cfg.actions["joint_pos"].windup_pose["neck_pitch"] > 0.0
    assert cfg.actions["joint_pos"].release_pose["neck_pitch"] < 0.0


def test_beak_reward_order_and_aligned_terminal_objective():
    cfg = make_microduck_beak_throw_env_cfg()
    rewards = cfg.rewards
    assert rewards["beak_predicted_range"].func is microduck_mdp.beak_grip_predicted_range
    assert rewards["beak_forward_carry"].func is microduck_mdp.beak_landing_forward_distance
    keys = list(rewards)
    owner = keys.index("beak_predicted_range")
    assert owner < keys.index("beak_release_range")
    assert owner < keys.index("beak_release_forward")
    assert owner < keys.index("beak_release_upward")
    assert rewards["beak_forward_carry"].weight > rewards["beak_release_range"].weight
    assert rewards["beak_reference_pose"].params["joint_indices"] == (5, 6)
    assert rewards["beak_predicted_range"].params["release_phase"] == RELEASE_PHASE


def test_reset_order_and_success_termination():
    cfg = make_microduck_beak_throw_env_cfg()
    keys = list(cfg.events)
    assert keys.index("reset_ball") > keys.index("set_ground_state")
    assert cfg.events["reset_ball"].func is microduck_mdp.reset_ball_at_beak
    term = cfg.terminations["beak_throw_complete"]
    assert term.func is microduck_mdp.beak_throw_complete_termination
    assert term.time_out is True


def test_play_starts_at_exact_deployment_phase_zero():
    cfg = make_microduck_beak_throw_env_cfg(play=True)
    command = cfg.commands["twist"]
    assert command.randomize_phase is False
    assert cfg.actions["joint_pos"].max_bias == 0.0


def test_actor_contract_stays_61d_family_layout():
    from mjlab_microduck.tasks.microduck_header_env_cfg import make_microduck_header_env_cfg

    beak = make_microduck_beak_throw_env_cfg()
    header = make_microduck_header_env_cfg()
    assert list(beak.observations["actor"].terms) == list(header.observations["actor"].terms)


def test_hardware_task_matches_runtime_limits_and_trains_recovery():
    cfg = make_microduck_beak_throw_hardware_env_cfg()
    action = cfg.actions["joint_pos"]
    assert action.clip == HARDWARE_ACTION_CLIP_RAD
    assert action.max_bias == 0.0
    assert cfg.episode_length_s == THROW_PERIOD_S
    assert cfg.rewards["action_over_limit"].func is microduck_mdp.action_over_limit_penalty
    assert cfg.rewards["action_over_limit"].weight < 0.0
    assert cfg.rewards["action_over_limit"].params["overshoot"] == 0.0
    assert cfg.rewards["beak_predicted_lateral"].func is (
        microduck_mdp.beak_grip_predicted_lateral_range
    )
    assert cfg.rewards["beak_predicted_lateral"].weight < 0.0
    assert cfg.rewards["beak_release_lateral"].func is (
        microduck_mdp.beak_release_lateral_velocity
    )
    assert cfg.rewards["beak_release_lateral"].weight < 0.0
    assert cfg.rewards["beak_lateral_carry"].func is (
        microduck_mdp.beak_landing_lateral_distance
    )
    assert cfg.rewards["beak_lateral_carry"].weight < 0.0
    assert cfg.rewards["beak_recovery_upright"].func is (
        microduck_mdp.beak_recovery_upright_linear
    )
    assert cfg.rewards["beak_recovery_upright"].weight > 0.0
    assert cfg.rewards["beak_recovery_pose"].weight > 0.0
    assert cfg.rewards["beak_forward_carry"].weight < 150.0
    assert "beak_throw_complete" not in cfg.terminations
    assert not cfg.curriculum

    assert np.allclose(
        HARDWARE_ACTION_CLIP_RAD[r"^left_hip_yaw$"],
        np.radians([-25.0, 30.0]),
    )
    assert np.allclose(
        HARDWARE_ACTION_CLIP_RAD[r"^head_roll$"],
        np.radians([-25.0, 25.0]),
    )


def test_ballistic_predictor_rewards_forward_and_loft_not_sideways():
    pos = torch.tensor([[0.0, 0.0, 0.22]]).repeat(3, 1)
    vel = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    direction = torch.tensor([[1.0, 0.0]]).repeat(3, 1)
    distance = microduck_mdp._vacuum_forward_range(pos, vel, direction, 0.012, 3.0)
    assert distance[1] > distance[0] > 0.0
    assert distance[2] == 0.0
    lateral = microduck_mdp._vacuum_lateral_range(
        pos, vel, direction, 0.012, 3.0
    )
    assert lateral[0] == 0.0
    assert lateral[1] == 0.0
    assert lateral[2] > 0.0


def test_penalty_signs_and_reference_curriculum():
    cfg = make_microduck_beak_throw_env_cfg()
    for name in ("body_ang_vel", "angular_momentum", "dof_pos_limits", "action_rate_l2", "self_collisions"):
        assert cfg.rewards[name].weight < 0
    stages = cfg.curriculum["reference_weight"].params["weight_stages"]
    weights = [stage["weight"] for stage in stages]
    assert weights == sorted(weights, reverse=True)
    assert stages[-3:] == [
        {"step": 1500 * 24, "weight": 0.5},
        {"step": 1875 * 24, "weight": 0.25},
        {"step": 2250 * 24, "weight": 0.10},
    ]
    assert weights[-1] == 0.10
