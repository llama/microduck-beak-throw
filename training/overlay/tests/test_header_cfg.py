"""Config-invariant tests for the Header (bill-tray toss) env.

Locks in the toss-model contact contract (the structural anti-cheat), the
kick-family reward recipe, and the measured tray geometry.
"""
import math

import mujoco
import numpy as np

from mjlab_microduck.robot.microduck_constants import (
    TOSS_TRAY_GEOMS,
    TOSS_TRAY_OFFSET_FROM_ROOT,
    get_toss_spec,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_header_env_cfg import (
    BALL_TARGET_SPEED,
    make_microduck_header_env_cfg,
    MicroduckHeaderRlCfg,
)


def _collides(g1, g2):
    """MuJoCo mask rule: (c1 & a2) | (c2 & a1)."""
    return bool(
        (int(g1.contype[0]) & int(g2.conaffinity[0]))
        | (int(g2.contype[0]) & int(g1.conaffinity[0]))
    )


def test_toss_model_contact_contract():
    """The ball may touch tray/floor/legs but NEVER the head hulls — the only
    way to accelerate the ball is through the bill. This is the anti-cheat."""
    spec = get_toss_spec()
    f = spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1]
    )
    b = spec.worldbody.add_body(name="ballbody", pos=[0.3, 0, 0.1])
    b.add_freejoint()
    bg = b.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0])
    bg.contype = 2
    bg.conaffinity = 1
    model = spec.compile()

    ball = model.geom("ball_geom")
    floor = model.geom("floor")
    tray = model.geom("tray_floor_collision")
    foot = model.geom("left_foot_collision")
    jaw_body = model.body("jaw_soft").id
    head_hulls = [
        model.geom(i)
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == jaw_body and int(model.geom(i).contype[0]) == 4
    ]
    assert len(head_hulls) == 3, "expected jaw + top/bottom head shells remapped"

    assert _collides(ball, floor)
    assert _collides(ball, tray)
    assert _collides(ball, foot)
    for h in head_hulls:
        assert not _collides(ball, h), "ball must not touch the head hulls"
        assert _collides(h, floor), "head must still hit the floor when falling"


def test_tray_geometry_is_the_measured_one():
    names = [n for n, _, _ in TOSS_TRAY_GEOMS]
    assert names == [
        "tray_floor_collision", "tray_lip_front_collision", "tray_rail_left_collision", "tray_rail_right_collision", "tray_back_collision",
    ]
    # tray sits above and in front of the trunk root
    ox, oy, oz = TOSS_TRAY_OFFSET_FROM_ROOT
    # head-top mount: slightly forward of center, well above the trunk
    assert 0.0 < ox < 0.05 and abs(oy) < 1e-6 and oz > 0.14


def test_ball_entity_and_reset_event_are_wired():
    cfg = make_microduck_header_env_cfg()
    assert list(cfg.scene.entities.keys())[0] == "robot", "robot must stay first entity"
    assert "ball" in cfg.scene.entities
    ev = cfg.events["reset_ball"]
    assert ev.func is microduck_mdp.reset_ball_in_tray
    assert ev.params["offset"] == TOSS_TRAY_OFFSET_FROM_ROOT
    # ball reset must run AFTER the ground-state reset (dict insertion order)
    keys = list(cfg.events.keys())
    assert keys.index("reset_ball") > keys.index("set_ground_state")


def test_toss_reward_recipe_matches_kick_family():
    cfg = make_microduck_header_env_cfg()
    fwd = cfg.rewards["ball_forward_velocity"]
    over = cfg.rewards["ball_speed_overshoot"]
    assert fwd.params["max_speed"] == BALL_TARGET_SPEED
    assert over.params["target_speed"] == BALL_TARGET_SPEED
    # at-target payoff ≈ +3/step (kick's scaling rule)
    assert abs(fwd.weight * BALL_TARGET_SPEED - 3.0) < 0.11
    # overshoot slope must stay below the forward slope (erring hard stays
    # cheaper than not tossing)
    assert over.weight < 0
    assert abs(over.weight) < fwd.weight


def test_toss_reward_is_launch_latched():
    """Anti-dump-hack regression (run nelovxk5): forward reward must be the
    LATCHED variant — a legit launch requires upward + forward exit velocity —
    and the latch-owning term must be registered before the read-only one."""
    cfg = make_microduck_header_env_cfg()
    fwd = cfg.rewards["ball_forward_velocity"]
    over = cfg.rewards["ball_speed_overshoot"]
    assert fwd.func is microduck_mdp.ball_forward_velocity_tossed
    assert over.func is microduck_mdp.ball_speed_overshoot_tossed
    # vz must NOT be required: the transition is detected mid-fall below the
    # tray where vz is always ~-1.2 (the throw_v4 postmortem bug). Forward
    # velocity is ballistically preserved and is the real throw/dump divider.
    assert fwd.params["min_launch_vz"] < -1.5
    assert fwd.params["min_launch_fwd"] >= 0.3
    keys = list(cfg.rewards.keys())
    assert keys.index("ball_forward_velocity") < keys.index("ball_speed_overshoot")


def test_ball_held_is_generous_early_and_decays():
    """v2 lesson (run wibozuwz): a small always-on hold reward is invisible
    against the standing stack — the policy ignores the ball forever. Hold
    must be a REAL gradient early, then decay so tossing wins the endgame."""
    cfg = make_microduck_header_env_cfg()
    held = cfg.rewards["ball_held"]
    assert held.func is microduck_mdp.ball_held_reward
    assert held.weight >= 1.0, "early hold gradient must be visible"
    stages = cfg.curriculum["held_weight"].params["weight_stages"]
    weights = [s["weight"] for s in stages]
    assert weights[0] == held.weight
    assert weights == sorted(weights, reverse=True)
    # endgame: hold-forever must lose to one completed toss
    final_hold = weights[-1] * cfg.episode_length_s * 50
    toss_value = cfg.rewards["launch_bonus"].weight + 3.0 * 60
    assert final_hold < toss_value, "hold-forever must lose to tossing at final weights"


def test_launch_bonus_is_one_time_and_after_forward_term():
    cfg = make_microduck_header_env_cfg()
    bonus = cfg.rewards["launch_bonus"]
    assert bonus.func is microduck_mdp.ball_toss_launch_bonus
    assert bonus.weight > 0
    keys = list(cfg.rewards.keys())
    # latch state owner must run first
    assert keys.index("ball_forward_velocity") < keys.index("launch_bonus")


def test_both_feet_have_anti_hop_rewards():
    cfg = make_microduck_header_env_cfg()
    sensor_names = {s.name for s in cfg.scene.sensors}
    for name in ("left_foot_grounded", "right_foot_grounded"):
        r = cfg.rewards[name]
        assert r.func is microduck_mdp.single_foot_grounded_reward
        assert r.weight > 0
        assert r.params["sensor_name"] in sensor_names


def test_neck_stays_flickable():
    # The flick IS a neck motion: neck pose reward must be looser/lighter than
    # kick's head-pinning values (std 0.3 / weight 1.0).
    cfg = make_microduck_header_env_cfg()
    neck = cfg.rewards["pose_stand_neck"]
    assert neck.params["std"] >= 0.5
    assert neck.weight < 1.0


def test_actor_is_ball_blind_critic_sees_ball():
    cfg = make_microduck_header_env_cfg()
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    assert "ball_position" not in actor and "ball_velocity" not in actor
    assert critic["ball_position"].func is microduck_mdp.ball_pos_in_base
    assert critic["ball_velocity"].func is microduck_mdp.ball_vel_in_base


def test_actor_observation_layout_parity_with_kick():
    from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
        make_microduck_ball_kick_env_cfg,
    )
    header = make_microduck_header_env_cfg()
    kick = make_microduck_ball_kick_env_cfg()
    assert list(header.observations["actor"].terms.keys()) == list(
        kick.observations["actor"].terms.keys()
    ), "actor layout must match the hot-swap family"


def test_penalty_sign_conventions():
    cfg = make_microduck_header_env_cfg()
    assert cfg.rewards["ball_speed_overshoot"].weight < 0   # cost ≥ 0
    assert cfg.rewards["action_rate_l2"].weight < 0
    assert cfg.rewards["body_ang_vel"].weight < 0
    assert cfg.rewards["angular_momentum"].weight < 0
    assert cfg.rewards["self_collisions"].weight < 0


def test_nan_guard_and_bam_events():
    cfg = make_microduck_header_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events
    assert "nan_state" in cfg.terminations
    assert "fell_over" in cfg.terminations
    critic = cfg.observations["critic"].terms
    if "foot_contact_forces" in critic:
        assert critic["foot_contact_forces"].func is microduck_mdp.foot_contact_forces_safe


def test_symmetry_augmentation_is_disabled():
    assert MicroduckHeaderRlCfg.algorithm.symmetry_cfg is None


def test_ball_rests_in_tray_physically():
    """End-to-end CPU physics check on the actual toss spec: the ball placed at
    the reset offset settles in the tray (base pinned; balance is the policy's
    job). Locks the tray geometry against model revisions."""
    spec = get_toss_spec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    b = spec.worldbody.add_body(name="ballbody", pos=[0.3, 0, 0.1])
    b.add_freejoint()
    bg = b.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0])
    bg.contype = 2
    bg.conaffinity = 1
    bg.density = 0.003 / (4 / 3 * np.pi * 0.012 ** 3)
    model = spec.compile()
    d = mujoco.MjData(model)
    home = {
        "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579, "left_knee": -0.0049,
        "left_ankle": 0.4530, "neck_pitch": 0.3491, "head_pitch": 0.3491,
        "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579, "right_knee": 0.0049,
        "right_ankle": -0.4530,
    }
    d.qpos[2] = 0.12
    d.qpos[3] = 1.0
    for j, v in home.items():
        d.qpos[model.joint(j).qposadr[0]] = v
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 15.0
        model.actuator_biasprm[i, 1] = -15.0
        model.actuator_biasprm[i, 2] = -0.5
        model.actuator_forcerange[i] = [-20, 20]
        jname = model.joint(model.actuator(i).trnid[0]).name
        d.ctrl[i] = home.get(jname, 0.0)
    d.qpos[-7:-4] = d.qpos[0:3] + np.array(TOSS_TRAY_OFFSET_FROM_ROOT) + [0, 0, 0.018]
    d.qpos[-4:] = [1, 0, 0, 0]
    mujoco.mj_forward(model, d)
    pin = d.qpos[:7].copy()
    for _ in range(int(1.5 / model.opt.timestep)):
        d.qpos[:7] = pin
        d.qvel[:6] = 0.0
        mujoco.mj_step(model, d)
    ball_z = d.xpos[model.body("ballbody").id][2]
    speed = np.linalg.norm(d.qvel[-6:-3])
    assert ball_z > 0.18, f"ball fell out of the tray (z={ball_z:.3f})"
    assert speed < 0.05, f"ball not settled (speed={speed:.3f})"


def test_toss_rehearsal_scene_builds():
    """infer_policy's --header scene: scene.xml + toss variant + ball_free.
    Locks the rehearsal path (viewer) to the same spec transform as training."""
    from mjlab_microduck.robot.microduck_constants import build_toss_scene_spec

    model = build_toss_scene_spec().compile()
    assert model.nu == 14, "action space must stay 14 servos"
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free") >= 0
    assert model.nkey >= 2  # scene keyframes survive (padded for the ball)
    ball = model.geom("toss_ball_geom")
    tray = model.geom("tray_floor_collision")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert _collides(ball, tray)
    if floor_id >= 0:
        assert _collides(ball, model.geom(floor_id))
    jaw_body = model.body("jaw_soft").id
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == jaw_body and int(model.geom(i).contype[0]) == 4:
            assert not _collides(ball, model.geom(i))


def test_toss_collision_cfg_masks_survive_mjlab_pipeline():
    """Header v1-v4 postmortem: FULL_COLLISION (disable_other_geoms=True)
    zeroed the tray masks and clobbered the head-hull masks at entity import —
    the GPU ball never touched the tray while CPU tests kept passing. The toss
    robot must use its own CollisionCfg whose dicts recreate the intended
    masks, and every tray/hull geom must carry a _collision name so the regex
    management can see it."""
    from mjlab_microduck.robot.microduck_constants import (
        MICRODUCK_TOSS_COLLISION,
        MICRODUCK_TOSS_ROBOT_CFG,
        get_toss_spec,
    )
    from mjlab.utils.string import filter_exp, resolve_expr

    assert MICRODUCK_TOSS_ROBOT_CFG.collisions == (MICRODUCK_TOSS_COLLISION,)
    spec = get_toss_spec()
    names = tuple(g.name for g in spec.geoms if g.name)
    managed = filter_exp(MICRODUCK_TOSS_COLLISION.geom_names_expr, names)
    for n, _, _ in TOSS_TRAY_GEOMS:
        assert n in managed, f"{n} not managed by CollisionCfg"
    hulls = [n for n in names if n.startswith("head_hull_")]
    assert len(hulls) == 3
    for h in hulls:
        assert h in managed
    # resolved masks: ball (contype 2 / conaffinity 1) must hit tray, not hulls
    test_names = tuple([TOSS_TRAY_GEOMS[0][0]] + hulls[:1])
    ct = resolve_expr(MICRODUCK_TOSS_COLLISION.contype, test_names, 1)
    ca = resolve_expr(MICRODUCK_TOSS_COLLISION.conaffinity, test_names, 1)
    tray_ct, hull_ct = ct
    tray_ca, hull_ca = ca
    BALL_CT, BALL_CA = 2, 1
    assert (BALL_CT & tray_ca) | (tray_ct & BALL_CA), "ball-tray must collide"
    assert not ((BALL_CT & hull_ca) | (hull_ct & BALL_CA)), "ball-hull must NOT collide"
    assert (hull_ct & 1) | (1 & hull_ca), "hull-terrain must still collide"


def test_header_throw_env_is_throw_only():
    """The throw companion must have NO hold income (the 13-variant lesson:
    hold pay extinguishes throwing), short episodes, no curriculum clocks
    (warm-start safe), and no pushes."""
    from mjlab_microduck.tasks.microduck_header_throw_env_cfg import (
        make_microduck_header_throw_env_cfg,
    )
    cfg = make_microduck_header_throw_env_cfg()
    # Retention income allowed but strictly dominated by one throw: the
    # max possible hold earnings per episode must stay below the launch bonus
    # alone (throw_v2 lesson: zero hold pay -> entropy strips the ball and
    # nothing re-tightens the grip).
    held_max = cfg.rewards["ball_held"].weight * cfg.episode_length_s * 50
    throw_value = (cfg.rewards["launch_bonus"].weight
                   + cfg.rewards["ball_distance"].weight * 0.5)  # bonus + 0.5m carry
    assert held_max < throw_value, "hold-forever must lose to one decent throw"
    assert cfg.episode_length_s <= 3.0
    assert "launch_bonus" in cfg.rewards and cfg.rewards["launch_bonus"].weight > 0
    assert "held_fwd_shaping" in cfg.rewards   # the ramp to the launch stays
    assert "push_robot" not in cfg.events
    for name in ("held_weight", "action_rate_weight", "push_magnitude"):
        assert name not in cfg.curriculum
    # obs contract unchanged (hot-swap with the hold policy)
    from mjlab_microduck.tasks.microduck_header_env_cfg import make_microduck_header_env_cfg
    hold = make_microduck_header_env_cfg()
    assert list(cfg.observations["actor"].terms.keys()) == list(
        hold.observations["actor"].terms.keys()
    )


def test_throw_env_ends_on_success_and_jackpot_dominates():
    """throw_v5 postmortem: full-length episodes made the standing annuity
    (~stack x steps) dwarf the launch jackpot, so PPO braked the throw.
    The env must (a) terminate shortly after a launch, (b) price the jackpot
    above the whole remaining standing annuity."""
    from mjlab_microduck.tasks.microduck_header_throw_env_cfg import (
        make_microduck_header_throw_env_cfg,
    )
    from mjlab_microduck.tasks import mdp as microduck_mdp
    cfg = make_microduck_header_throw_env_cfg()
    term = cfg.terminations["toss_complete"]
    assert term.func is microduck_mdp.toss_complete_termination
    assert term.time_out is True          # success, not failure
    assert term.params["settle_steps"] <= 100  # widened for full carry flight (distance objective)
    stand_stack = sum(cfg.rewards[n].weight for n in
                      ("pose_stand_legs", "pose_stand_neck", "upright",
                       "height_stand", "left_foot_grounded", "right_foot_grounded"))
    annuity = stand_stack * cfg.episode_length_s * 50
    # QUALITY ERA (v10 postmortem): the discovery jackpot is retired to a
    # nudge — throw VALUE now = bonus + distance pay, and with touchdown
    # termination the annuity is bounded by the short episode anyway. The
    # invariant: a decent 0.5m throw must beat sitting out the full clock.
    throw_value = (cfg.rewards["launch_bonus"].weight
                   + cfg.rewards["ball_distance"].weight * 0.5)
    assert throw_value > annuity * 0.15, \
        "a decent throw must decisively beat camping the episode"


def test_throw_env_pays_for_the_arc():
    """v7 style review: z-blind forward reward converged to a flat pour. The
    arc term must exist, be latch-gated upward velocity, and registered after
    the latch owner."""
    from mjlab_microduck.tasks.microduck_header_throw_env_cfg import (
        make_microduck_header_throw_env_cfg,
    )
    from mjlab_microduck.tasks import mdp as microduck_mdp
    cfg = make_microduck_header_throw_env_cfg()
    up = cfg.rewards["ball_up_velocity"]
    assert up.func is microduck_mdp.ball_upward_velocity_tossed
    assert up.weight > 0
    keys = list(cfg.rewards.keys())
    assert keys.index("ball_forward_velocity") < keys.index("ball_up_velocity")


def test_throw_v10_bundle_invariants():
    """The 1-4 improvement bundle: scaffolding annealed, legs freed, and the
    scaffolding-episode reward exclusions live in the mdp functions (distance
    and arc pay only on policy-earned launches — locked by code inspection of
    the gating flags here via the shared state names)."""
    from mjlab_microduck.tasks.microduck_header_throw_env_cfg import (
        make_microduck_header_throw_env_cfg,
    )
    cfg = make_microduck_header_throw_env_cfg()
    assert cfg.events["reset_ball"].params["windup_prob"] <= 0.05
    assert cfg.events["reset_ball"].params["launched_prob"] <= 0.10
    assert cfg.rewards["pose_stand_legs"].weight == 0.0
    # hold env keeps its legs anchored (only the throw frees them)
    from mjlab_microduck.tasks.microduck_header_env_cfg import make_microduck_header_env_cfg
    assert make_microduck_header_env_cfg().rewards["pose_stand_legs"].weight > 0


def test_throw_env_latch_demands_a_real_throw():
    """v11: price pressure alone left 3cm flicks optimal for 900 iters. The
    latch bar itself must define throw quality (gates over prices)."""
    from mjlab_microduck.tasks.microduck_header_throw_env_cfg import (
        make_microduck_header_throw_env_cfg,
    )
    from mjlab_microduck.tasks.microduck_header_env_cfg import make_microduck_header_env_cfg
    throw = make_microduck_header_throw_env_cfg()
    hold = make_microduck_header_env_cfg()
    assert throw.rewards["ball_forward_velocity"].params["min_launch_fwd"] >= 0.6
    # the hold env keeps the lenient bar (its latch is not the quality gate)
    assert hold.rewards["ball_forward_velocity"].params["min_launch_fwd"] < 0.6
