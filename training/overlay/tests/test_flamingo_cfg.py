"""Config-invariant tests for the Flamingo-nap env.

Each test locks in a decision that came from the pre-training physics
measurements (see the FLAMINGO section header in mdp.py) or from an
AGENTS.md reward-design rule.
"""
import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_flamingo_env_cfg import (
    CLEARANCE_TARGET,
    FLAMINGO_SPAWN_ROLL_RANGE,
    FLAMINGO_TUCK_OVERRIDES,
    MAX_TILT_COS,
    NAP_HEAD_OVERRIDES,
    make_microduck_flamingo_env_cfg,
    MicroduckFlamingoRlCfg,
)


def test_single_support_gate_wires_per_foot_sensors():
    cfg = make_microduck_flamingo_env_cfg()
    sensor_names = {s.name for s in cfg.scene.sensors}
    params = cfg.rewards["flamingo_composite"].params
    assert params["stance_sensor_name"] in sensor_names
    assert params["lifted_sensor_name"] in sensor_names
    assert params["stance_sensor_name"] != params["lifted_sensor_name"]
    # stance is the LEFT foot, lifted is the RIGHT (asymmetric task, v1)
    stance = next(s for s in cfg.scene.sensors if s.name == params["stance_sensor_name"])
    lifted = next(s for s in cfg.scene.sensors if s.name == params["lifted_sensor_name"])
    assert "left" in stance.primary.pattern
    assert "right" in lifted.primary.pattern


def test_tilt_ceiling_allows_the_mandatory_lean():
    # Measured: balanced one-foot configs need 10-25° of whole-body lean.
    # A tight uprightness gate would make the reward unreachable.
    assert MAX_TILT_COS < math.cos(math.radians(30.0))
    cfg = make_microduck_flamingo_env_cfg()
    assert cfg.rewards["flamingo_composite"].params["max_tilt_cos"] == MAX_TILT_COS
    # fell_over termination must also sit far outside the working lean band
    assert cfg.terminations["fell_over"].params["limit_angle"] > math.radians(45.0)


def test_spawn_roll_leans_toward_the_stance_foot():
    # Negative roll = toward the left (+y) foot in the spawn event's quat
    # convention — the measured equilibrium band is −25°…−10°.
    lo, hi = FLAMINGO_SPAWN_ROLL_RANGE
    assert lo < hi < 0.0
    cfg = make_microduck_flamingo_env_cfg()
    ev = cfg.events["set_flamingo_spawn"]
    assert ev.params["roll_range"] == FLAMINGO_SPAWN_ROLL_RANGE
    assert 0.0 < ev.params["flamingo_prob"] <= 0.5


def test_spawn_mix_ramps_toward_standing_heavy():
    cfg = make_microduck_flamingo_env_cfg()
    stages = cfg.curriculum["spawn_mix"].params["param_stages"]
    probs = [s["params"]["flamingo_prob"] for s in stages]
    assert probs[0] == cfg.events["set_flamingo_spawn"].params["flamingo_prob"]
    assert probs == sorted(probs, reverse=True)
    assert probs[-1] > 0.0  # hold data never disappears entirely


def test_tuck_targets_are_inside_joint_limits():
    # right hip_pitch / knee / ankle are all ±1.5708 hinges
    for idx, val in FLAMINGO_TUCK_OVERRIDES.items():
        assert idx in (11, 12, 13)
        assert abs(val) < 1.57
    for idx, val in NAP_HEAD_OVERRIDES.items():
        assert idx in (5, 6, 7, 8)
    # head_yaw target leans over the STANCE-side shoulder (the counterweight
    # direction, +y): positive yaw
    assert NAP_HEAD_OVERRIDES[7] > 0.0


def test_style_and_calm_rewards_start_at_zero_and_ramp_late():
    # AGENTS.md: attempt-taxes and style constraints only after the skill
    # exists. The head is the balance arm — pinning it early kills discovery.
    cfg = make_microduck_flamingo_env_cfg()
    for name, curriculum in (("head_nap", "nap_weight"),
                             ("flamingo_stillness", "stillness_weight")):
        assert cfg.rewards[name].weight == 0.0
        stages = cfg.curriculum[curriculum].params["weight_stages"]
        assert stages[0]["weight"] == 0.0
        assert stages[1]["step"] >= 2000 * 24
        weights = [s["weight"] for s in stages]
        assert weights == sorted(weights)


def test_penalty_sign_conventions():
    # Self-negating microduck penalties (return ≤ 0) → POSITIVE weight;
    # cost functions (return ≥ 0) → NEGATIVE weight. (The standup/sitstand
    # double-negation bug class.)
    cfg = make_microduck_flamingo_env_cfg()
    assert cfg.rewards["gentle_motion"].weight > 0.0        # returns -|a_z|
    assert cfg.rewards["pose_stance_l1"].weight > 0.0       # returns -L1
    assert cfg.rewards["action_rate_l2"].weight < 0.0       # cost ≥ 0
    assert cfg.rewards["hip_roll_limit"].weight < 0.0       # cost ≥ 0
    assert cfg.rewards["body_ang_vel"].weight < 0.0
    assert cfg.rewards["angular_momentum"].weight < 0.0
    assert cfg.rewards["self_collisions"].weight < 0.0


def test_stance_pose_reward_leaves_hip_roll_free():
    # The lean rides on the stance hip_roll — pricing it toward HOME would
    # fight the mandatory 10-25° lean.
    cfg = make_microduck_flamingo_env_cfg()
    assert 1 not in cfg.rewards["pose_stance_l1"].params["joint_indices"]


def test_hip_roll_limit_penalty_targets_only_hip_rolls():
    cfg = make_microduck_flamingo_env_cfg()
    pattern = cfg.rewards["hip_roll_limit"].params["asset_cfg"].joint_names[0]
    assert "hip_roll" in pattern
    assert cfg.rewards["hip_roll_limit"].params["margin"] > 0.0


def test_bridge_reward_shares_gates_with_the_composite():
    # lift_progress must hand over continuously to flamingo_composite: same
    # clearance target and the same not-fallen gate values.
    cfg = make_microduck_flamingo_env_cfg()
    comp = cfg.rewards["flamingo_composite"].params
    bridge = cfg.rewards["lift_progress"].params
    assert bridge["clearance_target"] == comp["clearance_target"] == CLEARANCE_TARGET
    assert bridge["max_tilt_cos"] == comp["max_tilt_cos"]
    assert bridge["min_height"] == comp["min_height"]
    assert bridge["stance_sensor_name"] == comp["stance_sensor_name"]


def test_pushes_ramp_from_zero():
    # A ±0.3 shove on one leg is a guaranteed fall — pushes must start OFF
    # and stay smaller than the standup env's final range.
    cfg = make_microduck_flamingo_env_cfg()
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)
    final = stages[-1]["velocity_range"]["x"]
    assert final[1] <= 0.2


def test_symmetry_augmentation_is_disabled():
    # Left-stance is intrinsically asymmetric; mirroring would train the
    # wrong-leg version into the same policy.
    assert MicroduckFlamingoRlCfg.algorithm.symmetry_cfg is None


def test_actor_observation_keeps_the_61d_slot_layout():
    cfg = make_microduck_flamingo_env_cfg()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert "height_scan" not in terms
    assert "head_command" in terms
    assert "body_command" in terms
    assert terms["body_command"].params["dim"] == 6


def test_obs_parity_with_standup():
    # Exact term-order parity — the condition for the exported ONNX to load
    # into the runtime's hot-swap slot.
    from mjlab_microduck.tasks.microduck_standup_env_cfg import (
        make_microduck_standup_env_cfg,
    )

    flam = make_microduck_flamingo_env_cfg()
    standup = make_microduck_standup_env_cfg()
    for grp in ("actor", "critic"):
        assert list(flam.observations[grp].terms.keys()) == list(
            standup.observations[grp].terms.keys()
        ), f"observation layout diverges in group {grp}"


def test_nan_guard_and_bam_events_are_registered():
    # Standalone-env checklist from AGENTS.md: BAM startup event + NaN-safe
    # critic terms + nan_state termination with sensor names.
    cfg = make_microduck_flamingo_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events
    assert "nan_state" in cfg.terminations
    assert cfg.terminations["nan_state"].params["sensor_names"]
    critic = cfg.observations["critic"].terms
    if "foot_contact_forces" in critic:
        assert critic["foot_contact_forces"].func is microduck_mdp.foot_contact_forces_safe
    if "foot_air_time" in critic:
        assert critic["foot_air_time"].func is microduck_mdp.foot_air_time_safe
