"""Microduck HeaderThrow task — throw-ONLY companion to the Header hold env.

Why a separate policy (2026-08-30, after 13 single-policy variants): a policy
paid to hold extinguishes throwing during the hold phase and PPO never brings
it back — the launch fired only at iterations 2-10 (random flailing) and never
again once holding was learned. The deployed family already solves this shape
of problem with HOT-SWAPPED policies (walk / kick / roulade): so the Header
splits in two — the Header env's policy HOLDS; this env's policy THROWS.

Deployment/viewer: duck balances the ball (hold policy), a trigger swaps to
this policy for ~2s, it launches the ball hard and settles, swap back.

Deltas from the hold env:
  - NO ball_held income (throwing is the only ball-related pay: launch bonus
    + max-power forward reward + the sweep-shaping ramp toward the launch).
  - Short episodes (3s): a throw opportunity, not a lifestyle.
  - FLAT weights, no iteration-indexed curricula (safe to warm-start from any
    checkpoint without curriculum-clock mismatches), no pushes.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp

from mjlab_microduck.tasks.microduck_header_env_cfg import (
    MicroduckHeaderRlCfg,
    make_microduck_header_env_cfg,
)

THROW_EPISODE_LENGTH_S = 3.0


def make_microduck_header_throw_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_header_env_cfg(play)
    cfg.episode_length_s = THROW_EPISODE_LENGTH_S

    # Small RETENTION income (throw_v2 lesson: with zero hold pay, discovery
    # entropy knocks the ball off in the first moments and no gradient re-
    # tightens the grip — ball_rel_z went negative, episodes ran ball-less).
    # Capped by the short episode: 0.2 x 150 steps = 30 max, strictly less
    # than one launch (bonus 60 + roll), so hold-forever still always loses.
    cfg.rewards["ball_held"].weight = 0.2

    # Discovery economics: the one-time launch jackpot doubled vs the hold env
    # (stall postmortem, throw_v1 iters 3500-3700: sweeps plateaued — the
    # regularizer/stand stack taxes big sweeps before the launch pays, so the
    # first launches must be LOUD in the advantage signal).
    cfg.rewards["launch_bonus"].weight = 60.0

    # Flat weights: mild smoothness from step 0 (the throw is a fast one-shot
    # motion; heavy action_rate blocks discovery, none at all thrashes).
    cfg.rewards["action_rate_l2"].weight = -0.3

    # No fall termination (throw_v6 postmortem): fell_over spiked to 9/period
    # exactly at the bonus peak — throw attempts often stumble, and if a
    # stumble is an episode death sentence the policy can never PRACTICE
    # recoil management, so it retreats to not-throwing. A fallen duck here
    # just earns the slim nothing until timeout; deployment hands falls to
    # the recovery policy anyway.
    cfg.terminations.pop("fell_over", None)
    # Gentler wind-ups: 4-9 rad/s incoming whip was beyond what the policy
    # could ride out; 2.5-6.5 is completable without a guaranteed stumble.
    cfg.events["reset_ball"].params["windup_omega"] = (2.5, 6.5)

    # "Trick done" termination 1s after a successful launch + a slim standing
    # stack + a big jackpot: the annuity-vs-jackpot fix (see
    # toss_complete_termination docstring; throw_v5 postmortem).
    cfg.terminations["toss_complete"] = TerminationTermCfg(
        func=microduck_mdp.toss_complete_termination,
        time_out=True,
        params={"settle_steps": 80},
    )
    # HARD GATE on throw quality (v11 postmortem: 900 iters flat at ~3cm
    # dinky flicks under pure price pressure). A launch only COUNTS (latch,
    # bonus, termination) if the ball exits at >= 0.6 m/s forward — a dinky
    # flick now wastes the ball and earns nothing, so the nearest profitable
    # behavior is a harder version of the same swing. Gates over prices.
    cfg.rewards["ball_forward_velocity"].params["min_launch_fwd"] = 0.6

    # Discovery jackpot RETIRED to a nudge (v10 postmortem: with throwing
    # universal and touchdown termination, a 200 bonus made rapid dinky
    # flicks optimal — throughput farming). Distance pay now dominates:
    # a 0.6m carry pays ~48 vs the 20 bonus.
    cfg.rewards["launch_bonus"].weight = 20.0
    # Arc styling: pay for upward ball velocity post-launch (z-blind forward
    # reward alone converged to a flat pour/roll — Doug's review of v7).
    cfg.rewards["ball_up_velocity"] = RewardTermCfg(
        func=microduck_mdp.ball_upward_velocity_tossed,
        weight=1.5,
        params={"asset_name": "ball", "max_vz": 2.0},
    )
    # Distance objective (Doug's spec: throw as far as possible): one-time
    # bonus proportional to measured landing distance — 50/m, so a 2m carry
    # pays 100 on top of the launch bonus. The aligned objective; the arc
    # term above is kept small as early shaping toward lofted releases.
    cfg.rewards["ball_distance"] = RewardTermCfg(
        func=microduck_mdp.ball_landing_distance_bonus,
        weight=80.0,
        params={"asset_name": "ball", "ball_radius": 0.012, "max_distance": 3.0},
    )
    for name, wgt in (("pose_stand_legs", 0.5), ("pose_stand_neck", 0.3),
                      ("upright", 0.5), ("height_stand", 0.3),
                      ("left_foot_grounded", 0.5), ("right_foot_grounded", 0.5)):
        cfg.rewards[name].weight = wgt

    # Wind-up spawns: 25% of episodes begin mid-swing (see reset_ball_in_tray)
    # — the exploration lives in the spawn distribution, so entropy stays at
    # the calm default where ball retention is easy.
    # Training wheels annealed (the skill exists; ~85% of experience is now
    # the deployment case: cold-start throws).
    cfg.events["reset_ball"].params["windup_prob"] = 0.05
    cfg.events["reset_ball"].params["launched_prob"] = 0.10

    # Whole-body unlock: stop paying the legs to stay at the standing pose
    # during the throw — crouch-and-extend leg drive is now free to emerge.
    cfg.rewards["pose_stand_legs"].weight = 0.0

    # No iteration-indexed schedules (warm-start safe) and no pushes (the
    # 3s throw window isn't the place to practice shove recovery — the hold
    # policy owns that).
    for name in ("held_weight", "action_rate_weight", "com_range",
                 "head_com_range", "push_magnitude"):
        cfg.curriculum.pop(name, None)
    cfg.events.pop("push_robot", None)

    return cfg


MicroduckHeaderThrowRlCfg = RslRlOnPolicyRunnerCfg(
    actor=MicroduckHeaderRlCfg.actor,
    critic=MicroduckHeaderRlCfg.critic,
    algorithm=MicroduckHeaderRlCfg.algorithm,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_header_throw",
    run_name="microduck_header_throw",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)
