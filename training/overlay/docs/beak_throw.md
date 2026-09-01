# Beak throw runbook

`Mjlab-BeakThrow-Flat-MicroDuck` trains Microduck to throw a 24 mm, 3 g ball
from its beak. The 14-action policy drives the body and head. The physical mouth
servo remains outside the policy and begins opening at phase `0.30` of a 2.4 s
cycle.

The ball is a free rigid body from reset onward: it is never welded, clamped,
or teleported during the episode. Explicit upper/lower palate contacts, rounded
front lips, a rear stop, and side cheeks form a shallow pocket around it. The
restored lower-bill hinge follows the hardware controller range (-5 degrees
closed, +30 degrees open), holds the ball at +20 degrees, and uses the same
delayed XL330/BAM actuator model as the robot's other servos. A release only
counts after the ball physically separates from the pocket.

The pocket is load-bearing in both simulation and hardware. Two truly flat,
rigid bill plates can retain a sphere only through precisely controlled squeeze
and friction; that is too sensitive for reliable sim-to-real transfer. The
modeled rounded lip makes the throat slightly smaller than the ball while the
rear/side surfaces prevent it escaping in other directions. If the stock
hardware bill does not provide equivalent compliant concavity, fit a small TPU
or silicone liner that matches this geometry before hardware trials.

This is intentionally not the old Header objective. Header declared a launch
only after the ball was already about 10 cm below the tray and rewarded radial
distance, so a sideways/downward pour could score. The beak task measures
forward carry from the actual release point to first touchdown. Invalid releases
while the trunk is fallen score zero.

## Bootstrap and smoke test

The best Header-hold checkpoint already knows how to balance the heavy head.
Its 61D actor and 80D critic match BeakThrow, but its three twist-command inputs
were nearly zero during training. `prepare_beak_bootstrap.py` preserves the
balance network, zeros only those three first-layer columns, resets their
normalizer statistics, clears the old optimizer/curriculum clocks, and restores
exploration noise.

```bash
uv run python scripts/prepare_beak_bootstrap.py \
  checkpoints_hf/microduck_header/2026-08-30_14-16-27_header_v13_maxpower/model_2999.pt \
  logs/rsl_rl/microduck_beak_throw/bootstrap/model_0.pt

uv run train Mjlab-BeakThrow-Flat-MicroDuck \
  --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 5 \
  --agent.logger tensorboard --agent.run-name beak-bootstrap-smoke \
  --agent.resume True --agent.load-run '^bootstrap$' \
  --agent.load-checkpoint '^model_0.pt$'
```

The checked-in HF Jobs wrapper performs that surgery and private bundling
automatically. The long-run command is:

```bash
uv run train Mjlab-BeakThrow-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 3000 \
  --agent.run-name beak-v1 \
  --bootstrap-checkpoint \
    checkpoints_hf/microduck_header/2026-08-30_14-16-27_header_v13_maxpower/model_2999.pt \
  --hf-jobs --run-name beak-v1
```

Do not launch the long run without the bootstrap. A random policy falls before
the scheduled mouth-open instant and learns from almost no valid attempts.

To continue a trained BeakThrow body policy against the collision-resolved
mouth without modifying its actor, normalizers, optimizer, or iteration clock:

```bash
uv run train Mjlab-BeakThrow-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 2000 \
  --agent.run-name beak-physical-v1-20260831 \
  --resume-checkpoint checkpoints_hf/beak-v1-20260831/microduck_beak_throw/2026-08-31_16-54-12_beak-v1-20260831/model_1250.pt \
  --hf-jobs --run-name beak-physical-v1-20260831 \
  --flavor a10g-large --timeout 12h
```

`--resume-checkpoint` is intentionally different from `--bootstrap-checkpoint`:
the former preserves an existing BeakThrow checkpoint verbatim; the latter is
only for converting a Header checkpoint into a fresh BeakThrow iteration zero.
The physical-release sweep selected `model_1250`: at phase 0.30 it carried the
ball about 0.59 m, while the later model had already traded away some forward
carry. `--agent.max-iterations` is additional training after the resumed
iteration, so the command above trains from iteration 1250 through 3250.

The slow-fade recovery branch starts from the best physical checkpoint before
the zero-reference regression:

```bash
uv run train Mjlab-BeakThrow-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 1750 \
  --agent.run-name beak-physical-slowfade-v2-20260831 \
  --resume-checkpoint checkpoints_hf/beak-physical-v1-20260831/microduck_beak_throw/2026-08-31_19-50-26_beak-physical-v1-20260831/model_1500.pt \
  --hf-jobs --run-name beak-physical-slowfade-v2-20260831 \
  --flavor a10g-large --timeout 12h
```

This adds 1750 iterations and therefore finishes at iteration 3250. The
original physical continuation remains independent and is not canceled.

## What to watch

The main W&B channels are:

- `Episode_Reward/beak_forward_carry`: the terminal objective; it must grow.
- `Episode_Reward/beak_release_range`: one-shot ballistic estimate at release.
- `Episode_Reward/beak_predicted_range`: dense pre-release discovery bridge.
- `Episode_Reward/beak_reference_pose`: tapers from 0.5 at iteration 1500 to
  0.25 at 1875 and a 0.1 floor at 2250. An abrupt drop to zero caused physical
  carry to regress in the first collision-resolved continuation.
- `Episode_Termination/beak_throw_complete`: confirms real release/landing
  episodes instead of timeout-only batches.

Every penalty channel must remain non-positive. Total reward rising without
forward carry rising is not success.

## Export and acceptance gate

Always export with `scripts/export.py`; it bakes in the observation normalizer.
The HF job also uploads its final export as `exported/policy.onnx`.

```bash
uv run python scripts/eval_beak.py beak_throw.onnx \
  --episodes 50 --joint-noise 0.01 --spawn-z-noise 0.005 \
  --yaw-noise-deg 180 --min-carry 0.50 --min-success-rate 0.90 \
  --require-gate
```

Promotion gate for the first beak policy:

- at least 90% of rollouts land at least 0.50 m forward;
- upward release velocity is positive in every counted success;
- trunk tilt remains below 50 degrees through each counted rollout;
- inspect the printed p10 and lateral statistics, not only the median.

If the policy has a clean, repeatable motion but plateaus below 0.50 m, retain
the aligned reward and extend training before changing the task. If the beak
mechanism cannot physically retain the ball through the wind-up, the same body
policy architecture can be adapted to a head tray, but tray distance must use
the physical-separation evaluator in `scripts/eval_header.py`, never the retired
reward latch.

## Cinematic simulator tasks

The cinematic pair is intentionally separate from the hardware-oriented task:

- `Mjlab-BeakThrow-Cinematic-MicroDuck` is the stable showcase for the original
  `model_1250`. An invisible latch keeps the ball at the visible pocket during
  wind-up; the beak starts opening at phase 0.35 and releases at phase 0.38 with
  the pocket's measured velocity. From that instant through touchdown, the ball
  is an ordinary free MuJoCo body.
- `Mjlab-BeakThrow-Cinematic-Train-MicroDuck` uses the same 61-observation and
  14-action ABI, but adds terminal lateral-error cost and stronger distance,
  uprightness, and smoothness terms. Its runner resumes the networks and
  iteration clock while resetting the optimizer to `2e-4`, reducing action std
  to `0.18`, and disabling random episode-age starts.

Watch the known-good checkpoint on macOS:

```bash
.venv/bin/mjpython .venv/bin/play Mjlab-BeakThrow-Cinematic-MicroDuck \
  --checkpoint-file checkpoints_hf/beak-v1-20260831/microduck_beak_throw/2026-08-31_16-54-12_beak-v1-20260831/model_1250.pt
```

Launch a short continuation on HF Jobs:

```bash
uv run train Mjlab-BeakThrow-Cinematic-Train-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 750 \
  --agent.run-name beak-cinematic-v1-20260831 \
  --resume-checkpoint checkpoints_hf/beak-v1-20260831/microduck_beak_throw/2026-08-31_16-54-12_beak-v1-20260831/model_1250.pt \
  --hf-jobs --run-name beak-cinematic-v1-20260831 \
  --flavor a10g-large --timeout 6h
```

## Hardware integration contract

Deployment must provide the same phase command in observation slots 48–50:
`[cos(2π phase), sin(2π phase), 0]`, starting from phase zero and advancing at
50 Hz over 2.4 s. Hold the mouth at +20 degrees while `phase < 0.30`; from
phase `0.30` to `0.34`, smoothly command it to the hardware +30-degree limit,
then keep it open through follow-through. Hand back to the standing policy
after recovery. The mouth is actuated, but it is not a fifteenth policy action.

Before deploying, measure the real liner dimensions and mouth opening trace.
Update the simulated pivot/liner if those measurements disagree, and randomize
opening latency, ball diameter, friction, and liner compliance around the
measured values rather than around guesses.

Start hardware trials with an empty beak, then a foam ball behind a clear
shield, then the 3 g training ball. Record actual release video and distance;
do not increase payload until the release direction and recovery are repeatable.
