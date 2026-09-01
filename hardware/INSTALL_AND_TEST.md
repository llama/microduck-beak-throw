# Microduck beak throw — hardware test handoff v1

This bundle is an experimental, supervised first-hardware test. It contains the
selected `model_2150.pt` export from
`q2p/beak-hardware-straight-v3-20260901-a100`, a 61-input / 14-output ONNX body
policy, the exact standing policy used for handoff evaluation, and a runtime
patch that synchronizes the real mouth servo with the learned 2.4-second throw.
The mouth is held at +20°, then opens to +30° over phase 0.30–0.34. The body
policy does not command the mouth directly.

The runtime's throw-specific anatomical target clamp is mandatory. The
randomized simulator trial landed straight and recovered 50/50 times, but the
unclamped network exceeded the strict raw-target limit in every trial. This is
not a production-accepted policy; begin with a supported, empty-beak test.

## Mechanical contract

The simulated projectile is a 24 mm diameter, 3 g sphere. It is retained by a
shallow compliant beak liner with upper/lower pads, a rounded front lip, a rear
stop, and side cheeks. Two flat hard plates do **not** provide the same grip.
The `liner/` directory contains the editable OpenSCAD model, two watertight
reference STLs, print settings, and a mandatory fit/release check. Its default
contact dimensions match the simulator but its mounting face has not been
verified against the tester's hardware revision; measure and adapt it before a
loaded trial. An empty-beak trial validates body motion and mouth timing without
pretending the stock bill matches this collision geometry.

Do not substitute a heavier ball for the first tests.

## Install

The runtime patch is based on `pollen-robotics/microduck` commit
`590b986bd8c0d50ae02cb3ea2f59c463b6828168`.

1. Apply `microduck-runtime.patch` to that checkout.
2. Copy `beak_throw.onnx` to `policies/beak_throw.onnx` and
   `alpha_stand.onnx` to `policies/alpha_stand.onnx`. The latter makes the
   throw-to-stand handoff identical to the evaluated sequence.
3. Build and install the development artifact using the repository's normal
   signed developer flow, for example:

   ```bash
   scripts/dev-push.sh --docker radxa@ROBOT_IP
   ```

4. The updater preserves the robot's existing `/etc/robot/robotd.toml`. Add the
   following explicit opt-in and restart `robotd`:

   ```toml
   [policy]
   beak_throw = "/opt/robot/current/policies/beak_throw.onnx"
   beak_throw_period = 2.4
   beak_throw_action_scale = 1.0
   beak_throw_gain_ratio = 1.0
   ```

Do not reduce `beak_throw_action_scale` as a safety shortcut. The policy is a
closed-loop dynamic motion; simulated reduced-scale trials fell. The patched
runtime instead clamps this skill to the same per-joint anatomical limits used
during its final training.

## First test protocol

Use a full battery, a padded floor, eye protection, and a clear exclusion zone.
Keep people, animals, glass, and electronics out of the throw line and both
sides of it. One operator should remain at the terminal with both the normal
abort and torque-off commands pretyped in separate shells. Do not hold or tether
the robot during the motion.

Before installation, complete `HARDWARE_COMPATIBILITY.md`. A software version,
joint calibration, mouth range, liner fit, or projectile field left unknown is
a stop condition for a loaded test.

1. Remove the ball. Place the duck on a level, high-friction surface, facing
   into the cleared area.
2. Verify the daemon and policy load:

   ```bash
   robotctl health
   robotctl monitor
   ```

3. In a second terminal, enable the robot and perform one empty-beak throw:

   ```bash
   robotctl robot enable
   robotctl robot do beak-throw
   ```

4. Use the normal abort for a bad trajectory or mistimed mouth. It stops policy
   control and returns toward home while retaining torque:

   ```bash
   robotctl robot disable
   ```

   If a joint chatters, strikes a stop, or continues pushing, cut torque instead
   (the duck will go limp, so only do this over the padded floor):

   ```bash
   robotctl robot relax --yes
   ```

5. Review the empty motion and motor temperatures. Repeat at most three empty
   trials before adding a projectile.
6. Follow `liner/README.md` and pass its unpowered retention and +30-degree
   release checks. Then test one 24 mm lightweight foam ball. Only progress
   toward the simulated 3 g ball after repeatable upright trials.

Record a side view at 120/240 fps if possible, plus a front view of at least one
trial. For every trial note battery voltage, projectile diameter/mass, whether
the ball was retained until mouth opening, whether every foot stayed planted,
and whether the duck recovered upright.

## Stop conditions

Stop the session after any servo overload/shutdown, linkage collision, torn or
shifted liner, repeated foot slip, fall, or release toward the operator. Do not
raise gain, action scale, projectile mass, or close the mouth farther to solve a
retention problem; those changes invalidate the trained dynamics.

## Simulator acceptance

`ACCEPTANCE.txt` records the randomized collision evaluation of the exact ONNX
in this directory, including the 2.4-second switch to the packaged standing
policy and the following two-second settle. It distinguishes the 50/50 clamped
dynamic result from the failed raw-output gate. `SHA256SUMS` binds the policy,
patch, instructions, liner, and acceptance report together.
