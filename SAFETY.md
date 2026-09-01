# Safety

This release is **simulation-validated and hardware-unvalidated**. It is an
experimental research artifact, not a safety-certified robot skill.

The exact ONNX completed 50/50 randomized simulated throws and hybrid
recoveries when commanded targets were clipped to the same anatomical limits
as the patched runtime. The raw network exceeded the strict target-limit gate
in 50/50 trials, with a median excess of 0.751 rad. Consequently:

- do not run the throw through an unpatched or differently configured runtime;
- do not disable or widen the throw-specific anatomical clamp;
- do not compensate by reducing action scale, raising gain, closing the beak
  farther, or using a heavier projectile;
- do not treat simulator success as evidence of physical safety.

Any hardware work must be supervised on a padded, level, high-friction surface
with eye protection and a clear exclusion zone in front and to both sides. Keep
people, animals, glass, and electronics outside that zone. Have the ordinary
abort and torque-off commands ready in separate terminals. Do not hold or tether
the robot during the motion.

The first powered trial must have an empty beak. Before adding a ball, verify
hardware/software revision, calibration, battery, physical mouth range and
direction, liner clearance, liner retention/release, and projectile diameter
and mass. The reference liner is not a guaranteed fit for any physical robot.

Stop after any servo overload or shutdown, linkage collision, liner movement or
damage, repeated foot slip, fall, unexpected release direction, chatter, or
continued pushing against a stop. See
[hardware/INSTALL_AND_TEST.md](hardware/INSTALL_AND_TEST.md) for the complete
protocol and [hardware/HARDWARE_COMPATIBILITY.md](hardware/HARDWARE_COMPATIBILITY.md)
for the mandatory preflight record.
