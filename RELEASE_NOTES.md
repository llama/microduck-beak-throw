# Microduck Beak Throw v0.1.0-sim

First public research release of the selected Microduck beak-throw policy.

The attached tester ZIP includes the throw and recovery ONNX policies, runtime
patch, configuration, reference liner, checksums, validation report, and a
supervised empty-beak-first test protocol. The MP4 is the exact selected policy
running in MuJoCo.

**Status: simulation-validated, hardware-unvalidated.** The policy requires the
patched runtime's anatomical target clamp and failed the strict raw-output gate.
Read SAFETY.md and INSTALL_AND_TEST.md before any physical experiment.
