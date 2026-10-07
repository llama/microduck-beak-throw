# Changelog

## Unreleased — 2026-10-07

- Fixed the beak hinge geometry in the training overlay: the moving lower jaw
  now carries the `jaw` mesh (bill + side links) and the soft `jaw_soft` bill
  pad, while the yellow `bottom_head_shell` stays fixed to the head. Previously
  the lower shell was hinged (its rear swung up into the top shell) and the
  soft pad stayed behind. Zero-mass visuals only: physics, checkpoints and the
  ONNX policy are unchanged.
- Re-rendered `media/preview.mp4` and `media/preview.png` from the same
  checkpoint, seed and locked cameras with the corrected jaw.

## v0.1.0-sim — 2026-09-01

- Published selected beak-throw checkpoint `model_2150.pt` and normalized ONNX.
- Packaged the exact standing policy used for the 2.4-second recovery handoff.
- Added the mandatory runtime target clamp and scripted mouth release.
- Added a reference compliant beak liner and supervised test protocol.
- Recorded 50-episode randomized simulator acceptance and software validation.

This release is simulation-validated and hardware-unvalidated.
