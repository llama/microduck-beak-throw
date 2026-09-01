# Runtime integration

`microduck-runtime.patch` is based on `pollen-robotics/microduck` commit
`590b986bd8c0d50ae02cb3ea2f59c463b6828168`.

Apply it to that exact checkout, download `beak_throw.onnx` and
`alpha_stand.onnx` from the Hugging Face repository, and copy both into the
runtime's `policies/` directory. The supplied `robotd-beak-throw.toml` shows the
explicit opt-in configuration. The throw-specific anatomical target clamp in
the patch is mandatory.

Do not deploy before reading [../SAFETY.md](../SAFETY.md) and
[../hardware/INSTALL_AND_TEST.md](../hardware/INSTALL_AND_TEST.md).
