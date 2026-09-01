# Reconstructing the training checkout

This directory preserves the experiment as a patch plus an overlay against an
exact upstream base. It is not a fork of every upstream file.

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git
cd microduck_rl
git checkout d424a0c899f6b33cbd3daeb279913134349c0b63
git apply /absolute/path/to/training/microduck_rl-shared-files.patch
rsync -a /absolute/path/to/training/overlay/ ./
uv sync
```

The patch contains edits to shared upstream files. The overlay contains new
task, model, evaluation, video, and test files. Run the relevant tests with:

```bash
uv run pytest tests/test_beak_bootstrap.py tests/test_beak_throw_cfg.py \
  tests/test_beak_cinematic_cfg.py tests/test_header_cfg.py tests/test_hf_jobs.py
```

Download `model_2150.pt` from
[`q2p/microduck-beak-throw`](https://huggingface.co/q2p/microduck-beak-throw),
then visualize the hardware-oriented task on macOS:

```bash
.venv/bin/mjpython .venv/bin/play Mjlab-BeakThrow-Hardware-MicroDuck \
  --checkpoint-file /absolute/path/to/model_2150.pt --num-envs 1
```

The source snapshot reflects the broader development checkout, so it also
contains the precursor head-tray/cinematic tasks and one incidental co-developed
flamingo task. The released deployable here is specifically the
`Mjlab-BeakThrow-Hardware-MicroDuck` policy selected at checkpoint 2150.
