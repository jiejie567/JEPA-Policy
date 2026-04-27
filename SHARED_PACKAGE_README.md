## Shared Code Package

This package is the cleaned shared subset of the repository for robomimic and LIBERO image-policy training.

Included:
- `mip/` core code
- `examples/train_robomimic.py`
- `examples/configs/` needed for robomimic and LIBERO runs
- `README.md`, `LICENSE`, `pyproject.toml`, `uv.lock`

Excluded:
- experiment outputs and checkpoints
- plotting / comparison scripts
- test files
- PushT-specific code and configs
- cluster-only launcher presets

### Verified

- `py_compile` passes for the main updated files
- `mip/losses_test.py` passes
- Hydra presets resolve correctly

### Short commands

Baseline:

```bash
./.venv/bin/python examples/train_robomimic.py \
  -cn exps/libero_mip_baseline.yaml \
  task=moka_moka_image \
  task.dataset_path=/path/to/dataset.hdf5 \
  task.libero_root=/path/to/LIBERO
```

Future4 direct:

```bash
./.venv/bin/python examples/train_robomimic.py \
  -cn exps/libero_mip_future4_direct.yaml \
  task=moka_moka_image \
  task.dataset_path=/path/to/dataset.hdf5 \
  task.libero_root=/path/to/LIBERO
```

Future4 mip two-step:

```bash
./.venv/bin/python examples/train_robomimic.py \
  -cn exps/libero_mip_future4_mip_twostep.yaml \
  task=moka_moka_image \
  task.dataset_path=/path/to/dataset.hdf5 \
  task.libero_root=/path/to/LIBERO
```
