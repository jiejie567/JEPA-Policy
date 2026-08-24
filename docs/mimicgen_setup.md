# MimicGen D1 setup

This workspace uses three MimicGen core image datasets:

| Task config | Official dataset | Rollout horizon | Expected bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| `coffee_preparation_d1_image` | `core/coffee_preparation_d1.hdf5` | 800 | 6,923,699,908 | `0e9e1eac8d969c05a5fff90358f702b6530ea80bb7a9977497e3a3740b88cf55` |
| `three_piece_assembly_d1_image` | `core/three_piece_assembly_d1.hdf5` | 500 | 3,237,234,348 | `7f5cad32fdf492b210c181b84b4856eaff1a90573ffbc5eeae871cbe8e01e586` |
| `kitchen_d1_image` | `core/kitchen_d1.hdf5` | 800 | 7,069,704,890 | `e43e339f85283aca458a2455acfe013a0642d72b117412d3218844ffd7d82dfb` |

The files live under:

```text
/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy/datasets/mimicgen/core/<task>/<task>.hdf5
```

## Isolation

The Python environment is `/mnt/data_nas/ykj_jepa_policy/venvs/mimicgen`.
Pinned sources are kept separately from the existing Robomimic and LIBERO
sources:

| Package | Source / version |
| --- | --- |
| MimicGen | official `NVlabs/mimicgen`, commit `72bd767c255545f462e7ccfb2731f2e5d4c1d9bb` (v1.0.1 code) |
| robosuite | official commit `b9d8d3de5e3dfd1724f4a0e6555246c460407daa` (v1.4.1) |
| robomimic | official commit `d0b37cf214bd24fb590d182edb6384333f67b661` |
| robosuite-task-zoo | official commit `74eab7f88214c21ca1ae8617c2b2f8d19718a9ed` |
| MuJoCo | 3.3.6 |
| NumPy / Numba | 1.26.4 / 0.61.2 |

The host has Python 3.12, for which the official MimicGen recommendation of
MuJoCo 2.3.2 has no binary wheel. MuJoCo 3.3.6 is used instead and has passed a
real reset, two-camera render, and one-step rollout for all three Panda tasks.

`tools/run_mimicgen_python.sh` exposes only the verified PPU Torch packages at
the front of `PYTHONPATH`. JEPA project dependencies from `venvs/jepa_ppu` are
appended as a last-resort read-only fallback by `tools/mimicgen_sitecustomize`.
Pinned MimicGen NumPy, MuJoCo, robosuite, and robomimic paths take precedence,
and the training entry point rejects version mismatches before creating an
environment.

## Download or resume

```bash
cd /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy
tools/download_mimicgen_core.sh
```

Downloads use `.hdf5.inprogress`, support HTTP range resume, verify the official
byte count and SHA-256, and only then rename atomically to `.hdf5`.

The downloader and task YAMLs intentionally use this single canonical path.
This prevents a complete Hugging Face `--local-dir` download from being missed
because a second, flat data directory was configured elsewhere on the NAS.

## Environment smoke test

Always use the wrapper; invoking the venv Python directly does not configure
the bundled headless EGL runtime.

```bash
tools/run_mimicgen_python.sh - <<'PY'
import mimicgen
import robosuite
print(mimicgen.__version__, robosuite.__version__)
PY
```

Verified observations for all three environments are:

```text
agentview_image             (84, 84, 3)
robot0_eye_in_hand_image    (84, 84, 3)
robot0_eef_pos              (3,)
robot0_eef_quat             (4,)
robot0_gripper_qpos         (2,)
action                      (7,)
```

## Training entry points

Baseline example:

```bash
tools/run_mimicgen_train.sh task=coffee_preparation_d1_image
```

Future-4, ratio-0.1 should use the same experiment overrides used by the
Robomimic Ablation-2 runs, with only the task changed. Keep a same-seed baseline
for each new task.
