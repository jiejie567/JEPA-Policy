# robomimic / LIBERO / MimicGen environment isolation

The two benchmark families intentionally use separate Python environments:

| Task family | Environment | Key versions | Launcher |
| --- | --- | --- | --- |
| robomimic (square, lift, can, tool_hang, transport) | `venvs/jepa_ppu` | PyTorch 2.6.0, robosuite 1.5.1, MuJoCo 3.3.6 | `tools/run_robomimic_train.sh` |
| LIBERO (mug_mug, moka_moka) | `venvs/libero` | PyTorch 2.9.0+cu128, robosuite 1.4.0, MuJoCo 3.10.0, NumPy 1.26.0 | `tools/run_libero_train.sh` |
| MimicGen (Coffee Preparation D1, Three Piece Assembly D1, Kitchen D1) | `venvs/mimicgen` | PPU PyTorch 2.6.0 overlay, robosuite 1.4.1, robomimic 0.3.x, MuJoCo 3.3.6, NumPy 1.26.4 | `tools/run_mimicgen_train.sh` |

Do not install `third_party/LIBERO/requirements.txt` into `jepa_ppu`. The
robosuite and MuJoCo versions conflict with the verified robomimic runtime.

Do not install MimicGen into either existing environment. MimicGen explicitly
does not support robosuite 1.5+, and its Kitchen environment additionally needs
the pinned robosuite-task-zoo source. Use `tools/run_mimicgen_python.sh` for all
MimicGen Python commands so the pinned sources, PPU Torch overlay, and bundled
Mesa EGL runtime are selected in the correct order. See `docs/mimicgen_setup.md`.

The LIBERO PyTorch, TorchVision, TorchAudio, Triton, and CUDA 12.8 Python
packages are installed directly in `venvs/libero`. Do not remove them in favor
of the DLC image's system PyTorch: the LIBERO venv was originally created with
`--system-site-packages`, so doing so makes its PyTorch version change when the
base image changes.

On PPU nodes, `run_14_ablation_single_node.sh` automatically builds a narrow
overlay under `/tmp/jepa_policy_ablation_14` that exposes only the verified
PPU builds of PyTorch, TorchVision, TorchAudio, and Triton from `venvs/jepa_ppu`
to the LIBERO interpreter. LIBERO's own NumPy, MuJoCo, robosuite, bddl, and gym
packages remain isolated in `venvs/libero`. Set
`LIBERO_PPU_TORCH_OVERLAY=off` only on a CUDA-compatible NVIDIA node, or `on`
to require the overlay explicitly.

To retry the failed GPU 1-13 portion of the 14-run launcher without deleting
the original logs or duplicating the GPU 0 run:

```bash
START_RUN_INDEX=1 RUN_SUFFIX=retry1 ./run_14_ablation_single_node.sh
```

The 14-run ablation launcher uses the persistent Ubuntu 24.04 x86_64 Mesa EGL
runtime at `venvs/egl_noble_x86_64`. It runs an EGL context smoke test with both
benchmark environments before creating any training process. This fallback uses
CPU llvmpipe rendering because the current DLC does not provide a GPU-vendor EGL
implementation.

The launcher deliberately leaves `MUJOCO_EGL_DEVICE_ID` unset and exports
`JEPA_POLICY_EGL_DEVICE_ID=0`. The latter is handled inside JEPA-Policy so that
robomimic and LIBERO render on Mesa EGL device 0 while PyTorch continues to use
the physical card selected by `CUDA_VISIBLE_DEVICES`.

For GPU-accelerated EGL instead, the host image needs its vendor EGL driver and
the headless GLVND libraries:

```bash
apt-get install -y --no-install-recommends \
  libegl1 libegl-mesa0 libgl1 libglx0 libopengl0 libosmesa6 libglu1-mesa
```

Examples:

```bash
tools/run_robomimic_train.sh task=square_ph_image \
  ~task.dataset_repo \
  +task.dataset_path=/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/square/ph/image.hdf5

tools/run_libero_train.sh \
  -cn exps/libero_mip_future4_mip_twostep.yaml \
  task=mug_mug_image \
  task.dataset_path=/mnt/data_nas/ykj_jepa_policy/datasets/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5
```

The training entry point validates these versions before it creates a logger,
dataset, or simulator. `MIP_ALLOW_ENV_MISMATCH=1` bypasses the check for an
intentional dependency experiment.

## 2026-07-17 recovery record

The LIBERO requirements were written into `jepa_ppu` on 2026-07-16 around
17:30, replacing robosuite 1.5.1 / MuJoCo 3.3.6 with robosuite 1.4.0 / MuJoCo
3.10.0 and adding the non-headless OpenCV package. The polluted environment is
preserved at:

`/mnt/data_nas/ykj_jepa_policy/venv_backup/jepa_ppu_polluted_20260717`

The repaired environment was restored from the verified 2026-07-03 backup.
