# RoboCasa selected-task setup

This workspace uses RoboCasa 1.0.1 target-human datasets for
`SteamInMicrowave`, `StoreLeftoversInBowl`, and `LoadDishwasher`.

- Python environment: `/mnt/data_nas/ykj_jepa_policy/venvs/robocasa`
- RoboCasa checkout: `third_party/robocasa_v1_0_1`
- Dedicated robosuite checkout: `third_party/robocasa_robosuite`
- Dataset root: `datasets/robocasa`
- Runtime wrapper: `tools/run_robocasa_python.sh`
- Task manifest: `configs/robocasa_selected_tasks.yaml`

Never activate or add the MimicGen, LIBERO, or JEPA PPU environments to
`PYTHONPATH`. The wrapper clears those environment markers and uses isolated
cache directories.

The runtime uses a no-system-site-packages Python 3.12 virtual environment.
RoboCasa 1.0.1 supports Python 3, and its official exact dependency pins are
kept intact. In particular, do not upgrade NumPy, Numba, SciPy, MuJoCo, or
LeRobot independently.

RoboCasa is loaded directly from its pinned checkout. This deliberately avoids
running its `find_packages()` over the multi-GB asset tree. Its declared
dependencies are recorded in `configs/robocasa_requirements.txt`; the dedicated
robosuite checkout is installed editable into the isolated environment. The
wrapper also fixes the bundled Mesa EGL runtime and ignores inherited Python,
Conda, OpenGL, and dynamic-library paths.

Download and verify:

```bash
bash tools/download_robocasa_selected.sh
bash tools/download_robocasa_assets.sh
tools/run_robocasa_python.sh tools/check_robocasa_setup.py --env-reset
```

## JEPA-Policy training

Always launch RoboCasa through the dedicated wrapper; do not activate the
virtual environment in a shared shell. The adapter uses PyAV explicitly
because TorchCodec cannot load the system FFmpeg libraries on the current DSW
image.

Compose a configuration without starting training:

```bash
tools/run_robocasa_train.sh --config-name exps/robocasa_mip_baseline \
  task=steam_in_microwave_robocasa_image --cfg job
```

The available task overrides are:

- `steam_in_microwave_robocasa_image`
- `store_leftovers_in_bowl_robocasa_image`
- `load_dishwasher_robocasa_image`

The available experiment configurations are
`exps/robocasa_mip_baseline` and
`exps/robocasa_mip_future4_ratio010`. They default to disabled W&B logging and
serial evaluation. Keep those defaults until a short GPU and rollout-capacity
smoke test has passed on the actual DLC training node.

On the current DLC PPU image, export `ROBOCASA_USE_PPU_TORCH=1` before using
the launcher. This selects the verified read-only Torch 2.6 overlay without
modifying the RoboCasa virtual environment. The venv's Torch 2.7.1 remains the
default outside DLC.

## Shared array training cache

Shortening the source videos' GOP removes long seeks but still opens and
decodes three videos for every shuffled sample. That remains much slower than
the in-memory HDF5 paths used by LIBERO, robomimic, and MimicGen.

Formal node launches therefore build a 128x128 uint8 NumPy array cache under
`/dev/shm/jepa_policy_robocasa_node<N>/datasets`. The three tasks occupy about
171 GiB in total and are shared by all nine training processes on the node.
State and action are cached as float32 arrays too. The NAS datasets are never
modified.

To build or validate one cache manually:

```bash
tools/run_robocasa_python.sh tools/prepare_robocasa_array_cache.py \
  --source datasets/robocasa/v1.0/target/composite/SteamInMicrowave/20250814/lerobot \
  --destination /tmp/robocasa-steam-array/lerobot \
  --workers 16 --height 128 --width 128 --force
```

The cache keeps RGB as uint8 through DataLoader IPC, pinned memory, and H2D;
conversion to float32 and `[-1, 1]` normalization happen on the GPU. Future4
reads only offsets `[0, 1, 5]`. The formal launcher refuses to train if its
batch-256, 12-worker steady-state wait averages 0.30 seconds or more.

RoboCasa evaluation renders at 128x128, which matches RoboCasa's official
evaluation helper and avoids rendering 256x256 only to resize it immediately.
The launcher also smoke-tests the configured ten-worker persistent rollout
pool before starting training.

Formal concurrent runs use W&B offline mode. Inspect `metrics.jsonl` for live
loss and timing information; the launcher finalizes and serially synchronizes
the offline W&B runs after training. Always use a new submission suffix/run tag
when relaunching so existing logs are retained.
