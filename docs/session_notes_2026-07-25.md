# JEPA-Policy session notes — 2026-07-25

## RoboCasa stopped-run diagnosis

- The two formal DLC jobs were stopped by the user and were not restarted:
  - node 1: `dlcaagqa2c6bhl0x`
  - node 2: `dlcaufx1m1rea0if`
- All 18 runs produced one local training record at step 999. The baseline
  losses were approximately 0.657–0.723 and the future total losses were
  approximately 0.717–0.784.
- Reaching step 999 took approximately 35,500–38,400 seconds per run, or about
  36–38 seconds per optimization step.
- GPU system samples showed that the GPUs were idle almost all of the time
  despite holding model memory. The bottleneck was input decoding, not the
  model update.

## Root cause

- Every training sample reads two frames from each of three MP4 cameras.
- The published RoboCasa MP4 files normally have a 250-frame GOP. PyAV must
  seek to the preceding keyframe and decode up to roughly 250 frames to return
  a small randomly selected window.
- Nine runs on each node also started DataLoader workers and eight rollout
  environment workers per run at the same time. This added CPU/process
  contention even though rollout workers were unused until evaluation.
- W&B online logging was overloaded by the concurrent runs. Local
  `metrics.jsonl` files contained loss, while the W&B filestream showed
  timeouts, resets, and capacity errors. The single step-999 points appeared
  online only after a delay.

## Implemented speed fix

- Added `tools/prepare_robocasa_fast_cache.py`.
  - It copies non-video LeRobot files to node-local storage.
  - It transcodes MP4 files to H.264 with an 8-frame GOP.
  - It validates video count, records a content marker, reuses a complete
    cache, and never modifies the NAS source dataset.
- `run_robocasa_dlc_node.sh` now builds and benchmarks the three node-local
  caches before launching training, then trains from those paths.
- Parallel rollout pools are lazy during training and start only at the first
  evaluation.
- Formal training uses `batch_size=256` and 12 DataLoader workers per run on
  the 160-CPU node, matching the earlier experiments' batch size.
- Parallel rollout uses ten workers per run and is configurable via
  `EVAL_WORKERS`. Using 20 workers for each of nine concurrent runs would
  create 180 active rollout environments and oversubscribe the node.
- The original 300,000-step training budget and 10,000-step evaluation/save
  intervals are preserved.
- Per-step `data_wait_seconds`, `preprocess_seconds`, and `update_seconds`
  metrics were added for the next run.

## Validation

- A complete 1,533-video SteamInMicrowave cache was generated successfully.
- On 24 identical random samples:
  - original mean sample time: 0.425069 seconds
  - short-GOP cache mean sample time: 0.091404 seconds
  - measured speedup: 4.65x
- A 50-batch steady-state diagnostic using `batch_size=128` and 12 workers
  averaged 1.156 seconds of data wait per batch after startup. The stopped runs
  averaged roughly 36–38 seconds per complete step, so the dominant input
  bottleneck was reduced substantially; the final `batch_size=256` setting is
  benchmarked separately because it doubles both decoding and model work.
- With the final `batch_size=256` and 12-worker setting, a 40-batch test
  averaged 1.928 seconds of data wait beyond the initial prefetch window.
  This is input time only; the complete CUDA step must still be measured on
  the DLC node.
- The future-state cache path averaged 0.109518 seconds per random sample.
- Actions and all non-image observations matched exactly. Re-encoded image
  tensors kept identical shapes; mean absolute image difference was 0.00989
  and the maximum per-sample mean difference was 0.02193.
- Two-step end-to-end CPU smoke training passed for both baseline and future
  configurations. Both branches performed backward updates and wrote loss,
  timing metrics, and complete offline W&B history.
- Both node launch plans passed Hydra composition, Bash syntax, Python compile,
  and dry-run checks.

## W&B behavior for a future run

- Training writes loss immediately to each run's `metrics.jsonl`.
- W&B is configured offline during the concurrent training phase to avoid
  losing history to connection/capacity errors.
- Runs are finalized explicitly and synchronized serially afterward. Offline
  data remains on NAS if synchronization fails, including when another run on
  the node fails.
- A unique run tag defaults to the submission suffix so an old log directory
  cannot be overwritten accidentally.

## Current state

The speed and logging fixes are complete locally. No formal RoboCasa job was
submitted or restarted; restart remains a user decision.

## Later live-run update

Two new jobs were subsequently started outside this diagnostic turn:

- node 1: `dlc1rlpceiyqy3g9`
- node 2: `dlclphgqymbhye0i`

All 18 runs reached step 9,999. Recent training throughput was approximately
0.319 seconds/step for baseline and 0.358 seconds/step for future, versus
36–38 seconds/step in the stopped run. Recent `data_wait_seconds` was below
0.001 seconds, confirming that the short-GOP cache removed the input bottleneck.

At the first step-10,000 evaluation, all runs hit the same parallel-evaluation
error while reducing RoboCasa's nested object-array success payload. They fell
back to serial evaluation and therefore stopped writing training metrics while
the long serial rollout ran. `_extract_success_info` was updated to reduce
nested success values safely; five focused success/parallel-rollout tests pass.
Already-running Python processes cannot load this source edit, so the two live
jobs remain on their serial fallback until they finish or are restarted/resumed.

## Follow-up: short GOP was still too slow

The earlier short-GOP result was not sufficient for the user's two-day target.
Cross-benchmark completed-run logs showed:

- LIBERO baseline: 0.314 seconds/step, about 26.2 hours for 300k.
- robomimic baseline: 0.256 seconds/step, about 21.4 hours.
- MimicGen baseline: 0.232 seconds/step, about 19.3 hours.
- stopped RoboCasa baseline: 36.38 seconds/step.

At batch 256, the short-GOP RoboCasa cache still averaged 1.928 seconds of
input wait alone. LIBERO, robomimic, and MimicGen read in-memory arrays, while
RoboCasa was still opening and decoding 768 video streams per batch. RoboCasa
also transferred three 256x256 float cameras before resizing on the GPU.

## Final array-cache implementation

- Added `tools/prepare_robocasa_array_cache.py`.
  - Sequential PyAV decoding matches LeRobot's decoded source pixels exactly.
  - Frames are resized once with torchvision bilinear antialiasing, quantized
    to 128x128 uint8, and written to one global CHW `.npy` mmap per camera.
  - State and action are materialized as global float32 arrays.
  - Episode ends are recorded for exact boundary clamping.
  - The cache is staged atomically, content-marked, structurally validated,
    reusable, and never changes the NAS source.
- The full three-task cache is about 171 GiB and is placed in the node's
  1.6-TiB `/dev/shm`, shared by all nine runs.
- `--force` always rebuilds the node-local destination; without it, a
  structurally current cache is reused.
- `mip/datasets/robocasa_dataset.py` bypasses LeRobot and PyAV when the array
  marker is present. It performs only clamped mmap indexing.
- Future4 now reads offsets `[0, 1, 5]`, rather than decoding unused frames
  2–4.
- Cached RGB stays uint8 through DataLoader collation, pinning, IPC, and H2D.
  `examples/train_robomimic.py` converts it to float32 and `[-1, 1]` on GPU.
- Evaluation renders directly at 128x128, matching RoboCasa's official
  evaluation helper. This is consistent across all new baseline/future runs,
  but is not pixel-for-pixel identical to the stopped launcher's 256-to-128
  evaluation path. A ten-worker persistent rollout smoke tool was added.

## Final validation and timing

- Full SteamInMicrowave cache: 487,012 frames, 511 episodes, about 67 GiB.
- State/action values and episode-boundary clamping match the original path
  exactly.
- Image error versus the original float resize:
  - mean absolute error: 0.000891 on `[0, 1]`
  - maximum absolute error: 0.001961
  - this is only uint8 quantization and is substantially smaller than the
    earlier short-GOP re-encoding error.
- Batch 256, 12-worker DataLoader long test:
  - baseline mean standalone wait: 0.0270 seconds
  - future mean standalone wait: 0.0429 seconds
- Real cached-batch CUDA update, including input wait, H2D, normalization,
  forward, backward, optimizer, and synchronization:
  - baseline: 0.3192 seconds/step, 26.6 hours per 300k training steps
  - future: 0.3577 seconds/step, 29.8 hours
  - steady-state wait hidden behind GPU compute: about 0.0007 seconds
  - peak memory: 12.1 GiB baseline and 15.4 GiB future
- Ten rollout workers started, reset, and stepped successfully. At direct
  128x128 rendering, one eight-action parallel cycle took about 3.05 seconds
  on the 11-CPU DSW host. With 40 episodes every 10k steps, worst-case
  full-horizon rollout overhead gives an overall estimate of roughly 2.1–2.6
  days depending on task. Parallel evaluation now stops and removes workers
  from the active set as soon as their episode succeeds, matching the official
  RoboCasa evaluation behavior and reducing later evaluation cost.
- The DLC job timeout was raised from 48 to 72 hours so the slowest
  full-horizon evaluation case is not killed near completion.
- The final targeted regression suite passes all 36 tests, including array
  cache indexing, baseline/future behavior, active-worker rollout dispatch,
  evaluation RNG, joint future loss, runtime isolation, and temporal crops.
- Both formal node plans pass dry-run with batch 256, 12 DataLoader workers,
  ten rollout workers, and 300k steps.

No formal job was submitted or restarted. The user will decide whether to
restart after reviewing these results.

## Formal online launch

- The first `formal1` submission used the launcher's offline W&B mode. After
  confirming that losses were being written locally, the user requested a
  clean restart with live W&B.
- The two `formal1` jobs were stopped without deleting their NAS logs:
  `dlc5biy2yeftdomf` and `dlc5vi4uiqrj34mw`.
- The launcher now defaults to `ROBOCASA_WANDB_MODE=online`; its resolved Hydra
  configs assert `log.wandb_mode=online`, and the DLC command explicitly sets
  `SYNC_WANDB=0`.
- Active online jobs:
  - node 1: `dlc1rlpceiyqy3g9`
  - node 2: `dlclphgqymbhye0i`
- Both jobs reached `Running`, contain an injected nonempty `WANDB_API_KEY`,
  and use the `jepa-policy/robocasa` project. A direct W&B API authentication
  and project read check succeeded as viewer `ykj`.
- At handoff, both jobs were still building their one-time node-local array
  caches. W&B run pages appear only after the cache and launch preflights
  finish and the first training process initializes W&B.

## Transport four-camera dataset audit

- DLC job `dlc1k8f0nevdurb1`
  (`transport-4cam-dataset-20260723-retry1`) succeeded after about 10 hours
  57 minutes.
- Final output:
  `/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/transport/ph/image_4cam.hdf5`
  (3,319,270,588 bytes).
- Independent HDF5 validation passed:
  - 200 demos and 93,752 samples;
  - four 84x84 RGB camera streams per sample;
  - all image arrays are gzip-compressed `uint8`;
  - all demo shapes match their action lengths;
  - no duplicated `next_obs` groups;
  - sampled frames from the first, middle, and last demos are non-constant;
  - no `.inprogress` file remains.
- `transport_ph_image_4cam.yaml` and the new-seed launcher point to this final
  file. The current launcher deliberately retains the older single-camera
  Transport task for seed 42, so that must be changed before running a fully
  comparable four-camera three-seed matrix.

## Planned Transport four-camera Ablation-2 matrix

- Added `run_transport_4cam_ablation2_dlc_node.sh` and
  `tools/submit_transport_4cam_ablation2_dlc.sh`.
- The dedicated matrix always uses
  `transport_ph_image_4cam` for all three seeds, including seed 42.
- Each seed has six runs:
  - baseline;
  - future4 at future/action ratios 0.05, 0.1, and 0.2;
  - future2 at ratio 0.1;
  - future6 at ratio 0.1.
- Total: 3 seeds x 6 variants = 18 runs. The seed/variant matrix is crossed
  between two DLC nodes with nine runs per node and no overlap.
- The scientific and logging settings match the existing Ablation-2 runs:
  batch 256, 300,000 optimization steps, temporal-consistent 90% crop,
  10 persistent rollout workers, 40 evaluation episodes, online W&B entity
  `jepa-policy`, and project `ablation2`.
- The node launcher checks the validated dataset size and SHA-256, all 200
  demos, 93,752 samples, four RGB streams, six low-dimensional observations,
  dual-arm action dimensions, resolved Hydra settings, hardware, environment
  versions, W&B access, and a four-camera Transport EGL reset.
- Both node launchers passed dry-run with nine configs each. The submission
  helper also passed dry-run.
- Submitted the two formal DLC nodes on 2026-07-25:
  - node 1: `dlc15eygwaqzbnul`
    (`transport-4cam-ablation2-node1-20260725-formal1`);
  - node 2: `dlc15oy2a2s6wcam`
    (`transport-4cam-ablation2-node2-20260725-formal1`).
- Both jobs were accepted by DLC and were `Queuing` at handoff on resource
  group `quotaa5c3d64iror`, with no reported status error.

## Transport four-camera Ablation-2 failure and local fix

### Failed formal jobs

- The two `formal1` jobs subsequently entered `Failed`:
  - node 1: `dlc15eygwaqzbnul`
    (`transport-4cam-ablation2-node1-20260725-formal1`);
  - node 2: `dlc15oy2a2s6wcam`
    (`transport-4cam-ablation2-node2-20260725-formal1`).
- Both failures happened during launcher preflight, before any of the 18
  training runs started. No formal training results were produced and no
  run log directories were created by these jobs.
- The dataset itself was present and passed the full 200-demo, 93,752-sample
  four-camera validation and SHA-256 check on both nodes. The 16-device
  PyTorch and package-version checks also passed.

### Root causes

1. The EGL environment-reset preflight composed
   `transport_ph_image_4cam` without the local
   `task.dataset_path=/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/transport/ph/image_4cam.hdf5`
   override used by the real training commands. It therefore fell back to
   `dataset_repo` and attempted to download
   `robomimic/transport/ph/image_4cam.hdf5` from
   `ChaoyiPan/mip-dataset`. That file is not published there, so both jobs
   exited on an HTTP 404 during the reset preflight.
2. The W&B preflight heredoc placed `||` at the end of the command line and
   the `fail` command on the following line. Bash included that shell line in
   the Python heredoc, producing `IndentationError: unexpected indent`.
   Because of the surrounding shell conditional semantics, the launcher
   continued to the later EGL preflight instead of stopping cleanly at this
   first error.

### Implemented fix

- `run_transport_4cam_ablation2_dlc_node.sh` now passes the validated NAS
  dataset path into the EGL reset process, removes `task.dataset_repo`, adds
  `task.dataset_path`, and asserts both resolved values before constructing
  the environment.
- The W&B preflight now keeps `|| fail ...` on the same command line as the
  Python heredoc invocation, so a real W&B failure terminates the launcher and
  shell text cannot be parsed as Python.

### Validation after the fix

- Bash syntax checks pass for the node launcher and submission helper.
- Both node dry runs pass, resolving all 18 unique Hydra configurations.
- The submission-helper dry run passes with the proposed `fix1` names.
- The corrected four-camera reset preflight was executed locally with the
  same NAS dataset override and bundled Mesa EGL environment. It successfully
  reset Transport and verified all four observation tensors as
  `(3, 84, 84)`.

### Current handoff state

- The launcher is fixed locally.
- No replacement Transport DLC jobs have been submitted yet.
- The intended next action is to submit two new jobs with unique names, for
  example suffix `fix1`, then verify that both pass preflight and enter
  training before considering the relaunch complete.

## RoboTwin isolated setup and formal matrix

- RoboTwin is isolated under `third_party/robotwin` and pinned to upstream
  commit `c3ddfa8b97d5519efa828b075999bd0006778e5e`.
- The policy rollout integration keeps MPLib/TOPP motion planning while making
  CuRobo optional. Only the Aloha embodiment and object assets needed by the
  three clean tasks are copied into each node-local runtime.
- The official 50-episode clean datasets were downloaded and content-checked
  for:
  - `stack_bowls_three`
  - `handover_block`
  - `put_object_cabinet`
- Compact array caches contain 23,550, 14,084, and 13,460 transitions,
  respectively. State and actions are 14-dimensional float32 arrays, and
  observations are pre-resized 128x128 NCHW uint8 arrays. Batch 256 with 12
  workers averaged about 0.090 seconds of standalone input wait locally.
- The formal matrix has exactly 18 unique runs: three tasks x seeds 41/42/43 x
  baseline/Future4-ratio0.1. It is split into nine runs on each of two nodes,
  uses 300,000 gradient steps, batch 256, ten persistent rollout workers, 40
  evaluation episodes, and online W&B at `jepa-policy/robotwin`.
- Stable W&B run IDs and `model_latest.pt` auto-resume prevent a spot
  interruption from creating duplicate histories when a job is resubmitted.
- The isolated Python 3.12 runtime archive is
  `artifacts/robotwin/robotwin_runtime_py312.tar.gz`, SHA-256
  `7615b3c42d32af7ed092490539d6a89bc21912674bc3d9ad3e9cc5caf9e72039`.
- The Ubuntu 24 graphics-library archive is
  `artifacts/robotwin/robotwin_system_libs_ubuntu24_x86_64.tar.gz`, SHA-256
  `1c368772abfa4d73431a17e8a4c2da1a91f248019f4212cf32b631e7a6cf0e6f`.
- The targeted RoboTwin/runtime/future-loss regression suite passes all 35
  tests. All caches pass structural validation, all 18 Hydra configurations
  resolve correctly, and both nine-run launch plans pass dry-run.

### GPU infrastructure diagnosis

- The current PAI quota nodes expose PPU accelerators rather than NVIDIA
  Vulkan/EGL and therefore cannot render SAPIEN.
- Lingjun H20 and L20X nodes run CUDA correctly, but their injected NVIDIA
  driver does not expose a usable Vulkan ICD to the container. SAPIEN fails at
  `vkCreateInstance`, so those nodes are not used for RoboTwin.
- A standard ECS `ecs.gn8is.4xlarge` spot node with one NVIDIA L20 48 GiB
  passed CUDA, torchvision, and a real SAPIEN renderer creation test.
- The formal launcher targets standard ECS L20 nodes, packages the missing
  Ubuntu graphics libraries, requests NVIDIA graphics driver capabilities,
  and runs a full three-environment reset/step preflight before training.
- For the two-GPU `ecs.gn8is-2x.8xlarge` layout, each node schedules its nine
  experiments in waves of at most four, with at most two processes per GPU.
  The preflight tests baseline and Future concurrently on one GPU at the
  formal batch size before any 300k run is allowed to start.

## RoboTwin progress checkpoint — 2026-07-25 11:26 UTC

### Completed

- The three requested RoboTwin tasks are configured:
  `stack_bowls_three`, `handover_block`, and `put_object_cabinet`.
- The official clean 50-episode datasets, required task assets, pinned
  RoboTwin source, isolated Python runtime, JEPA dataset/environment adapters,
  baseline/Future configurations, tests, and two-node launcher are present on
  NAS.
- The 18-run formal matrix is fixed at:
  three tasks x seeds 41/42/43 x
  `{baseline, Future4 ratio=0.1}`.
- Formal settings are batch 256, 300,000 optimization steps, online W&B
  entity `jepa-policy`, project `robotwin`, 40 evaluation episodes, and ten
  persistent rollout workers.
- The two nodes each own nine non-overlapping runs. On a two-L20 node the
  launcher runs waves of `4 + 4 + 1`, with at most two processes per GPU and
  six DataLoader workers per process.
- Hydra composition and both nine-run dry plans pass. The targeted local
  regression suite passes all 35 tests.
- W&B uses a stable hash-derived run ID and `resume=allow`. A run with
  `model_latest.pt` resumes model, optimizer, step, and evaluation state. If a
  spot reclaim happens before the first checkpoint, the same uniquely tagged
  run restarts from step zero instead of blocking the entire node.

### GPU and DLC findings

- Existing workspace quota nodes are PPU rather than NVIDIA GPUs, so they
  cannot run the SAPIEN NVIDIA rendering path.
- Lingjun H20 and L20X probes passed CUDA but failed Vulkan instance creation
  because their injected NVIDIA driver/ICD path is not usable by SAPIEN.
- Standard ECS L20 works:
  - one-L20 probe `dlcdpg8i2ns2vzsn` succeeded;
  - two-L20 probe `dlc4jz956ezf4ddp` succeeded and reported two
    `NVIDIA L20` 48-GiB devices plus `SAPIEN_RENDERER_OK`.
- Workspace `594560` rejects standard ECS pay-as-you-go jobs. The working ECS
  L20 path therefore uses `SpotWithPriceLimit` with discount limit `1.0`,
  meaning it accepts up to the normal price to maximize allocation
  probability but remains reclaimable.
- A temporary one-GPU fallback preflight was submitted while the two-GPU
  resource was being purchased, then deliberately stopped once the two-GPU
  preflight obtained a node. It never launched formal training.

### Full-preflight issue and fix

- First full preflight:
  `dlc1ljxwy2czmmpa`
  (`robotwin-formal-node1-20260725-ecs-preflight1`).
- It reached a real two-L20 node and passed:
  - all nine resolved node-1 configurations;
  - PyTorch 2.6 CUDA arithmetic on two GPUs;
  - runtime archive verification and extraction;
  - source/assets/cache copy into node-local `/dev/shm`.
- It then failed before creating a RoboTwin task because importing OpenCV
  reported:
  `ImportError: libGL.so.1: cannot open shared object file`.
- Root cause: the ChatLearn Ubuntu 24 image does not install the GLVND OpenGL
  runtime required by the packaged non-headless OpenCV wheel. This is separate
  from the already validated NVIDIA Vulkan path.
- The isolated system-library archive was rebuilt to include:
  `libGL.so.1`, `libGLX.so.0`, `libGLdispatch.so.0`, EGL/Vulkan/X11
  dependencies, and OpenCV's GLib/Pcre dependencies.
- Current graphics-library artifact:
  `artifacts/robotwin/robotwin_system_libs_ubuntu24_x86_64.tar.gz`,
  SHA-256
  `1c368772abfa4d73431a17e8a4c2da1a91f248019f4212cf32b631e7a6cf0e6f`.
- Both the node launcher and Python wrapper now verify those libraries before
  starting. Bash syntax, archive contents, hash, and local shared-object
  dependency checks pass.

### Current external state

- Second full preflight:
  `dlcnzi2h6lpwytfz`
  (`robotwin-formal-node1-20260725-ecs-preflight2`).
- At 2026-07-25 11:26 UTC it was `Creating`, waiting for the standard ECS
  two-L20 spot node. It had not started a container and therefore had not read
  a stale copy of the repaired archive.
- No RoboTwin formal 300k training job has been submitted yet.
- No formal RoboTwin W&B training run has been created yet. The launcher will
  only submit/start formal runs after the second full preflight passes:
  three real environment reset/steps, array-cache benchmark, nonzero-GPU
  mapping, ten-worker rollout, concurrent batch-256 baseline/Future optimizer
  updates, and W&B online write/read verification.

### Next actions

1. Wait for `dlcnzi2h6lpwytfz` to reach `Running`.
2. Inspect logs for `ROBOTWIN_SETUP_OK`,
   `ROBOTWIN_ARRAY_CACHE_BENCHMARK`, `GPU_MAPPING_OK`,
   `ROBOTWIN_ROLLOUT_POOL_OK`, `OPTIMIZER_SMOKE_OK`,
   `WANDB_ONLINE_OK`, and final `PREFLIGHT_ONLY_OK`.
3. If all markers pass, submit the two formal nine-run node jobs with a new
   common run tag and online W&B.
4. Confirm both formal jobs reach `Running`, pass their repeated preflight,
   and create their first W&B training runs before handing off.
5. If a spot node is reclaimed, resubmit the affected node with the same run
   tag so stable W&B IDs and local checkpoints resume the original runs.
