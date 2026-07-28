# JEPA-Policy session notes — 2026-07-22

## Camera-input audit

### LIBERO

- MugMug policy input uses two RGB observations:
  - `agentview_rgb`
  - `eye_in_hand_rgb`
- MokaMoka policy input uses the same two RGB observations.
- `moka_moka_image.yaml` incorrectly used `render_obs_key: agentview_image` even
  though the LIBERO wrapper exposes `agentview_rgb`; this was corrected to
  `render_obs_key: agentview_rgb`.
- LIBERO scene XML defines four fixed scene cameras (`frontview`, `birdview`,
  `agentview`, and `sideview`) and the robot provides an eye-in-hand camera, but
  the policy configurations above consume only the two listed RGB observations.

### Robomimic

- Square consumes two camera images:
  - `agentview_image`
  - `robot0_eye_in_hand_image`
- Tool Hang consumes two camera images:
  - `sideview_image`
  - `robot0_eye_in_hand_image`
- The initial MIP Transport configuration consumed four camera images:
  - `shouldercamera0_image`
  - `shouldercamera1_image`
  - `robot0_eye_in_hand_image`
  - `robot1_eye_in_hand_image`
- Commit `998f05e845b9801709270275e2df2f188107aca3`, authored on
  2026-05-12 13:35:43 +0800 with subject `Use RGB-only future targets`, changed
  Transport to use only `robot1_eye_in_hand_image`. It also changed the configured
  action dimension from 14 to 20 for absolute-action conversion.
- The current NAS dataset
  `/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/transport/ph/image.hdf5`
  is approximately 4.0 GB and actually contains only
  `robot1_eye_in_hand_image`. Its stored actions are 14-dimensional; the training
  pipeline converts them to 20-dimensional absolute actions.
- Consequently, restoring four keys in the old task YAML while continuing to use
  `image.hdf5` would fail because three observations are absent.

## Four-camera Transport configuration

- Added `examples/configs/task/transport_ph_image_4cam.yaml`.
- It uses all four official Transport camera viewpoints and points to a separate
  dataset:
  `/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/transport/ph/image_4cam.hdf5`.
- The existing `transport_ph_image.yaml` was deliberately left unchanged so the
  currently running single-camera experiments and their checkpoints remain
  resumable.
- Baseline and Future experiments using the four-camera setup must both be
  retrained with the same new data and configuration.

## Four-camera dataset generation

- Added executable helper:
  `tools/generate_transport_4cam.sh`.
- It restores simulator states from `image.hdf5`, renders four 84x84 cameras,
  copies rewards and dones, excludes duplicated `next_obs`, enables gzip
  compression, and refuses to overwrite an existing output.
- Intended command on a GPU node:

  ```bash
  CUDA_VISIBLE_DEVICES=15 \
  bash /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy/tools/generate_transport_4cam.sh \
  2>&1 | tee /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy/logs/generate_transport_4cam.log
  ```

- Initial attempts from the current execution environment failed before creating
  output because no EGL/OpenGL backend was available:

  ```text
  AttributeError: 'NoneType' object has no attribute 'eglQueryString'
  ```

- The helper now automatically prefers NVIDIA EGL and falls back to CPU OSMesa.
  The current PPU DSW has `libosmesa6` installed, and a two-frame smoke test
  successfully rendered all four 84x84 RGB observations with non-constant pixel
  values. The Robomimic and Robosuite private-macro warnings are unrelated.
- OSMesa is suitable for validation but can be substantially slower. Full dataset
  generation should still prefer a node with working NVIDIA EGL when available.
- Useful diagnostics on the intended DLC GPU node:

  ```bash
  nvidia-smi
  find /usr /lib /opt -name 'libEGL.so*' 2>/dev/null
  echo "$NVIDIA_DRIVER_CAPABILITIES"
  ```

- If NVIDIA libraries are under `/usr/local/nvidia/lib64`, use:

  ```bash
  export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  ```

- If `libEGL.so` is absent, recreate the DLC container with
  `NVIDIA_DRIVER_CAPABILITIES=all`, or install the appropriate EGL/GLVND runtime
  libraries when root access is available.

## Ablation-2 run status and preliminary results

- Examined:
  - `logs/ablation2/libero` (8 runs)
  - `logs/ablation2/robomimic` (12 runs)
- All runs target 300k gradient steps. At inspection they had reached roughly
  190k–271k and their log files were still advancing. They should continue rather
  than be restarted.
- No Python traceback, CUDA OOM, or non-finite-loss failure was observed.
- Approximate recent success rate (mean of the last five 40-episode evaluations
  at inspection time):

  | Task | Baseline | ratio=.05 | ratio=.10 | ratio=.20 |
  |---|---:|---:|---:|---:|
  | MugMug | 78.5% | 81.0% | 90.0% | 92.5% |
  | MokaMoka | 77.5% | 81.5% | 82.0% | 74.5% |
  | Square | 86.5% | 88.0% | 86.0% | 91.5% |
  | Tool Hang | 28.5% | 38.5% | 43.0% | 38.0% |
  | Transport | 42.0% | 43.0% | 41.0% | 26.5% |

- Preliminary interpretation:
  - MugMug benefits clearly from `.10` / `.20`.
  - MokaMoka favors `.05` / `.10`; `.20` appears too strong.
  - Square currently favors `.20`.
  - Tool Hang `.10` is the most stable improvement; `.20` has a high but unstable
    peak.
  - Single-camera Transport shows no convincing Future improvement, and `.20`
    degrades late performance. This result should not be generalized to the
    original four-camera MIP setting.
- Final comparisons should use an equal 300k training budget. Winning settings
  should subsequently be repeated with additional seeds because the current
  matrix uses only seed 42 and each 40-episode evaluation changes in increments
  of 2.5 percentage points.

## Follow-up seeds

- The Ablation-2 launcher now accepts `SEED=<non-negative integer>` while
  retaining seed 42 as its backward-compatible default. Run names, manifests,
  status files, and temporary caches are isolated by seed.
- Planned follow-up matrices use seeds 41 and 43 for all five tasks.
- Transport is the exception that must first move to the four-camera dataset.
  Do not launch new Transport seeds against the old single-camera `image.hdf5`;
  generate `image_4cam.hdf5` and switch the Transport task entry deliberately.

## Future-horizon ablation

- Planned a separate non-Transport horizon matrix at future-loss ratio 0.1:
  future steps 2 and 6, seeds 41/42/43, across MugMug, MokaMoka, Tool Hang,
  and Square (24 runs total).
- The two-node launcher assigns the 12 LIBERO runs to one node and the 12
  Robomimic runs to another. Each seed uses four GPUs, with the benchmark-
  specific Python environments and dependency preflights retained.

## Files changed during this session

- `examples/configs/task/moka_moka_image.yaml`
- `examples/configs/task/transport_ph_image_4cam.yaml` (new)
- `tools/generate_transport_4cam.sh` (new)
- `docs/session_notes_2026-07-22.md` (this file)

## Later-session summary: remote access and proxy automation

- The DSW SSH server permits TCP forwarding (`AllowTcpForwarding yes` and
  `GatewayPorts no`). The Mac-side SSH config was set up to create this reverse
  tunnel automatically whenever VS Code Remote-SSH connects to `dsw`:

  ```sshconfig
  Host dsw
      HostName 47.98.194.208
      User root
      Port 1024
      ServerAliveInterval 60
      ServerAliveCountMax 3
      TCPKeepAlive yes
      ExitOnForwardFailure yes
      RemoteForward 7897 127.0.0.1:7897
  ```

- The Mac must have BitzNet running with its mixed proxy listening on local port
  7897 before connecting.
- DSW `/root/.bashrc` sources `/root/.dsw_proxy.sh`. That script now exports
  HTTP, HTTPS, and SOCKS proxy variables pointing at `127.0.0.1:7897`.
- A new interactive shell successfully loaded all proxy variables, and a GitHub
  request through the tunnel returned HTTP 200.

## Later-session summary: Transport four-camera generation

- The difference between fixing the generation pipeline and generating the full
  dataset was clarified: only a two-frame smoke dataset had initially been
  rendered; the complete 93,752-frame `image_4cam.hdf5` did not yet exist.
- The current DSW is a PPU node rather than an NVIDIA EGL node. `libosmesa6` and
  minimal Mesa runtime packages were installed for CPU rendering tests.
- A real two-frame conversion successfully rendered all four 84x84 RGB cameras
  with non-constant pixels under both OSMesa and the bundled Mesa EGL runtime:
  `shouldercamera0_image`, `shouldercamera1_image`,
  `robot0_eye_in_hand_image`, and `robot1_eye_in_hand_image`.
- `tools/generate_transport_4cam.sh` now:
  - selects native NVIDIA EGL when available;
  - supports OSMesa fallback;
  - uses the NAS-bundled Mesa EGL/llvmpipe runtime when EGL is requested in a
    PPU/XPU image without NVIDIA EGL libraries;
  - writes to `image_4cam.hdf5.inprogress` first;
  - validates demo count, total sample count, all four image keys, 84x84 RGB
    shapes, `uint8` dtype, and absence of duplicated `next_obs`;
  - atomically renames the validated file to `image_4cam.hdf5`.
- This fixed an audit finding that the submitted PPU/XPU job could otherwise
  repeat the old `AttributeError: 'NoneType' object has no attribute
  'eglQueryString'` failure.
- A dedicated DLC data-generation job was submitted:
  - name: `transport-4cam-dataset-20260722`
  - Job ID: `dlcj93vnu1gr5we6`
  - resources: 1 GPU, 10 CPU, 96 GiB memory/shared memory
  - status at final audit: `Queuing / JobEnqueued`
  - log: `logs/generate_transport_4cam.log`
  - output: `datasets/robomimic/transport/ph/image_4cam.hdf5`
- Reusable submission helper: `tools/submit_transport_4cam_dlc.sh`.

## Later-session summary: seed 41 and 43 replications

- The common Ablation-2 launcher originally hard-coded seed 42. It now accepts
  `SEED=<non-negative integer>`, while retaining 42 as the default so the
  existing seed-42 launch path remains unchanged.
- Run names, manifest/status files, and temporary caches are isolated by seed.
- `run_ablation2_robomimic_12.sh` now supports `SKIP_TRANSPORT=1`.
- New-seed Transport runs are protected from accidentally using the old
  single-camera data: when Transport is included for a non-42 seed, the launcher
  selects `transport_ph_image_4cam` and requires `image_4cam.hdf5`.
- Because the four-camera dataset was still pending, seeds 41 and 43 were
  submitted without Transport. Each seed therefore contains 8 LIBERO runs and
  8 Robomimic runs, or 32 runs in total:

  | Seed | Benchmark | Runs | Job ID |
  |---:|---|---:|---|
  | 41 | LIBERO | 8 | `dlc1q9jytmkj8rhe` |
  | 41 | Robomimic (no Transport) | 8 | `dlc1qjjk7e67t9bd` |
  | 43 | LIBERO | 8 | `dlc1qtj5l65ctxo6` |
  | 43 | Robomimic (no Transport) | 8 | `dlc1r3iqyyb4xz07` |

- Every node requests 16 GPUs, 160 CPUs, and 1600 GiB memory/shared memory.
- Reusable submission helper: `tools/submit_ablation2_followup_dlc.sh`.
- All four jobs were `Queuing / JobEnqueued` at the final audit.

## Later-session summary: future-horizon ablation

- The current main Future experiment predicts step 4. A separate horizon
  ablation was prepared for steps 2 and 6 while keeping the future-loss ratio at
  0.1.
- Matrix: seeds 41/42/43 x horizons 2/6 x MugMug/MokaMoka/Tool Hang/Square =
  24 runs. Transport is excluded.
- Two DLC nodes were submitted:

  | Benchmark | Runs | GPU assignment | Job ID |
  |---|---:|---|---|
  | LIBERO | 12 | seed 41: 0-3; seed 42: 4-7; seed 43: 8-11 | `dlcetk7iiv06ciri` |
  | Robomimic | 12 | seed 41: 0-3; seed 42: 4-7; seed 43: 8-11 | `dlcfdjea2b8n9bev` |

- Both jobs request 16 GPUs, 160 CPUs, and 1600 GiB memory/shared memory and were
  `Queuing / JobEnqueued` at the final audit.
- `run_ablation_future_horizon_12.sh` runs one benchmark per node and starts
  three seed-specific launchers with disjoint GPU offsets.
- `tools/ablation2_launcher_common.sh` now supports GPU offsets and the
  future-horizon mode while preserving the original default matrix.
- `tools/submit_ablation_future_horizon_dlc.sh` submits the two nodes.
- All 24 resolved configs were parsed and checked for the correct seed, horizon,
  ratio 0.1, embedding target, one future token, 300k training steps, and absence
  of Transport.

## Later-session summary: environment isolation and final audit

- LIBERO jobs use
  `/mnt/data_nas/ykj_jepa_policy/venvs/libero/bin/python`.
- Robomimic jobs use
  `/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu/bin/python`.
- Each path retains its environment-version, dataset, CUDA, EGL, and W&B
  preflights. Dry runs passed for the new matrices.
- The original default future-step-4 seed-42 matrix was dry-run again after the
  launcher changes; all 8 LIBERO and 12 Robomimic configs still validated.
- No new run directory conflicted with existing logs. Horizon runs include the
  `future_horizon` suffix and explicit `future2_ratio010` or
  `future6_ratio010` names.
- None of the seven newly submitted DLC jobs had started at the final audit; all
  were still queued, so the NAS-side fixes apply when they begin.
- No plaintext W&B credential was stored in repository scripts. A W&B API key
  was pasted into chat during the session; it should be rotated in W&B even
  though it was omitted from notes and submission files.

## Remaining follow-up

1. Monitor the seven queued DLC jobs and inspect preflight output as each starts.
2. Validate the completed four-camera HDF5 before launching Transport training.
3. Submit the missing four-camera Transport baseline/Future runs for all desired
   seeds, using a consistent camera configuration across compared methods.
4. Let the original seed-42 20-run matrix finish at 300k; do not restart it.
5. Aggregate equal-budget final metrics and then compare seeds 41/42/43 and the
   future horizons 2/4/6.
