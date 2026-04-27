## Known Issues

- Dataset paths and `task.libero_root` remain machine-specific and are intentionally not hardcoded.
- If image training hits cuDNN initialization problems, the shared presets already set `optimization.disable_cudnn=true`.
