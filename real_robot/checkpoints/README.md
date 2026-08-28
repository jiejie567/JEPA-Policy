# Checkpoints

Model weights are distributed separately from the Git repository. Preserve
the task and filename layout documented in the root README and publish a
SHA-256 checksum for every released artifact.

`ARTIFACT_MANIFEST.json` records the exact method/task/step matrix and expected
file sizes used for the reported evaluation. It deliberately contains no
private source paths and no model payloads.
