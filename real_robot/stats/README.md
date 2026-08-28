# Normalization statistics

JEPA and MIP require the `stats.json` generated from the matching training
dataset. Store it as `stats/<task>/stats.json`. These files may reveal dataset
statistics and are intentionally not included by default.

After obtaining the five released files, run `sha256sum -c MANIFEST.sha256`
from this directory to verify that each task uses the reported statistics.
