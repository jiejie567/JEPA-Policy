# Much Ado About Noising: Do Flow Models Actually Make Better Control Policies?

This repository contains the code for the paper "Much Ado About Noising: Do Flow Models Actually Make Better Control Policies?".
We include common robot behavior cloning algorithms and easy to use dataset loading and training scripts.
This repository contains all the best practices discovered in paper and is designed to be a reference for future research on flow/consistency model/flow map model/shortcut model/regression model for robot learning.

## Installation

```
uv sync
```

## Training

```
uv run examples/train_robomimic.py
```

On the datanas training host, robomimic and LIBERO use incompatible simulator
dependencies. Use `tools/run_robomimic_train.sh` for robomimic tasks and
`tools/run_libero_train.sh` for LIBERO tasks. See
[`docs/environment_isolation.md`](docs/environment_isolation.md).
