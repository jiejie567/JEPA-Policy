# Diffusion Policy runtime snapshot

This directory contains the Diffusion Policy modules needed to restore the
reported ARX-R5 transformer checkpoints. It is derived from
`real-stanford/diffusion_policy` and includes project-specific dataset,
transformer, temporal-crop, and workspace changes. The upstream MIT license is
preserved.

The authoritative real-robot entry points live under
`../../tools/arx4_jepa_eval`; this directory is loaded as a model runtime.
