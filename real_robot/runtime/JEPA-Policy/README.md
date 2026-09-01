# JEPA Policy and MIP runtime snapshot

This directory contains the MIP baseline and the future-supervised JEPA variant
used by the real-robot inference adapter. It is derived from
`simchowitzlabpublic/much-ado-about-noising` and includes project-specific
ARX-R5 model/configuration changes. The upstream MIT license is preserved.

The authoritative real-robot entry points live under
`../../tools/arx4_jepa_eval`; this directory is loaded as a model runtime.
