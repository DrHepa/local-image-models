# Model Licenses and Upstream Terms

This repository provides **code and integration scaffolding only**.

It does **not** ship the weights for Stable Diffusion 1.5, SDXL Base 1.0, or FLUX.1-schnell.

## Core rule

The repository code is licensed under **MIT**, but the referenced model files remain governed by their **own upstream licenses and access conditions**.

You must obtain, access, and use those model files according to the original upstream terms.

## Referenced models

| Model | Upstream identifier | Upstream license / terms | Notes |
| --- | --- | --- | --- |
| Stable Diffusion 1.5 | `runwayml/stable-diffusion-v1-5` | `creativeml-openrail-m` | Weights are NOT included here. Use remains subject to the CreativeML Open RAIL-M terms. |
| SDXL Base 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` | `openrail++` / CreativeML Open RAIL++-M | Weights are NOT included here. Use remains subject to the SDXL upstream terms. |
| FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` | `apache-2.0` | Access to model files/model card on Hugging Face requires accepting upstream conditions and sharing contact information. Weights are NOT included here. |

## What this means in practice

- You may use, modify, and distribute the **repository code** under MIT.
- You may **not** assume the same license automatically applies to model weights.
- Downloading or using a model referenced by this repo may require separate acceptance of upstream terms.
- If you publish a product or workflow that uses these models, review the upstream model licenses directly before distribution.

## No weight redistribution in this repo

This repository intentionally avoids bundling:

- checkpoints
- weight files
- safetensors bundles
- model snapshots
- other upstream model artifacts

That boundary is intentional so the repository can remain public while preserving the original licensing obligations of each model family.
