# Local Image Models

> Public-facing project name for this repository: `local-image-models`.

This repository is a **local image model extension bundle**: a source bundle for 3 self-contained model extensions plus one shared runtime.

It is **not** a weights mirror and it does **not** redistribute the upstream model weights.

## What this repo contains

- `extensions/sd15/`
- `extensions/sdxl-base/`
- `extensions/flux-schnell/`
- `shared/runtime/local_image_runtime/`
- `tools/sync_extension_runtime.py`

Each extension is the installable unit and contains its own:

- `manifest.json`
- `generator.py`
- `setup.py`
- `src/local_image_runtime/` vendored runtime copy

The canonical shared code lives in `shared/runtime/local_image_runtime/` and is synced into each extension root.

## Included model families

| Extension ID | Visible name | Capabilities |
| --- | --- | --- |
| `sd15` | Stable Diffusion 1.5 | `text-to-image`, `image-to-image` |
| `sdxl-base` | SDXL Base 1.0 | `text-to-image`, `image-to-image` |
| `flux-schnell` | FLUX.1-schnell | `text-to-image` |

The existing capability IDs remain unchanged by design:

- `sd15`
- `sdxl-base`
- `flux-schnell`

## Important licensing boundary

This repository distributes:

- repository code
- manifests
- setup/integration scaffolding
- runtime glue code
- documentation

This repository does **not** distribute:

- model weights
- checkpoints
- safetensors bundles
- upstream model artifacts

The repository code is licensed under **MIT**. See [`LICENSE`](./LICENSE).

The referenced models remain subject to their **original upstream licenses and access conditions**. See [`MODEL_LICENSES.md`](./MODEL_LICENSES.md).

If you use any referenced model, YOU are responsible for obtaining and using its files in compliance with the applicable upstream terms.

## Architecture summary

- The architecture is **model-first**.
- Extension identity is the family identity.
- `params.model_id` is legacy compatibility only and must match the fixed extension.
- Capabilities are declared in each manifest, not by central branching.

## Runtime responsibilities

Shared runtime responsibilities:

- bootstrap of `.local-image-runtime/`
- state normalization and migration
- Modly `Install GitHub` / `Repair` contract handling
- legacy local install validation
- payload/request validation helpers
- backend dispatch boundary

Per-family responsibilities:

- manifest identity
- exposed nodes
- node defaults and help text
- minimum local source requirements

## State and weight layout

Current persisted state version: **v2**.

- `extensions`: ownership by family/extension ID
- `legacy_models`: retained legacy residue for fallback, audit, and later cleanup

Each child keeps its own `.local-image-runtime/`, but the canonical weight location for Modly is external to this repository:

```text
modelsDir/<ext.id>/<node.id>/...
```

Examples:

- `modelsDir/sd15/text-to-image/model_index.json`
- `modelsDir/sdxl-base/image-to-image/model_index.json`
- `modelsDir/flux-schnell/text-to-image/model_index.json`

## Sync flow for the shared runtime

When `shared/runtime/local_image_runtime/` changes, resync the vendored copies:

```bash
python3 tools/sync_extension_runtime.py
python3 tools/sync_extension_runtime.py --check
```

## Installation flow

The operational flow has **two separate steps**:

1. **Install GitHub / Repair**: run each extension's `setup.py` with a Modly JSON payload to create `venv`, install dependencies, and persist readiness.
2. **Install Weights**: download model files outside this repo into `modelsDir/<ext.id>/<node.id>/...`.

Example setup invocation:

```bash
python3 extensions/sd15/setup.py '{"python_exe":"/usr/bin/python3","ext_dir":"/tmp/modly-sd15","gpu_sm":"87","cuda_version":"128"}'
```

Legacy local CLI commands still exist for scaffold/manual use, but they are not the main Modly contract.

## Minimum local source layouts

### `sd15`

- `model_index.json`
- `scheduler/`
- `text_encoder/`
- `tokenizer/`
- `unet/`
- `vae/`

### `sdxl-base`

- `model_index.json`
- `scheduler/`
- `text_encoder/`
- `text_encoder_2/`
- `tokenizer/`
- `tokenizer_2/`
- `unet/`
- `vae/`

### `flux-schnell`

- `model_index.json`
- `scheduler/`
- `text_encoder/`
- `text_encoder_2/`
- `tokenizer/`
- `tokenizer_2/`
- `transformer/`
- `vae/`

## Scope

Included here:

- source bundle for 3 local image model extensions
- syncable shared runtime
- persisted v1 -> v2 state migration
- manifests and CLIs per family

Out of scope:

- Modly core changes
- real backend generation integration
- model weight hosting or redistribution
- build/release automation

## Manual verification checklist

1. Sync the shared runtime and verify vendored copies.
2. Verify the `Install GitHub` / `Repair` JSON contract for each child root.
3. Verify generator protocol behavior with valid JSON over `stdin`.
4. Verify node-scoped weight readiness under `modelsDir`.
5. Recheck the same baseline for `sd15`, `sdxl-base`, and `flux-schnell`.

This repository remains scaffold-first: honest structure now, backend integration later.
