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

| Extension ID | Visible name | Hugging Face model | Capabilities | Practical VRAM guidance |
| --- | --- | --- | --- | --- |
| `sd15` | Stable Diffusion 1.5 | [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) | `text-to-image`, `image-to-image` | ~6GB+ recommended |
| `sdxl-base` | SDXL Base 1.0 | [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | `text-to-image`, `image-to-image` | ~12GB+ recommended |
| `flux-schnell` | FLUX.1-schnell | [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | `text-to-image` | ~16GB+ recommended; 24GB is more comfortable |

These VRAM values are practical recommendations, not hard guarantees. Actual usage depends on resolution, precision, CPU/GPU offload, driver/runtime behavior, and memory optimizations available in the local environment.

Current baseline bundle flows are verified on the active Linux ARM64 path and remain intended to support the existing Windows candidate path. On Linux ARM64 with NVIDIA GB10 and `torch==2.11.0+cu130`, both SDXL Base `image-to-image` `Style reference` (`sdxl_ip_adapter_style`) and SD1.5 `image-to-image` `Style reference` (`sd15_ip_adapter_style`) have passed installed local-only smoke. SD1.5 promotion is scoped to `image-to-image` only; SD1.5 `text-to-image` is unchanged. This does **not** promote ControlNet, Windows compatibility, or public release readiness. Windows remains prepared/intended but candidate/unverified until GitHub-installed Install/Repair, readiness, and generation evidence is collected on Windows.

For FLUX.1-schnell weights, open [`https://huggingface.co/black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell), log in, accept the model conditions, and share contact information if Hugging Face requests it. Use the same Hugging Face account/token that Modly uses for the download; otherwise the extension cannot download the weights.

## Current Modly compatibility notes

- This is a bundled local image model extension pack for Modly-compatible local model workflows.
- Host-side bundle/model support is still ahead of upstream Modly in a few areas and is being discussed in [Modly issue #114](https://github.com/lightningpixel/modly/issues/114), so use a compatible Modly branch/fork until that support lands upstream.
- Current preview flows use the primary generated image (`output_path`). The runtime can preserve additional output metadata internally, but richer multi-output display is a future Modly integration task.

## Generation controls

- SD15 and SDXL expose text-to-image and image-to-image controls such as prompt, negative prompt, size, steps, guidance scale, seed, and output format.
- SDXL and SD1.5 image-to-image support the optional named `Style reference` input backed by IP-Adapter on the verified Linux ARM64/GB10/cu130 local path. SD1.5 text-to-image is unchanged. ControlNet remains future explicit-node work.
- FLUX.1-schnell exposes text-to-image controls tuned for its pipeline, including prompt, size, steps, guidance scale, maximum sequence length, seed, and output format.
- PNG is the default output format. JPEG output is available with a configurable JPEG quality value.
- FLUX.1-schnell recommends low step counts (`1`-`4`) and guidance `0.0`, but higher values are allowed for experimentation and may not improve quality.

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

Generation is local-only/no-download after Install/Repair and weight installation have acquired the required baseline and optional assets. Install/Repair may acquire supported optional IP-Adapter assets for SDXL and SD1.5 style-reference readiness; generation must use local files only. Windows installation support is prepared as a `windows-amd64` candidate path for later validation and must not be described as verified until real Windows evidence exists. ControlNet is intentionally separate future work with explicit nodes.

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
- model weight hosting or redistribution
- build/release automation

## Manual verification checklist

1. Sync the shared runtime and verify vendored copies.
2. Verify the `Install GitHub` / `Repair` JSON contract for each child root.
3. Verify generator protocol behavior with valid JSON over `stdin`.
4. Verify node-scoped weight readiness under `modelsDir`.
5. Recheck the same baseline for `sd15`, `sdxl-base`, and `flux-schnell`.

This repository remains source-first: it provides the extension bundle and runtime integration while leaving model weight hosting and redistribution to the upstream model providers.
