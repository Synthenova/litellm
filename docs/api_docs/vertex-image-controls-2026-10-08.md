# Vertex image generation controls

Contract checked against official documentation on 2026-10-08

LiteLLM documents `imageConfig` on image generation requests and the flat aliases
`aspect_ratio`, `aspectRatio`, `image_size`, and `imageSize`. Flat values override
the corresponding nested fields. The provider request carries these controls
inside `generationConfig.imageConfig`

Source: [LiteLLM Vertex image generation](https://docs.litellm.ai/docs/providers/vertex_image#passing-imageconfig-gemini-models)

Google's REST example places `aspectRatio` inside
`generationConfig.imageConfig`. Its image generation page lists
`gemini-3.1-flash-image` and `gemini-3.1-flash-lite-image` among supported models

Source: [Google image generation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/image-generation)

The optional-parameter filter must retain flat aliases when the selected provider
advertises them, allowing its existing mapper to normalize the request. Providers
that do not advertise these aliases retain their existing extra-parameter behavior

An automatic client choice omits the corresponding flat field; omission and `None`
must not create a provider image configuration. This compatibility repair covers
image generation, with no claim about image editing or native Interactions media
controls

## Offline verification

Base: `6c3368cb6fb981d918c49ac8f2244639fa6a2d44`

All 16 spelling/model/nested-override regression cases fail on the base and pass
with the filter repair. Vertex image generation/edit suites pass 81 tests with 2
existing skips; focused filter tests pass 15 tests. Existing Interactions, router,
proxy and Vertex integration suites pass 108 tests with 37 network/provider skips
when run with `--disable-socket --allow-unix-socket`

A supplementary local native contract gate passes 21 checks covering provider
registration, URL construction, unsupported operations, media payloads,
continuation IDs, tamper rejection, response wrapping and modality-aware costs

The broad utility suite passes 349 tests with 2 skips and one pre-existing schema
failure: its model-map schema rejects the native Omni token-estimation fields.
The same failure reproduces from the exact base test source. Production Ruff and
format checks pass; existing utility-test lint violations are not increased

`make bootstrap` and `make pre-commit` cannot complete because the installed Rust
1.92.0 is below the locked AWS crates' required Rust 1.94.1. Source-based Python
tests ran in an isolated environment. The real provider-backed production gate
was not run: this validation made no paid requests or production mutations
