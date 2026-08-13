# Vertex Omni Production Release Gate Evidence

**Date:** 2026-08-13

**Branch:** `codex/vertex-interactions`

**Tested base commit:** `600aff15059cb71e4f7de2c645791f2d15b0f728`

**Topology:** Two LiteLLM proxy processes, shared PostgreSQL, shared coordination Redis, stable salt, two equal-weight real Vertex deployments

**Release decision:** **GO for the bucket-capable account topology tested on 2026-08-14.** The LiteLLM integration passed the supported contract, security, routing, limiter, persistence, background, stop/stream, inline-media and GCS URI boundaries. The retest duplicated the bucket-capable service account across two logical deployments, so independent-account redundancy remains unverified and the non-bucket-capable account must not be included in the production Omni pool.

No credentials, virtual keys, raw interaction IDs, inline media or database URLs are recorded here.

## Results

| Suite | Result | Real evidence |
|---|---|---|
| 1. Basic endpoints | PASS | Authenticated create, `in_progress` response, terminal poll, repeated retrieve, stable opaque ID, continuation and missing/invalid auth/model/payload assertions passed. Missing model now returns `400`, not `500`. |
| 2. Complete modality smoke | PARTIAL / URI BLOCKED | Text and image understanding passed. Text-to-video, image-to-video, two-reference-image video and video edit produced real inline MP4 model outputs. Audio input was rejected by both deployments as unsupported. Video input behaved as video edit/generation, not text understanding. Account A URI output failed GCS authorization. One GCS reference job remained empty/nonterminal beyond ten minutes. |
| 3. Service-account routing and affinity | PASS | New work selected both deployment aliases. Polling and `previous_interaction_id` continuation stayed on the creating deployment. Direct deployment alias selection returned `403`. |
| 4. Ownership and opaque IDs | PASS | Cross-key GET and continuation returned `403`. Cross-key cancel/delete returned `403`. Tampered/truncated IDs returned `400`. Model-restricted keys cannot use raw provider IDs. IDs remained stable across GET, replica and restart. |
| 5. RPM | PASS | With RPM 3, requests returned `200, 200, 200, 429` across two replicas. With RPM 1, reset behavior returned `200, 429, 200` after the one-minute window. |
| 6. TPM | PASS | Terminal usage entered the shared limiter once. Eight concurrent terminal GETs did not duplicate usage or authoritative settlement. Oversized input returned `429`. A second key remained independent. The original key succeeded after reset. |
| 7. Parallel requests | PASS | With `max_parallel_requests=1`, a 20-request burst admitted one and rejected nineteen. A later request through the other replica succeeded. Slots also released after provider validation error and client disconnect. |
| 8. Exact-once persistence | PASS | Create on replica A, poll/retrieve through both replicas, eight concurrent GETs and process restart preserved ID, usage and affinity. PostgreSQL contained one SpendLogs row for the 122-token interaction with duplicate count 1. |
| 9. Background execution | PASS | Background create returned `in_progress`, survived the creating client's disconnect, and reached `completed` through a new client. Restart and second-replica retrieval passed. |
| 10. Streaming and stop contract | PASS | Live create streaming returned `400`; one-shot SSE snapshot returned `200`; cursor resume returned `400`; owner cancel returned explicit client-side-only `400`. |
| 11. Validation and media errors | PASS | Missing input, corrupt base64, invalid image bytes/MIME, invalid video task and unsupported audio output all returned deliberate `400`, never `500`. Unsupported owner delete now returns `400`; cross-owner mutation returns `403`. |

## Real media evidence

| Scenario | Deployment | Result | Output proof | Usage proof |
|---|---|---|---|---|
| Text to video, inline | Account A | PASS | 2,682,082-byte MP4 model output | 16 input, 57,920 video output, 266 thought, 58,202 total tokens |
| Image to video, inline | Account B | PASS | 2,535,417-byte MP4 model output with SHA-256 recorded in local evidence | 1,107 input, 57,920 video output, 313 thought, 59,340 total tokens |
| Two references to video, inline | Account B | PASS | 2,716,786-byte MP4 model output | 2,217 input, 57,920 video output, 481 thought, 60,618 total tokens |
| Obvious video edit, inline | Account A | PASS | 952,830-byte MP4 model output; SHA-256 differs from source | 23,160 input, 23,168 video output, 253 thought, 46,581 total tokens |
| Text to video, GCS URI | Account A | FAIL | Provider terminal error: service account lacks `storage.objects.create` on configured bucket | No terminal usage |
| Two references to video, GCS URI | Account B | FAIL / STALLED | Accepted, then returned empty snapshots without status, steps or usage beyond ten minutes | No terminal usage |
| Audio input | Accounts A and B | PROVIDER LIMITATION | Correctly typed WAV returned terminal provider error that the model does not support audio input | No terminal usage |

The test helper originally accepted an echoed `user_input` video as an edit output. The release pass caught this false positive. It now validates only video objects inside `model_output` steps. The corrected edit output is byte-distinct from the source.

## Defects found and fixed during the gate

1. Missing `model` and `agent` fell through to generic Gemini and returned `500`. The proxy now rejects the request with `400` before provider selection.
2. A model-restricted key could submit an unattributed raw interaction ID and reach the generic provider path. Restricted keys now require LiteLLM-issued opaque IDs.
3. GET minted a new randomized opaque ID for the same interaction. GET now preserves the caller's opaque resource ID.
4. Cancel/delete skipped managed-resource ownership. They now enforce attribution, ownership and model authorization first.
5. Unsupported Vertex delete returned `500`. It now returns a deliberate `400`.
6. Vertex Interactions metadata advertised audio input despite both real deployments rejecting it. The Vertex Interactions entry now advertises text, image and video input only.
7. The media integration test could validate the echoed input video instead of generated output. It now requires a `model_output` video.
8. Create accepted both `model` and `agent` and silently prioritized one. It now requires exactly one selector.
9. Raw-ID protection originally checked only a key's direct model list. It now also treats inherited team-model and access-group policy as restricted.

## Original required remediation and retest

- Grant both Vertex service accounts `storage.objects.create` for the exact production output bucket/prefix, or disable URI delivery and commit to inline output.
- Rerun text-to-video, image-to-video, reference-to-video and video-edit URI delivery until both deployments have independently completed at least one URI job.
- Rerun the previously stalled reference job shape. A response with no status, steps or usage after the terminal timeout remains a failure.
- Deploy only a commit containing the fixes listed above; rerun this gate against the deployed revision rather than the local process.

The 2026-08-14 retest completed the URI matrix using the bucket-capable account. The remaining deployment requirement is to exclude the account that cannot write the bucket until its URI matrix passes independently.

## Verification limitation

Targeted Ruff, Python compilation, JSON parsing, staged diff checks and all real proxy/provider boundaries described above passed. Repository `make pre-commit` could not reach its lint phase because the workstation has Rust 1.93.1 while the locked AWS crates require Rust 1.94.1. No Rust code or dependency was changed for this integration.

## Local artifacts

Generated media and raw redacted test JSON remain under `/tmp/omni-gate-*`. They are intentionally not committed because responses contain large media and opaque live interaction IDs.

## 2026-08-14 production retest

The latest branch was retested with two logical equal-weight deployments backed by the bucket-capable service account. This validates LiteLLM routing, deployment affinity, two-replica persistence, continuation, GCS delivery and limiter behavior; it does not validate independent Google account redundancy.

| Boundary | Result | Evidence |
|---|---|---|
| Complete real integration gate | PASS | Both contract tests passed in 199.42 seconds. Inline image-to-video, GCS reference-to-video and GCS video edit completed. The edit used a synthetic non-human source because a human-containing fixture triggered a provider safety rejection with zero usage. |
| GCS delivery | PASS | Reference-to-video and video edit returned model-output URIs under a fresh test prefix in the existing bucket. |
| Routing and affinity | PASS | New interactions selected both logical deployment IDs. Polling and `previous_interaction_id` continuation stayed on the creating deployment. |
| Two replicas | PASS | Create, terminal polling, repeated GET and continuation succeeded across ports 4011 and 4012 with shared PostgreSQL and Redis. |
| Settlement | PASS | Each terminal generation produced one authoritative SpendLogs row. Repeated GETs did not change spend. A checker discrepancy was traced to two requests settling before a cumulative assertion, not duplicate billing. |
| Failure release | PASS | Malformed media reached terminal `failed` with no usage and no terminal spend. A subsequent real generation completed with the exact expected spend increment. |
| User limits | PASS | The disposable restricted key reported 60 RPM, 500,000 TPM and 5 parallel requests. An approximately 600,000-token prompt was rejected with 429 before provider dispatch. Existing exact-boundary shared limiter tests cover RPM and parallel reset behavior. |
| Cached input | PASS / none reported | A separate six-turn, 25,212-token conversation reported only each new turn's input tokens and no cache fields. Omni cost and TPM accounting now use all provider-reported input without an undocumented cached-input discount. |

The real test now runs all long media interactions with `background=true`, redacts inline media from terminal-failure output and uses short text interactions to cover both logical routing slots instead of generating redundant videos.
