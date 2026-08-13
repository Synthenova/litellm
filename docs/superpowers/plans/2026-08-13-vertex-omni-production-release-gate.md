# Vertex Omni Production Release Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute and document all eleven real integration suites required to release Vertex Gemini Omni through LiteLLM.

**Architecture:** Expand the existing provider-backed integration boundary rather than create a second test framework. Start two local LiteLLM replicas against shared PostgreSQL and Redis only for persistence checks; all other suites use one proxy and disposable virtual keys. Fix defects at their shared production code path, rerun the affected suite, then run the complete gate.

**Tech Stack:** LiteLLM proxy, Python 3, httpx, pytest integration tests, PostgreSQL, Redis, Vertex AI, GCS, ffprobe

## Global Constraints

- Test the LiteLLM fork only; do not change Infiknit or Electron
- Use real provider calls; no mocks, fake providers, or unit-test substitutes
- Never commit service-account JSON, virtual keys, database URLs, access tokens, or inline media
- Preserve LiteLLM meanings: RPM counts HTTP requests, TPM charges configured token type, and `max_parallel_requests` limits concurrent HTTP requests
- Generated media stays in `/tmp`; committed evidence contains only redacted metadata and assertions

---

### Task 1: Basic contract, ownership, and validation

**Files:**
- Modify: `tests/integration_tests/test_vertex_interactions.py`
- Modify if defects surface: `litellm/proxy/google_endpoints/endpoints.py`

**Interfaces:**
- Consumes: `POST /v1beta/interactions`, `GET /v1beta/interactions/{id}`, virtual-key auth
- Produces: passing suites 1, 4, and the non-media portion of suite 11

- [ ] Add real assertions for valid create/poll/retrieve/continuation, invalid auth/model/payload, unknown/malformed/tampered IDs, and cross-key ownership
- [ ] Run the focused integration cases against the real proxy and confirm every failure is a stable 4xx rather than 500
- [ ] Fix any shared proxy defect and rerun the focused cases

### Task 2: Modality and media matrix

**Files:**
- Modify: `tests/integration_tests/test_vertex_interactions.py`
- Modify if defects surface: `litellm/llms/vertex_ai/interactions/transformation.py`

**Interfaces:**
- Consumes: local PNG, MP3/WAV and MP4 fixtures plus writable GCS output
- Produces: passing suite 2 and media validation cases from suite 11

- [ ] Exercise text, image, multiple-image, audio, video and mixed-input understanding through the real provider
- [ ] Exercise text-to-video, image-to-video, reference-to-video and obvious video edit
- [ ] Validate inline and GCS outputs using MIME/file signatures, non-zero duration/resolution and requested edit evidence
- [ ] Exercise corrupt base64, MIME mismatch, corrupt MP4 and invalid media configuration and assert stable errors

### Task 3: Routing, affinity, background, and stream contract

**Files:**
- Modify: `tests/integration_tests/test_vertex_interactions.py`
- Modify if defects surface: `litellm/router.py`, `litellm/interactions/id_utils.py`, `litellm/llms/vertex_ai/interactions/transformation.py`

**Interfaces:**
- Consumes: two equal-weight `vertex-omni` deployments and opaque interaction IDs
- Produces: passing suites 3, 9, and 10

- [ ] Prove both deployments receive new interactions and that GET plus previous-interaction calls retain creator affinity
- [ ] Prove background state progression, concurrent polling and client disconnect behavior
- [ ] Prove live create streaming, cursor resume and provider cancellation are rejected with the documented client-side-only contract

### Task 4: Configured limiter enforcement

**Files:**
- Modify: `tests/integration_tests/test_vertex_interactions.py`
- Modify if defects surface: `litellm/proxy/hooks/parallel_request_limiter_v3.py`, `litellm/proxy/interactions/settlement.py`

**Interfaces:**
- Consumes: disposable virtual keys with small `rpm_limit`, `tpm_limit`, and `max_parallel_requests`
- Produces: passing suites 5, 6, and 7

- [ ] Prove exact RPM boundary, polling behavior, independent keys and reset
- [ ] Prove terminal text/media TPM charge, repeat and concurrent retrieval idempotency, boundary rejection, independent keys and reset
- [ ] Prove concurrent HTTP boundary and slot release after success, error and disconnect

### Task 5: Restart, replica, and exact-once persistence

**Files:**
- Modify: `tests/integration_tests/test_vertex_interactions.py`
- Modify if defects surface: `litellm/proxy/interactions/settlement.py`, `litellm/proxy/db/db_spend_update_writer.py`

**Interfaces:**
- Consumes: two LiteLLM processes sharing PostgreSQL, Redis, salt and configuration
- Produces: passing suite 8 plus restart cases from suites 3 and 9

- [ ] Create on replica A and retrieve concurrently through A and B
- [ ] Restart A and retrieve/continue the same interaction without losing affinity
- [ ] Assert one SpendLogs record and one TPM terminal charge after concurrent/repeated retrieval
- [ ] Fix any persistence defect and repeat the complete boundary

### Task 6: Full gate and evidence

**Files:**
- Create: `docs/superpowers/evidence/2026-08-13-vertex-omni-production-release-gate.md`
- Modify: `tests/integration_tests/test_vertex_interactions.py`

**Interfaces:**
- Consumes: results from tasks 1-5
- Produces: auditable pass/fail evidence for all eleven suites

- [ ] Run the complete real integration gate from a clean proxy start
- [ ] Record redacted scenario results, commit SHA, deployment aliases, terminal states, usage and limit/persistence assertions
- [ ] Run Ruff, compile checks, diff review and the integration gate again for any changed production path
- [ ] Commit and push the reviewed LiteLLM changes and evidence
