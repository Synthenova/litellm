# Vertex Omni Production Release Gate Design

## Goal

Prove the LiteLLM Vertex Gemini Omni integration is safe to release using real proxy, PostgreSQL, Redis, two service accounts, and real provider calls. No mocked provider behavior is accepted as release evidence.

## Evidence rules

- Each suite records the LiteLLM commit, request result, selected deployment alias, terminal state, usage, and relevant limiter or persistence evidence
- Secrets, raw credentials, access tokens, virtual keys, and inline media are never stored in committed evidence
- A test passes only when the client-visible behavior is asserted; HTTP 200 alone is insufficient
- Provider limitations are accepted only when LiteLLM rejects the operation clearly and does not advertise support
- Infiknit and Electron are outside this gate; this gate ends at the LiteLLM API boundary

## Release suites

1. **Basic endpoints:** authentication, create, poll, repeated retrieve, continuation, malformed/tampered/unknown IDs, invalid models and payloads
2. **Complete modality smoke:** text, image, multiple images, audio, video and mixed input understanding; text-to-video, image-to-video, reference-to-video, obvious video edit, inline and GCS output validation
3. **Service-account routing and affinity:** both deployments receive new work; poll and continuation stay on the creating deployment; clients cannot select accounts
4. **Ownership and opaque IDs:** cross-key access and continuation fail; tampering, truncation and raw provider IDs fail without leaking routing or credentials
5. **RPM:** exact configured boundary, reset, independent keys, polling policy and LiteLLM-local 429 evidence
6. **TPM:** exact terminal usage, repeat/concurrent retrieval idempotency, media usage, configured boundary, reset and independent keys
7. **Parallel requests:** configured HTTP concurrency boundary and slot release after success, failure and disconnect
8. **Exact-once persistence:** concurrent settlement, restart retrieval, two-proxy retrieval and shared database/limiter state
9. **Background execution:** immediate ID, state progression, concurrent polling, disconnect and restart survival, terminal failure handling
10. **Streaming and stop contract:** unsupported create streaming, one-shot SSE snapshot, rejected cursor resume, and explicit client-side-only stop/cancel behavior
11. **Validation and media errors:** empty/missing/unsupported inputs, invalid base64/MIME/media/configuration, GCS errors, oversized input, and stable non-500 responses

## Architecture

Extend the existing real integration boundary in `tests/integration_tests/test_vertex_interactions.py`. Reuse its polling, media validation, spend and affinity helpers. Add only the minimum orchestration needed for disposable keys, concurrent calls, proxy restart and a second replica. Produce one redacted Markdown result table under `docs/superpowers/evidence/`; keep large generated media in `/tmp`.

## Release decision

All applicable assertions in suites 1-11 must pass. Any failure is either fixed in LiteLLM and rerun, or recorded as a precise provider limitation with a deliberate client-visible rejection. Unknown, untested, skipped, or indirectly inferred behavior does not pass the gate.
