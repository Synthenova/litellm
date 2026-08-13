import base64
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest


BASE_URL = os.getenv("LITELLM_VERTEX_INTERACTIONS_BASE_URL")
VIRTUAL_KEY = os.getenv("LITELLM_VERTEX_INTERACTIONS_VIRTUAL_KEY")
MODEL = os.getenv("LITELLM_VERTEX_INTERACTIONS_MODEL", "vertex-omni")
GCS_OUTPUT = os.getenv("LITELLM_VERTEX_INTERACTIONS_GCS_OUTPUT")
MP4_PATH = os.getenv("LITELLM_VERTEX_INTERACTIONS_MP4_PATH")
PNG_PATH = Path(__file__).parents[1] / "llm_translation" / "duck.png"

REQUIRED_ENV = {
    "LITELLM_VERTEX_INTERACTIONS_BASE_URL": BASE_URL,
    "LITELLM_VERTEX_INTERACTIONS_VIRTUAL_KEY": VIRTUAL_KEY,
    "LITELLM_VERTEX_INTERACTIONS_GCS_OUTPUT": GCS_OUTPUT,
    "LITELLM_VERTEX_INTERACTIONS_MP4_PATH": MP4_PATH,
    "LITELLM_VERTEX_INTERACTIONS_RPM_LIMIT": os.getenv("LITELLM_VERTEX_INTERACTIONS_RPM_LIMIT"),
    "LITELLM_VERTEX_INTERACTIONS_TPM_LIMIT": os.getenv("LITELLM_VERTEX_INTERACTIONS_TPM_LIMIT"),
    "LITELLM_VERTEX_INTERACTIONS_MAX_PARALLEL_REQUESTS": os.getenv(
        "LITELLM_VERTEX_INTERACTIONS_MAX_PARALLEL_REQUESTS"
    ),
}
pytestmark = pytest.mark.skipif(
    any(not value for value in REQUIRED_ENV.values()),
    reason=f"Set {', '.join(name for name, value in REQUIRED_ENV.items() if not value)}",
)

TERMINAL_FAILURES = {"failed", "cancelled", "incomplete", "budget_exceeded"}


def _without_inline_media(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "<omitted>" if key == "data" else _without_inline_media(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_without_inline_media(child) for child in value]
    return value


def _poll_terminal(
    client: httpx.Client,
    interaction_id: str,
    deployment_id: str,
) -> dict[str, Any]:
    for _ in range(90):
        retrieve = client.get(f"/v1beta/interactions/{interaction_id}")
        retrieve.raise_for_status()
        body = retrieve.json()
        assert body["id"] == interaction_id
        assert retrieve.headers["x-litellm-model-id"] == deployment_id
        status = body.get("status")
        if status == "completed":
            return body
        if status in TERMINAL_FAILURES:
            pytest.fail(f"interaction {interaction_id} ended as {status}: {_without_inline_media(body)}")
        time.sleep(5)
    pytest.fail(f"interaction {interaction_id} did not complete")


def _video_outputs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        matches = [value] if str(value.get("type", "")).lower() == "video" else []
        return matches + [match for child in value.values() for match in _video_outputs(child)]
    if isinstance(value, list):
        return [match for child in value for match in _video_outputs(child)]
    return []


def _model_video_outputs(terminal: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        output
        for step in terminal.get("steps") or []
        if isinstance(step, dict) and step.get("type") == "model_output"
        for output in _video_outputs(step.get("content") or [])
    ]


def _find_duration(value: Any) -> float:
    if isinstance(value, dict):
        for key in ("duration_seconds", "durationSeconds"):
            duration = value.get(key)
            if isinstance(duration, (int, float)) and duration > 0:
                return float(duration)
        return max((_find_duration(child) for child in value.values()), default=0.0)
    if isinstance(value, list):
        return max((_find_duration(child) for child in value), default=0.0)
    return 0.0


def _expected_cost(terminal: dict[str, Any]) -> float:
    usage = terminal["usage"]
    modalities = usage.get("output_tokens_by_modality") or []
    text_tokens = usage.get("text_output_tokens") or sum(
        item.get("tokens") or 0 for item in modalities if str(item.get("modality", "")).lower() == "text"
    )
    if not text_tokens and not modalities:
        text_tokens = usage.get("total_output_tokens") or 0
    video_tokens = usage.get("video_output_tokens") or sum(
        item.get("tokens") or 0 for item in modalities if str(item.get("modality", "")).lower() == "video"
    )
    duration = 0.0 if video_tokens else _find_duration(terminal)
    return (
        (usage.get("total_input_tokens") or 0) * 0.0000015
        + (
            text_tokens
            + (
                usage.get("total_reasoning_tokens")
                or usage.get("total_thought_tokens")
                or usage.get("thought_tokens")
                or 0
            )
        )
        * 0.000009
        + video_tokens * 0.0000175
        + duration * 0.10
    )


def _key_spend(client: httpx.Client) -> float:
    key_info = client.get("/key/info", params={"key": VIRTUAL_KEY})
    key_info.raise_for_status()
    return float(key_info.json()["info"]["spend"])


def _key_rate_limits(client: httpx.Client) -> tuple[int | None, int | None, int | None]:
    key_info = client.get("/key/info", params={"key": VIRTUAL_KEY})
    key_info.raise_for_status()
    info = key_info.json()["info"]
    return info.get("rpm_limit"), info.get("tpm_limit"), info.get("max_parallel_requests")


def _wait_for_spend(client: httpx.Client, expected: float) -> float:
    for _ in range(60):
        spend = _key_spend(client)
        if spend == pytest.approx(expected, abs=1e-12):
            return spend
        if spend > expected + 1e-12:
            pytest.fail(f"spend exceeded expected value: got {spend}, expected {expected}")
        time.sleep(1)
    pytest.fail(f"spend did not reach expected value {expected}; last value was {spend}")


def _assert_spend_stable(client: httpx.Client, expected: float) -> None:
    for _ in range(10):
        assert _key_spend(client) == pytest.approx(expected, abs=1e-12)
        time.sleep(1)


def test_real_vertex_interactions_rejects_missing_model_or_agent() -> None:
    response = httpx.post(
        f"{BASE_URL}/v1beta/interactions",
        headers={"Authorization": f"Bearer {VIRTUAL_KEY}"},
        json={"input": "Hello"},
        timeout=30,
    )
    assert response.status_code == 400
    assert "model or agent" in response.text

    both = httpx.post(
        f"{BASE_URL}/v1beta/interactions",
        headers={"Authorization": f"Bearer {VIRTUAL_KEY}"},
        json={"model": MODEL, "agent": "agent-id", "input": "Hello"},
        timeout=30,
    )
    assert both.status_code == 400
    assert "model or agent" in both.text


def test_real_vertex_interactions_create_retrieve_resume_and_affinity() -> None:
    assert GCS_OUTPUT.startswith("gs://") and GCS_OUTPUT.rstrip("/") != "gs://cloud-samples-data", (
        "LITELLM_VERTEX_INTERACTIONS_GCS_OUTPUT must be a writable GCS location"
    )
    assert PNG_PATH.is_file()
    mp4_path = Path(MP4_PATH)
    assert mp4_path.is_file(), "LITELLM_VERTEX_INTERACTIONS_MP4_PATH does not exist"
    mp4 = mp4_path.read_bytes()
    assert len(mp4) > 12 and mp4[4:8] == b"ftyp", "LITELLM_VERTEX_INTERACTIONS_MP4_PATH must point to a valid MP4"
    png = base64.b64encode(PNG_PATH.read_bytes()).decode()
    video = base64.b64encode(mp4).decode()

    headers = {"Authorization": f"Bearer {VIRTUAL_KEY}"}
    deployment_ids: set[str] = set()
    affinity_interaction_id = ""
    affinity_deployment_id = ""
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=900) as client:
        assert _key_rate_limits(client) == (
            int(os.environ["LITELLM_VERTEX_INTERACTIONS_RPM_LIMIT"]),
            int(os.environ["LITELLM_VERTEX_INTERACTIONS_TPM_LIMIT"]),
            int(os.environ["LITELLM_VERTEX_INTERACTIONS_MAX_PARALLEL_REQUESTS"]),
        )
        baseline_spend = _key_spend(client)
        shape_payloads = (
            (
                {
                    "model": MODEL,
                    "input": [
                        {"type": "image", "data": png, "mime_type": "image/png"},
                        {"type": "text", "text": "Animate this."},
                    ],
                    "response_format": {"type": "video", "delivery": "inline"},
                    "generation_config": {"video_config": {"task": "image_to_video"}},
                    "background": True,
                },
                "inline",
            ),
            (
                {
                    "model": MODEL,
                    "input": [
                        {
                            "type": "image",
                            "uri": "gs://cloud-samples-data/generative-ai/image/scones.jpg",
                            "mime_type": "image/jpeg",
                        },
                        {
                            "type": "image",
                            "uri": "gs://cloud-samples-data/generative-ai/image/character.jpg",
                            "mime_type": "image/jpeg",
                        },
                        {"type": "text", "text": "Use both reference images."},
                    ],
                    "response_format": {"type": "video", "delivery": "uri", "gcs_uri": GCS_OUTPUT},
                    "generation_config": {"video_config": {"task": "reference_to_video"}},
                    "background": True,
                },
                "uri",
            ),
            (
                {
                    "model": MODEL,
                    "input": [
                        {"type": "video", "data": video, "mime_type": "video/mp4"},
                        {
                            "type": "text",
                            "text": (
                                "Recolor the blue robot bright neon magenta and replace the white background "
                                "with vivid lime green."
                            ),
                        },
                    ],
                    "response_format": {"type": "video", "delivery": "uri", "gcs_uri": GCS_OUTPUT},
                    "generation_config": {"video_config": {"task": "edit"}},
                    "background": True,
                },
                "uri",
            ),
        )
        for payload, delivery in shape_payloads:
            shaped = client.post("/v1beta/interactions", json=payload)
            shaped.raise_for_status()
            interaction_id = shaped.json()["id"]
            assert interaction_id.startswith("int_")
            terminal = _poll_terminal(
                client,
                interaction_id,
                shaped.headers["x-litellm-model-id"],
            )
            outputs = _model_video_outputs(terminal)
            assert outputs, f"completed {delivery} interaction returned no video output: {terminal}"
            if delivery == "inline":
                assert any(isinstance(output.get("data"), str) and output["data"] for output in outputs)
            else:
                assert any(
                    isinstance(output.get("uri"), str) and output["uri"].startswith(GCS_OUTPUT) for output in outputs
                )
            baseline_spend = _wait_for_spend(
                client,
                baseline_spend + _expected_cost(terminal),
            )

        for _ in range(8):
            create = client.post(
                "/v1beta/interactions",
                json={
                    "model": MODEL,
                    "input": "Remember that the marble is blue. Reply only READY.",
                    "background": True,
                },
            )
            create.raise_for_status()
            interaction = create.json()
            assert interaction["id"].startswith("int_")
            creator_deployment_id = create.headers["x-litellm-model-id"]
            deployment_ids.add(creator_deployment_id)
            terminal = _poll_terminal(client, interaction["id"], creator_deployment_id)

            expected_spend = baseline_spend + _expected_cost(terminal)
            first_spend = _wait_for_spend(client, expected_spend)
            baseline_spend = first_spend
            repeated = client.get(f"/v1beta/interactions/{interaction['id']}")
            repeated.raise_for_status()
            _assert_spend_stable(client, first_spend)

            affinity_interaction_id = interaction["id"]
            affinity_deployment_id = creator_deployment_id
            if len(deployment_ids) == 2:
                break

        assert len(deployment_ids) == 2

        previous = client.post(
            "/v1beta/interactions",
            json={
                "model": MODEL,
                "input": "What color was the marble? Reply with the color only.",
                "previous_interaction_id": affinity_interaction_id,
                "background": True,
            },
        )
        previous.raise_for_status()
        previous_interaction_id = previous.json()["id"]
        assert previous_interaction_id.startswith("int_")
        assert previous.headers["x-litellm-model-id"] == affinity_deployment_id
        _poll_terminal(client, previous_interaction_id, affinity_deployment_id)

        stream = client.post(
            "/v1beta/interactions",
            json={"model": MODEL, "input": "Say hello.", "stream": True, "store": True},
        )
        assert stream.status_code == 400
        assert "does not support live streaming" in stream.text
        assert "poll GET /v1beta/interactions/{interaction_id}" in stream.text

        snapshot = client.get(
            f"/v1beta/interactions/{affinity_interaction_id}",
            params={"stream": "true"},
        )
        snapshot.raise_for_status()
        assert snapshot.headers["x-litellm-model-id"] == affinity_deployment_id
        assert tuple(line for line in snapshot.iter_lines() if line.startswith("data:"))

        resume = client.get(
            f"/v1beta/interactions/{affinity_interaction_id}",
            params={"stream": "true", "last_event_id": "unsupported-cursor"},
        )
        assert resume.status_code == 400
        assert "single SSE-formatted snapshot" in resume.text
        assert "does not support last_event_id resume" in resume.text

        cancel = client.post(f"/v1beta/interactions/{affinity_interaction_id}/cancel")
        assert cancel.status_code == 400
        assert "does not support provider-side cancellation" in cancel.text
        assert "does not stop generation or billing" in cancel.text
