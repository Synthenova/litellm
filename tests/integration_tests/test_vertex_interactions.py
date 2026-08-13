import base64
import json
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
}
pytestmark = pytest.mark.skipif(
    any(not value for value in REQUIRED_ENV.values()),
    reason=f"Set {', '.join(name for name, value in REQUIRED_ENV.items() if not value)}",
)

TERMINAL_FAILURES = {"failed", "cancelled", "incomplete", "budget_exceeded"}


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
            pytest.fail(f"interaction {interaction_id} ended as {status}: {body}")
        time.sleep(5)
    pytest.fail(f"interaction {interaction_id} did not complete")


def _video_outputs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        matches = [value] if str(value.get("type", "")).lower() == "video" else []
        return matches + [match for child in value.values() for match in _video_outputs(child)]
    if isinstance(value, list):
        return [match for child in value for match in _video_outputs(child)]
    return []


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
    assert video_tokens or duration, f"completed video interaction has no billable video usage: {usage}"
    return (
        (usage.get("total_input_tokens") or 0) * 0.0000015
        + (text_tokens + (usage.get("total_reasoning_tokens") or usage.get("thought_tokens") or 0)) * 0.000009
        + video_tokens * 0.0000175
        + duration * 0.10
    )


def _key_spend(client: httpx.Client) -> float:
    key_info = client.get("/key/info", params={"key": VIRTUAL_KEY})
    key_info.raise_for_status()
    return float(key_info.json()["info"]["spend"])


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


def _parse_sse_events(lines: tuple[str, ...]) -> list[dict[str, Any]]:
    events = []
    for line in lines:
        payload = line[5:].strip()
        if payload and payload != "[DONE]":
            event = json.loads(payload)
            assert isinstance(event, dict)
            events.append(event)
    return events


def _stream_interaction_id(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        interaction = event.get("interaction")
        candidates = (
            interaction.get("id") if isinstance(interaction, dict) else None,
            event.get("interaction_id"),
            event.get("id"),
        )
        if interaction_id := next(
            (candidate for candidate in candidates if isinstance(candidate, str) and candidate.startswith("int_")),
            None,
        ):
            return interaction_id
    return None


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
                        {"type": "text", "text": "Edit this video."},
                    ],
                    "response_format": {"type": "video", "delivery": "uri", "gcs_uri": GCS_OUTPUT},
                    "generation_config": {"video_config": {"task": "video_edit"}},
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
            outputs = _video_outputs(terminal.get("steps") or terminal.get("outputs") or [])
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
                    "input": "A three-second cinematic video of one blue marble rolling across white paper.",
                    "response_format": {"type": "video", "delivery": "uri", "aspect_ratio": "16:9"},
                    "generation_config": {"video_config": {"task": "text_to_video"}},
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
                "input": "Make the marble red.",
                "previous_interaction_id": affinity_interaction_id,
                "background": True,
            },
        )
        previous.raise_for_status()
        assert previous.json()["id"].startswith("int_")
        assert previous.headers["x-litellm-model-id"] == affinity_deployment_id

        with client.stream(
            "POST",
            "/v1beta/interactions",
            json={"model": MODEL, "input": "Say hello.", "stream": True, "store": True},
        ) as stream:
            stream.raise_for_status()
            stream_deployment_id = stream.headers["x-litellm-model-id"]
            lines = tuple(line for line in stream.iter_lines() if line.startswith("data:"))
        parsed_events = _parse_sse_events(lines)
        assert parsed_events

        streamed_interaction_id = _stream_interaction_id(parsed_events)
        assert streamed_interaction_id, f"stream returned no opaque interaction ID: {parsed_events}"
        event_id = next((event["event_id"] for event in parsed_events if event.get("event_id")), None)
        if not event_id:
            pytest.fail(f"Vertex stream emitted no event_id; resume is unsupported or unverified: {parsed_events}")

        try:
            with client.stream(
                "GET",
                f"/v1beta/interactions/{streamed_interaction_id}",
                params={"stream": "true", "last_event_id": event_id},
            ) as resumed:
                resumed.raise_for_status()
                assert resumed.headers["x-litellm-model-id"] == stream_deployment_id
                resumed_lines = tuple(line for line in resumed.iter_lines() if line.startswith("data:"))
        except httpx.HTTPStatusError as exc:
            pytest.fail(f"Vertex stream resume is unsupported: {exc.response.text}")
        resumed_events = _parse_sse_events(resumed_lines)
        assert resumed_events, "Vertex stream resume returned no events"
