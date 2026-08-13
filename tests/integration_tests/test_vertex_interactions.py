import base64
import json
import os
import time

import httpx
import pytest


BASE_URL = os.getenv("LITELLM_VERTEX_INTERACTIONS_BASE_URL")
VIRTUAL_KEY = os.getenv("LITELLM_VERTEX_INTERACTIONS_VIRTUAL_KEY")
MODEL = os.getenv("LITELLM_VERTEX_INTERACTIONS_MODEL", "vertex-omni")
GCS_OUTPUT = os.getenv("LITELLM_VERTEX_INTERACTIONS_GCS_OUTPUT", "gs://cloud-samples-data/")
PNG = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
).decode()

pytestmark = pytest.mark.skipif(
    not BASE_URL or not VIRTUAL_KEY,
    reason="Set LITELLM_VERTEX_INTERACTIONS_BASE_URL and LITELLM_VERTEX_INTERACTIONS_VIRTUAL_KEY",
)


def test_real_vertex_interactions_create_retrieve_resume_and_affinity() -> None:
    headers = {"Authorization": f"Bearer {VIRTUAL_KEY}"}
    deployment_ids: set[str] = set()
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=900) as client:
        # Documented native content/output shapes must reach Vertex unchanged. A
        # provider 4xx is acceptable preview-availability evidence; proxy 4xx is not.
        shape_payloads = (
            {
                "model": MODEL,
                "input": [{"type": "image", "data": PNG, "mime_type": "image/png"}, {"type": "text", "text": "Animate this."}],
                "response_format": {"type": "video", "delivery": "inline"},
                "generation_config": {"video_config": {"task": "image_to_video"}},
            },
            {
                "model": MODEL,
                "input": [
                    {"type": "image", "uri": "gs://cloud-samples-data/generative-ai/image/scones.jpg", "mime_type": "image/jpeg"},
                    {"type": "image", "uri": "gs://cloud-samples-data/generative-ai/image/character.jpg", "mime_type": "image/jpeg"},
                    {"type": "text", "text": "Use both reference images."},
                ],
                "response_format": {"type": "video", "delivery": "uri", "gcs_uri": GCS_OUTPUT},
                "generation_config": {"video_config": {"task": "reference_to_video"}},
            },
            {
                "model": MODEL,
                "input": [{"type": "video", "data": base64.b64encode(b"boundary-video").decode(), "mime_type": "video/mp4"}, {"type": "text", "text": "Edit this video."}],
                "response_format": {"type": "video", "delivery": "uri", "gcs_uri": GCS_OUTPUT},
                "generation_config": {"video_config": {"task": "video_edit"}},
            },
        )
        for payload in shape_payloads:
            shaped = client.post("/v1beta/interactions", json=payload)
            assert shaped.status_code < 500
            if shaped.status_code >= 400:
                assert "vertex" in shaped.text.lower() or "google" in shaped.text.lower()

        baseline_info = client.get("/key/info", params={"key": VIRTUAL_KEY})
        baseline_info.raise_for_status()
        baseline_spend = float(baseline_info.json()["info"]["spend"])
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
            deployment_ids.add(create.headers["x-litellm-model-id"])

            for _ in range(90):
                retrieve = client.get(f"/v1beta/interactions/{interaction['id']}")
                retrieve.raise_for_status()
                assert retrieve.json()["id"] == interaction["id"]
                assert retrieve.headers["x-litellm-model-id"] == create.headers["x-litellm-model-id"]
                if retrieve.json().get("status") == "completed":
                    terminal = retrieve.json()
                    break
                time.sleep(5)
            else:
                pytest.fail("background interaction did not complete")

            key_info = client.get("/key/info", params={"key": VIRTUAL_KEY})
            key_info.raise_for_status()
            first_spend = float(key_info.json()["info"]["spend"])
            usage = terminal["usage"]
            modalities = usage.get("output_tokens_by_modality") or []
            expected_cost = (
                (usage.get("total_input_tokens") or 0) * 0.0000015
                + ((usage.get("text_output_tokens") or sum((m.get("tokens") or 0) for m in modalities if str(m.get("modality", "")).lower() == "text")) + (usage.get("total_reasoning_tokens") or usage.get("thought_tokens") or 0)) * 0.000009
                + (usage.get("video_output_tokens") or sum((m.get("tokens") or 0) for m in modalities if str(m.get("modality", "")).lower() == "video")) * 0.0000175
            )
            assert first_spend - baseline_spend == pytest.approx(expected_cost, abs=1e-12)
            baseline_spend = first_spend
            repeated = client.get(f"/v1beta/interactions/{interaction['id']}")
            repeated.raise_for_status()
            repeated_key_info = client.get("/key/info", params={"key": VIRTUAL_KEY})
            repeated_key_info.raise_for_status()
            assert float(repeated_key_info.json()["info"]["spend"]) == first_spend

            previous = client.post(
                "/v1beta/interactions",
                json={
                    "model": MODEL,
                    "input": "Make the marble red.",
                    "previous_interaction_id": interaction["id"],
                    "background": True,
                },
            )
            previous.raise_for_status()
            assert previous.headers["x-litellm-model-id"] == create.headers["x-litellm-model-id"]
            if len(deployment_ids) == 2:
                break

        assert len(deployment_ids) == 2

        with client.stream(
            "POST",
            "/v1beta/interactions",
            json={"model": MODEL, "input": "Say hello.", "stream": True},
        ) as stream:
            stream.raise_for_status()
            events = tuple(line for line in stream.iter_lines() if line.startswith("data:"))
        assert events

        parsed_events = [json.loads(event[5:].strip()) for event in events]
        event_id = next((event["event_id"] for event in parsed_events if event.get("event_id")), None)
        if event_id:
            with client.stream(
                "GET",
                f"/v1beta/interactions/{interaction['id']}",
                params={"stream": "true", "last_event_id": event_id},
            ) as resumed:
                resumed.raise_for_status()
                assert resumed.headers["x-litellm-model-id"] == create.headers["x-litellm-model-id"]
                resumed_events = tuple(line for line in resumed.iter_lines() if line.startswith("data:"))
            assert resumed_events
        else:
            assert all("event_id" not in event for event in parsed_events)
