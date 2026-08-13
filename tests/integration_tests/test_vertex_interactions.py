import os
import time

import httpx
import pytest


BASE_URL = os.getenv("LITELLM_VERTEX_INTERACTIONS_BASE_URL")
VIRTUAL_KEY = os.getenv("LITELLM_VERTEX_INTERACTIONS_VIRTUAL_KEY")
MODEL = os.getenv("LITELLM_VERTEX_INTERACTIONS_MODEL", "vertex-omni")

pytestmark = pytest.mark.skipif(
    not BASE_URL or not VIRTUAL_KEY,
    reason="Set LITELLM_VERTEX_INTERACTIONS_BASE_URL and LITELLM_VERTEX_INTERACTIONS_VIRTUAL_KEY",
)


def test_real_vertex_interactions_create_retrieve_resume_and_affinity() -> None:
    headers = {"Authorization": f"Bearer {VIRTUAL_KEY}"}
    deployment_ids: set[str] = set()
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=900) as client:
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
                    break
                time.sleep(5)
            else:
                pytest.fail("background interaction did not complete")

            key_info = client.get("/key/info", params={"key": VIRTUAL_KEY})
            key_info.raise_for_status()
            first_spend = float(key_info.json()["info"]["spend"])
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

        event_id = next(
            (event.split('"event_id":"', 1)[1].split('"', 1)[0] for event in events if '"event_id":"' in event),
            None,
        )
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
