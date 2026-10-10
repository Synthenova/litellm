"""Offline coverage for the Infiknit Vertex Omni behavior carried on top of upstream.

Live coverage lives in tests/integration_tests/test_vertex_interactions.py.
"""

from unittest.mock import AsyncMock, patch

import pytest

from litellm import Router
from litellm.cost_calculator import response_cost_calculator
from litellm.interactions.id_utils import (
    decode_interaction_id,
    encode_interaction_id,
    wrap_interaction_response_id,
)
from litellm.llms.vertex_ai.interactions.transformation import VertexAIInteractionsConfig
from litellm.types.interactions import InteractionsAPIResponse
from litellm.types.router import GenericLiteLLMParams

USAGE = {
    "total_tokens": 175,
    "total_input_tokens": 100,
    "input_tokens_by_modality": [{"modality": "text", "tokens": 100}],
    "total_cached_tokens": 0,
    "total_output_tokens": 50,
    "output_tokens_by_modality": [{"modality": "video", "tokens": 50}],
    "total_tool_use_tokens": 0,
    "total_thought_tokens": 25,
}


@pytest.fixture(autouse=True)
def salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-infiknit-test-salt-key-0123456789")


def _completed(interaction_id: str = "v1_upstream") -> InteractionsAPIResponse:
    return InteractionsAPIResponse(
        id=interaction_id, model="gemini-omni-flash-preview", status="completed", steps=[], usage=dict(USAGE)
    )


def test_interaction_id_round_trips_deployment_and_upstream_id():
    encoded = encode_interaction_id("vertex-gemini-omni-account-b", "v1_abc")
    assert encoded.startswith("int_")
    assert "v1_abc" not in encoded
    assert decode_interaction_id(encoded) == {
        "deployment_id": "vertex-gemini-omni-account-b",
        "upstream_id": "v1_abc",
    }


@pytest.mark.parametrize("value", [None, "", "v1_raw_vertex_id", "int_not-encrypted", "int_"])
def test_decode_rejects_raw_and_tampered_ids(value):
    assert decode_interaction_id(value) is None


def test_decode_rejects_a_flipped_ciphertext():
    encoded = encode_interaction_id("vertex-gemini-omni-account-a", "v1_abc")
    tampered = encoded[:-3] + ("A" if encoded[-3] != "A" else "B") + encoded[-2:]
    assert decode_interaction_id(tampered) is None


def test_wrap_is_idempotent_and_skips_unrouted_responses():
    response = _completed()
    wrap_interaction_response_id(response, deployment_id=None)
    assert response.id == "v1_upstream"

    wrap_interaction_response_id(response, deployment_id="vertex-gemini-omni-account-a")
    first = response.id
    wrap_interaction_response_id(response, deployment_id="vertex-gemini-omni-account-a")
    assert response.id == first
    assert decode_interaction_id(first)["upstream_id"] == "v1_upstream"


@pytest.mark.asyncio
async def test_router_routes_a_poll_to_the_creating_deployment():
    router = Router(model_list=[{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}])
    encoded = encode_interaction_id("vertex-gemini-omni-account-b", "v1_abc")
    with patch.object(router, "_ageneric_api_call_with_fallbacks", new=AsyncMock(return_value="ok")) as routed:
        result = await router._init_interactions_api_endpoints(
            original_function=AsyncMock(), interaction_id=encoded, custom_llm_provider="vertex_ai"
        )
    assert result == "ok"
    kwargs = routed.call_args.kwargs
    assert kwargs["model"] == "vertex-gemini-omni-account-b"
    assert kwargs["interaction_id"] == "v1_abc"
    assert kwargs["custom_llm_provider"] == "vertex_ai"


@pytest.mark.asyncio
async def test_router_rejects_a_tampered_interaction_id():
    router = Router(model_list=[{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}])
    with pytest.raises(ValueError, match="tampered"):
        await router._init_interactions_api_endpoints(original_function=AsyncMock(), interaction_id="int_garbage")


def test_vertex_interaction_is_billed_only_when_settled():
    unsettled = _completed()
    assert response_cost_calculator(
        response_object=unsettled,
        model="vertex_ai/gemini-omni-flash-preview",
        custom_llm_provider="vertex_ai",
        call_type="acreate_interaction",
        optional_params={},
        cache_hit=False,
        base_model=None,
        custom_pricing=False,
        prompt=None,
    ) == 0.0

    settled = _completed()
    settled._hidden_params["settle_interaction_cost"] = True
    cost = response_cost_calculator(
        response_object=settled,
        model="vertex_ai/gemini-omni-flash-preview",
        custom_llm_provider="vertex_ai",
        call_type="acreate_interaction",
        optional_params={},
        cache_hit=False,
        base_model=None,
        custom_pricing=False,
        prompt=None,
    )
    # 100 input * 1.5e-6 + 25 thought * 9e-6 + 50 video * 1.75e-5
    assert cost == pytest.approx(100 * 1.5e-6 + 25 * 9e-6 + 50 * 1.75e-5)


def test_non_vertex_interactions_keep_upstream_costing():
    response = _completed()
    cost = response_cost_calculator(
        response_object=response,
        model="gemini/gemini-2.5-flash",
        custom_llm_provider="gemini",
        call_type="acreate_interaction",
        optional_params={},
        cache_hit=False,
        base_model=None,
        custom_pricing=False,
        prompt=None,
    )
    assert cost is None or cost > 0


@pytest.mark.asyncio
async def test_vertex_create_skips_upstream_background_polling():
    from litellm.interactions import main as interactions_main

    pending = InteractionsAPIResponse(id="v1_pending", model="gemini-omni-flash-preview", status="in_progress")
    with (
        patch.object(interactions_main, "create", return_value=pending),
        patch.object(interactions_main, "maybe_schedule_background_interaction_cost_polling") as scheduled,
    ):
        response = await interactions_main.acreate(
            model="vertex_ai/gemini-omni-flash-preview", input="hi", background=True, custom_llm_provider="vertex_ai"
        )
    scheduled.assert_not_called()
    assert response.id == "v1_pending"  # no deployment metadata, so the raw id is kept


def test_vertex_requests_select_the_steps_schema():
    config = VertexAIInteractionsConfig(mint_access_token=lambda credentials, project: ("token", "project-a"))
    headers = config.validate_environment({}, "gemini-omni-flash-preview", GenericLiteLLMParams())
    assert headers["Api-Revision"] == "2026-05-20"
    assert headers["Authorization"] == "Bearer token"
