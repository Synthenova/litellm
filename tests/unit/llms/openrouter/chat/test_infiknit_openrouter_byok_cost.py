"""Infiknit: OpenRouter BYOK billing.

With BYOK, ``usage.cost`` is only OpenRouter's fee and the provider charge is in
``cost_details.upstream_inference_cost``. A missing or invalid charge is unknown, so the
deployment's configured pricing must apply instead of a provider-stated 0.
"""

import math
from unittest.mock import MagicMock, Mock, patch

import pytest

from litellm import Router
from litellm.cost_calculator import response_cost_calculator
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openrouter.chat.transformation import (
    OpenRouterChatCompletionStreamingHandler,
    OpenrouterConfig,
)
from litellm.llms.openrouter.common_utils import openrouter_billed_cost
from litellm.types.utils import Choices, Message, ModelResponse, Usage

COST_HEADER = "llm_provider-x-litellm-response-cost"
GLM = "openrouter/z-ai/glm-5.3-flash:nitro"
GLM_FALLBACK = 1000 * 1.5e-7 + 1000 * 5e-7  # 0.00065 at 1k in / 1k out


def _byok(upstream: object, *, details: bool = True, fee: object = 0) -> dict:
    usage: dict = {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000, "cost": fee, "is_byok": True}
    if details:
        usage["cost_details"] = {"upstream_inference_cost": upstream}
    return usage


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, None),
        ({}, None),
        ({"cost": 0.0000038, "is_byok": False}, 0.0000038),
        ({"cost": 0, "is_byok": False}, 0.0),
        ({"cost": 0.0000038}, 0.0000038),
        (_byok(0.0000036), 0.0000036),
        (_byok(0.00002, fee=0.000001), 0.000021),
        (_byok(0), 0.0),  # explicit zero provider charge is authoritative
        (_byok(None), None),
        (_byok(True), None),
        (_byok("0.1"), None),
        (_byok(-0.01), None),
        (_byok(math.nan), None),
        (_byok(math.inf), None),
        (_byok(None, details=False), None),
        ({**_byok(0.1), "cost_details": None}, None),
        (_byok(0.1, fee=-1), None),
        (_byok(0.1, fee=math.nan), None),
        ({"cost": True, "is_byok": False}, None),
        ({"cost": -0.5, "is_byok": False}, None),
        ({"cost": math.inf}, None),
    ],
)
def test_openrouter_billed_cost(usage, expected):
    result = openrouter_billed_cost(usage)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def _transform(usage: dict) -> ModelResponse:
    raw = Mock()
    raw.json.return_value = {"id": "gen-1", "model": GLM, "choices": [], "usage": usage}
    raw.headers = {}
    model_response = ModelResponse(
        id="gen-1",
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="pong", role="assistant"))],
        model=GLM,
        usage=Usage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000, cost=usage.get("cost")),
    )
    with patch.object(OpenAIGPTConfig, "transform_response", return_value=model_response):
        return OpenrouterConfig().transform_response(
            model=GLM,
            raw_response=raw,
            model_response=model_response,
            logging_obj=Mock(),
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )


def test_non_stream_byok_bills_fee_plus_upstream():
    result = _transform(_byok(0.0000036))
    assert result._hidden_params["additional_headers"][COST_HEADER] == pytest.approx(0.0000036)
    assert result.usage.cost == pytest.approx(0.0000036)


@pytest.mark.parametrize("usage", [_byok(None), _byok(None, details=False), _byok(-1), _byok(math.nan)])
def test_non_stream_unknown_byok_charge_is_not_billed_as_zero(usage):
    result = _transform(usage)
    assert COST_HEADER not in result._hidden_params.get("additional_headers", {})
    assert getattr(result.usage, "cost", None) is None


def _stream_usage(usage: dict) -> Usage | None:
    handler = OpenRouterChatCompletionStreamingHandler(streaming_response=iter(()), sync_stream=True)
    return handler.chunk_parser({"id": "gen-1", "created": 1, "model": GLM, "choices": [], "usage": usage}).usage


def test_stream_chunk_carries_billed_cost():
    assert _stream_usage(_byok(0.0000036)).cost == pytest.approx(0.0000036)


@pytest.mark.parametrize("usage", [_byok(None), _byok(True), _byok(-0.01), _byok(math.inf), _byok(None, details=False)])
def test_stream_chunk_drops_unknown_charge(usage):
    assert getattr(_stream_usage(usage), "cost", None) is None


def test_stream_chunk_without_cost_is_unchanged():
    usage = _stream_usage({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    assert getattr(usage, "cost", None) is None
    assert usage.total_tokens == 2


@pytest.fixture
def glm_deployment_id():
    router = Router(
        model_list=[
            {
                "model_name": "infiknit/openrouter/glm-5.3-flash-nitro",
                "litellm_params": {"model": GLM, "api_key": "sk-test"},
                "model_info": {"id": "byok-glm-test", "input_cost_per_token": 1.5e-7, "output_cost_per_token": 5e-7},
            }
        ]
    )
    return router.model_list[0]["model_info"]["id"]


def _bill(response: ModelResponse, deployment_id: str) -> float | None:
    logging_obj = MagicMock()
    logging_obj.litellm_params = {}
    return response_cost_calculator(
        response_object=response,
        model=GLM,
        custom_llm_provider="openrouter",
        call_type="acompletion",
        optional_params={},
        cache_hit=False,
        base_model=None,
        custom_pricing=True,
        prompt=None,
        router_model_id=deployment_id,
        litellm_model_name=GLM,
        litellm_logging_obj=logging_obj,
    )


def test_unknown_byok_charge_falls_back_to_configured_pricing(glm_deployment_id):
    cost = _bill(_transform(_byok(None)), glm_deployment_id)
    assert cost == pytest.approx(GLM_FALLBACK)


def test_known_byok_charge_overrides_configured_pricing(glm_deployment_id):
    cost = _bill(_transform(_byok(0.0000171)), glm_deployment_id)
    assert cost == pytest.approx(0.0000171)
