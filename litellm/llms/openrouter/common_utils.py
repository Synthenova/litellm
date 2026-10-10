from collections.abc import Mapping

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class OpenRouterException(BaseLLMException):
    pass


def openrouter_billed_cost(usage: Mapping[str, object] | None) -> float | None:
    """Total charge for one OpenRouter call.

    With BYOK, ``usage.cost`` is only OpenRouter's fee (often 0) and the provider
    bills the account's own key for ``cost_details.upstream_inference_cost``.
    """
    if not usage:
        return None
    cost = usage.get("cost")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        return None
    total = float(cost)
    if usage.get("is_byok") is True:
        details = usage.get("cost_details")
        upstream = details.get("upstream_inference_cost") if isinstance(details, Mapping) else None
        if isinstance(upstream, (int, float)) and not isinstance(upstream, bool):
            total += float(upstream)
    return total
