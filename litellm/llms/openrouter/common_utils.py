import math
from collections.abc import Mapping

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class OpenRouterException(BaseLLMException):
    pass


def _charge(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    amount = float(value)
    return amount if math.isfinite(amount) and amount >= 0 else None


def openrouter_billed_cost(usage: Mapping[str, object] | None) -> float | None:
    """Total charge for one OpenRouter call, or None when it is not known.

    With BYOK, ``usage.cost`` is only OpenRouter's fee (often 0) and the provider
    bills the account's own key for ``cost_details.upstream_inference_cost``. A BYOK
    response without a valid upstream charge is unknown, so configured pricing applies.
    """
    if not usage:
        return None
    fee = _charge(usage.get("cost"))
    if fee is None:
        return None
    if usage.get("is_byok") is not True:
        return fee
    details = usage.get("cost_details")
    upstream = _charge(details.get("upstream_inference_cost")) if isinstance(details, Mapping) else None
    return None if upstream is None else fee + upstream
