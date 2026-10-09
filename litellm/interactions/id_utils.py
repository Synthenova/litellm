import json
from typing import Any, TypedDict

from litellm.types.interactions import InteractionsAPIResponse, InteractionsAPIStreamingResponse


class InteractionId(TypedDict):
    deployment_id: str
    upstream_id: str


def encode_interaction_id(
    deployment_id: str,
    upstream_id: str,
) -> str:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    payload = json.dumps({"v": 3, "d": deployment_id, "u": upstream_id}, separators=(",", ":"))
    encrypted = encrypt_value_helper(payload)
    if not isinstance(encrypted, str) or not encrypted:
        raise ValueError("LITELLM_SALT_KEY or master_key is required to issue interaction IDs")
    return f"int_{encrypted}"


def decode_interaction_id(interaction_id: str | None) -> InteractionId | None:
    if not interaction_id or not interaction_id.startswith("int_"):
        return None
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper

    try:
        raw = decrypt_value_helper(interaction_id[4:], key="interaction_id")
        if not isinstance(raw, str):
            return None
        payload = json.loads(raw)
        if (
            payload.get("v") != 3
            or not isinstance(payload.get("d"), str)
            or not isinstance(payload.get("u"), str)
        ):
            return None
        return {
            "deployment_id": payload["d"],
            "upstream_id": payload["u"],
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def interaction_id_context(metadata: dict[str, Any] | None) -> dict[str, str]:
    metadata = metadata or {}
    model_info = metadata.get("model_info") or {}
    return {
        "deployment_id": str(model_info.get("id") or ""),
    }


def wrap_interaction_response_id(
    response: InteractionsAPIResponse | InteractionsAPIStreamingResponse,
    deployment_id: str | None,
) -> InteractionsAPIResponse | InteractionsAPIStreamingResponse:
    if not deployment_id:
        return response

    def _wrap(value: str) -> str:
        return encode_interaction_id(deployment_id, value)

    response_id = getattr(response, "id", None)
    if response_id and decode_interaction_id(response_id) is None:
        response.id = _wrap(response_id)
    interaction_id = getattr(response, "interaction_id", None)
    if interaction_id and decode_interaction_id(interaction_id) is None:
        response.interaction_id = _wrap(interaction_id)
    interaction = getattr(response, "interaction", None)
    for key in ("id", "interaction_id", "previous_interaction_id"):
        if interaction and isinstance(interaction.get(key), str) and decode_interaction_id(interaction[key]) is None:
            interaction[key] = _wrap(interaction[key])
    return response
