import base64
import hashlib
import hmac
import json
from typing import Any, TypedDict

from litellm.types.interactions import InteractionsAPIResponse, InteractionsAPIStreamingResponse


class InteractionId(TypedDict):
    deployment_id: str
    upstream_id: str
    model: str
    user_id: str
    team_id: str


def encode_interaction_id(
    deployment_id: str,
    upstream_id: str,
    *,
    model: str = "",
    user_id: str = "",
    team_id: str = "",
) -> str:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import _get_salt_key

    payload = json.dumps(
        {"v": 2, "d": deployment_id, "u": upstream_id, "m": model, "i": user_id, "t": team_id},
        separators=(",", ":"),
    ).encode()
    key = _get_salt_key()
    if not isinstance(key, str) or not key:
        raise ValueError("LITELLM_SALT_KEY or master_key is required to issue interaction IDs")
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    signature = base64.urlsafe_b64encode(hmac.new(key.encode(), payload, hashlib.sha256).digest()).decode().rstrip("=")
    return f"int_{encoded}.{signature}"


def decode_interaction_id(interaction_id: str | None) -> InteractionId | None:
    if not interaction_id or not interaction_id.startswith("int_"):
        return None
    from litellm.proxy.common_utils.encrypt_decrypt_utils import _get_salt_key

    try:
        encoded, signature = interaction_id[4:].split(".", 1)
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        key = _get_salt_key()
        if not isinstance(key, str) or not key:
            return None
        expected = base64.urlsafe_b64encode(hmac.new(key.encode(), raw, hashlib.sha256).digest()).decode().rstrip("=")
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(raw)
        if (
            payload.get("v") != 2
            or not isinstance(payload.get("d"), str)
            or not isinstance(payload.get("u"), str)
            or not isinstance(payload.get("m", ""), str)
            or not isinstance(payload.get("i", ""), str)
            or not isinstance(payload.get("t", ""), str)
        ):
            return None
        return {
            "deployment_id": payload["d"],
            "upstream_id": payload["u"],
            "model": payload.get("m", ""),
            "user_id": payload.get("i", ""),
            "team_id": payload.get("t", ""),
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def interaction_id_context(metadata: dict[str, Any] | None) -> dict[str, str]:
    metadata = metadata or {}
    model_info = metadata.get("model_info") or {}
    return {
        "deployment_id": str(model_info.get("id") or ""),
        "model": str(metadata.get("model_group") or ""),
        "user_id": str(metadata.get("user_api_key_user_id") or ""),
        "team_id": str(metadata.get("user_api_key_team_id") or ""),
    }


def wrap_interaction_response_id(
    response: InteractionsAPIResponse | InteractionsAPIStreamingResponse,
    deployment_id: str | None,
    *,
    model: str = "",
    user_id: str = "",
    team_id: str = "",
) -> InteractionsAPIResponse | InteractionsAPIStreamingResponse:
    if not deployment_id:
        return response

    def _wrap(value: str) -> str:
        return encode_interaction_id(
            deployment_id,
            value,
            model=model,
            user_id=user_id,
            team_id=team_id,
        )

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
