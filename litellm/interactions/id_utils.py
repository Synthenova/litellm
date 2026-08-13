import base64
import json
from typing import TypedDict

from litellm.types.interactions import InteractionsAPIResponse, InteractionsAPIStreamingResponse


class InteractionId(TypedDict):
    deployment_id: str
    upstream_id: str


def encode_interaction_id(deployment_id: str, upstream_id: str) -> str:
    payload = json.dumps({"v": 1, "d": deployment_id, "u": upstream_id}, separators=(",", ":")).encode()
    return "int_" + base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_interaction_id(interaction_id: str | None) -> InteractionId | None:
    if not interaction_id or not interaction_id.startswith("int_"):
        return None
    try:
        encoded = interaction_id[4:]
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if payload.get("v") != 1 or not payload.get("d") or not payload.get("u"):
            return None
        return {"deployment_id": payload["d"], "upstream_id": payload["u"]}
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def wrap_interaction_response_id(
    response: InteractionsAPIResponse | InteractionsAPIStreamingResponse,
    deployment_id: str | None,
) -> InteractionsAPIResponse | InteractionsAPIStreamingResponse:
    if not deployment_id:
        return response
    response_id = getattr(response, "id", None)
    if response_id and decode_interaction_id(response_id) is None:
        response.id = encode_interaction_id(deployment_id, response_id)
    interaction_id = getattr(response, "interaction_id", None)
    if interaction_id and decode_interaction_id(interaction_id) is None:
        response.interaction_id = encode_interaction_id(deployment_id, interaction_id)
    interaction = getattr(response, "interaction", None)
    if interaction and isinstance(interaction.get("id"), str) and decode_interaction_id(interaction["id"]) is None:
        interaction["id"] = encode_interaction_id(deployment_id, interaction["id"])
    if interaction and isinstance(interaction.get("interaction_id"), str):
        if decode_interaction_id(interaction["interaction_id"]) is None:
            interaction["interaction_id"] = encode_interaction_id(deployment_id, interaction["interaction_id"])
    return response
