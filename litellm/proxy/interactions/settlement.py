import json
from datetime import datetime, timezone
from typing import Any

from litellm.interactions.id_utils import InteractionId
from litellm.types.interactions import InteractionsAPIResponse, InteractionsAPIStreamingResponse


def _interaction_row_id(upstream_id: str, deployment_id: str) -> str:
    return f"interactions:{deployment_id}:{upstream_id}"


async def record_interaction_generation(
    response: InteractionsAPIResponse | InteractionsAPIStreamingResponse,
    decoded_id: InteractionId,
    metadata: dict[str, Any],
) -> None:
    from litellm.proxy.proxy_server import prisma_client
    from litellm.repositories.table_repositories import ManagedObjectRepository

    if prisma_client is None:
        return
    upstream_id = decoded_id["upstream_id"]
    interaction = getattr(response, "interaction", None)
    status = getattr(response, "status", None) or (
        interaction.get("status") if isinstance(interaction, dict) else None
    )
    unified_id = getattr(response, "id", None) or getattr(response, "interaction_id", None)
    if not unified_id and isinstance(interaction, dict):
        unified_id = interaction.get("id") or interaction.get("interaction_id")
    await ManagedObjectRepository(prisma_client).table.upsert(
        where={"model_object_id": _interaction_row_id(upstream_id, decoded_id["deployment_id"])},
        data={
            "create": {
                "unified_object_id": unified_id or upstream_id,
                "model_object_id": _interaction_row_id(upstream_id, decoded_id["deployment_id"]),
                "file_object": json.dumps(
                    {
                        "deployment_id": decoded_id["deployment_id"],
                        "upstream_id": upstream_id,
                        "model": decoded_id["model"],
                        "api_key": metadata.get("user_api_key"),
                        "user_id": metadata.get("user_api_key_user_id"),
                        "team_id": metadata.get("user_api_key_team_id"),
                        "org_id": metadata.get("user_api_key_org_id"),
                        "end_user_id": metadata.get("user_api_key_end_user_id"),
                    }
                ),
                "file_purpose": "interaction",
                "status": status,
                "batch_processed": False,
                "created_by": decoded_id["user_id"] or None,
                "team_id": decoded_id["team_id"] or None,
                "updated_by": decoded_id["user_id"] or None,
            },
            "update": {"status": status},
        },
    )


def is_terminal_interaction(
    response: InteractionsAPIResponse | InteractionsAPIStreamingResponse,
) -> bool:
    interaction = getattr(response, "interaction", None)
    nested_status = interaction.get("status") if isinstance(interaction, dict) else None
    return getattr(response, "status", None) == "completed" or (
        getattr(response, "event_type", None) in {"interaction.completed", "interaction.complete"}
        and nested_status == "completed"
    )


async def settle_terminal_interaction_once(
    response: InteractionsAPIResponse | InteractionsAPIStreamingResponse,
    decoded_id: InteractionId,
) -> bool:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.proxy.proxy_server import prisma_client

    if not is_terminal_interaction(response) or prisma_client is None:
        return False
    row = await prisma_client.db.litellm_managedobjecttable.find_first(
        where={"model_object_id": _interaction_row_id(decoded_id["upstream_id"], decoded_id["deployment_id"])}
    )
    if row is None:
        return False
    snapshot = json.loads(row.file_object) if isinstance(row.file_object, str) else row.file_object
    request_id = f"interaction:{decoded_id['deployment_id']}:{decoded_id['upstream_id']}"
    now = datetime.now(timezone.utc)
    claimed = await prisma_client.db.litellm_managedobjecttable.update_many(
        where={"id": row.id, "batch_processed": False},
        data={"batch_processed": True, "status": "completed"},
    )
    if claimed != 1:
        return False

    model = snapshot.get("model") or getattr(response, "model", None) or ""
    logging_obj = LiteLLMLogging(
        model=model,
        messages=[{"role": "user", "content": "<interaction>"}],
        stream=False,
        call_type="acreate_interaction",
        start_time=row.created_at,
        litellm_call_id=request_id,
        function_id=request_id,
    )
    logging_obj.update_environment_variables(
        litellm_params={
            "custom_llm_provider": "vertex_ai",
            "metadata": {
                "user_api_key": snapshot.get("api_key"),
                "user_api_key_user_id": snapshot.get("user_id"),
                "user_api_key_team_id": snapshot.get("team_id"),
                "user_api_key_org_id": snapshot.get("org_id"),
                "user_api_key_end_user_id": snapshot.get("end_user_id"),
                "model_group": model,
                "model_info": {"id": decoded_id["deployment_id"]},
            },
        },
        optional_params={},
    )
    response._hidden_params["settle_interaction_cost"] = True
    await logging_obj.async_success_handler(result=response, start_time=row.created_at, end_time=now)
    return True


async def observe_interaction_generation(
    response: InteractionsAPIResponse | InteractionsAPIStreamingResponse,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Persist attribution when first observed and settle the same generation at terminal."""
    from litellm.interactions.id_utils import decode_interaction_id

    interaction = getattr(response, "interaction", None)
    unified_id = getattr(response, "id", None) or getattr(response, "interaction_id", None)
    if not unified_id and isinstance(interaction, dict):
        unified_id = interaction.get("id") or interaction.get("interaction_id")
    decoded = decode_interaction_id(unified_id) if isinstance(unified_id, str) else None
    if decoded is None:
        return False
    if metadata is not None:
        await record_interaction_generation(response, decoded, metadata)
    return await settle_terminal_interaction_once(response, decoded)
