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
    status = getattr(response, "status", None) or (interaction.get("status") if isinstance(interaction, dict) else None)
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
                        "model": metadata.get("model_group"),
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
                "created_by": metadata.get("user_api_key_user_id"),
                "team_id": metadata.get("user_api_key_team_id"),
                "updated_by": metadata.get("user_api_key_user_id"),
            },
            "update": {"status": status},
        },
    )


async def get_interaction_attribution(decoded_id: InteractionId) -> dict[str, Any] | None:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return None
    row = await prisma_client.db.litellm_managedobjecttable.find_first(
        where={"model_object_id": _interaction_row_id(decoded_id["upstream_id"], decoded_id["deployment_id"])}
    )
    if row is None:
        return None
    return json.loads(row.file_object) if isinstance(row.file_object, str) else row.file_object


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
    from litellm.proxy.proxy_server import (
        increment_spend_counters,
        prisma_client,
        proxy_logging_obj,
        update_cache,
    )
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

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
    start_time, end_time, response = logging_obj._success_handler_helper_fn(
        result=response,
        start_time=row.created_at,
        end_time=now,
        cache_hit=False,
    )
    payload = get_logging_payload(
        kwargs=logging_obj.model_call_details,
        response_obj=response,
        start_time=start_time,
        end_time=end_time,
    )
    from litellm.interactions.cost_calculator import interactions_cost

    payload["spend"] = interactions_cost(response)
    inserted = await proxy_logging_obj.db_spend_update_writer.insert_spend_log_and_increment_counters_once(
        prisma_client=prisma_client,
        payload=payload,
        request_id=request_id,
    )
    if inserted:
        await increment_spend_counters(
            token=payload.get("api_key") or None,
            team_id=payload.get("team_id") or None,
            user_id=payload.get("user") or None,
            response_cost=payload["spend"],
            org_id=payload.get("organization_id") or None,
            end_user_id=payload.get("end_user") or None,
        )
        await update_cache(
            token=payload.get("api_key") or None,
            user_id=payload.get("user") or None,
            end_user_id=payload.get("end_user") or None,
            team_id=payload.get("team_id") or None,
            response_cost=payload["spend"],
            parent_otel_span=None,
        )
    await prisma_client.db.litellm_managedobjecttable.update_many(
        where={"id": row.id, "batch_processed": False},
        data={"batch_processed": True, "status": "completed"},
    )
    return inserted


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
