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
    await ManagedObjectRepository(prisma_client).table.upsert(
        where={"model_object_id": _interaction_row_id(upstream_id, decoded_id["deployment_id"])},
        data={
            "create": {
                "unified_object_id": response.id,
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
                "status": response.status,
                "batch_processed": False,
                "created_by": decoded_id["user_id"] or None,
                "team_id": decoded_id["team_id"] or None,
                "updated_by": decoded_id["user_id"] or None,
            },
            "update": {"status": response.status},
        },
    )


async def settle_terminal_interaction_once(
    response: InteractionsAPIResponse | InteractionsAPIStreamingResponse,
    decoded_id: InteractionId,
) -> bool:
    from litellm.interactions.cost_calculator import interactions_cost
    from litellm.proxy.proxy_server import increment_spend_counters, prisma_client
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    if response.status != "completed" or prisma_client is None:
        return False
    row = await prisma_client.db.litellm_managedobjecttable.find_first(
        where={"model_object_id": _interaction_row_id(decoded_id["upstream_id"], decoded_id["deployment_id"])}
    )
    if row is None:
        return False
    snapshot = json.loads(row.file_object) if isinstance(row.file_object, str) else row.file_object
    request_id = f"interaction:{decoded_id['deployment_id']}:{decoded_id['upstream_id']}"
    cost = interactions_cost(response)
    now = datetime.now(timezone.utc)
    response._hidden_params["settle_interaction_cost"] = True
    payload = get_logging_payload(
        kwargs={
            "litellm_call_id": request_id,
            "call_type": "acreate_interaction",
            "custom_llm_provider": "vertex_ai",
            "model": snapshot.get("model") or response.model or "",
            "response_cost": cost,
            "litellm_params": {
                "metadata": {
                    "user_api_key": snapshot.get("api_key"),
                    "user_api_key_user_id": snapshot.get("user_id"),
                    "user_api_key_team_id": snapshot.get("team_id"),
                    "user_api_key_org_id": snapshot.get("org_id"),
                    "model_group": snapshot.get("model"),
                    "model_info": {"id": decoded_id["deployment_id"]},
                }
            },
        },
        response_obj=response,
        start_time=row.created_at,
        end_time=now,
    )
    payload["spend"] = cost
    async with prisma_client.db.tx(timeout=60) as transaction:
        claimed = await transaction.litellm_managedobjecttable.update_many(
            where={"id": row.id, "batch_processed": False},
            data={"batch_processed": True, "status": "completed", "file_object": response.model_dump_json()},
        )
        if claimed != 1:
            return False
        await transaction.litellm_spendlogs.create(data=prisma_client.jsonify_object(payload))
        if snapshot.get("api_key"):
            await transaction.litellm_verificationtoken.update_many(
                where={"token": snapshot["api_key"]},
                data={"spend": {"increment": cost}, "last_active": now},
            )
        if snapshot.get("user_id"):
            await transaction.litellm_usertable.update_many(
                where={"user_id": snapshot["user_id"]}, data={"spend": {"increment": cost}}
            )
        if snapshot.get("team_id"):
            await transaction.litellm_teamtable.update_many(
                where={"team_id": snapshot["team_id"]}, data={"spend": {"increment": cost}}
            )
        if snapshot.get("user_id") and snapshot.get("team_id"):
            await transaction.litellm_teammembership.update_many(
                where={"user_id": snapshot["user_id"], "team_id": snapshot["team_id"]},
                data={"spend": {"increment": cost}, "total_spend": {"increment": cost}},
            )
        if snapshot.get("org_id"):
            await transaction.litellm_organizationtable.update_many(
                where={"organization_id": snapshot["org_id"]}, data={"spend": {"increment": cost}}
            )
        if snapshot.get("end_user_id"):
            await transaction.litellm_endusertable.update_many(
                where={"user_id": snapshot["end_user_id"]}, data={"spend": {"increment": cost}}
            )
    await increment_spend_counters(
        token=snapshot.get("api_key"),
        team_id=snapshot.get("team_id"),
        user_id=snapshot.get("user_id"),
        response_cost=cost,
        org_id=snapshot.get("org_id"),
        end_user_id=snapshot.get("end_user_id"),
        tags=None,
    )
    return True
