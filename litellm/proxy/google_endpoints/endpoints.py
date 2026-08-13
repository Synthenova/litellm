from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.types.llms.vertex_ai import TokenCountDetailsResponse

router = APIRouter(
    tags=["google genai endpoints"],
)


@router.post(
    "/v1beta/models/{model_name:path}:generateContent",
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/models/{model_name:path}:generateContent",
    dependencies=[Depends(user_api_key_auth)],
)
async def google_generate_content(
    request: Request,
    model_name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    if "model" not in data:
        data["model"] = model_name

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="agenerate_content",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model_name,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1beta/models/{model_name:path}:streamGenerateContent",
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/models/{model_name:path}:streamGenerateContent",
    dependencies=[Depends(user_api_key_auth)],
)
async def google_stream_generate_content(
    request: Request,
    model_name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    if "model" not in data:
        data["model"] = model_name
    data["stream"] = True
    # google-genai SDK (?alt=sse) must not receive OpenAI's data: [DONE] terminator.
    data["_litellm_skip_openai_stream_done"] = True
    data["_litellm_raw_sse_stream"] = True

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="agenerate_content_stream",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model_name,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1beta/models/{model_name:path}:countTokens",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenCountDetailsResponse,
)
@router.post(
    "/models/{model_name:path}:countTokens",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenCountDetailsResponse,
)
async def google_count_tokens(request: Request, model_name: str):
    """
    ```json
    return {
        "totalTokens": 31,
        "totalBillableCharacters": 96,
        "promptTokensDetails": [
            {
            "modality": "TEXT",
            "tokenCount": 31
            }
        ]
    }
    ```
    """
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
    from litellm.proxy.proxy_server import token_counter as internal_token_counter

    data = await _read_request_body(request=request)
    contents = data.get("contents", [])
    # Create TokenCountRequest for the internal endpoint
    from litellm.proxy._types import TokenCountRequest

    # Translate contents to openai format messages using the adapter
    messages = GoogleGenAIAdapter().translate_generate_content_to_completion(model_name, contents).get("messages", [])

    token_request = TokenCountRequest(
        model=model_name,
        contents=contents,
        messages=messages,  # compatibility when use openai-like endpoint
    )

    # Call the internal token counter function with direct request flag set to False
    token_response = await internal_token_counter(
        request=token_request,
        call_endpoint=True,
    )
    if token_response is not None:
        # cast the response to the well known format
        original_response: dict = token_response.original_response or {}
        if original_response:
            return TokenCountDetailsResponse(
                totalTokens=original_response.get("totalTokens", 0),
                promptTokensDetails=original_response.get("promptTokensDetails", []),
            )
        else:
            return TokenCountDetailsResponse(
                totalTokens=token_response.total_tokens or 0,
                promptTokensDetails=[],
            )

    #########################################################
    # Return the response in the well known format
    #########################################################
    return TokenCountDetailsResponse(
        totalTokens=0,
        promptTokensDetails=[],
    )


# ============================================================
# Google Interactions API Endpoints
# Per OpenAPI spec: https://ai.google.dev/static/api/interactions.openapi.json
# ============================================================


@router.post(
    "/v1beta/interactions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.post(
    "/interactions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def create_interaction(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new interaction using Google's Interactions API.
    
    Per OpenAPI spec: POST /{api_version}/interactions
    
    Supports both model interactions and agent interactions:
    - Model: Provide `model` parameter (e.g., "gemini-2.5-flash")
    - Agent: Provide `agent` parameter (e.g., "deep-research-pro-preview-12-2025")
    
    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1beta/interactions" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gemini/gemini-2.5-flash",
            "input": "Hello, how are you?"
        }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    data["defer_interaction_settlement"] = True
    previous_interaction_id = data.get("previous_interaction_id")
    if isinstance(previous_interaction_id, str) and previous_interaction_id.startswith("int_"):
        from litellm.interactions.id_utils import decode_interaction_id
        from litellm.llms.base_llm.managed_resources.isolation import can_access_resource
        from litellm.proxy.auth.auth_checks import can_key_call_model
        from litellm.proxy.interactions.settlement import get_interaction_attribution
        from litellm.proxy.proxy_server import llm_model_list, llm_router

        decoded = decode_interaction_id(previous_interaction_id)
        if decoded is None:
            raise HTTPException(status_code=400, detail="Invalid or tampered interaction ID")
        attribution = await get_interaction_attribution(decoded)
        if attribution is None:
            raise HTTPException(status_code=404, detail="Interaction attribution not found")
        if not can_access_resource(user_api_key_dict, attribution.get("user_id"), attribution.get("team_id")):
            raise HTTPException(status_code=403, detail="Access denied to interaction")
        if attribution.get("model"):
            data["model"] = attribution["model"]
            await can_key_call_model(
                model=data["model"],
                llm_model_list=llm_model_list,
                valid_token=user_api_key_dict,
                llm_router=llm_router,
            )

    if "custom_llm_provider" not in data:
        model = data.get("model")
        previous_interaction_id = data.get("previous_interaction_id")
        data["custom_llm_provider"] = (
            "vertex_ai"
            if isinstance(model, str)
            and model.startswith("vertex_ai/")
            or isinstance(previous_interaction_id, str)
            and previous_interaction_id.startswith("int_")
            else "gemini"
        )

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        response = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acreate_interaction",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
        if isinstance(response, InteractionsAPIResponse) and response.id.startswith("int_"):
            from litellm.interactions.id_utils import decode_interaction_id
            from litellm.proxy.interactions.settlement import observe_interaction_generation

            decoded = decode_interaction_id(response.id)
            if decoded:
                metadata = processor.data.get("litellm_metadata") or processor.data.get("metadata") or {}
                await observe_interaction_generation(response, metadata)
        return response
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1beta/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.get(
    "/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def get_interaction(
    request: Request,
    interaction_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get an interaction by ID.

    Per OpenAPI spec: GET /{api_version}/interactions/{interaction_id}
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_model_list,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    from litellm.interactions.id_utils import decode_interaction_id
    from litellm.llms.base_llm.managed_resources.isolation import can_access_resource
    from litellm.proxy.auth.auth_checks import can_key_call_model
    from litellm.proxy.interactions.settlement import get_interaction_attribution

    decoded = decode_interaction_id(interaction_id) if interaction_id.startswith("int_") else None
    if interaction_id.startswith("int_") and decoded is None:
        raise HTTPException(status_code=400, detail="Invalid or tampered interaction ID")
    attribution = await get_interaction_attribution(decoded) if decoded else None
    if decoded and attribution is None:
        raise HTTPException(status_code=404, detail="Interaction attribution not found")
    if decoded and not can_access_resource(
        user_api_key_dict, attribution.get("user_id"), attribution.get("team_id")
    ):
        raise HTTPException(status_code=403, detail="Access denied to interaction")
    if attribution and attribution.get("model"):
        await can_key_call_model(
            model=attribution["model"],
            llm_model_list=llm_model_list,
            valid_token=user_api_key_dict,
            llm_router=llm_router,
        )
    data = {
        "interaction_id": interaction_id,
        "model": attribution.get("model") if attribution else None,
        "custom_llm_provider": "vertex_ai" if interaction_id.startswith("int_") else "gemini",
        "stream": request.query_params.get("stream") == "true",
        "last_event_id": request.query_params.get("last_event_id"),
        "defer_interaction_settlement": True,
    }

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        response = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_interaction",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
        if decoded and isinstance(response, InteractionsAPIResponse):
            from litellm.proxy.interactions.settlement import observe_interaction_generation

            await observe_interaction_generation(response)
        return response
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.delete(
    "/v1beta/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.delete(
    "/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def delete_interaction(
    request: Request,
    interaction_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete an interaction by ID.

    Per OpenAPI spec: DELETE /{api_version}/interactions/{interaction_id}
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = {
        "interaction_id": interaction_id,
        "custom_llm_provider": "vertex_ai" if interaction_id.startswith("int_") else "gemini",
    }

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_interaction",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1beta/interactions/{interaction_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.post(
    "/interactions/{interaction_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def cancel_interaction(
    request: Request,
    interaction_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cancel an interaction by ID.

    Per OpenAPI spec: POST /{api_version}/interactions/{interaction_id}:cancel
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = {
        "interaction_id": interaction_id,
        "custom_llm_provider": "vertex_ai" if interaction_id.startswith("int_") else "gemini",
    }

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acancel_interaction",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
