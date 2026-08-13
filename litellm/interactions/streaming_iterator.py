"""
Streaming iterators for the Interactions API.

This module provides streaming iterators that properly stream SSE responses
from the Google Interactions API, similar to the responses API streaming iterator.
"""

import asyncio
import json
from datetime import datetime
from typing import Any

import httpx

from litellm._logging import verbose_logger
from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.interactions.id_utils import interaction_id_context, wrap_interaction_response_id
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.types.interactions import (
    InteractionsAPIStreamingResponse,
)
from litellm.utils import CustomStreamWrapper


class BaseInteractionsAPIStreamingIterator:
    """
    Base class for streaming iterators that process responses from the Interactions API.

    This class contains shared logic for both synchronous and asynchronous iterators.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str | None,
        interactions_api_config: BaseInteractionsAPIConfig,
        logging_obj: LiteLLMLoggingObj,
        litellm_metadata: dict[str, Any] | None = None,
        custom_llm_provider: str | None = None,
        settle_on_terminal: bool = False,
        generation_owned_settlement: bool = False,
        persist_generation_attribution: bool = False,
    ):
        self.response = response
        self.model = model
        self.logging_obj = logging_obj
        self.finished = False
        self.interactions_api_config = interactions_api_config
        self.completed_response: InteractionsAPIStreamingResponse | None = None
        self.start_time = datetime.now()

        # set request kwargs
        self.litellm_metadata = litellm_metadata
        self.custom_llm_provider = custom_llm_provider
        self.settle_on_terminal = settle_on_terminal
        self.generation_owned_settlement = generation_owned_settlement
        self.persist_generation_attribution = persist_generation_attribution

        # set hidden params for response headers
        _api_base = get_api_base(
            model=model or "",
            optional_params=self.logging_obj.model_call_details.get("litellm_params", {}),
        )
        _model_info: dict = litellm_metadata.get("model_info", {}) if litellm_metadata else {}
        self._hidden_params = {
            "model_id": _model_info.get("id", None),
            "api_base": _api_base,
        }
        self._hidden_params["additional_headers"] = process_response_headers(self.response.headers or {})

    def _process_chunk(self, chunk: str) -> InteractionsAPIStreamingResponse | None:
        """Process a single chunk of data from the stream."""
        if not chunk:
            return None

        # Handle SSE format (data: {...})
        stripped_chunk = CustomStreamWrapper._strip_sse_data_from_chunk(chunk)
        if stripped_chunk is None:
            return None

        # Handle "[DONE]" marker
        if stripped_chunk == STREAM_SSE_DONE_STRING:
            self.finished = True
            return None

        try:
            # Parse the JSON chunk
            parsed_chunk = json.loads(stripped_chunk)

            # Format as InteractionsAPIStreamingResponse
            if isinstance(parsed_chunk, dict):
                streaming_response = self.interactions_api_config.transform_streaming_response(
                    model=self.model,
                    parsed_chunk=parsed_chunk,
                    logging_obj=self.logging_obj,
                )
                id_context = interaction_id_context(self.litellm_metadata)
                wrap_interaction_response_id(streaming_response, **id_context)
                streaming_response._hidden_params.update(self._hidden_params)

                # Store the completed response.
                # Legacy schema signals completion via status="completed".
                # New schema (Api-Revision: 2026-05-20) uses event_type="interaction.completed".
                # Remove the legacy check after June 8, 2026.
                if streaming_response and (
                    getattr(streaming_response, "status", None) == "completed"
                    or getattr(streaming_response, "event_type", None)
                    in {"interaction.completed", "interaction.complete"}
                    and isinstance(getattr(streaming_response, "interaction", None), dict)
                    and streaming_response.interaction.get("status") == "completed"
                ):
                    if self.settle_on_terminal and not self.generation_owned_settlement:
                        streaming_response._hidden_params["settle_interaction_cost"] = True
                    self.completed_response = streaming_response
                    if self.settle_on_terminal and not self.generation_owned_settlement:
                        self._handle_logging_completed_response()

                return streaming_response

            return None
        except json.JSONDecodeError:
            # If we can't parse the chunk, continue
            verbose_logger.debug(f"Failed to parse streaming chunk: {stripped_chunk[:200]}...")
            return None

    def _handle_logging_completed_response(self):
        """Base implementation - should be overridden by subclasses."""


class InteractionsAPIStreamingIterator(BaseInteractionsAPIStreamingIterator):
    """
    Async iterator for processing streaming responses from the Interactions API.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str | None,
        interactions_api_config: BaseInteractionsAPIConfig,
        logging_obj: LiteLLMLoggingObj,
        litellm_metadata: dict[str, Any] | None = None,
        custom_llm_provider: str | None = None,
        settle_on_terminal: bool = False,
        generation_owned_settlement: bool = False,
        persist_generation_attribution: bool = False,
    ):
        super().__init__(
            response=response,
            model=model,
            interactions_api_config=interactions_api_config,
            logging_obj=logging_obj,
            litellm_metadata=litellm_metadata,
            custom_llm_provider=custom_llm_provider,
            settle_on_terminal=settle_on_terminal,
            generation_owned_settlement=generation_owned_settlement,
            persist_generation_attribution=persist_generation_attribution,
        )
        self.stream_iterator = response.aiter_lines()

    def __aiter__(self):
        return self

    async def __anext__(self) -> InteractionsAPIStreamingResponse:
        try:
            while True:
                # Get the next chunk from the stream
                try:
                    chunk = await self.stream_iterator.__anext__()
                except StopAsyncIteration:
                    self.finished = True
                    raise StopAsyncIteration

                result = self._process_chunk(chunk)

                if self.finished:
                    raise StopAsyncIteration
                elif result is not None:
                    if self.generation_owned_settlement:
                        from litellm.proxy.interactions.settlement import (
                            observe_interaction_generation,
                        )

                        await observe_interaction_generation(
                            result,
                            self.litellm_metadata if self.persist_generation_attribution else None,
                        )
                    return result
                # If result is None, continue the loop to get the next chunk

        except httpx.HTTPError as e:
            # Handle HTTP errors
            self.finished = True
            raise e

    def _handle_logging_completed_response(self):
        """Handle logging for completed responses in async context."""
        import copy

        logging_response = copy.deepcopy(self.completed_response)

        asyncio.create_task(
            self.logging_obj.dispatch_success_handlers(
                logging_response,
                start_time=self.start_time,
                end_time=datetime.now(),
                cache_hit=None,
                prefer_async_handlers=True,
            )
        )


class SyncInteractionsAPIStreamingIterator(BaseInteractionsAPIStreamingIterator):
    """
    Synchronous iterator for processing streaming responses from the Interactions API.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str | None,
        interactions_api_config: BaseInteractionsAPIConfig,
        logging_obj: LiteLLMLoggingObj,
        litellm_metadata: dict[str, Any] | None = None,
        custom_llm_provider: str | None = None,
        settle_on_terminal: bool = False,
        generation_owned_settlement: bool = False,
        persist_generation_attribution: bool = False,
    ):
        super().__init__(
            response=response,
            model=model,
            interactions_api_config=interactions_api_config,
            logging_obj=logging_obj,
            litellm_metadata=litellm_metadata,
            custom_llm_provider=custom_llm_provider,
            settle_on_terminal=settle_on_terminal,
            generation_owned_settlement=generation_owned_settlement,
            persist_generation_attribution=persist_generation_attribution,
        )
        self.stream_iterator = response.iter_lines()

    def __iter__(self):
        return self

    def __next__(self) -> InteractionsAPIStreamingResponse:
        try:
            while True:
                # Get the next chunk from the stream
                try:
                    chunk = next(self.stream_iterator)
                except StopIteration:
                    self.finished = True
                    raise StopIteration

                result = self._process_chunk(chunk)

                if self.finished:
                    raise StopIteration
                elif result is not None:
                    if self.generation_owned_settlement:
                        from litellm.proxy.interactions.settlement import (
                            observe_interaction_generation,
                        )

                        run_async_function(
                            async_function=observe_interaction_generation,
                            response=result,
                            metadata=(self.litellm_metadata if self.persist_generation_attribution else None),
                        )
                    return result
                # If result is None, continue the loop to get the next chunk

        except httpx.HTTPError as e:
            # Handle HTTP errors
            self.finished = True
            raise e

    def _handle_logging_completed_response(self):
        """Handle logging for completed responses in sync context."""
        import copy

        logging_response = copy.deepcopy(self.completed_response)

        run_async_function(
            async_function=self.logging_obj.async_success_handler,
            result=logging_response,
            start_time=self.start_time,
            end_time=datetime.now(),
            cache_hit=None,
        )

        executor.submit(
            self.logging_obj.success_handler,
            result=logging_response,
            cache_hit=None,
            start_time=self.start_time,
            end_time=datetime.now(),
        )
