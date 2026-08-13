from typing import Any

import litellm
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.gemini.interactions.transformation import GoogleAIStudioInteractionsConfig
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class VertexAIInteractionsConfig(GoogleAIStudioInteractionsConfig, VertexBase):
    def __init__(self) -> None:
        VertexBase.__init__(self)

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.VERTEX_AI

    @property
    def api_version(self) -> str:
        return "v1beta1"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:
        params = dict(litellm_params) if litellm_params else {}
        token, _ = self._ensure_access_token(
            credentials=self.safe_get_vertex_ai_credentials(params),
            project_id=self.safe_get_vertex_ai_project(params),
            custom_llm_provider="vertex_ai",
        )
        authenticated = self.set_headers(token, headers)
        authenticated["Api-Revision"] = "2026-05-07" if litellm.use_legacy_interactions_schema else "2026-05-20"
        return authenticated

    def _interactions_url(self, api_base: str | None, litellm_params: dict[str, Any]) -> str:
        project = self.safe_get_vertex_ai_project(litellm_params)
        location = self.safe_get_vertex_ai_location(litellm_params) or "global"
        if not project:
            raise ValueError("vertex_project is required for Vertex AI Interactions")
        base = (api_base or "https://aiplatform.googleapis.com").rstrip("/")
        return f"{base}/{self.api_version}/projects/{project}/locations/{location}/interactions"

    def get_complete_url(
        self,
        api_base: str | None,
        model: str | None,
        agent: str | None = None,
        litellm_params: dict | None = None,
        stream: bool | None = None,
    ) -> str:
        if stream:
            raise litellm.BadRequestError(
                message=(
                    "Vertex AI Gemini Omni Interactions does not support live streaming. "
                    "Use background=true and poll GET /v1beta/interactions/{interaction_id}."
                ),
                model=model,
                llm_provider="vertex_ai",
            )
        url = self._interactions_url(api_base, litellm_params or {})
        return url

    def transform_get_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        encoded = encode_url_path_segment(interaction_id, field_name="interaction_id")
        return f"{self._interactions_url(api_base, dict(litellm_params))}/{encoded}", {}

    def transform_delete_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        raise ValueError("Vertex AI Interactions does not document a delete operation")

    def transform_cancel_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        raise ValueError(
            "Vertex AI Interactions does not document a cancel operation; stopping SSE is client-side only "
            "and the provider generation may continue and incur charges"
        )
