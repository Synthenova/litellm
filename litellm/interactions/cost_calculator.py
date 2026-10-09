from typing import Any

from litellm.types.interactions import InteractionsAPIResponse, InteractionsAPIStreamingResponse


def interactions_cost(response: InteractionsAPIResponse | InteractionsAPIStreamingResponse) -> float:
    from litellm.utils import _get_bundled_model_cost_map

    pricing = _get_bundled_model_cost_map()["gemini-omni-flash-preview"]
    interaction = getattr(response, "interaction", None)
    usage = response.usage or (interaction.get("usage", {}) if isinstance(interaction, dict) else {})
    input_tokens = usage.get("total_input_tokens") or 0
    reasoning_tokens = (
        usage.get("total_reasoning_tokens") or usage.get("total_thought_tokens") or usage.get("thought_tokens") or 0
    )
    modalities = usage.get("output_tokens_by_modality") or []
    text_tokens = usage.get("text_output_tokens") or sum(
        item.get("tokens") or 0 for item in modalities if str(item.get("modality", "")).lower() == "text"
    )
    video_tokens = usage.get("video_output_tokens") or sum(
        item.get("tokens") or 0 for item in modalities if str(item.get("modality", "")).lower() == "video"
    )
    if not text_tokens and not modalities:
        text_tokens = usage.get("total_output_tokens") or 0
    video_seconds = 0.0 if video_tokens else _find_video_duration(response.model_dump())
    return (
        input_tokens * pricing["input_cost_per_token"]
        + (text_tokens + reasoning_tokens) * pricing["output_cost_per_token"]
        + video_tokens * pricing["output_cost_per_video_token"]
        + video_seconds * pricing["output_cost_per_video_per_second"]
    )


def _find_video_duration(value: Any) -> float:
    if isinstance(value, dict):
        for key in ("duration_seconds", "durationSeconds"):
            duration = value.get(key)
            if isinstance(duration, (int, float)) and duration > 0:
                return float(duration)
        return max((_find_video_duration(item) for item in value.values()), default=0.0)
    if isinstance(value, list):
        return max((_find_video_duration(item) for item in value), default=0.0)
    return 0.0
