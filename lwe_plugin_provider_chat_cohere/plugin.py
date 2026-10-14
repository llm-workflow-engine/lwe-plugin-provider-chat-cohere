"""LWE chat cohere provider integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_cohere import ChatCohere
from langchain_core.messages import BaseMessage
from typing_extensions import override
import cohere

from lwe.core.provider import PresetValue, Provider

COHERE_DEFAULT_MODEL = "command-r-plus"


class CustomChatCohere(ChatCohere):

    @property
    @override
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chat_cohere"


class ProviderChatCohere(Provider):
    """
    Access to Cohere chat models.
    """

    @property
    @override
    def model_property_name(self) -> str:
        return "model"

    @property
    @override
    def capabilities(self) -> dict[str, bool]:
        return {
            "chat": True,
            "validate_models": True,
        }

    @property
    @override
    def default_model(self) -> str:
        return COHERE_DEFAULT_MODEL

    def fetch_models(self) -> dict[str, dict[str, float | None]]:
        try:
            client = cohere.Client()
            model_data = client.models.list()
            models = {
                model.name: {"max_tokens": model.context_length}
                for model in model_data.models
                if model.name is not None and "chat" in (model.endpoints or [])
            }
            return models
        except Exception as e:
            raise ValueError(f"Could not retrieve models: {e}")

    @override
    def prepare_messages_method(
        self,
    ) -> Callable[[list[dict[str, Any]]], list[BaseMessage]]:
        return self.prepare_messages_for_llm_chat

    @override
    def llm_factory(self) -> type[CustomChatCohere]:
        return CustomChatCohere

    def transform_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        return self.transform_openai_tool_spec_to_json_schema_spec(tool)

    @override
    def customization_config(self) -> dict[str, Any]:
        return {
            "model": PresetValue(str, options=self.available_models),
            "temperature": PresetValue(float, min_value=0.0, max_value=5.0),
            "cohere_api_key": PresetValue(str, include_none=True, private=True),
            "tools": None,
            "tool_choice": None,
        }
