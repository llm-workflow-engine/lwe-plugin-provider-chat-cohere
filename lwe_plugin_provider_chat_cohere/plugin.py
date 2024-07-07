import cohere

from langchain_cohere import ChatCohere

from lwe.core.provider import Provider, PresetValue

COHERE_DEFAULT_MODEL = "command-r-plus"


class CustomChatCohere(ChatCohere):

    @property
    def _llm_type(self):
        """Return type of llm."""
        return "chat_cohere"


class ProviderChatCohere(Provider):
    """
    Access to Cohere chat models.
    """

    @property
    def model_property_name(self):
        return 'model'

    @property
    def capabilities(self):
        return {
            "chat": True,
            'validate_models': True,
        }

    @property
    def default_model(self):
        return COHERE_DEFAULT_MODEL

    def fetch_models(self):
        try:
            client = cohere.Client()
            model_data = client.models.list()
            models = {model.name: {'max_tokens': model.context_length} for model in model_data.models if 'chat' in model.endpoints}
            return models
        except Exception as e:
            raise ValueError(f"Could not retrieve models: {e}")

    def prepare_messages_method(self):
        return self.prepare_messages_for_llm_chat

    def llm_factory(self):
        return CustomChatCohere

    def transform_tool(self, tool):
        return self.transform_openai_tool_spec_to_json_schema_spec(tool)

    def customization_config(self):
        return {
            'model': PresetValue(str, options=self.available_models),
            'temperature': PresetValue(float, min_value=0.0, max_value=5.0),
            'cohere_api_key': PresetValue(str, include_none=True, private=True),
            "tools": None,
            "tool_choice": None,
        }
