from langchain_cohere import ChatCohere

from lwe.core.provider import Provider, PresetValue

COHERE_DEFAULT_MODEL = "command"


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
            'models': {
                'command': {
                    'max_tokens': 4096,
                },
                'command-light': {
                    'max_tokens': 4096,
                },
                'command-r': {
                    "max_tokens": 131072,
                },
                'command-r-plus': {
                    "max_tokens": 131072,
                },
            }
        }

    @property
    def default_model(self):
        return COHERE_DEFAULT_MODEL

    def llm_factory(self):
        return ChatCohere

    def customization_config(self):
        return {
            'model': PresetValue(str, options=self.available_models),
            'temperature': PresetValue(float, min_value=0.0, max_value=5.0),
            'cohere_api_key': PresetValue(str, include_none=True, private=True),
        }
