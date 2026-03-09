from .openai_compat import OpenAICompatBackend


class OllamaBackend(OpenAICompatBackend):
    def __init__(self, model: str, base_url: str = "http://ollama:11434") -> None:
        super().__init__(model=model, base_url=f"{base_url}/v1", api_key="ollama")
