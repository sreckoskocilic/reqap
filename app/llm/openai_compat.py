"""
Generic OpenAI-compatible backend.
Works with Ollama, Groq, Gemini (via their OpenAI-compat endpoint), Together, etc.
"""

from typing import AsyncIterator
from openai import AsyncOpenAI

from .base import LLMEvent


class OpenAICompatBackend:
    def __init__(self, model: str, base_url: str, api_key: str = "none") -> None:
        self.model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def stream_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        use_cache: bool = False,  # not supported by OpenAI-compat APIs
        capture_thinking: bool = False,  # not supported by OpenAI-compat APIs
    ) -> AsyncIterator[LLMEvent]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
            max_tokens=max_tokens,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield LLMEvent(type="text_delta", text=delta)

    async def complete(self, system: str, user: str, *, max_tokens: int = 2048) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=False,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
