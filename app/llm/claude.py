from typing import AsyncIterator
import anthropic

from .base import LLMEvent

_THINKING_MODELS = ("opus",)  # models that support adaptive thinking


class ClaudeBackend:
    def __init__(self, model: str) -> None:
        self.model = model
        self._client = anthropic.AsyncAnthropic()
        self._supports_thinking = any(k in model for k in _THINKING_MODELS)

    async def stream_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        use_cache: bool = False,
        capture_thinking: bool = False,
    ) -> AsyncIterator[LLMEvent]:
        system_content = [{"type": "text", "text": system}]
        if use_cache:
            system_content[0]["cache_control"] = {"type": "ephemeral"}  # type: ignore[index]

        user_content = [{"type": "text", "text": user}]
        if use_cache:
            user_content[0]["cache_control"] = {"type": "ephemeral"}  # type: ignore[index]

        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system_content,
            messages=[{"role": "user", "content": user_content}],
        )

        if self._supports_thinking and capture_thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield LLMEvent(type="text_delta", text=event.delta.text)
                    elif event.delta.type == "thinking_delta":
                        yield LLMEvent(type="thinking_delta", text=event.delta.thinking)

            final = await stream.get_final_message()
            yield LLMEvent(
                type="usage",
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                cache_read_tokens=getattr(final.usage, "cache_read_input_tokens", 0)
                or 0,
            )

    async def complete(self, system: str, user: str, *, max_tokens: int = 2048) -> str:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")
