from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Protocol, runtime_checkable


@dataclass
class LLMEvent:
    type: Literal["text_delta", "thinking_delta", "usage"]
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


@runtime_checkable
class LLMBackend(Protocol):
    async def stream_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        use_cache: bool = False,
        capture_thinking: bool = False,
    ) -> AsyncIterator[LLMEvent]:
        ...

    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
    ) -> str:
        """Non-streaming completion, returns full text."""
        ...
