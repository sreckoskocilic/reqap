"""Tests for app/review/engine.py"""

import tempfile
from pathlib import Path
from typing import AsyncIterator

import pytest

from app.llm.base import LLMEvent
from app.review.engine import REPORT_TYPES, run_review


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tmp_txt(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", encoding="utf-8", delete=False
    )
    f.write(content)
    f.close()
    return Path(f.name)


def _small_book_txt() -> Path:
    text = "Chapter 1\n" + ("word " * 200) + "\n\nChapter 2\n" + ("word " * 200)
    return _write_tmp_txt(text)


class MockBackend:
    """LLM backend that returns a fixed response without hitting any API."""

    def __init__(self, response: str = "Mock review text."):
        self.response = response
        self.calls: list[dict] = []

    async def stream_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        use_cache: bool = False,
        capture_thinking: bool = False,
    ) -> AsyncIterator[LLMEvent]:
        self.calls.append({"system": system, "user": user})
        yield LLMEvent(type="text_delta", text=self.response)
        yield LLMEvent(type="usage", input_tokens=10, output_tokens=5)

    async def complete(self, system: str, user: str, *, max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "user": user})
        return self.response


class MockRouter:
    def __init__(self, context_tokens: int = 900_000):
        self.backend = MockBackend()
        self.notes_backend = self.backend
        self.synthesis_backend = self.backend
        self.notes_context_tokens = context_tokens
        self.synthesis_context_tokens = context_tokens


# ---------------------------------------------------------------------------
# run_review — basic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_review_yields_done_event():
    path = _small_book_txt()
    router = MockRouter()
    events = [e async for e in run_review(path, router)]
    types = [e.type for e in events]
    assert "done" in types


@pytest.mark.asyncio
async def test_run_review_yields_text_event():
    path = _small_book_txt()
    router = MockRouter()
    events = [e async for e in run_review(path, router)]
    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) > 0
    combined = "".join(e.data["text"] for e in text_events)
    assert "Mock review text." in combined


@pytest.mark.asyncio
async def test_run_review_done_contains_title_and_author():
    path = _small_book_txt()
    router = MockRouter()
    events = [e async for e in run_review(path, router)]
    done = next(e for e in events if e.type == "done")
    assert "title" in done.data
    assert "author" in done.data
    assert "elapsed_seconds" in done.data


@pytest.mark.asyncio
async def test_run_review_progress_events_emitted():
    path = _small_book_txt()
    router = MockRouter()
    events = [e async for e in run_review(path, router)]
    progress = [e for e in events if e.type == "progress"]
    assert len(progress) >= 2  # at least parsing + synthesis


@pytest.mark.asyncio
async def test_run_review_bad_file_yields_error():
    events = [e async for e in run_review(Path("/nonexistent/file.txt"), MockRouter())]
    assert any(e.type == "error" for e in events)


# ---------------------------------------------------------------------------
# Chapter filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_review_chapter_indices_filters_chapters():
    text = "\n\n".join(f"Chapter {i}\n" + (f"content{i} " * 100) for i in range(5))
    path = _write_tmp_txt(text)
    router = MockRouter()
    backend = MockBackend()
    router.synthesis_backend = backend

    events = [e async for e in run_review(path, router, chapter_indices=[0, 1])]
    assert any(e.type == "done" for e in events)

    # The synthesis prompt should only mention chapters 0 and 1
    call_user = backend.calls[-1]["user"]
    assert "content0" in call_user or "Chapter 0" in call_user


@pytest.mark.asyncio
async def test_run_review_no_chapters_after_filter_yields_error():
    path = _small_book_txt()
    router = MockRouter()
    # Index 999 doesn't exist — results in empty chapter list
    events = [e async for e in run_review(path, router, chapter_indices=[999])]
    assert any(e.type == "error" for e in events)


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_review_uses_report_type_label_in_prompt():
    path = _small_book_txt()
    router = MockRouter()
    backend = MockBackend()
    router.synthesis_backend = backend

    await _collect(run_review(path, router, report_type="characters"))
    call_user = backend.calls[-1]["user"]
    assert "Character Map" in call_user


@pytest.mark.asyncio
async def test_run_review_unknown_report_type_falls_back_to_review():
    path = _small_book_txt()
    router = MockRouter()
    backend = MockBackend()
    router.synthesis_backend = backend

    events = [e async for e in run_review(path, router, report_type="nonexistent")]
    # Should not error — falls back to review type
    assert any(e.type == "done" for e in events)


def test_all_report_types_have_required_fields():
    for key, config in REPORT_TYPES.items():
        assert config.label, f"{key}: missing label"
        assert config.structure, f"{key}: missing structure"
        assert config.notes_hint, f"{key}: missing notes_hint"


# ---------------------------------------------------------------------------
# Multi-pass (small context forces chunking)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_review_multipass_emits_notes_progress():
    # Force multi-pass by setting a very small context window
    text = "Chapter 1\n" + ("word " * 500) + "\n\nChapter 2\n" + ("word " * 500)
    path = _write_tmp_txt(text)

    router = MockRouter(context_tokens=50)  # tiny window → multi-pass
    events = [e async for e in run_review(path, router)]
    stages = [e.data.get("stage") for e in events if e.type == "progress"]
    assert "notes" in stages
    assert "synthesis" in stages


@pytest.mark.asyncio
async def test_run_review_multipass_calls_notes_then_synthesis():
    text = "Chapter 1\n" + ("word " * 500) + "\n\nChapter 2\n" + ("word " * 500)
    path = _write_tmp_txt(text)

    notes_backend = MockBackend("chapter notes")
    synth_backend = MockBackend("final review")

    router = MockRouter(context_tokens=50)
    router.notes_backend = notes_backend
    router.synthesis_backend = synth_backend

    events = [e async for e in run_review(path, router)]
    assert any(e.type == "done" for e in events)
    assert len(notes_backend.calls) >= 1
    assert len(synth_backend.calls) >= 1
    # Synthesis prompt should contain the notes output
    synth_prompt = synth_backend.calls[-1]["user"]
    assert "chapter notes" in synth_prompt


# ---------------------------------------------------------------------------
# Thinking events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_review_thinking_suppressed_by_default():
    path = _small_book_txt()

    class ThinkingBackend(MockBackend):
        async def stream_text(self, system, user, **kwargs):
            yield LLMEvent(type="thinking_delta", text="deep thought")
            yield LLMEvent(type="text_delta", text="answer")

    router = MockRouter()
    router.synthesis_backend = ThinkingBackend()
    events = [e async for e in run_review(path, router, show_thinking=False)]
    assert not any(e.type == "thinking" for e in events)


@pytest.mark.asyncio
async def test_run_review_thinking_emitted_when_enabled():
    path = _small_book_txt()

    class ThinkingBackend(MockBackend):
        async def stream_text(self, system, user, **kwargs):
            yield LLMEvent(type="thinking_delta", text="deep thought")
            yield LLMEvent(type="text_delta", text="answer")

    router = MockRouter()
    router.synthesis_backend = ThinkingBackend()
    events = [e async for e in run_review(path, router, show_thinking=True)]
    thinking = [e for e in events if e.type == "thinking"]
    assert len(thinking) > 0
    assert thinking[0].data["text"] == "deep thought"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _collect(gen):
    return [e async for e in gen]
