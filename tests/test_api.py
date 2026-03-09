"""Tests for app/api/routes.py"""
import io
import json
import tempfile
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.base import LLMEvent
from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_router(response: str = "Mock review."):
    """Return a mock LLMRouter whose backends yield a fixed text response."""

    async def _stream(*args, **kwargs) -> AsyncIterator[LLMEvent]:
        yield LLMEvent(type="text_delta", text=response)
        yield LLMEvent(type="usage", input_tokens=5, output_tokens=3)

    backend = MagicMock()
    backend.stream_text = _stream

    router = MagicMock()
    router.notes_backend = backend
    router.synthesis_backend = backend
    router.notes_context_tokens = 900_000
    router.synthesis_context_tokens = 900_000
    return router


def _make_client(router=None):
    """Return a TestClient with a pre-configured mock router."""
    if router is None:
        router = _make_mock_router()

    # Inject state before the lifespan runs
    with TestClient(app, raise_server_exceptions=True) as client:
        app.state.config = Settings()
        app.state.router = router
        yield client


def _small_epub_bytes() -> bytes:
    """Minimal valid EPUB bytes (enough for the parser to not crash)."""
    # We'll use a .txt upload instead — simpler for test fixtures
    return None


def _txt_upload(content: str = "Chapter 1\nSome content here.\n") -> tuple:
    return ("file.txt", io.BytesIO(content.encode()), "text/plain")


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    with TestClient(app) as client:
        app.state.config = Settings()
        app.state.router = _make_mock_router()
        r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /api/epub/chapters
# ---------------------------------------------------------------------------

def test_chapters_endpoint_returns_chapter_list():
    content = "Chapter 1\nFirst chapter text.\n\nChapter 2\nSecond chapter text.\n"
    with TestClient(app) as client:
        app.state.config = Settings()
        app.state.router = _make_mock_router()
        r = client.post(
            "/api/epub/chapters",
            files={"epub_file": _txt_upload(content)},
        )
    assert r.status_code == 200
    data = r.json()
    assert "chapters" in data
    assert len(data["chapters"]) == 2
    assert data["chapters"][0]["title"] == "Chapter 1"
    assert data["chapters"][1]["title"] == "Chapter 2"


def test_chapters_endpoint_returns_token_counts():
    content = "Chapter 1\n" + ("word " * 100) + "\n"
    with TestClient(app) as client:
        app.state.config = Settings()
        app.state.router = _make_mock_router()
        r = client.post(
            "/api/epub/chapters",
            files={"epub_file": _txt_upload(content)},
        )
    data = r.json()
    assert data["chapters"][0]["tokens"] > 0


def test_chapters_endpoint_returns_title_and_author():
    with TestClient(app) as client:
        app.state.config = Settings()
        app.state.router = _make_mock_router()
        r = client.post(
            "/api/epub/chapters",
            files={"epub_file": _txt_upload("Some content.")},
        )
    data = r.json()
    assert "title" in data
    assert "author" in data


def test_chapters_endpoint_oversized_file_returns_error():
    cfg = Settings()
    cfg = cfg.model_copy(update={"upload_max_mb": 0})  # 0 MB limit

    with TestClient(app) as client:
        app.state.config = cfg
        app.state.router = _make_mock_router()
        r = client.post(
            "/api/epub/chapters",
            files={"epub_file": _txt_upload("x" * 100)},
        )
    data = r.json()
    assert "error" in data


# ---------------------------------------------------------------------------
# /api/reviews/stream — SSE
# ---------------------------------------------------------------------------

def _collect_sse(response) -> list[dict]:
    """Parse SSE stream into list of {type, data} dicts."""
    events = []
    current_type = None
    for line in response.iter_lines():
        if line.startswith("event: "):
            current_type = line[7:].strip()
        elif line.startswith("data: "):
            payload = json.loads(line[6:])
            events.append({"type": current_type, "data": payload})
    return events


def test_stream_review_returns_sse_content_type():
    with TestClient(app) as client:
        app.state.config = Settings()
        app.state.router = _make_mock_router()
        r = client.post(
            "/api/reviews/stream",
            data={"show_thinking": "false"},
            files={"epub_file": _txt_upload()},
        )
    assert "text/event-stream" in r.headers["content-type"]


def test_stream_review_yields_done_event():
    with TestClient(app) as client:
        app.state.config = Settings()
        app.state.router = _make_mock_router()
        r = client.post(
            "/api/reviews/stream",
            data={"show_thinking": "false"},
            files={"epub_file": _txt_upload()},
        )
    events = _collect_sse(r)
    types = [e["type"] for e in events]
    assert "done" in types


def test_stream_review_yields_text_event():
    with TestClient(app) as client:
        app.state.config = Settings()
        app.state.router = _make_mock_router("Hello from mock")
        r = client.post(
            "/api/reviews/stream",
            data={"show_thinking": "false"},
            files={"epub_file": _txt_upload()},
        )
    events = _collect_sse(r)
    text_events = [e for e in events if e["type"] == "text"]
    combined = "".join(e["data"]["text"] for e in text_events)
    assert "Hello from mock" in combined


def test_stream_review_accepts_report_type():
    with TestClient(app) as client:
        app.state.config = Settings()
        app.state.router = _make_mock_router()
        r = client.post(
            "/api/reviews/stream",
            data={"show_thinking": "false", "report_type": "summary"},
            files={"epub_file": _txt_upload()},
        )
    events = _collect_sse(r)
    assert any(e["type"] == "done" for e in events)


def test_stream_review_accepts_chapter_indices():
    content = "Chapter 1\nContent one.\n\nChapter 2\nContent two.\n"
    with TestClient(app) as client:
        app.state.config = Settings()
        app.state.router = _make_mock_router()
        r = client.post(
            "/api/reviews/stream",
            data={"show_thinking": "false", "chapter_indices": "0"},
            files={"epub_file": _txt_upload(content)},
        )
    events = _collect_sse(r)
    assert any(e["type"] == "done" for e in events)


def test_stream_review_oversized_file_yields_error_event():
    cfg = Settings()
    cfg = cfg.model_copy(update={"upload_max_mb": 0})

    with TestClient(app) as client:
        app.state.config = cfg
        app.state.router = _make_mock_router()
        r = client.post(
            "/api/reviews/stream",
            data={"show_thinking": "false"},
            files={"epub_file": _txt_upload("x" * 100)},
        )
    events = _collect_sse(r)
    assert any(e["type"] == "error" for e in events)


def test_stream_review_llm_mode_override():
    """Per-request mode override should build a new router without affecting global."""
    cfg = Settings()
    # Provide dummy keys so the router doesn't raise on init
    cfg = cfg.model_copy(update={
        "llm_mode": "free-gemini",
        "gemini_api_key": "dummy",
        "groq_api_key": "dummy",
    })

    with TestClient(app) as client:
        app.state.config = cfg
        app.state.router = _make_mock_router()

        with patch("app.api.routes.LLMRouter") as MockLLMRouter:
            MockLLMRouter.return_value = _make_mock_router()
            r = client.post(
                "/api/reviews/stream",
                data={"show_thinking": "false", "llm_mode": "free-groq"},
                files={"epub_file": _txt_upload()},
            )
        # A new router was created for the override
        MockLLMRouter.assert_called_once()
