"""Tests for app/llm/router.py"""

import pytest

from app.config import Settings
from app.llm.claude import ClaudeBackend
from app.llm.ollama import OllamaBackend
from app.llm.openai_compat import OpenAICompatBackend
from app.llm.router import LLMRouter


def _cfg(**overrides) -> Settings:
    base = {
        "anthropic_api_key": "dummy",
        "gemini_api_key": "dummy",
        "groq_api_key": "dummy",
    }
    base.update(overrides)
    return Settings(**base)


def test_online_mode_uses_claude_opus():
    router = LLMRouter(_cfg(llm_mode="online"))
    assert isinstance(router.notes_backend, ClaudeBackend)
    assert isinstance(router.synthesis_backend, ClaudeBackend)


def test_hybrid_mode_haiku_notes_opus_synthesis():
    router = LLMRouter(_cfg(llm_mode="hybrid", hybrid_notes_backend="haiku"))
    assert isinstance(router.notes_backend, ClaudeBackend)
    assert isinstance(router.synthesis_backend, ClaudeBackend)
    # notes model is haiku, synthesis is opus
    assert router.notes_backend.model == _cfg().claude_haiku_model
    assert router.synthesis_backend.model == _cfg().claude_opus_model


def test_hybrid_mode_ollama_notes():
    router = LLMRouter(_cfg(llm_mode="hybrid", hybrid_notes_backend="ollama"))
    assert isinstance(router.notes_backend, OllamaBackend)
    assert isinstance(router.synthesis_backend, ClaudeBackend)


def test_offline_mode_uses_ollama():
    router = LLMRouter(_cfg(llm_mode="offline"))
    assert isinstance(router.notes_backend, OllamaBackend)
    assert isinstance(router.synthesis_backend, OllamaBackend)


def test_free_gemini_mode():
    router = LLMRouter(_cfg(llm_mode="free-gemini"))
    assert isinstance(router.notes_backend, OpenAICompatBackend)
    assert isinstance(router.synthesis_backend, OpenAICompatBackend)
    assert router.synthesis_context_tokens == _cfg().gemini_context_tokens


def test_free_groq_mode():
    router = LLMRouter(_cfg(llm_mode="free-groq"))
    assert isinstance(router.notes_backend, OpenAICompatBackend)
    assert isinstance(router.synthesis_backend, OpenAICompatBackend)


def test_free_hybrid_mode():
    router = LLMRouter(_cfg(llm_mode="free-hybrid"))
    assert isinstance(router.notes_backend, OpenAICompatBackend)
    assert isinstance(router.synthesis_backend, OpenAICompatBackend)
    # synthesis uses gemini context
    assert router.synthesis_context_tokens == _cfg().gemini_context_tokens
    # notes uses groq context
    assert router.notes_context_tokens < _cfg().gemini_context_tokens


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown LLM_MODE"):
        LLMRouter(_cfg(llm_mode="magic-unicorn"))


def test_context_tokens_reduced_by_reserve_for_groq():
    cfg = _cfg(llm_mode="free-groq", chunk_reserve_tokens=1000)
    router = LLMRouter(cfg)
    assert router.notes_context_tokens == cfg.groq_context_tokens - 1000


def test_context_tokens_reduced_by_reserve_for_ollama():
    cfg = _cfg(llm_mode="offline", chunk_reserve_tokens=500)
    router = LLMRouter(cfg)
    assert router.notes_context_tokens == cfg.ollama_context_tokens - 500


def test_mode_stored_on_router():
    router = LLMRouter(_cfg(llm_mode="offline"))
    assert router.mode == "offline"


def test_mode_case_insensitive():
    router = LLMRouter(_cfg(llm_mode="FREE-GEMINI"))
    assert router.mode == "free-gemini"
