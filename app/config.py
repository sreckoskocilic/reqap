from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM mode ──────────────────────────────────────────────────────────
    # online | hybrid | offline | free-gemini | free-groq | free-hybrid
    llm_mode: str = "free-gemini"

    # ── Claude (paid) ─────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    claude_opus_model: str = "claude-opus-4-6"
    claude_haiku_model: str = "claude-haiku-4-5"
    claude_max_tokens: int = 4096
    claude_max_input_tokens: int = 150_000

    # In hybrid mode, which backend handles notes extraction
    hybrid_notes_backend: str = "haiku"  # haiku | ollama

    # ── Gemini (free tier) ────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    # 1M context; we cap conservatively to leave room for prompts + output
    gemini_context_tokens: int = 900_000

    # ── Groq (free tier) ──────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_notes_model: str = "llama-3.1-8b-instant"  # fastest, free
    groq_synthesis_model: str = "llama-3.3-70b-versatile"  # best quality on Groq free
    groq_context_tokens: int = 128_000

    # ── Ollama (local / offline) ──────────────────────────────────────────
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2"
    ollama_context_tokens: int = 8192

    # ── Shared ────────────────────────────────────────────────────────────
    chunk_reserve_tokens: int = 2048  # reserved per chunk for prompt + response
    upload_max_mb: int = 50

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
