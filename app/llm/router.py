from .base import LLMBackend
from .claude import ClaudeBackend
from .ollama import OllamaBackend
from .openai_compat import OpenAICompatBackend
from app.config import Settings

# Public Groq endpoint (OpenAI-compatible)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Google Gemini OpenAI-compatible endpoint
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class LLMRouter:
    """
    Maps LLM_MODE → (notes_backend, synthesis_backend).

    Supported modes
    ───────────────
    online          Claude Opus for everything (paid)
    hybrid          Claude Haiku notes + Claude Opus synthesis (paid, cheaper)
    offline         Ollama for everything (free, local)

    free-gemini     Gemini 1.5 Flash for everything
                    1M context → single-pass for almost all books
                    Free tier: 15 RPM, 1.5M tokens/day

    free-groq       Groq for notes + Groq for synthesis
                    Fast inference, free tier: 30 RPM, 6K tokens/min
                    128K context (llama-3.1) — may need chunking

    free-hybrid     Groq for notes (fast) + Gemini for synthesis (1M ctx, best quality)
                    Best free-tier quality — recommended free option
    """

    notes_backend: LLMBackend
    synthesis_backend: LLMBackend
    synthesis_context_tokens: int
    notes_context_tokens: int

    def __init__(self, cfg: Settings) -> None:
        mode = cfg.llm_mode.lower()
        self.mode = mode

        # --- Paid backends ---
        opus  = ClaudeBackend(model=cfg.claude_opus_model)
        haiku = ClaudeBackend(model=cfg.claude_haiku_model)

        # --- Free backends ---
        gemini = OpenAICompatBackend(
            model=cfg.gemini_model,
            base_url=GEMINI_BASE_URL,
            api_key=cfg.gemini_api_key,
        )
        groq_notes = OpenAICompatBackend(
            model=cfg.groq_notes_model,
            base_url=GROQ_BASE_URL,
            api_key=cfg.groq_api_key,
        )
        groq_synth = OpenAICompatBackend(
            model=cfg.groq_synthesis_model,
            base_url=GROQ_BASE_URL,
            api_key=cfg.groq_api_key,
        )

        # --- Offline ---
        ollama = OllamaBackend(model=cfg.ollama_model, base_url=cfg.ollama_url)
        ollama_ctx = cfg.ollama_context_tokens - cfg.chunk_reserve_tokens

        if mode == "online":
            self.notes_backend     = opus
            self.synthesis_backend = opus
            self.notes_context_tokens     = cfg.claude_max_input_tokens
            self.synthesis_context_tokens = cfg.claude_max_input_tokens

        elif mode == "hybrid":
            notes_be = (
                ollama if cfg.hybrid_notes_backend == "ollama" else haiku
            )
            notes_ctx = (
                ollama_ctx if cfg.hybrid_notes_backend == "ollama"
                else cfg.claude_max_input_tokens
            )
            self.notes_backend     = notes_be
            self.synthesis_backend = opus
            self.notes_context_tokens     = notes_ctx
            self.synthesis_context_tokens = cfg.claude_max_input_tokens

        elif mode == "offline":
            self.notes_backend     = ollama
            self.synthesis_backend = ollama
            self.notes_context_tokens     = ollama_ctx
            self.synthesis_context_tokens = ollama_ctx

        elif mode == "free-gemini":
            # Gemini 1.5 Flash: 1M context → almost always single-pass
            self.notes_backend     = gemini
            self.synthesis_backend = gemini
            self.notes_context_tokens     = cfg.gemini_context_tokens
            self.synthesis_context_tokens = cfg.gemini_context_tokens

        elif mode == "free-groq":
            # Groq only — smaller context, may chunk
            self.notes_backend     = groq_notes
            self.synthesis_backend = groq_synth
            self.notes_context_tokens     = cfg.groq_context_tokens - cfg.chunk_reserve_tokens
            self.synthesis_context_tokens = cfg.groq_context_tokens - cfg.chunk_reserve_tokens

        elif mode == "free-hybrid":
            # Groq for fast notes + Gemini for high-quality synthesis
            self.notes_backend     = groq_notes
            self.synthesis_backend = gemini
            self.notes_context_tokens     = cfg.groq_context_tokens - cfg.chunk_reserve_tokens
            self.synthesis_context_tokens = cfg.gemini_context_tokens

        else:
            raise ValueError(
                f"Unknown LLM_MODE: {mode!r}. "
                "Use: online | hybrid | offline | free-gemini | free-groq | free-hybrid"
            )
