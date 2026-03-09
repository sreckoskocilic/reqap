from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.epub.parser import estimate_tokens, extract_book_content
from app.llm.router import LLMRouter
from app.review.engine import SSEEvent, run_review

router = APIRouter(prefix="/api")


def _format_sse(event: SSEEvent) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"


@router.post("/epub/chapters")
async def epub_chapters(request: Request, epub_file: UploadFile):
    cfg = request.app.state.config
    max_bytes = cfg.upload_max_mb * 1024 * 1024
    content = await epub_file.read(max_bytes + 1)
    if len(content) > max_bytes:
        return {"error": f"File exceeds {cfg.upload_max_mb} MB limit."}

    suffix = Path(epub_file.filename).suffix.lower() or ".epub"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        title, author, chapters = extract_book_content(tmp_path)
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        os.unlink(tmp_path)

    return {
        "title": title,
        "author": author,
        "chapters": [
            {"index": i, "title": t, "tokens": estimate_tokens(text)}
            for i, (t, text) in enumerate(chapters)
        ],
    }


@router.post("/reviews/stream")
async def stream_review(
    request: Request,
    epub_file: UploadFile,
    show_thinking: bool = Form(False),
    llm_mode: str = Form(""),
    chapter_indices: str = Form(""),
    report_type: str = Form("review"),
):
    cfg = request.app.state.config

    # Allow per-request mode override from the portal UI
    if llm_mode and llm_mode != cfg.llm_mode:
        override_cfg = cfg.model_copy(update={"llm_mode": llm_mode})
        llm_router = LLMRouter(override_cfg)
    else:
        llm_router = request.app.state.router

    # Enforce upload size limit
    max_bytes = cfg.upload_max_mb * 1024 * 1024
    content = await epub_file.read(max_bytes + 1)
    if len(content) > max_bytes:

        async def size_error():
            yield _format_sse(
                SSEEvent(
                    "error",
                    {
                        "code": "file_too_large",
                        "message": f"File exceeds {cfg.upload_max_mb} MB limit.",
                    },
                )
            )

        return StreamingResponse(size_error(), media_type="text/event-stream")

    async def generate():
        file_suffix = Path(epub_file.filename).suffix.lower() or ".epub"
        with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        indices = (
            [int(i) for i in chapter_indices.split(",") if i.strip()]
            if chapter_indices.strip()
            else None
        )

        try:
            async for event in run_review(
                epub_path=Path(tmp_path),
                router=llm_router,
                show_thinking=show_thinking,
                chapter_indices=indices,
                report_type=report_type,
            ):
                yield _format_sse(event)
                if event.type in ("done", "error"):
                    break
        except Exception as exc:
            yield _format_sse(
                SSEEvent("error", {"code": "internal_error", "message": str(exc)})
            )
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/health")
async def health(request: Request):
    return {"status": "ok", "mode": request.app.state.config.llm_mode}


@router.get("/info")
async def info(request: Request):
    cfg = request.app.state.config
    return {
        "mode": cfg.llm_mode,
        "claude_opus_model": cfg.claude_opus_model,
        "claude_haiku_model": cfg.claude_haiku_model,
        "ollama_model": cfg.ollama_model,
        "ollama_url": cfg.ollama_url,
        "hybrid_notes_backend": cfg.hybrid_notes_backend,
    }
