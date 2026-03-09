#!/usr/bin/env python3
"""
EPUB Book Review Service
Reads an EPUB file comprehensively and generates a detailed review.

Online mode  (default): uses Claude API via Anthropic SDK
Offline mode (--offline): uses a local Ollama model with multi-pass chunking
"""

import sys
import re
import argparse
import anthropic
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

SYSTEM_PROMPT = """You are an expert literary critic and book reviewer with deep knowledge of literature,
narrative craft, themes, and cultural context. You read books comprehensively and produce insightful,
well-structured reviews that are both analytically rigorous and accessible to general readers."""


# ---------------------------------------------------------------------------
# EPUB parsing
# ---------------------------------------------------------------------------


def extract_epub_content(epub_path: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Return (title, author, chapters) from an EPUB file."""
    book = epub.read_epub(epub_path)

    title = book.get_metadata("DC", "title")
    title = title[0][0] if title else "Unknown Title"

    author = book.get_metadata("DC", "creator")
    author = author[0][0] if author else "Unknown Author"

    chapters: list[tuple[str, str]] = []
    spine_ids = {item_id for item_id, _ in book.spine}

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        if item.get_id() not in spine_ids:
            continue

        soup = BeautifulSoup(item.get_content(), "lxml")

        heading = soup.find(["h1", "h2", "h3"])
        chapter_title = heading.get_text(strip=True) if heading else item.get_name()

        for tag in soup(["script", "style", "nav"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if text.strip():
            chapters.append((chapter_title, text))

    return title, author, chapters


def build_book_content(title: str, author: str, chapters: list[tuple[str, str]]) -> str:
    parts = [f"BOOK: {title}", f"AUTHOR: {author}", "=" * 60, ""]
    for i, (chapter_title, text) in enumerate(chapters, 1):
        parts.append(f"--- Chapter {i}: {chapter_title} ---")
        parts.append(text)
        parts.append("")
    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


# ---------------------------------------------------------------------------
# Online mode — Claude API
# ---------------------------------------------------------------------------

REVIEW_STRUCTURE = """
1. **Overview** — Brief synopsis without major spoilers; genre and intended audience
2. **Writing Style & Voice** — Prose quality, narrative technique, pacing, dialogue
3. **Characters** — Depth, development, believability of main and supporting characters
4. **Themes & Ideas** — Central themes, underlying messages, philosophical or social commentary
5. **Structure & Plot** — Story architecture, plot effectiveness, strengths and weaknesses
6. **Strengths** — What the book does exceptionally well
7. **Weaknesses** — Areas where the book falls short
8. **Overall Assessment** — Verdict and rating (out of 10), who should read this and why"""


def review_online(
    book_content: str,
    output,
    show_thinking: bool = False,
) -> None:
    client = anthropic.Anthropic()

    user_prompt = (
        f"Please read the following book in its entirety and write a comprehensive review.\n\n"
        f"<book>\n{book_content}\n</book>\n\n"
        f"Write a thorough review covering:{REVIEW_STRUCTURE}\n\n"
        f"Be specific with examples from the text. Write the review in an engaging, professional tone."
    )

    thinking_buffer: list[str] = []
    current_block_type: str | None = None

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    ) as stream:
        for event in stream:
            if event.type == "content_block_start":
                current_block_type = event.content_block.type
                if current_block_type == "thinking" and show_thinking:
                    print("\n[THINKING]\n", file=sys.stderr)

            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    thinking_buffer.append(event.delta.thinking)
                    if show_thinking:
                        print(event.delta.thinking, end="", flush=True, file=sys.stderr)
                elif event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True, file=output)

            elif event.type == "content_block_stop":
                if current_block_type == "thinking" and show_thinking:
                    print("\n[/THINKING]\n", file=sys.stderr)
                current_block_type = None

    final = stream.get_final_message()

    full_thinking = "".join(thinking_buffer)
    if full_thinking:
        print(f"\nThinking captured: {len(full_thinking):,} chars", file=sys.stderr)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(
        f"Tokens used — input: {final.usage.input_tokens:,}, output: {final.usage.output_tokens:,}",
        file=sys.stderr,
    )
    if getattr(final.usage, "cache_read_input_tokens", None):
        print(
            f"Cache read: {final.usage.cache_read_input_tokens:,} tokens",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Offline mode — Ollama (local model, multi-pass chunking)
# ---------------------------------------------------------------------------


def _ollama_stream(model: str, system: str, prompt: str, base_url: str):
    """Call Ollama via its OpenAI-compatible /v1/chat/completions endpoint."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _group_chapters(
    chapters: list[tuple[str, str]], max_tokens: int
) -> list[list[tuple[str, str]]]:
    """
    Group chapters into batches that each fit within max_tokens.
    A single oversized chapter is placed in its own batch (it will be truncated).
    """
    max_chars = max_tokens * 4
    groups: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_chars = 0

    for chapter in chapters:
        chapter_chars = len(chapter[1])
        if current and current_chars + chapter_chars > max_chars:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(chapter)
        current_chars += chapter_chars

    if current:
        groups.append(current)

    return groups


def _format_chapter_group(
    title: str, author: str, chapters: list[tuple[str, str]], max_chars: int
) -> str:
    parts = [f"BOOK: {title}", f"AUTHOR: {author}", "=" * 60, ""]
    remaining = max_chars - len("\n".join(parts))
    for chapter_title, text in chapters:
        header = f"--- {chapter_title} ---\n"
        if remaining <= len(header):
            break
        remaining -= len(header)
        parts.append(header.rstrip())
        if len(text) > remaining:
            text = text[:remaining] + "\n[...truncated]"
            remaining = 0
        else:
            remaining -= len(text)
        parts.append(text)
        parts.append("")
    return "\n".join(parts)


def review_offline(
    title: str,
    author: str,
    chapters: list[tuple[str, str]],
    output,
    model: str,
    context_tokens: int,
    base_url: str,
) -> None:
    # Reserve tokens for prompts and response
    chunk_tokens = context_tokens - 2048

    groups = _group_chapters(chapters, chunk_tokens)
    total = len(groups)
    print(
        f"Offline mode: {total} chapter group(s), model={model}, context={context_tokens:,} tokens",
        file=sys.stderr,
    )
    for i, g in enumerate(groups, 1):
        names = ", ".join(t for t, _ in g)
        print(f"  Group {i}: {names}", file=sys.stderr)

    max_chars = chunk_tokens * 4

    if total == 1:
        # All chapters fit in one pass — direct review
        print("Single-pass review...", file=sys.stderr)
        book_text = _format_chapter_group(title, author, groups[0], max_chars)
        prompt = (
            f"Please read the following book and write a comprehensive review.\n\n"
            f"<book>\n{book_text}\n</book>\n\n"
            f"Write a thorough review covering:{REVIEW_STRUCTURE}\n\n"
            f"Be specific with examples from the text."
        )
        for token in _ollama_stream(model, SYSTEM_PROMPT, prompt, base_url):
            print(token, end="", flush=True, file=output)

    else:
        # Multi-pass: extract notes per chapter group, then synthesize
        all_notes: list[str] = []

        for i, group in enumerate(groups, 1):
            chapter_names = ", ".join(t for t, _ in group)
            print(
                f"  Pass {i}/{total}: extracting notes ({chapter_names})...",
                file=sys.stderr,
            )
            group_text = _format_chapter_group(title, author, group, max_chars)
            note_prompt = (
                f'You are reading "{title}" by {author}. '
                f"This is chapter group {i} of {total} "
                f"(chapters: {chapter_names}).\n\n"
                f"<text>\n{group_text}\n</text>\n\n"
                f"Extract concise reading notes covering:\n"
                f"- Key plot events and turning points\n"
                f"- Character introductions and development\n"
                f"- Recurring themes and motifs\n"
                f"- Notable writing style observations\n"
                f"- Memorable quotes (with context)\n\n"
                f"Be specific. These notes will be used to write a full book review."
            )
            notes = "".join(_ollama_stream(model, SYSTEM_PROMPT, note_prompt, base_url))
            all_notes.append(f"=== Notes: {chapter_names} ===\n{notes}")
            print(f"  Notes captured: {len(notes):,} chars", file=sys.stderr)

        # Final synthesis pass
        print("  Synthesis pass: writing final review...", file=sys.stderr)
        combined_notes = "\n\n".join(all_notes)
        synthesis_prompt = (
            f'You have finished reading "{title}" by {author}. '
            f"Below are your reading notes from all {total} parts of the book.\n\n"
            f"<notes>\n{combined_notes}\n</notes>\n\n"
            f"Now write a comprehensive, cohesive book review covering:{REVIEW_STRUCTURE}\n\n"
            f"Draw on specific examples from your notes. "
            f"Write in an engaging, professional tone as a literary critic."
        )
        for token in _ollama_stream(model, SYSTEM_PROMPT, synthesis_prompt, base_url):
            print(token, end="", flush=True, file=output)

    print(f"\n{'=' * 60}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def review_epub(
    epub_path: str,
    output_file: str | None = None,
    show_thinking: bool = False,
    offline: bool = False,
    ollama_model: str = "llama3.2",
    ollama_context: int = 8192,
    ollama_url: str = "http://localhost:11434",
) -> None:
    print(f"Reading: {epub_path}", file=sys.stderr)

    title, author, chapters = extract_epub_content(epub_path)
    print(f"Title:    {title}", file=sys.stderr)
    print(f"Author:   {author}", file=sys.stderr)
    print(f"Chapters: {len(chapters)}", file=sys.stderr)

    book_content = build_book_content(title, author, chapters)
    token_estimate = estimate_tokens(book_content)
    print(f"Estimated tokens: ~{token_estimate:,}", file=sys.stderr)

    output = sys.stdout if output_file is None else open(output_file, "w")

    try:
        print(
            f"\nGenerating review ({'offline' if offline else 'online'})...\n",
            file=sys.stderr,
        )
        print("=" * 60)

        if offline:
            review_offline(
                title=title,
                author=author,
                chapters=chapters,
                output=output,
                model=ollama_model,
                context_tokens=ollama_context,
                base_url=ollama_url,
            )
        else:
            # Cap at 150K tokens for Claude's context window
            if token_estimate > 150_000:
                print(
                    "Warning: Book is very large — truncating to 150K tokens.",
                    file=sys.stderr,
                )
                book_content = book_content[: 150_000 * 4] + "\n\n[... truncated ...]"

            review_online(book_content, output, show_thinking=show_thinking)

    finally:
        if output_file is not None:
            output.close()
            print(f"\nReview saved to: {output_file}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive review of an EPUB book."
    )
    parser.add_argument("epub_file", help="Path to the EPUB file")
    parser.add_argument(
        "-o", "--output", help="Save review to file (default: stdout)", default=None
    )

    # Online options
    parser.add_argument(
        "--thinking", action="store_true", help="Stream Claude's thinking to stderr"
    )

    # Offline options
    parser.add_argument(
        "--offline", action="store_true", help="Use local Ollama model (no internet)"
    )
    parser.add_argument(
        "--model", default="llama3.2", help="Ollama model name (default: llama3.2)"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=8192,
        help="Local model context window in tokens (default: 8192). "
        "Use a higher value if your model supports it, e.g. 32768 for llama3.2:latest.",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434", help="Ollama base URL"
    )

    args = parser.parse_args()

    try:
        review_epub(
            epub_path=args.epub_file,
            output_file=args.output,
            show_thinking=args.thinking,
            offline=args.offline,
            ollama_model=args.model,
            ollama_context=args.context,
            ollama_url=args.ollama_url,
        )
    except FileNotFoundError:
        print(f"Error: File not found: {args.epub_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
