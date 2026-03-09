from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, NamedTuple

from app.epub.parser import (
    CHARS_PER_TOKEN,
    estimate_tokens,
    extract_book_content,
    format_chapter_group,
    group_chapters,
)
from app.llm.router import LLMRouter

SYSTEM_PROMPT = """You are an expert literary critic and analyst with deep knowledge of literature,
narrative craft, themes, and cultural context. You read books comprehensively and produce insightful,
well-structured reports that are both analytically rigorous and accessible to general readers."""


class ReportConfig(NamedTuple):
    label: str
    structure: str
    notes_hint: str


REPORT_TYPES: dict[str, ReportConfig] = {
    "review": ReportConfig(
        label="Book Review",
        notes_hint="key events, characters, themes, writing style observations, memorable quotes",
        structure="""
1. **Overview** — Brief synopsis without major spoilers; genre and intended audience
2. **Writing Style & Voice** — Prose quality, narrative technique, pacing, dialogue
3. **Characters** — Depth, development, believability of main and supporting characters
4. **Themes & Ideas** — Central themes, underlying messages, philosophical or social commentary
5. **Structure & Plot** — Story architecture, plot effectiveness, strengths and weaknesses
6. **Strengths** — What the book does exceptionally well
7. **Weaknesses** — Areas where the book falls short
8. **Overall Assessment** — Verdict and rating (out of 10), who should read this and why""",
    ),
    "summary": ReportConfig(
        label="Summary",
        notes_hint="plot events in order, character actions, key turning points, resolution",
        structure="""Write a clear, flowing summary in 4-6 paragraphs:
1. **Opening** — setting, main characters, initial situation
2. **Rising Action** — key developments and complications
3. **Climax** — the central conflict or turning point
4. **Resolution** — how things conclude
5. **Significance** — what the book ultimately conveys
Be specific and concrete.""",
    ),
    "chapter-summaries": ReportConfig(
        label="Chapter Summaries",
        notes_hint="what happens in each chapter, character actions, key reveals, scene descriptions",
        structure="""For each chapter or section write a concise summary (2-4 sentences) covering:
- Key events
- Character developments
- Important reveals or shifts

Format as:
**[Chapter/Section Title]**
Summary text.""",
    ),
    "characters": ReportConfig(
        label="Character Map",
        notes_hint="character names, descriptions, actions, dialogue, relationships, development arcs",
        structure="""For each significant character provide a profile:

**[Name]** — [role/archetype]
- **Description:** physical and personality traits
- **Arc:** how they change through the story
- **Relationships:** key connections to other characters
- **Notable moments:** 1-2 defining scenes or quotes

Group as: Main Characters → Supporting Characters → Minor Characters.""",
    ),
    "timeline": ReportConfig(
        label="Timeline",
        notes_hint="events with timing, dates, sequence, cause and effect, any flashbacks or time jumps",
        structure="""Extract all significant events in chronological story-time order.

Format each entry as:
**[Time reference]** — Event description *(chapter/location)*

Group into phases or acts where clear. Note any flashbacks or non-linear structure.""",
    ),
    "themes": ReportConfig(
        label="Themes & Motifs",
        notes_hint="themes, symbols, recurring images, philosophical ideas, social commentary, motifs",
        structure="""
1. **Central Themes** — 3-5 major themes with specific textual evidence
2. **Recurring Motifs** — symbols or images that repeat and their significance
3. **Underlying Messages** — what the author is arguing or exploring
4. **Social/Philosophical Context** — broader ideas the work engages with
5. **Theme Development** — where each theme is introduced, built, and resolved""",
    ),
    "quotes": ReportConfig(
        label="Key Quotes",
        notes_hint="memorable quotes, significant dialogue, notable passages — copy them exactly with context",
        structure="""Select 15-20 of the most significant or memorable passages.

For each:
**"[Quote]"**
- *Context:* who says/writes it and when
- *Significance:* why this passage matters

Organise by: Character Voice / Thematic Significance / Stylistic Excellence""",
    ),
    "reading-guide": ReportConfig(
        label="Reading Guide",
        notes_hint="plot events, character motivations, themes, writing techniques, emotional moments",
        structure="""Create a reading guide with 20-25 discussion questions organised into:

**Plot & Comprehension** (5 questions)
**Character & Motivation** (5 questions)
**Themes & Meaning** (5 questions)
**Style & Craft** (5 questions)
**Personal Response & Connection** (5 questions)

Each question should be open-ended and thought-provoking.""",
    ),
    "argument-map": ReportConfig(
        label="Argument Map (non-fiction)",
        notes_hint="claims, evidence, examples, statistics, counterarguments, conclusions, assumptions",
        structure="""
1. **Central Thesis** — the book's main claim in 1-2 sentences
2. **Supporting Arguments** — the 3-6 main pillars
3. **Evidence & Examples** — key evidence for each argument
4. **Assumptions** — unstated premises the argument relies on
5. **Counterarguments Addressed** — objections the author acknowledges
6. **Conclusion** — what the author asks the reader to believe or do
7. **Strength Assessment** — how well-supported the overall argument is""",
    ),
    "action-items": ReportConfig(
        label="Action Items (self-help/business)",
        notes_hint="advice, action steps, exercises, principles, warnings, frameworks, routines",
        structure="""
1. **Core Principles** — the fundamental ideas or frameworks presented
2. **Action Items** — specific things the reader is advised to do (checklist format)
3. **Exercises & Practices** — any exercises, routines, or practices described
4. **Key Insights** — the most important reframings or "aha" moments
5. **Mistakes to Avoid** — pitfalls or anti-patterns warned against
6. **Quick Reference** — one-page cheat sheet of the book's advice""",
    ),
}


@dataclass
class SSEEvent:
    type: str  # progress | text | thinking | done | error
    data: dict[str, Any] = field(default_factory=dict)


async def run_review(
    epub_path: Path,
    router: LLMRouter,
    show_thinking: bool = False,
    chapter_indices: list[int] | None = None,
    report_type: str = "review",
) -> AsyncIterator[SSEEvent]:
    started = time.monotonic()

    # --- Parse EPUB ---
    yield SSEEvent("progress", {"stage": "parsing", "message": "Reading EPUB file…"})
    try:
        title, author, chapters = extract_book_content(str(epub_path))
    except Exception as exc:
        yield SSEEvent("error", {"code": "epub_parse_error", "message": str(exc)})
        return

    if chapter_indices is not None:
        selected = set(chapter_indices)
        chapters = [c for i, c in enumerate(chapters) if i in selected]
        if not chapters:
            yield SSEEvent("error", {"code": "no_chapters", "message": "No chapters matched the selected indices."})
            return

    total_text = " ".join(t for _, t in chapters)
    token_estimate = estimate_tokens(total_text)
    yield SSEEvent(
        "progress",
        {
            "stage": "parsing",
            "message": f"Parsed {len(chapters)} chapters (~{token_estimate:,} tokens)",
        },
    )

    report = REPORT_TYPES.get(report_type, REPORT_TYPES["review"])

    # --- Decide: single-pass or multi-pass ---
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}

    if token_estimate <= router.synthesis_context_tokens:
        # Single pass — entire book fits
        yield SSEEvent("progress", {"stage": "synthesis", "message": "Writing review (single-pass)…"})

        max_chars = router.synthesis_context_tokens * CHARS_PER_TOKEN
        book_text = format_chapter_group(title, author, chapters, max_chars)
        prompt = (
            f"Please read the following book and produce a {report.label}.\n\n"
            f"<book>\n{book_text}\n</book>\n\n"
            f"Produce the {report.label} covering:{report.structure}\n\n"
            f"Be specific with examples from the text."
        )
        async for evt in router.synthesis_backend.stream_text(
            SYSTEM_PROMPT, prompt,
            max_tokens=4096,
            use_cache=True,
            capture_thinking=show_thinking,
        ):
            if evt.type == "text_delta":
                yield SSEEvent("text", {"text": evt.text})
            elif evt.type == "thinking_delta" and show_thinking:
                yield SSEEvent("thinking", {"text": evt.text})
            elif evt.type == "usage":
                for k in usage_totals:
                    usage_totals[k] += getattr(evt, k, 0)

    else:
        # Multi-pass: notes per chapter group → synthesis
        groups = group_chapters(chapters, router.notes_context_tokens)
        total_groups = len(groups)
        all_notes: list[str] = []

        for i, group in enumerate(groups, 1):
            chapter_names = ", ".join(t for t, _ in group)
            yield SSEEvent(
                "progress",
                {
                    "stage": "notes",
                    "message": f"Extracting notes: group {i}/{total_groups} ({chapter_names})",
                    "current": i,
                    "total": total_groups,
                },
            )

            max_chars = router.notes_context_tokens * CHARS_PER_TOKEN
            group_text = format_chapter_group(title, author, group, max_chars)
            note_prompt = (
                f"You are reading \"{title}\" by {author}. "
                f"This is chapter group {i} of {total_groups} (chapters: {chapter_names}).\n\n"
                f"<text>\n{group_text}\n</text>\n\n"
                f"Extract concise reading notes. Focus especially on: {report.notes_hint}.\n\n"
                f"Be specific. These notes will be used to produce a {report.label}."
            )

            notes_text = ""
            async for evt in router.notes_backend.stream_text(
                SYSTEM_PROMPT, note_prompt, max_tokens=2048
            ):
                if evt.type == "text_delta":
                    notes_text += evt.text
                elif evt.type == "usage":
                    for k in usage_totals:
                        usage_totals[k] += getattr(evt, k, 0)

            all_notes.append(f"=== Notes: {chapter_names} ===\n{notes_text}")

        # Synthesis pass
        yield SSEEvent("progress", {"stage": "synthesis", "message": "Writing final review…"})

        combined_notes = "\n\n".join(all_notes)
        synthesis_prompt = (
            f"You have finished reading \"{title}\" by {author}. "
            f"Below are your reading notes from all {total_groups} parts.\n\n"
            f"<notes>\n{combined_notes}\n</notes>\n\n"
            f"Produce a comprehensive {report.label} covering:{report.structure}\n\n"
            f"Draw on specific examples. Write in an engaging, professional tone."
        )
        async for evt in router.synthesis_backend.stream_text(
            SYSTEM_PROMPT, synthesis_prompt,
            max_tokens=4096,
            use_cache=True,
            capture_thinking=show_thinking,
        ):
            if evt.type == "text_delta":
                yield SSEEvent("text", {"text": evt.text})
            elif evt.type == "thinking_delta" and show_thinking:
                yield SSEEvent("thinking", {"text": evt.text})
            elif evt.type == "usage":
                for k in usage_totals:
                    usage_totals[k] += getattr(evt, k, 0)

    yield SSEEvent(
        "done",
        {
            **usage_totals,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "title": title,
            "author": author,
        },
    )
