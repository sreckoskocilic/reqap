"""Tests for app/epub/parser.py"""
import tempfile
from pathlib import Path

import pytest

from app.epub.parser import (
    CHARS_PER_TOKEN,
    estimate_tokens,
    extract_text_content,
    extract_book_content,
    format_chapter_group,
    group_chapters,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_basic():
    text = "a" * 400
    assert estimate_tokens(text) == 100


def test_estimate_tokens_consistent_with_constant():
    text = "x" * 1000
    assert estimate_tokens(text) == 1000 // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# group_chapters
# ---------------------------------------------------------------------------

def _make_chapters(sizes):
    """Create (title, text) tuples with given character sizes."""
    return [(f"Ch {i}", "x" * n) for i, n in enumerate(sizes)]


def test_group_chapters_single_group_when_all_fit():
    chapters = _make_chapters([100, 200, 100])
    groups = group_chapters(chapters, max_tokens=1000)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_group_chapters_splits_when_exceeds_limit():
    # Each chapter is 400 chars = 100 tokens; limit is 150 tokens → max 1 per group
    chapters = _make_chapters([400, 400, 400])
    groups = group_chapters(chapters, max_tokens=150)
    assert len(groups) == 3


def test_group_chapters_packs_multiple_into_group():
    # 4 chapters of 100 chars = 25 tokens each; limit is 60 tokens → pairs
    chapters = _make_chapters([100, 100, 100, 100])
    groups = group_chapters(chapters, max_tokens=60)
    assert len(groups) == 2
    assert all(len(g) == 2 for g in groups)


def test_group_chapters_empty():
    assert group_chapters([], max_tokens=1000) == []


def test_group_chapters_single_oversized_chapter():
    # A chapter larger than max still goes in its own group
    chapters = _make_chapters([10000])
    groups = group_chapters(chapters, max_tokens=10)
    assert len(groups) == 1


# ---------------------------------------------------------------------------
# format_chapter_group
# ---------------------------------------------------------------------------

def test_format_chapter_group_contains_title_and_author():
    chapters = [("Introduction", "Some introductory text.")]
    result = format_chapter_group("My Book", "Jane Doe", chapters, max_chars=10000)
    assert "My Book" in result
    assert "Jane Doe" in result
    assert "Introduction" in result
    assert "Some introductory text." in result


def test_format_chapter_group_respects_max_chars():
    chapters = [("Ch 1", "x" * 5000)]
    result = format_chapter_group("T", "A", chapters, max_chars=200)
    assert len(result) <= 220  # small tolerance for header overhead


def test_format_chapter_group_truncates_long_chapter():
    chapters = [("Long", "word " * 2000)]
    result = format_chapter_group("T", "A", chapters, max_chars=500)
    assert "[...truncated]" in result


# ---------------------------------------------------------------------------
# extract_text_content
# ---------------------------------------------------------------------------

def _write_tmp_txt(content: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".txt", mode="w",
                                    encoding="utf-8", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_extract_text_no_chapters_treated_as_single():
    text = "Just a block of plain text with no headings at all.\n" * 20
    path = _write_tmp_txt(text)
    title, author, chapters = extract_text_content(path)
    assert len(chapters) == 1
    assert chapters[0][1].strip() != ""


def test_extract_text_detects_chapter_headings():
    text = "Chapter 1\nFirst chapter content.\n\nChapter 2\nSecond chapter content.\n"
    path = _write_tmp_txt(text)
    _, _, chapters = extract_text_content(path)
    assert len(chapters) == 2
    assert chapters[0][0] == "Chapter 1"
    assert chapters[1][0] == "Chapter 2"


def test_extract_text_title_from_filename():
    text = "Some content here."
    path = _write_tmp_txt(text)
    title, _, _ = extract_text_content(path)
    assert title == Path(path).stem


def test_extract_text_author_unknown():
    path = _write_tmp_txt("Content.")
    _, author, _ = extract_text_content(path)
    assert author == "Unknown Author"


def test_extract_text_skips_empty_chapters():
    text = "Chapter 1\n\nChapter 2\nActual content here.\n"
    path = _write_tmp_txt(text)
    _, _, chapters = extract_text_content(path)
    # Empty Chapter 1 should be dropped
    assert all(text.strip() for _, text in chapters)


# ---------------------------------------------------------------------------
# extract_book_content dispatch
# ---------------------------------------------------------------------------

def test_extract_book_content_dispatches_txt():
    path = _write_tmp_txt("Hello world content.")
    title, author, chapters = extract_book_content(path)
    assert len(chapters) >= 1


def test_extract_book_content_unknown_extension_treated_as_text(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\nContent here.")
    # Non-epub, non-txt extension falls through to text parser
    title, author, chapters = extract_book_content(str(f))
    assert len(chapters) >= 1
