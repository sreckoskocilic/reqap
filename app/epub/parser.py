import re
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

CHARS_PER_TOKEN = 4  # rough average for English prose

_HEADING = re.compile(
    r"^(?:"
    r"(?:chapter|part|book|section|prologue|epilogue|introduction|preface|appendix)\b.*"
    r"|[IVXLC]+\.?\s*$"
    r"|\d+\.?\s*$"
    r"|[A-Z][A-Z\s]{2,50}$"
    r")$",
    re.IGNORECASE,
)


def extract_text_content(text_path: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Return (title, author, chapters) from a plain text file."""
    with open(text_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    title = Path(text_path).stem
    author = "Unknown Author"

    lines = content.split("\n")
    chapters: list[tuple[str, str]] = []
    current_title = title
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) < 80 and _HEADING.match(stripped):
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    chapters.append((current_title, text))
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            chapters.append((current_title, text))

    if len(chapters) <= 1:
        chapters = [(title, content.strip())]

    return title, author, chapters


def extract_book_content(path: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Dispatch to epub or text parser based on file extension."""
    if Path(path).suffix.lower() == ".epub":
        return extract_epub_content(path)
    return extract_text_content(path)


def extract_epub_content(epub_path: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Return (title, author, chapters) from an EPUB file."""
    book = epub.read_epub(epub_path)

    title_meta = book.get_metadata("DC", "title")
    title = title_meta[0][0] if title_meta else "Unknown Title"

    author_meta = book.get_metadata("DC", "creator")
    author = author_meta[0][0] if author_meta else "Unknown Author"

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


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def group_chapters(
    chapters: list[tuple[str, str]], max_tokens: int
) -> list[list[tuple[str, str]]]:
    """Group chapters into batches that each fit within max_tokens."""
    max_chars = max_tokens * CHARS_PER_TOKEN
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


def format_chapter_group(
    title: str, author: str, chapters: list[tuple[str, str]], max_chars: int
) -> str:
    parts = [f"BOOK: {title}", f"AUTHOR: {author}", "=" * 60, ""]
    remaining = max_chars - sum(len(p) + 1 for p in parts)

    for chapter_title, text in chapters:
        header = f"\n--- {chapter_title} ---\n"
        if remaining <= len(header):
            break
        remaining -= len(header)
        parts.append(header.strip())

        if len(text) > remaining:
            text = text[:remaining] + "\n[...truncated]"
            remaining = 0
        else:
            remaining -= len(text)
        parts.append(text)
        parts.append("")

    return "\n".join(parts)
