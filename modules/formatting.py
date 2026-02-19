"""
Text formatting helpers used by the TUI.

Provides `apply_format` that wraps the current selection (или слово под курсором)
разметкой для жирного/ссылок или меняет регистр.
Возвращает объект с полями `text` и `caret`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class FormatResult:
    text: str
    caret: int


def _word_bounds(text: str, sel_start: int, sel_end: int, caret: int) -> Tuple[int, int]:
    """Return (a, b) slice for the active selection or word under caret."""
    if sel_start != sel_end:
        a, b = sorted((sel_start, sel_end))
        return a, b
    n = len(text)
    caret = max(0, min(n, caret))
    # If caret is on whitespace, keep that whitespace as a boundary span
    if caret < n and text[caret].isspace():
        return caret, min(caret + 1, n)
    left = caret
    while left > 0 and not text[left - 1].isspace():
        left -= 1
    right = caret
    while right < n and not text[right].isspace():
        right += 1
    return (caret, caret) if left == right else (left, right)


def _format_link(segment: str, link_text: str, link_url: str) -> str:
    label = (link_text or segment or link_url or "").strip()
    url = (link_url or "").strip()
    if not label and segment:
        label = segment
    if not label and url:
        label = url
    if url:
        formatted = f"[{label or 'link'}]({url})"
    else:
        formatted = label or segment
    if segment and segment.isspace():
        return segment + formatted + " "
    return formatted


def _apply_case(kind: str, segment: str, text: str) -> str:
    source = segment or text
    if kind == "upper":
        return source.upper()
    if kind == "lower":
        return source.lower()
    return segment


def _format_segment(kind: str, segment: str, text: str, link_text: str, link_url: str) -> str:
    if kind == "bold":
        return f"**{segment or 'текст'}**"
    if kind == "italic":
        return f"_{segment or 'текст'}_"
    if kind == "link":
        return _format_link(segment, link_text, link_url)
    if kind in ("upper", "lower"):
        return _apply_case(kind, segment, text)
    return segment


def apply_format(
    kind: str,
    text: str,
    caret: int,
    sel_start: int,
    sel_end: int,
    link_text: str = "",
    link_url: str = "",
) -> FormatResult:
    """Apply formatting token to selection/word and return updated text+caret."""
    text = text or ""
    a, b = _word_bounds(text, sel_start, sel_end, caret)
    segment = text[a:b]
    formatted = _format_segment(kind, segment, text, link_text, link_url)

    new_text = text[:a] + formatted + text[b:]
    if kind == "link" and segment and segment.isspace():
        new_caret = a + len(formatted.rstrip())
    else:
        new_caret = a + len(formatted)
    return FormatResult(new_text, new_caret)


__all__ = ["apply_format", "FormatResult"]
