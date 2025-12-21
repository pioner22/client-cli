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

    if kind == "bold":
        formatted = f"**{segment or 'текст'}**"
    elif kind == "italic":
        formatted = f"_{segment or 'текст'}_"
    elif kind == "link":
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
        # Preserve surrounding space when the original span was whitespace
        if segment and segment.isspace():
            formatted = segment + formatted + " "
    elif kind == "upper":
        formatted = (segment or text).upper()
    elif kind == "lower":
        formatted = (segment or text).lower()
    else:
        formatted = segment

    new_text = text[:a] + formatted + text[b:]
    if kind == "link" and segment and segment.isspace():
        new_caret = a + len(formatted.rstrip())
    else:
        new_caret = a + len(formatted)
    return FormatResult(new_text, new_caret)


__all__ = ["apply_format", "FormatResult"]
