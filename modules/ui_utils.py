from __future__ import annotations

"""
UI helper utilities shared between client and tests.
"""

import unicodedata
import functools
from typing import Optional


_ZERO_WIDTH = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u200e",  # LEFT-TO-RIGHT MARK
    "\u200f",  # RIGHT-TO-LEFT MARK
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\ufe0e",  # VARIATION SELECTOR-15
    "\ufe0f",  # VARIATION SELECTOR-16
}


def _is_zero_width(ch: str) -> bool:
    if ch in _ZERO_WIDTH:
        return True
    try:
        # Cf includes many format/zero-width chars
        return unicodedata.category(ch) == "Cf"
    except Exception:
        return False


def _wcwidth(ch: str) -> int:
    # Resolve optional wcwidth backend once (avoid per-char import overhead).
    # If wcwidth isn't installed, we fall back to a simple East Asian width heuristic.
    global _WCWIDTH_FUNC
    try:
        if _is_zero_width(ch) or unicodedata.combining(ch):
            return 0
    except Exception:
        pass
    try:
        fn = _WCWIDTH_FUNC
        if fn is not None:
            w = fn(ch)
            if w is not None and w >= 0:
                return int(w)
    except Exception:
        pass
    try:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            return 2
    except Exception:
        pass
    return 1


def _load_wcwidth_func():
    try:
        import wcwidth as _wc  # type: ignore

        return getattr(_wc, "wcwidth", None)
    except Exception:
        return None


_WCWIDTH_FUNC = _load_wcwidth_func()


@functools.lru_cache(maxsize=4096)
def _wcwidth_cached(ch: str) -> int:
    return _wcwidth(ch)


def display_width(s: str) -> int:
    try:
        return sum(_wcwidth_cached(ch) for ch in (s or ""))
    except Exception:
        return len(s or "")


def truncate_to_width(s: str, width: int) -> str:
    """Trim string so its display width is <= width (Unicode-aware)."""
    if width <= 0:
        return ""
    out: list[str] = []
    cols = 0
    for ch in (s or ""):
        try:
            if _is_zero_width(ch) or unicodedata.combining(ch):
                if out:
                    out.append(ch)
                continue
        except Exception:
            pass
        w = max(0, _wcwidth_cached(ch))
        if cols + w > width:
            break
        out.append(ch)
        cols += w
    return "".join(out)


def right_truncate_to_width(s: str, width: int) -> str:
    """Keep the rightmost part of string that fits into width (Unicode-aware)."""
    if width <= 0:
        return ""
    out_rev: list[str] = []
    cols = 0
    pending: list[str] = []
    for ch in reversed(s or ""):
        try:
            if _is_zero_width(ch) or unicodedata.combining(ch):
                pending.append(ch)
                continue
        except Exception:
            pass
        w = max(0, _wcwidth_cached(ch))
        if cols + w > width:
            break
        seg = ch + "".join(reversed(pending))
        pending.clear()
        out_rev.append(seg)
        cols += w
    out_rev.reverse()
    return "".join(out_rev)


def pad_to_width(s: str, width: int) -> str:
    """Pad or trim string to exactly `width` display cells (Unicode-aware)."""
    if width <= 0:
        return ""
    trimmed = truncate_to_width(s, width)
    cur = display_width(trimmed)
    if cur >= width:
        return trimmed
    return trimmed + (" " * (width - cur))


def fit_contact_label(s: str, width: int) -> str:
    """Trim contact label to fit `width`, preserving trailing " [ID]" if present.

    - If the string fits, return as-is.
    - If there is a trailing " [ID]" suffix and not enough space, trim the
      prefix and keep the suffix visible, inserting an ellipsis before it.
    - Otherwise, trim right and append an ellipsis.
    """
    try:
        if width <= 0:
            return ""
        if display_width(s) <= width:
            return s
        if width == 1:
            return "…"
        idx = s.rfind(" [")
        if idx != -1 and s.endswith("]"):
            suffix = s[idx:]
            # If suffix itself is too long, hard trim
            if display_width(suffix) >= max(4, width):
                return truncate_to_width(s, max(0, width - 1)) + "…"
            avail = width - display_width(suffix) - 1
            if avail <= 0:
                return "…" + right_truncate_to_width(suffix, max(0, width - 1))
            prefix = s[:idx].rstrip()
            return truncate_to_width(prefix, avail) + "…" + suffix
        return truncate_to_width(s, max(0, width - 1)) + "…"
    except Exception:
        return s[:width]

__all__ = [
    "display_width",
    "truncate_to_width",
    "right_truncate_to_width",
    "pad_to_width",
    "fit_contact_label",
]


def search_status_line(query: str, live_ok: bool, live_id: Optional[str] = None) -> tuple[str, str]:
    """Compute status message and category for search overlay.

    Returns (message, category) where category ∈ {"", "warn", "success", "error"}.
    - <3 symbols (after '@' removal) → warn: "Минимум 3 символа..." (if non-empty)
    - found (live_ok and live_id)      → success: "Найден: <id> — Enter/A: выбрать"
    - >=3 and not found                → error:   "Не найден — исправьте ID/@логин"
    - empty query                      → ("", "")
    """
    try:
        q = (query or "").strip()
        if not q:
            return "", ""
        qcore = q[1:] if q.startswith('@') else q
        if len(qcore) < 3:
            return "Минимум 3 символа для живого поиска", "warn"
        if live_ok and (live_id or "").strip():
            return f"Найден: {live_id} — Enter/A: выбрать", "success"
        return "Не найден — исправьте ID/@логин", "error"
    except Exception:
        return "", ""

__all__.append("search_status_line")
