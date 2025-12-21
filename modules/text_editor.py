from __future__ import annotations

"""
Lightweight text editor helper for single-line/multiline input with a caret.

Supports insertion, deletion, and caret movement by characters and newlines.
Wrapping is left to the UI; up/down operate on logical lines separated by '\n'.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TextEditor:
    text: str = ""
    caret: int = 0  # caret index in [0..len(text)]

    def clamp(self) -> None:
        if self.caret < 0:
            self.caret = 0
        if self.caret > len(self.text):
            self.caret = len(self.text)

    def set(self, text: str, caret: Optional[int] = None) -> None:
        self.text = text or ""
        if caret is not None:
            self.caret = int(caret)
        self.clamp()

    def insert(self, s: str) -> None:
        if not s:
            return
        self.text = self.text[: self.caret] + s + self.text[self.caret :]
        self.caret += len(s)

    def backspace(self) -> None:
        if self.caret <= 0:
            return
        self.text = self.text[: self.caret - 1] + self.text[self.caret :]
        self.caret -= 1

    def delete(self) -> None:
        if self.caret >= len(self.text):
            return
        self.text = self.text[: self.caret] + self.text[self.caret + 1 :]

    def move_left(self) -> None:
        if self.caret > 0:
            self.caret -= 1

    def move_right(self) -> None:
        if self.caret < len(self.text):
            self.caret += 1

    def _line_col(self) -> tuple[int, int]:
        # Return (line_index, col_in_line) based on '\n' splitting
        before = self.text[: self.caret]
        lines = before.split("\n")
        line_idx = len(lines) - 1
        col = len(lines[-1]) if lines else 0
        return line_idx, col

    def move_up(self) -> None:
        line_idx, col = self._line_col()
        if line_idx <= 0:
            return
        # Compute absolute index of start of target line
        lines = self.text.split("\n")
        # Clamp col to target line length
        new_col = min(col, len(lines[line_idx - 1]))
        # Index of start of target line
        new_index = sum(len(l) + 1 for l in lines[: line_idx - 1]) + new_col
        self.caret = max(0, min(len(self.text), new_index))

    def move_down(self) -> None:
        line_idx, col = self._line_col()
        lines = self.text.split("\n")
        if line_idx >= len(lines) - 1:
            return
        new_col = min(col, len(lines[line_idx + 1]))
        new_index = sum(len(l) + 1 for l in lines[: line_idx + 1]) + new_col
        self.caret = max(0, min(len(self.text), new_index))

    # ---- Word-wise navigation helpers ----
    def move_word_left(self) -> None:
        if self.caret <= 0:
            self.caret = 0
            return
        i = self.caret
        s = self.text
        # Skip spaces left
        while i > 0 and s[i - 1].isspace():
            i -= 1
        # Skip a word left (non-space)
        while i > 0 and not s[i - 1].isspace():
            i -= 1
        self.caret = i

    def move_word_right(self) -> None:
        n = len(self.text)
        i = self.caret
        s = self.text
        if i >= n:
            self.caret = n
            return
        # Skip current word
        while i < n and not s[i].isspace():
            i += 1
        # Skip trailing spaces to the beginning of next word
        while i < n and s[i].isspace():
            i += 1
        self.caret = i

    def delete_word_left(self) -> None:
        if self.caret <= 0:
            return
        j = self.caret
        s = self.text
        i = j
        while i > 0 and s[i - 1].isspace():
            i -= 1
        while i > 0 and not s[i - 1].isspace():
            i -= 1
        self.text = s[:i] + s[j:]
        self.caret = i

    def delete_word_right(self) -> None:
        n = len(self.text)
        if self.caret >= n:
            return
        s = self.text
        i = self.caret
        # Skip spaces to the start of the next word
        j = i
        while j < n and s[j].isspace():
            j += 1
        # Then skip the word itself
        while j < n and not s[j].isspace():
            j += 1
        self.text = s[:i] + s[j:]
        self.caret = i

    def delete_to_line_end(self) -> None:
        # Remove from caret to the end of logical line (or text end)
        s = self.text
        if self.caret >= len(s):
            return
        i = self.caret
        nl = s.find('\n', i)
        end = nl if nl != -1 else len(s)
        self.text = s[:i] + s[end:]

def wrap_text_simple(text: str, width: int) -> List[str]:
    """Simple greedy word-wrap used by the TUI.

    - Breaks lines at spaces where possible within `width`.
    - Falls back to hard cut if no space found.
    - Preserves explicit newlines by processing line-by-line.
    """
    if width <= 0:
        return [text]
    lines: List[str] = []
    for raw_line in text.splitlines() or [""]:
        s = raw_line
        while len(s) > width:
            cut = s.rfind(' ', 0, width)
            if cut == -1:
                cut = width
            lines.append(s[:cut])
            s = s[cut:].lstrip()
        lines.append(s)
    return lines


def caret_view_window(text: str, caret: int, width: int, visible_rows: int) -> Tuple[int, int, int]:
    """Compute a scrolling window for the input box that keeps caret visible.

    Returns (start_row, caret_row_in_box, col_in_line).
    - `start_row` is the first wrapped row to display.
    - `caret_row_in_box` is zero-based row index within the visible window.
    - `col_in_line` is caret column within the current wrapped line (logical col modulo width).
    """
    width = max(1, int(width))
    visible_rows = max(1, int(visible_rows))
    text = text or ""
    caret = max(0, min(len(text), int(caret)))

    def _rows_for(t: str) -> int:
        rows = 0
        for raw in t.split('\n') or [""]:
            chunks = wrap_text_simple(raw, width)
            rows += max(1, len(chunks))
        return rows

    pre = text[:caret]
    pre_rows = 0
    for raw in pre.split('\n')[:-1]:
        pre_rows += max(1, len(wrap_text_simple(raw, width)))
    cur_line = pre.split('\n')[-1] if pre else ""
    cur_row_offset = len(wrap_text_simple(cur_line, width)) - 1 if cur_line else 0
    total_rows = _rows_for(text)
    display_row = pre_rows + cur_row_offset
    start_row = min(max(0, display_row - visible_rows + 1), max(0, total_rows - visible_rows))
    caret_row_in_box = max(0, min(visible_rows - 1, display_row - start_row))
    col_in_line = len(cur_line) % width
    return start_row, caret_row_in_box, col_in_line


__all__ = [
    "TextEditor",
    "wrap_text_simple",
    "caret_view_window",
]
