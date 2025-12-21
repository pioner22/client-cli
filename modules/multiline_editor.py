from __future__ import annotations

"""
Multiline editor helpers to introspect caret position in wrapped view.

Interface mirrors Codex Ink's editor signals used by terminal-chat-input:
- get_row(): current wrapped row index of caret
- get_col(): column within current wrapped row
- is_cursor_at_last_row(): True if caret is on the last wrapped row
"""

from dataclasses import dataclass
from typing import Tuple

from .text_editor import wrap_text_simple


@dataclass
class EditorView:
    text: str
    caret: int
    width: int

    def _compute(self) -> Tuple[int, int, int]:
        width = max(1, int(self.width))
        text = self.text or ""
        caret = max(0, min(len(text), int(self.caret)))

        def _rows_for(t: str) -> int:
            rows = 0
            for raw in t.split('\n') or [""]:
                rows += max(1, len(wrap_text_simple(raw, width)))
            return rows

        pre = text[:caret]
        pre_rows = 0
        for raw in pre.split('\n')[:-1]:
            pre_rows += max(1, len(wrap_text_simple(raw, width)))
        cur_line = pre.split('\n')[-1] if pre else ""
        cur_row_offset = len(wrap_text_simple(cur_line, width)) - 1 if cur_line else 0
        total_rows = _rows_for(text)
        display_row = pre_rows + cur_row_offset
        col_in_line = len(cur_line) % width
        return display_row, col_in_line, total_rows

    def get_row(self) -> int:
        row, _, _ = self._compute()
        return row

    def get_col(self) -> int:
        _, col, _ = self._compute()
        return col

    def is_cursor_at_last_row(self) -> bool:
        row, _, total = self._compute()
        return row >= max(0, total - 1)


__all__ = ["EditorView"]

