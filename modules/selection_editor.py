from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelectEditor:
    text: str = ""
    caret: int = 0
    sel_start: int = 0
    sel_end: int = 0

    def _clamp_all(self) -> None:
        n = len(self.text)
        self.caret = max(0, min(n, int(self.caret)))
        self.sel_start = max(0, min(n, int(self.sel_start)))
        self.sel_end = max(0, min(n, int(self.sel_end)))

    def has_selection(self) -> bool:
        return int(self.sel_start) != int(self.sel_end)

    def clear_selection(self) -> None:
        self.sel_start = self.caret
        self.sel_end = self.caret

    def _delete_range(self, a: int, b: int) -> None:
        a, b = min(a, b), max(a, b)
        self.text = (self.text[:a] + self.text[b:])
        self.caret = a
        self.sel_start = self.caret
        self.sel_end = self.caret
        self._clamp_all()

    def insert(self, s: str) -> None:
        if not s:
            return
        if self.has_selection():
            a, b = int(self.sel_start), int(self.sel_end)
            a, b = min(a, b), max(a, b)
            self.text = self.text[:a] + s + self.text[b:]
            self.caret = a + len(s)
            self.sel_start = self.caret
            self.sel_end = self.caret
        else:
            a = int(self.caret)
            self.text = self.text[:a] + s + self.text[a:]
            self.caret = a + len(s)
        self._clamp_all()

    def backspace(self) -> None:
        if self.has_selection():
            self._delete_range(self.sel_start, self.sel_end)
            return
        if self.caret <= 0:
            return
        a = int(self.caret) - 1
        self.text = self.text[:a] + self.text[a + 1:]
        self.caret = a
        self.clear_selection()
        self._clamp_all()

    def delete(self) -> None:
        if self.has_selection():
            self._delete_range(self.sel_start, self.sel_end)
            return
        if self.caret >= len(self.text):
            return
        a = int(self.caret)
        self.text = self.text[:a] + self.text[a + 1:]
        self.clear_selection()
        self._clamp_all()

    def move_left(self) -> None:
        if self.caret > 0:
            self.caret -= 1
        self.clear_selection()
        self._clamp_all()

    def move_right(self) -> None:
        if self.caret < len(self.text):
            self.caret += 1
        self.clear_selection()
        self._clamp_all()

    def select_left(self) -> None:
        # Start a new selection or extend the existing one to the left by one.
        if not self.has_selection():
            anchor = int(self.caret)
            if self.caret > 0:
                self.caret -= 1
            # Anchor stays at original caret, active end follows caret
            self.sel_start = anchor
            self.sel_end = self.caret
        else:
            prev = int(self.caret)
            if self.caret > 0:
                self.caret -= 1
            # Move the side that previously matched the caret
            if prev == int(self.sel_start):
                self.sel_start = self.caret
            else:
                self.sel_end = self.caret
        self._clamp_all()

    def select_right(self) -> None:
        # Start a new selection or extend the existing one to the right by one.
        if not self.has_selection():
            anchor = int(self.caret)
            if self.caret < len(self.text):
                self.caret += 1
            # Anchor at original caret; active end follows caret
            self.sel_start = anchor
            self.sel_end = self.caret
        else:
            prev = int(self.caret)
            if self.caret < len(self.text):
                self.caret += 1
            # Move the side that previously matched the caret
            if prev == int(self.sel_end):
                self.sel_end = self.caret
            else:
                self.sel_start = self.caret
        self._clamp_all()

__all__ = ["SelectEditor"]
