from __future__ import annotations

"""
Chat input handling helpers shared by the TUI.

Provides a thin abstraction over the raw `state.input_*` fields so that
all editing operations (insert/delete/move/paste) share the same rules.
Also keeps the caret/cursor blink timer in sync.
"""

from dataclasses import dataclass
import time
from typing import Optional

from .selection_editor import SelectEditor  # type: ignore
from .text_editor import TextEditor  # type: ignore


CURSOR_BLINK_INTERVAL = 0.55  # seconds; exported for the renderer


@dataclass
class ChatInputSnapshot:
    text: str
    caret: int
    sel_start: int
    sel_end: int


def _snapshot(state) -> ChatInputSnapshot:
    text = getattr(state, 'input_buffer', '') or ''
    try:
        caret = int(getattr(state, 'input_caret', len(text)))
    except Exception:
        caret = len(text)
    caret = max(0, min(len(text), caret))
    try:
        sel_start = int(getattr(state, 'input_sel_start', caret))
    except Exception:
        sel_start = caret
    try:
        sel_end = int(getattr(state, 'input_sel_end', caret))
    except Exception:
        sel_end = caret
    sel_start = max(0, min(len(text), sel_start))
    sel_end = max(0, min(len(text), sel_end))
    return ChatInputSnapshot(text, caret, sel_start, sel_end)


def _mark_activity(state) -> None:
    try:
        state.input_cursor_visible = True
        state.input_cursor_last_toggle = time.time()
    except Exception:
        pass


def _apply(state, snap: ChatInputSnapshot) -> None:
    state.input_buffer = snap.text
    state.input_caret = snap.caret
    state.input_sel_start = snap.sel_start
    state.input_sel_end = snap.sel_end
    _mark_activity(state)


def set_text(state, text: str, caret_at_end: bool = True) -> None:
    text = text or ''
    caret = len(text) if caret_at_end else 0
    snap = ChatInputSnapshot(text, caret, caret, caret)
    _apply(state, snap)


def clear_selection(state) -> None:
    snap = _snapshot(state)
    snap.sel_start = snap.caret
    snap.sel_end = snap.caret
    _apply(state, snap)


def insert_text(state, data: str) -> None:
    if not data:
        return
    snap = _snapshot(state)
    normalized = data.replace('\r\n', '\n').replace('\r', '\n')
    sel = SelectEditor(snap.text, snap.caret, snap.sel_start, snap.sel_end)
    sel.insert(normalized)
    _apply(state, ChatInputSnapshot(sel.text, sel.caret, sel.sel_start, sel.sel_end))


def insert_newline(state) -> None:
    insert_text(state, '\n')


def backspace(state) -> None:
    snap = _snapshot(state)
    sel = SelectEditor(snap.text, snap.caret, snap.sel_start, snap.sel_end)
    sel.backspace()
    _apply(state, ChatInputSnapshot(sel.text, sel.caret, sel.sel_start, sel.sel_end))


def delete_forward(state) -> None:
    snap = _snapshot(state)
    sel = SelectEditor(snap.text, snap.caret, snap.sel_start, snap.sel_end)
    sel.delete()
    _apply(state, ChatInputSnapshot(sel.text, sel.caret, sel.sel_start, sel.sel_end))


def move_left(state) -> None:
    snap = _snapshot(state)
    if snap.sel_start != snap.sel_end:
        caret = min(snap.sel_start, snap.sel_end)
    else:
        caret = max(0, snap.caret - 1)
    _apply(state, ChatInputSnapshot(snap.text, caret, caret, caret))


def move_right(state) -> None:
    snap = _snapshot(state)
    if snap.sel_start != snap.sel_end:
        caret = max(snap.sel_start, snap.sel_end)
    else:
        caret = min(len(snap.text), snap.caret + 1)
    _apply(state, ChatInputSnapshot(snap.text, caret, caret, caret))


def move_up(state) -> None:
    snap = _snapshot(state)
    ed = TextEditor(snap.text, snap.caret)
    ed.move_up()
    caret = ed.caret
    _apply(state, ChatInputSnapshot(snap.text, caret, caret, caret))


def move_down(state) -> None:
    snap = _snapshot(state)
    ed = TextEditor(snap.text, snap.caret)
    ed.move_down()
    caret = ed.caret
    _apply(state, ChatInputSnapshot(snap.text, caret, caret, caret))


def move_word_left(state) -> None:
    snap = _snapshot(state)
    ed = TextEditor(snap.text, snap.caret)
    ed.move_word_left()
    caret = ed.caret
    _apply(state, ChatInputSnapshot(snap.text, caret, caret, caret))


def move_word_right(state) -> None:
    snap = _snapshot(state)
    ed = TextEditor(snap.text, snap.caret)
    ed.move_word_right()
    caret = ed.caret
    _apply(state, ChatInputSnapshot(snap.text, caret, caret, caret))


def delete_word_left(state) -> None:
    snap = _snapshot(state)
    ed = TextEditor(snap.text, snap.caret)
    ed.delete_word_left()
    _apply(state, ChatInputSnapshot(ed.text, ed.caret, ed.caret, ed.caret))


def delete_word_right(state) -> None:
    snap = _snapshot(state)
    ed = TextEditor(snap.text, snap.caret)
    ed.delete_word_right()
    _apply(state, ChatInputSnapshot(ed.text, ed.caret, ed.caret, ed.caret))


def delete_to_line_end(state) -> None:
    snap = _snapshot(state)
    ed = TextEditor(snap.text, snap.caret)
    ed.delete_to_line_end()
    _apply(state, ChatInputSnapshot(ed.text, ed.caret, ed.caret, ed.caret))


def move_line_start(state) -> None:
    snap = _snapshot(state)
    text, caret = snap.text, snap.caret
    pos = text.rfind('\n', 0, caret)
    caret = (pos + 1) if pos >= 0 else 0
    _apply(state, ChatInputSnapshot(text, caret, caret, caret))


def move_line_end(state) -> None:
    snap = _snapshot(state)
    text, caret = snap.text, snap.caret
    pos = text.find('\n', caret)
    caret = pos if pos >= 0 else len(text)
    _apply(state, ChatInputSnapshot(text, caret, caret, caret))


def ensure_cursor_tick(state, now: Optional[float] = None, force_visible: bool = False) -> None:
    """Toggle cursor blink; called from the renderer."""
    if now is None:
        now = time.time()
    if force_visible:
        state.input_cursor_visible = True
        state.input_cursor_last_toggle = now
        return
    last = getattr(state, 'input_cursor_last_toggle', 0.0) or 0.0
    visible = getattr(state, 'input_cursor_visible', True)
    if now - last >= CURSOR_BLINK_INTERVAL:
        state.input_cursor_visible = not visible
        state.input_cursor_last_toggle = now


__all__ = [
    'ChatInputSnapshot',
    'CURSOR_BLINK_INTERVAL',
    'set_text',
    'clear_selection',
    'insert_text',
    'insert_newline',
    'backspace',
    'delete_forward',
    'move_left',
    'move_right',
    'move_up',
    'move_down',
    'move_word_left',
    'move_word_right',
    'delete_word_left',
    'delete_word_right',
    'delete_to_line_end',
    'move_line_start',
    'move_line_end',
    'ensure_cursor_tick',
]
