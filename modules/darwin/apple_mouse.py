"""
Apple/macOS-specific mouse helpers for terminal UIs.

This module provides robust SGR mouse handling for macOS Terminal/iTerm2
where curses may not map mouse events to KEY_MOUSE properly and BUTTON1_*
constants may be zero. It focuses on the main UI (contacts/history pane)
selection and wheel scrolling.
"""
from __future__ import annotations

import re
import curses
from typing import Optional


def _compute_wheel_masks():
    """Return (wheel_up_mask, wheel_down_mask) for curses KEY_MOUSE bstate.

    Some macOS builds of curses expose only BUTTON4_* (no BUTTON5_*). In that
    case derive BUTTON5_* by shifting BUTTON4_* bit group by 6 bits.
    """
    def _sum(names: list[str]) -> int:
        m = 0
        for n in names:
            m |= getattr(curses, n, 0)
        return m

    up = _sum([
        'BUTTON4_PRESSED', 'BUTTON4_CLICKED', 'BUTTON4_RELEASED',
        'BUTTON4_DOUBLE_CLICKED', 'BUTTON4_TRIPLE_CLICKED',
    ])
    down = _sum([
        'BUTTON5_PRESSED', 'BUTTON5_CLICKED', 'BUTTON5_RELEASED',
        'BUTTON5_DOUBLE_CLICKED', 'BUTTON5_TRIPLE_CLICKED',
    ])
    if down == 0 and up:
        down = (up << 6)
    return up, down


_SGR_RE1 = re.compile(r"\x1b\[<(?P<b>\d+);(?P<x>\d+);(?P<y>\d+)(?P<t>[Mm])")
_SGR_RE2 = re.compile(r"\x1b\[(?P<b>\d+);(?P<x>\d+);(?P<y>\d+)(?P<t>[Mm])")
_X10_RE  = re.compile(r"\x1b\[M(?P<b>.)(?P<x>.)(?P<y>.)")


def _parse_mouse_event(ch: str):
    if not isinstance(ch, str) or not ch.startswith("\x1b["):
        return None
    m = _SGR_RE1.match(ch) or _SGR_RE2.match(ch)
    if m:
        try:
            cb = int(m.group("b"))
            mx = int(m.group("x")) - 1
            my = int(m.group("y")) - 1
            is_press = m.group("t") == "M"
            return cb, mx, my, is_press
        except Exception:
            return None
    m2 = _X10_RE.match(ch)
    if not m2:
        return None
    b = ord(m2.group("b")) - 32
    x = ord(m2.group("x")) - 32
    y = ord(m2.group("y")) - 32
    return b, x - 1, y - 1, True


def _mouse_flags(cb: int) -> tuple[bool, bool, bool, bool, bool]:
    is_wheel = bool(cb & 0x40)
    is_wheel_up = is_wheel and ((cb & 1) == 0)
    is_motion = bool(cb & 0x20)
    is_left_btn = (cb & 0x40) == 0 and ((cb & 3) == 0)
    ctrl_mod = bool(cb & 16)
    return is_wheel, is_wheel_up, is_motion, is_left_btn, ctrl_mod


def _left_width(stdscr) -> int:
    try:
        _h, w = stdscr.getmaxyx()
        return max(20, min(30, w // 4))
    except Exception:
        return 20


def _handle_wheel_event(state, *, ctrl_mod: bool, is_wheel_up: bool) -> None:
    try:
        state.mouse_ctrl_pressed = ctrl_mod
        if ctrl_mod:
            _wheel_contacts(state, -1 if is_wheel_up else 1)
        else:
            step = 1
            if is_wheel_up:
                state.history_scroll += step
            else:
                state.history_scroll = max(0, state.history_scroll - step)
            _clamp_history(state)
    except Exception:
        pass


def _handle_contacts_click(stdscr, state, net, *, mx: int, my: int, left_w: int, is_left_btn: bool, is_press: bool, is_motion: bool) -> bool:
    in_left = 0 <= mx < left_w
    if not (in_left and is_left_btn and is_press and (not is_motion)):
        return False
    rows = _rows(state)
    if rows is None:
        return False
    start_y = 2
    vis_h = _vis_h(stdscr)
    cs = max(0, int(getattr(state, "contacts_scroll", 0)))
    max_rows = min(max(0, len(rows) - cs), vis_h)
    if start_y <= my < start_y + max_rows:
        local_idx = my - start_y
        idx = cs + local_idx
        if 0 <= idx < len(rows):
            state.selected_index = idx
            _clamp_selection(state, prefer="down")
            sel = _current_selected_id(state)
            if sel and (sel not in getattr(state, "groups", {})):
                try:
                    net.send({"type": "message_read", "peer": sel})
                except Exception:
                    pass
            state.history_scroll = 0
    return True


def _history_geometry(stdscr, left_w: int) -> tuple[int, int, int, int]:
    h, w = stdscr.getmaxyx()
    hist_y = 2
    hist_x = left_w + 2
    hist_w = w - hist_x - 2
    hist_h = max(1, h - 3 - 2)
    return hist_y, hist_x, hist_w, hist_h


def _copy_to_clipboard_text(text: str) -> bool:
    try:
        import __main__ as _main  # type: ignore

        fn = getattr(_main, "copy_to_clipboard", None)
        if callable(fn):
            return bool(fn(text))
    except Exception:
        return False
    return False


def _selection_cols_for_row(*, sy: int, y0: int, y1: int, x0: int, x1: int, hist_x: int, hist_w: int) -> tuple[int, int]:
    if sy == y0 and sy == y1:
        return max(0, x0 - hist_x), min(hist_w, x1 - hist_x + 1)
    if sy == y0:
        return max(0, x0 - hist_x), hist_w
    if sy == y1:
        return 0, min(hist_w, x1 - hist_x + 1)
    return 0, hist_w


def _collect_selected_history_lines(state, *, y0: int, y1: int, x0: int, x1: int, hist_y: int, hist_x: int, hist_w: int) -> list[str]:
    lines: list[str] = []
    for sy in range(y0, y1 + 1):
        idx = sy - hist_y
        if 0 <= idx < len(getattr(state, "last_lines", [])):
            line = state.last_lines[idx]
            c0, c1 = _selection_cols_for_row(sy=sy, y0=y0, y1=y1, x0=x0, x1=x1, hist_x=hist_x, hist_w=hist_w)
            if c0 < c1:
                lines.append(line[c0:c1].rstrip())
    return lines


def _finalize_history_selection(state, *, my: int, mx: int, hist_y: int, hist_x: int, hist_w: int, hist_h: int) -> None:
    try:
        y0, y1 = sorted([state.sel_anchor_y, my])
        x0, x1 = sorted([state.sel_anchor_x, mx])
        y0 = max(y0, hist_y)
        y1 = min(y1, hist_y + hist_h - 1)
        x0 = max(x0, hist_x)
        x1 = min(x1, hist_x + hist_w - 1)
        if y0 == y1 and x0 == x1:
            x0 = hist_x
            x1 = hist_x + hist_w - 1
        selection_lines = _collect_selected_history_lines(
            state,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            hist_y=hist_y,
            hist_x=hist_x,
            hist_w=hist_w,
        )
        text_to_copy = "\n".join(selection_lines).rstrip("\n")
        if text_to_copy:
            if _copy_to_clipboard_text(text_to_copy):
                state.status = "Скопировано в буфер обмена"
            else:
                state.status = "Выделение пусто или копирование недоступно"
    except Exception:
        pass
    state.select_active = False


def _handle_history_drag(
    state,
    *,
    my: int,
    mx: int,
    is_left_btn: bool,
    is_press: bool,
    is_motion: bool,
    hist_y: int,
    hist_x: int,
    hist_w: int,
    hist_h: int,
) -> Optional[bool]:
    if not is_left_btn:
        return None
    if is_motion and getattr(state, "select_active", False):
        state.sel_cur_y, state.sel_cur_x = my, mx
        return True
    if is_press and not getattr(state, "select_active", False):
        state.select_active = True
        state.sel_anchor_y, state.sel_anchor_x = my, mx
        state.sel_cur_y, state.sel_cur_x = my, mx
        return True
    if (not is_press) and getattr(state, "select_active", False):
        _finalize_history_selection(state, my=my, mx=mx, hist_y=hist_y, hist_x=hist_x, hist_w=hist_w, hist_h=hist_h)
        return True
    return None


def _mark_history_as_read(state, net) -> None:
    try:
        sel = _current_selected_id(state)
        if sel and (sel not in getattr(state, "groups", {})):
            net.send({"type": "message_read", "peer": sel})
            state.unread[sel] = 0
    except Exception:
        pass


def _handle_history_area(stdscr, state, net, *, mx: int, my: int, left_w: int, is_left_btn: bool, is_press: bool, is_motion: bool) -> bool:
    hist_y, hist_x, hist_w, hist_h = _history_geometry(stdscr, left_w)
    in_hist = hist_y <= my < hist_y + hist_h and hist_x <= mx < hist_x + hist_w
    if not in_hist:
        return False
    drag = _handle_history_drag(
        state,
        my=my,
        mx=mx,
        is_left_btn=is_left_btn,
        is_press=is_press,
        is_motion=is_motion,
        hist_y=hist_y,
        hist_x=hist_x,
        hist_w=hist_w,
        hist_h=hist_h,
    )
    if drag is not None:
        return drag
    _mark_history_as_read(state, net)
    return True


def handle_sgr_mouse_main_ui(stdscr, state, net, ch: str) -> bool:
    """Process an SGR mouse sequence for the main UI.

    Returns True if the event was handled and should be consumed.
    """
    try:
        state.mouse_ctrl_pressed = False
    except Exception:
        pass
    parsed = _parse_mouse_event(ch)
    if parsed is None:
        return False
    mouse_on = bool(getattr(state, 'mouse_enabled', True))
    cb, mx, my, is_press = parsed
    is_wheel, is_wheel_up, is_motion, is_left_btn, ctrl_mod = _mouse_flags(cb)
    if is_wheel:
        _handle_wheel_event(state, ctrl_mod=ctrl_mod, is_wheel_up=is_wheel_up)
        return True
    left_w = _left_width(stdscr)
    if not mouse_on:
        return False
    if _handle_contacts_click(
        stdscr,
        state,
        net,
        mx=mx,
        my=my,
        left_w=left_w,
        is_left_btn=is_left_btn,
        is_press=is_press,
        is_motion=is_motion,
    ):
        return True
    if _handle_history_area(
        stdscr,
        state,
        net,
        mx=mx,
        my=my,
        left_w=left_w,
        is_left_btn=is_left_btn,
        is_press=is_press,
        is_motion=is_motion,
    ):
        return True
    return False


def _wheel_contacts(state, delta: int) -> None:
    try:
        rows = _rows(state) or []
        vis = int(getattr(state, 'last_left_h', 10) or 10)
        max_start = max(0, len(rows) - max(0, vis))
        cs = max(0, int(getattr(state, 'contacts_scroll', 0)))
        cs = max(0, min(max_start, cs + delta))
        state.contacts_scroll = cs
        _clamp_selection(state, prefer='down')
    except Exception:
        pass


def _clamp_history(state) -> None:
    try:
        total = int(getattr(state, 'last_history_lines_count', 0))
        hist_h = int(getattr(state, 'last_hist_h', 0))
        if total <= 0 or hist_h <= 0:
            # Не знаем высоту или количество строк — не обнуляем скролл, только помечаем грязным
            try:
                state.history_dirty = True
            except Exception:
                pass
            return
        max_scroll = max(0, total - max(1, hist_h))
    except Exception:
        max_scroll = 0
    try:
        hs = int(getattr(state, 'history_scroll', 0))
    except Exception:
        hs = 0
    if max_scroll > 0:
        hs = max(0, min(max_scroll, hs))
    try:
        state.history_scroll = hs
        state.history_dirty = True
    except Exception:
        pass


# Helper adapters (avoid importing client internals here)
def _rows(state):
    # Prefer bound helper if the client attached one
    try:
        return state.build_contact_rows()  # type: ignore[attr-defined]
    except AttributeError:
        pass
    try:
        import __main__ as _main  # type: ignore
    except Exception:
        return None
    try:
        fn = getattr(_main, "build_contact_rows", None)
        if callable(fn):
            return fn(state)
    except Exception:
        return None


def _clamp_selection(state, prefer: str = 'down') -> None:
    try:
        import __main__ as _main  # type: ignore
        fn = getattr(_main, "clamp_selection", None)
        if callable(fn):
            fn(state, prefer=prefer)
            return
    except Exception:
        # Best effort: bounds only
        rows = _rows(state)
        if rows is None:
            return
        try:
            if not rows:
                state.selected_index = 0
                return
            i = int(getattr(state, 'selected_index', 0))
            if i < 0:
                i = 0
            if i >= len(rows):
                i = len(rows) - 1
            state.selected_index = i
        except Exception:
            pass


def _current_selected_id(state):
    try:
        import __main__ as _main  # type: ignore
        fn = getattr(_main, "current_selected_id", None)
        if callable(fn):
            return fn(state)
    except Exception:
        return None


def _vis_h(stdscr) -> int:
    try:
        h, w = stdscr.getmaxyx()
        return max(0, h - 2 - 2)
    except Exception:
        return 10


def _keep_selection_visible(state, vis_h: int, start: int) -> None:
    try:
        sel = int(getattr(state, 'selected_index', 0))
        window = max(1, vis_h)
        if sel < start:
            state.selected_index = start
        elif sel >= start + window:
            state.selected_index = start + window - 1
    except Exception:
        pass
