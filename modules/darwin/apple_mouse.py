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


def handle_sgr_mouse_main_ui(stdscr, state, net, ch: str) -> bool:
    """Process an SGR mouse sequence for the main UI.

    Returns True if the event was handled and should be consumed.
    """
    if not isinstance(ch, str) or not ch.startswith("\x1b["):
        return False
    try:
        state.mouse_ctrl_pressed = False
    except Exception:
        pass
    mouse_on = bool(getattr(state, 'mouse_enabled', True))
    m = _SGR_RE1.match(ch) or _SGR_RE2.match(ch)
    if m:
        try:
            Cb = int(m.group('b'))
            mx = int(m.group('x')) - 1
            my = int(m.group('y')) - 1
            is_press = (m.group('t') == 'M')
        except Exception:
            return False
    else:
        m2 = _X10_RE.match(ch)
        if not m2:
            return False
        b = ord(m2.group('b')) - 32
        x = ord(m2.group('x')) - 32
        y = ord(m2.group('y')) - 32
        Cb, mx, my = b, x - 1, y - 1
        # X10 reports only press as 'M'; in our regex we don't capture t, so treat as press
        is_press = True
    # Прямая интерпретация SGR/X10 без опоры на curses BUTTON* (на macOS они могут быть 0)
    is_wheel = bool(Cb & 0x40)
    is_wheel_up = is_wheel and ((Cb & 1) == 0)
    is_wheel_down = is_wheel and ((Cb & 1) == 1)
    is_motion = bool(Cb & 0x20)
    is_left_btn = (Cb & 0x40) == 0 and ((Cb & 3) == 0)
    ctrl_mod = bool(Cb & 16)
    # Колёсико: плавный скролл (шаг 1), Ctrl+wheel — контакты
    if is_wheel:
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
        return True
    # Geometry (match client)
    try:
        h, w = stdscr.getmaxyx()
        left_w = max(20, min(30, w // 4))
    except Exception:
        left_w = 20
    in_left = (0 <= mx < left_w)
    # Wheel handled upstream
    # Если мышь в клиенте отключена — колёсико обработали, клики отдаём терминалу
    if not mouse_on:
        return False
    # Left-click selection in contacts list
    if in_left and is_left_btn and is_press and (not is_motion):
        rows = _rows(state)
        if rows is None:
            return False
        start_y = 2
        vis_h = _vis_h(stdscr)
        cs = max(0, int(getattr(state, 'contacts_scroll', 0)))
        max_rows = min(max(0, len(rows) - cs), vis_h)
        if start_y <= my < start_y + max_rows:
            local_idx = my - start_y
            idx = cs + local_idx
            if 0 <= idx < len(rows):
                state.selected_index = idx
                _clamp_selection(state, prefer='down')
                sel = _current_selected_id(state)
                if sel and (sel not in getattr(state, 'groups', {})):
                    try:
                        net.send({"type": "message_read", "peer": sel})
                    except Exception:
                        pass
                state.history_scroll = 0
        return True
    # History area activation
    hist_y = 2
    hist_x = left_w + 2
    hist_w = w - hist_x - 2
    hist_h = max(1, h - 3 - 2)
    if hist_y <= my < hist_y + hist_h and hist_x <= mx < hist_x + hist_w:
        # Прямоугольное выделение в истории (drag)
        if is_left_btn:
            if is_motion and getattr(state, 'select_active', False):
                state.sel_cur_y, state.sel_cur_x = my, mx
                return True
            if is_press and not getattr(state, 'select_active', False):
                state.select_active = True
                state.sel_anchor_y, state.sel_anchor_x = my, mx
                state.sel_cur_y, state.sel_cur_x = my, mx
                return True
            if (not is_press) and getattr(state, 'select_active', False):
                # Release: finalize copy
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
                    selection_lines: list[str] = []
                    # We need the same mapping as client: state.last_lines holds visible rows
                    for sy in range(y0, y1 + 1):
                        idx = sy - hist_y
                        if 0 <= idx < len(getattr(state, 'last_lines', [])):
                            line = state.last_lines[idx]
                            if sy == y0 and sy == y1:
                                c0 = max(0, x0 - hist_x)
                                c1 = min(hist_w, x1 - hist_x + 1)
                            elif sy == y0:
                                c0 = max(0, x0 - hist_x)
                                c1 = hist_w
                            elif sy == y1:
                                c0 = 0
                                c1 = min(hist_w, x1 - hist_x + 1)
                            else:
                                c0 = 0
                                c1 = hist_w
                            if c0 < c1:
                                selection_lines.append(line[c0:c1].rstrip())
                    text_to_copy = "\n".join(selection_lines).rstrip("\n")
                    if text_to_copy:
                        ok = False
                        try:
                            import __main__ as _main  # type: ignore
                            fn = getattr(_main, "copy_to_clipboard", None)
                            if callable(fn):
                                ok = bool(fn(text_to_copy))
                        except Exception:
                            ok = False
                        if ok:
                            state.status = "Скопировано в буфер обмена"
                        else:
                            state.status = "Выделение пусто или копирование недоступно"
                    state.select_active = False
                except Exception:
                    state.select_active = False
                return True
        # Обычный клик по истории — просто снять непрочитанные
        try:
            sel = _current_selected_id(state)
            if sel and (sel not in getattr(state, 'groups', {})):
                net.send({"type": "message_read", "peer": sel})
                state.unread[sel] = 0
        except Exception:
            pass
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
