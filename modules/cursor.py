from __future__ import annotations

import curses
import sys
from typing import Optional, Tuple


class CursorController:
    """Small helper to control terminal (hardware) cursor with minimal flicker.

    Usage per frame:
      ctrl.begin(stdscr)
      ctrl.want(y, x, vis=2)  # as many times as needed; last wins
      ctrl.apply(stdscr)      # once at the end, after drawing overlays
    """

    @staticmethod
    def _read_env_prefs() -> tuple[Optional[str], str, bool]:
        try:
            from os import getenv

            env_shape = getenv("CURSOR_SHAPE")
            env_style = getenv("CURSOR_STYLE", "bar").strip().lower()
            env_blink = getenv("CURSOR_BLINK")
        except Exception:
            return None, "bar", False
        prefer_blink = sys.platform.startswith("darwin")
        if env_blink is not None:
            prefer_blink = str(env_blink).strip().lower() in ("1", "true", "yes", "on")
        return env_shape, env_style, prefer_blink

    @staticmethod
    def _resolve_shape(shape: Optional[int], env_shape: Optional[str]) -> int:
        if env_shape is not None:
            try:
                shape = int(env_shape)
            except Exception:
                shape = None
        return 2 if shape not in (0, 1) else int(shape)

    @staticmethod
    def _style_code_for(style: str, prefer_blink: bool) -> int:
        # Cursor style handling via DECSCUSR (CSI Ps q):
        # 1/2 block, 3/4 underline, 5/6 bar. Default: steady thin bar (6).
        if style == "block":
            return 1 if prefer_blink else 2
        if style == "underline":
            return 3 if prefer_blink else 4
        return 5 if prefer_blink else 6

    @staticmethod
    def _show_cursor(vis: int) -> None:
        try:
            curses.curs_set(2 if vis >= 2 else 1)
            return
        except Exception:
            pass
        try:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()
        except Exception:
            pass

    @staticmethod
    def _hide_cursor() -> None:
        try:
            curses.curs_set(0)
            return
        except Exception:
            pass
        try:
            sys.stdout.write("\x1b[?25l")
            sys.stdout.flush()
        except Exception:
            pass

    @staticmethod
    def _move_cursor(stdscr, y: int, x: int) -> None:
        try:
            stdscr.move(int(y), int(x))
        except Exception:
            pass

    def __init__(self, shape: Optional[int] = None):
        env_shape, env_style, prefer_blink = self._read_env_prefs()
        self._shape = self._resolve_shape(shape, env_shape)
        self._style_code = self._style_code_for(env_style, prefer_blink)
        self._style_last = None  # last applied DECSCUSR code
        self._want: Tuple[int, Optional[int], Optional[int]] = (0, None, None)
        self._last: tuple[int, int, int] = (0, -1, -1)

    def _apply_style(self) -> None:
        # Apply DECSCUSR code (if supported by terminal). Use stdout directly to avoid drawing artifacts.
        try:
            code = int(self._style_code)
            if self._style_last != code:
                sys.stdout.write(f"\x1b[{code} q")
                sys.stdout.flush()
                self._style_last = code
        except Exception:
            pass

    def begin(self, stdscr) -> None:
        # Start of frame: reset desired state only (avoid toggling hardware per frame)
        self._want = (0, None, None)

    def want(self, y: int, x: int, vis: int = 1) -> None:
        try:
            self._want = (max(0, int(vis)), int(y), int(x))
        except Exception:
            self._want = (0, None, None)

    def apply(self, stdscr) -> None:
        try:
            vis, y, x = self._want
            last_vis, last_y, last_x = self._last
            if vis > 0 and y is not None and x is not None:
                self._apply_style()
                if vis != last_vis:
                    self._show_cursor(vis)
                self._move_cursor(stdscr, int(y), int(x))
            elif last_vis != 0:
                self._hide_cursor()
            self._last = (vis, y if y is not None else -1, x if x is not None else -1)
        except Exception:
            pass
