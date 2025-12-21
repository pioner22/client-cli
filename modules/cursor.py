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

    def __init__(self, shape: Optional[int] = None):
        # Cursor style handling via DECSCUSR (CSI Ps q):
        # 1/2 block, 3/4 underline, 5/6 bar. We'll default to steady thin bar (6).
        try:
            from os import getenv
            env_shape = getenv("CURSOR_SHAPE")
            env_style = getenv("CURSOR_STYLE", "bar").strip().lower()  # block|underline|bar
            env_blink = getenv("CURSOR_BLINK")
        except Exception:
            env_shape = None
            env_style = "bar"
            env_blink = None
        if env_shape is not None:
            try:
                shape = int(env_shape)
            except Exception:
                shape = None
        self._shape = 2 if shape not in (0, 1) else int(shape)
        # Map style to steady code; blinking handled by caller if desired
        # Blinking preference: default to blinking on macOS terminals
        try:
            prefer_blink = sys.platform.startswith('darwin')
            if env_blink is not None:
                prefer_blink = str(env_blink).strip().lower() in ('1','true','yes','on')
        except Exception:
            prefer_blink = False
        # Map style to DECSCUSR code; pick blinking variant when preferred
        if env_style == 'block':
            self._style_code = 1 if prefer_blink else 2
        elif env_style == 'underline':
            self._style_code = 3 if prefer_blink else 4
        else:  # 'bar' or unknown
            self._style_code = 5 if prefer_blink else 6
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
            # Always align curses' internal cursor position when visible
            if vis > 0 and y is not None and x is not None:
                # Ensure thin style before showing
                self._apply_style()
                # Only toggle visibility if it changed (avoid flicker)
                if vis != last_vis:
                    try:
                        curses.curs_set(2 if vis >= 2 else 1)
                    except Exception:
                        # Fallback: show cursor via DEC private mode
                        try:
                            sys.stdout.write("\x1b[?25h")
                            sys.stdout.flush()
                        except Exception:
                            pass
                # Always move to keep curses internal (y, x) in sync
                try:
                    stdscr.move(int(y), int(x))
                except Exception:
                    pass
            else:
                if last_vis != 0:
                    try:
                        curses.curs_set(0)
                    except Exception:
                        try:
                            sys.stdout.write("\x1b[?25l")
                            sys.stdout.flush()
                        except Exception:
                            pass
            self._last = (vis, y if y is not None else -1, x if x is not None else -1)
        except Exception:
            pass
