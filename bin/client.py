#!/usr/bin/env python3
from __future__ import annotations
import curses
import curses.textpad
import json
import logging
import logging.handlers
import os
import queue
import select
import socket
import ssl
import sys
import threading
import time
import locale
import ipaddress
from dataclasses import dataclass, field
from pathlib import Path
import base64
import functools
from typing import Dict, List, Optional, Tuple, Set, Callable
import urllib.request
import urllib.error
from urllib.parse import urlparse
import hashlib
import subprocess
import shutil
import tarfile
import tempfile
import re
import unicodedata

# Debug logging toggle (CLIENT_DEBUG_LOG=1 to enable verbose instrumentation)
DEBUG_LOG_ENABLED = str(os.environ.get('CLIENT_DEBUG_LOG', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
KEYLOG_ENABLED = str(os.environ.get('CLIENT_KEYLOG', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
DEBUG_LOGGER = logging.getLogger('client.debug')
def _dbg(msg: str) -> None:
    if DEBUG_LOG_ENABLED:
        try:
            DEBUG_LOGGER.debug(msg)
        except Exception:
            pass

def _term_write(seq: str, *, tmux_passthrough: bool = False) -> None:
    """Write a raw terminal control sequence.

    In some tmux setups xterm mouse enable sequences may be filtered. When
    tmux_passthrough=True we also emit a DCS passthrough so the outer terminal
    receives the sequence.
    """
    def _write_bytes(buf: bytes) -> None:
        # Prefer /dev/tty so control sequences reach the controlling terminal even
        # if stdout is wrapped (script/pty) or buffered.
        try:
            fd = os.open("/dev/tty", os.O_WRONLY | getattr(os, "O_NOCTTY", 0))
            try:
                os.write(fd, buf)
            finally:
                os.close(fd)
            return
        except Exception:
            pass
        try:
            out = sys.stdout
            if hasattr(out, "buffer"):
                out.buffer.write(buf)  # type: ignore[attr-defined]
                out.flush()
                return
        except Exception:
            pass
        try:
            out = getattr(sys, "__stdout__", None)
            if out is not None:
                if hasattr(out, "buffer"):
                    out.buffer.write(buf)  # type: ignore[attr-defined]
                else:
                    out.write(buf.decode("utf-8", "ignore"))
                out.flush()
        except Exception:
            pass

    try:
        if not seq:
            return
        b = seq.encode("utf-8", "ignore")
        _write_bytes(b)
        if tmux_passthrough and os.environ.get('TMUX'):
            payload = seq.replace("\x1b", "\x1b\x1b").encode("utf-8", "ignore")
            _write_bytes(b"\x1bPtmux;" + payload + b"\x1b\\")
    except Exception:
        pass

# Best-effort marker so F12 debug can show whether we attempted to arm mouse tracking.
__MOUSE_ARMED__ = False
__TMUX_MOUSE_PREV__ = None
__TMUX_MOUSE_AUTO_ENABLED__ = False

# Throttling helpers for verbose debug logs
_LAST_DRAW_SIG = None
_LAST_HISTORY_SIG = None

def _log_action(msg: str) -> None:
    """Higher-level action tracing (hotkeys, modes, sends)."""
    _dbg(f"[action] {msg}")

def _key_repr(ch) -> str:
    try:
        if isinstance(ch, str):
            if ch == '\n':
                return 'KEY=<LF>'
            if ch == '\x1b':
                return 'KEY=<ESC>'
            if ch == '\t':
                return 'KEY=<TAB>'
            return f"KEY=str({repr(ch)})"
        return f"KEY=int({ch})"
    except Exception:
        return f"KEY=unknown({ch})"

def _state_sig(state: ClientState) -> str:
    try:
        sel = current_selected_id(state)
        conv_len = len(state.conversations.get(sel, [])) if sel else 0
        return f"sel={sel} conv_len={conv_len} search={state.search_mode} modal={getattr(state,'action_menu_mode',False) or getattr(state,'search_action_mode',False)}"
    except Exception:
        return "state_sig_unavailable"
CLIENT_BIN_DIR = Path(__file__).resolve().parent
if CLIENT_BIN_DIR.name == 'bin':
    CLIENT_DIR = CLIENT_BIN_DIR.parent
else:
    CLIENT_DIR = CLIENT_BIN_DIR
MODULE_DIR = CLIENT_DIR / 'modules'
MODULE_PKG_DIR = MODULE_DIR / 'module'
CONFIG_DIR = CLIENT_DIR / 'config'
VAR_DIR = CLIENT_DIR / 'var'
RUNTIME_DIR = VAR_DIR / 'runtime'
DLL_DIR = RUNTIME_DIR / 'dll'
LOG_DIR = VAR_DIR / 'log'
USERS_DIR = VAR_DIR / 'users'
HISTORY_DIR = VAR_DIR / 'history'
FILES_DIR = VAR_DIR / 'files'

# Choose writable config/data dirs (macOS app bundle может быть read-only)
def _ensure_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".touch"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _select_dir(base: Path, *, env_var: str, fallbacks: List[Path]) -> Path:
    override = os.environ.get(env_var)
    if override:
        cand = Path(override).expanduser()
        if _ensure_writable_dir(cand):
            return cand
    if _ensure_writable_dir(base):
        return base
    for cand in fallbacks:
        if _ensure_writable_dir(cand):
            return cand
    return base


_home = Path.home()
_mac_app_support = _home / "Library" / "Application Support" / "Yagodka"
# macOS: keep config under Application Support; other OS — ~/.config/yagodka
CONFIG_DIR = _select_dir(
    CONFIG_DIR,
    env_var="CLIENT_CONFIG_DIR",
    fallbacks=[
        _mac_app_support / "config",
        _home / ".config" / "yagodka",
        _home / ".yagodka" / "config",
    ],
)
# Data (logs/history/runtime) — same fallback root
VAR_DIR = _select_dir(
    VAR_DIR,
    env_var="CLIENT_VAR_DIR",
    fallbacks=[
        _mac_app_support / "var",
        _home / ".local" / "share" / "yagodka",
        _home / ".yagodka" / "var",
    ],
)
RUNTIME_DIR = VAR_DIR / 'runtime'
DLL_DIR = RUNTIME_DIR / 'dll'
LOG_DIR = VAR_DIR / 'log'
USERS_DIR = VAR_DIR / 'users'
HISTORY_DIR = VAR_DIR / 'history'
FILES_DIR = VAR_DIR / 'files'

# Ensure modules/ is importable
try:
    import sys as _sys
    def _prepend_path(p: Path) -> None:
        if p.exists():
            sp = str(p)
            if sp not in _sys.path:
                _sys.path.insert(0, sp)
    # Add client root so `modules.*` works when running from bin/
    _prepend_path(CLIENT_DIR)
    # Prefer packaged module/ layout if present (as in dist bundles)
    _prepend_path(MODULE_PKG_DIR)
    # Fall back to flat modules/ and dll/ copies
    _prepend_path(MODULE_DIR)
    _prepend_path(DLL_DIR)
except Exception:
    pass

APPLE_MOUSE_HANDLER: Optional[Callable] = None
APPLE_MOUSE_AVAILABLE: Optional[bool] = None

# Formatting helpers (graceful fallback if module not available)
try:
    from modules.formatting import apply_format as apply_text_format  # type: ignore
except Exception:
    def _word_bounds_fb(text: str, sel_start: int, sel_end: int, caret: int):
        if sel_start != sel_end:
            a, b = sorted((sel_start, sel_end))
            return a, b
        n = len(text)
        caret = max(0, min(n, caret))
        l = caret
        while l > 0 and not text[l - 1].isspace():
            l -= 1
        r = caret
        while r < n and not text[r].isspace():
            r += 1
        if l == r:
            return 0, n
        return l, r

    def apply_text_format(kind: str, text: str, caret: int, sel_start: int, sel_end: int, link_text: str = "", link_url: str = ""):
        text = text or ""
        a, b = _word_bounds_fb(text, sel_start, sel_end, caret)
        segment = text[a:b]
        if kind == "bold":
            formatted = f"**{segment or 'текст'}**"
        elif kind == "link":
            t = (link_text or segment or link_url or "").strip()
            u = (link_url or "").strip()
            if not t and segment:
                t = segment
            if not t and u:
                t = u
            formatted = f"[{t or 'link'}]({u})" if u else t
        elif kind == "upper":
            formatted = (segment or text).upper()
        elif kind == "lower":
            formatted = (segment or text).lower()
        else:
            formatted = segment
        new_text = text[:a] + formatted + text[b:]
        new_caret = a + len(formatted)
        return type("FmtRes", (), {"text": new_text, "caret": new_caret})

# Toolbar actions: (key,label,kind)
FORMAT_ACTIONS = [
    ("B", "жирный", "bold"),
    ("U", "UPPER", "upper"),
    ("u", "lower", "lower"),
]

# Protocol types (shared). Provide a small fallback when modules/ is absent.
PROTOCOL_MODULE_FALLBACK = False
try:
    from modules.protocol import T  # type: ignore
except Exception:
    PROTOCOL_MODULE_FALLBACK = True
    _TFALLBACK_VALUES = {
        # Session / handshake
        "WELCOME": "welcome",
        "SESSION_REPLACED": "session_replaced",
        # Auth / register
        "AUTH": "auth",
        "AUTH_OK": "auth_ok",
        "AUTH_FAIL": "auth_fail",
        "REGISTER": "register",
        "REGISTER_OK": "register_ok",
        "REGISTER_FAIL": "register_fail",
        # Generic
        "ERROR": "error",
        "PING": "ping",
        "PONG": "pong",
        # Contacts / presence / roster
        "CONTACTS": "contacts",
        "CONTACT_JOINED": "contact_joined",
        "CONTACT_LEFT": "contact_left",
        "PRESENCE_UPDATE": "presence_update",
        "ROSTER": "roster",
        "ROSTER_FULL": "roster_full",
        "FRIENDS": "friends",
        "USERS": "users",
        # Profiles / prefs
        "PROFILE": "profile",
        "PROFILE_GET": "profile_get",
        "PROFILE_SET": "profile_set",
        "PROFILE_SET_RESULT": "profile_set_result",
        "PROFILE_UPDATED": "profile_updated",
        "PREFS": "prefs",
        "PREFS_GET": "prefs_get",
        "PREFS_SET": "prefs_set",
        "MUTE_SET": "mute_set",
        "MUTE_SET_RESULT": "mute_set_result",
        "BLOCK_SET": "block_set",
        "BLOCK_SET_RESULT": "block_set_result",
        # Messaging
        "SEND": "send",
        "MESSAGE": "message",
        "MESSAGE_DELIVERED": "message_delivered",
        "MESSAGE_QUEUED": "message_queued",
        "MESSAGE_BLOCKED": "message_blocked",
        "MESSAGE_READ": "message_read",
        "MESSAGE_READ_ACK": "message_read_ack",
        "UNREAD_COUNTS": "unread_counts",
        # Update
        "UPDATE_REQUIRED": "update_required",
        # History / search
        "HISTORY": "history",
        "HISTORY_RESULT": "history_result",
        "SEARCH": "search",
        "SEARCH_RESULT": "search_result",
        "LIST": "list",
        # Groups
        "GROUPS": "groups",
        "GROUP_CREATE": "group_create",
        "GROUP_CREATE_RESULT": "group_create_result",
        "GROUP_ADD": "group_add",
        "GROUP_ADD_RESULT": "group_add_result",
        "GROUP_LEAVE": "group_leave",
        "GROUP_LEAVE_RESULT": "group_leave_result",
        "GROUP_RENAME": "group_rename",
        "GROUP_RENAME_RESULT": "group_rename_result",
        "GROUP_DISBAND": "group_disband",
        "GROUP_DISBAND_RESULT": "group_disband_result",
        "GROUP_INFO": "group_info",
        "GROUP_INFO_RESULT": "group_info_result",
        "GROUP_ADDED": "group_added",
        "GROUP_UPDATED": "group_updated",
        "GROUP_REMOVED": "group_removed",
        "GROUP_INVITE": "group_invite",
        "GROUP_INVITE_RESULT": "group_invite_result",
        "GROUP_INVITE_RESPONSE": "group_invite_response",
        # Boards
        "BOARDS": "boards",
        "BOARD_CREATE": "board_create",
        "BOARD_CREATE_RESULT": "board_create_result",
        "BOARD_ADD": "board_add",
        "BOARD_ADD_RESULT": "board_add_result",
        "BOARD_REMOVE": "board_remove",
        "BOARD_REMOVE_RESULT": "board_remove_result",
        "BOARD_DISBAND": "board_disband",
        "BOARD_DISBAND_RESULT": "board_disband_result",
        "BOARD_INFO": "board_info",
        "BOARD_INFO_RESULT": "board_info_result",
        "BOARD_RENAME": "board_rename",
        "BOARD_RENAME_RESULT": "board_rename_result",
        "BOARD_SET_HANDLE": "board_set_handle",
        "BOARD_SET_HANDLE_RESULT": "board_set_handle_result",
        "BOARD_JOIN": "board_join",
        "BOARD_JOIN_RESULT": "board_join_result",
        "BOARD_LEAVE": "board_leave",
        "BOARD_LEAVE_RESULT": "board_leave_result",
        "BOARD_INVITE": "board_invite",
        "BOARD_INVITE_RESULT": "board_invite_result",
        "BOARD_ADDED": "board_added",
        "BOARD_UPDATED": "board_updated",
        "BOARD_REMOVED": "board_removed",
        # Authorization (friendship)
        "AUTHZ_REQUEST": "authz_request",
        "AUTHZ_RESPONSE": "authz_response",
        "AUTHZ_PENDING": "authz_pending",
        "AUTHZ_ACCEPTED": "authz_accepted",
        "AUTHZ_DECLINED": "authz_declined",
        "AUTHZ_CANCELLED": "authz_cancelled",
        # Files
        "FILE_OFFER": "file_offer",
        "FILE_OFFER_RESULT": "file_offer_result",
        "FILE_ACCEPT": "file_accept",
        "FILE_REJECT": "file_reject",
        "FILE_CHUNK": "file_chunk",
        "FILE_UPLOAD_COMPLETE": "file_upload_complete",
        "FILE_DOWNLOAD_BEGIN": "file_download_begin",
        "FILE_DOWNLOAD_COMPLETE": "file_download_complete",
        "FILE_ERROR": "file_error",
        "FILE_ACCEPT_NOTICE": "file_accept_notice",
        "FILE_RECEIVED": "file_received",
    }
    class T:  # mimic modules..T when shared module unavailable
        pass
    for _name, _value in _TFALLBACK_VALUES.items():
        setattr(T, _name, _value)
    del _name, _value, _TFALLBACK_VALUES

# Contact label fitting / display-width utilities (shared); fallback to local copy if unavailable
try:
    from modules.ui_utils import display_width, fit_contact_label, pad_to_width, right_truncate_to_width  # type: ignore
except Exception:
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
            return unicodedata.category(ch) == "Cf"
        except Exception:
            return False

    def _wcwidth(ch: str) -> int:
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

    def _truncate_to_width(s: str, width: int) -> str:
        if width <= 0:
            return ""
        out: List[str] = []
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

    def _right_truncate_to_width(s: str, width: int) -> str:
        if width <= 0:
            return ""
        out_rev: List[str] = []
        cols = 0
        pending: List[str] = []
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

    # Public alias for code that expects the shared helper name.
    right_truncate_to_width = _right_truncate_to_width

    def pad_to_width(s: str, width: int) -> str:
        if width <= 0:
            return ""
        trimmed = _truncate_to_width(s, width)
        cur = display_width(trimmed)
        if cur >= width:
            return trimmed
        return trimmed + (" " * (width - cur))

    def fit_contact_label(s: str, width: int) -> str:
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
                if display_width(suffix) >= max(4, width):
                    return _truncate_to_width(s, max(0, width - 1)) + "…"
                avail = width - display_width(suffix) - 1
                if avail <= 0:
                    return "…" + _right_truncate_to_width(suffix, max(0, width - 1))
                prefix = s[:idx].rstrip()
                return _truncate_to_width(prefix, avail) + "…" + suffix
            return _truncate_to_width(s, max(0, width - 1)) + "…"
        except Exception:
            return s[:width]
PROFILE_MODULE_FALLBACK = False
try:
    from modules.profile import make_profile_set_payload, normalize_handle  # type: ignore
except Exception:
    PROFILE_MODULE_FALLBACK = True
    # Fallback for single-file distribution
    import re as _re
    def _collapse_spaces(text: str) -> str:
        return " ".join(text.split())
    def _normalize_display_name(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        name = _collapse_spaces(name.strip())
        if not name:
            return None
        return name[:64]
    _HANDLE_RE = _re.compile(r"^@[a-z0-9_]{3,16}$")
    def normalize_handle(handle: Optional[str]) -> Optional[str]:
        if handle is None:
            return None
        h = handle.strip().lower()
        if not h:
            return None
        if not h.startswith('@'):
            h = '@' + h
        base = _re.sub(r"[^a-z0-9_]", "", h[1:])
        return '@' + base
    def _validate_display_name(name: Optional[str]) -> Tuple[bool, Optional[str]]:
        if name is None:
            return True, None
        if not name.strip():
            return False, 'empty'
        if len(name.strip()) > 64:
            return False, 'too_long'
        return True, None
    def _validate_handle(handle: Optional[str]) -> Tuple[bool, Optional[str]]:
        if handle is None:
            return True, None
        return (True, None) if _HANDLE_RE.match(handle) else (False, 'handle_invalid')
    def make_profile_set_payload(display_name: Optional[str], handle: Optional[str]) -> dict:
        nd = _normalize_display_name(display_name)
        ok, err = _validate_display_name(nd)
        if not ok:
            raise ValueError(f"invalid display_name: {err}")
        nh = normalize_handle(handle)
        ok, err = _validate_handle(nh)
        if not ok:
            raise ValueError(f"invalid handle: {err}")
        payload: dict = {"type": "profile_set"}
        if nd is not None:
            payload['display_name'] = nd
        if nh is not None:
            payload['handle'] = nh
        return payload

# Cursor controller (external module with fallback)
CURSOR_MODULE_FALLBACK = False
try:
    from modules.cursor import CursorController  # type: ignore
except Exception:
    CURSOR_MODULE_FALLBACK = True
    class CursorController:  # fallback minimal controller
        def __init__(self, shape: int = 2):
            self._want = (0, None, None)  # vis, y, x
            self._last = (0, -1, -1)
            self._shape = 2 if shape not in (0, 1) else shape

        def begin(self, stdscr):
            try:
                curses.curs_set(0)
            except Exception:
                pass
            self._want = (0, None, None)

        def want(self, y: int, x: int, vis: int = 1):
            try:
                self._want = (max(0, int(vis)), int(y), int(x))
            except Exception:
                self._want = (0, None, None)

        def apply(self, stdscr):
            try:
                vis, y, x = self._want
                last_vis, last_y, last_x = self._last
                if (vis, y, x) != (last_vis, last_y, last_x):
                    if vis > 0 and y is not None and x is not None:
                        try:
                            curses.curs_set(2 if vis >= 2 else 1)
                        except Exception:
                            pass
                        try:
                            stdscr.move(int(y), int(x))
                        except Exception:
                            pass
                    else:
                        try:
                            curses.curs_set(0)
                        except Exception:
                            pass
                    self._last = (vis, y if y is not None else -1, x if x is not None else -1)
            except Exception:
                pass

CURSOR = CursorController()

# Optional import for status icons
STATUS_MODULE_FALLBACK = False
try:
    from modules.status import status_icon  # type: ignore
except Exception:
    STATUS_MODULE_FALLBACK = True
    def status_icon(is_online: bool) -> str:  # fallback
        return '●' if is_online else '○'

# Optional text editor helper for input field (caret-aware insert/move)
TEXT_EDITOR_FALLBACK = False
try:
    from modules.text_editor import TextEditor  # type: ignore
except Exception:
    TEXT_EDITOR_FALLBACK = True
    from dataclasses import dataclass as _dataclass
    @_dataclass
    class TextEditor:  # minimal fallback (Py<3.10 safe)
        text: str = ""
        caret: int = 0
        def clamp(self):
            self.caret = max(0, min(len(self.text), int(self.caret)))
        def set(self, text, caret=None):
            self.text = text or ""
            self.caret = int(self.caret if caret is None else caret)
            self.clamp()
        def insert(self, s):
            if not s: return
            self.text = self.text[: self.caret] + s + self.text[self.caret :]
            self.caret += len(s)
        def backspace(self):
            if self.caret <= 0: return
            self.text = self.text[: self.caret - 1] + self.text[self.caret :]
            self.caret -= 1
        def delete(self):
            if self.caret >= len(self.text): return
            self.text = self.text[: self.caret] + self.text[self.caret + 1 :]
        def move_left(self):
            if self.caret > 0: self.caret -= 1
        def move_right(self):
            if self.caret < len(self.text): self.caret += 1
        def _line_col(self):
            before = self.text[: self.caret]
            parts = before.split('\n')
            return (len(parts) - 1, len(parts[-1]) if parts else 0)
        def move_up(self):
            li, col = self._line_col()
            if li <= 0: return
            lines = self.text.split('\n')
            new_col = min(col, len(lines[li-1]))
            self.caret = sum(len(l)+1 for l in lines[:li-1]) + new_col
        def move_down(self):
            li, col = self._line_col()
            lines = self.text.split('\n')
            if li >= len(lines)-1: return
            new_col = min(col, len(lines[li+1]))
            self.caret = sum(len(l)+1 for l in lines[:li+1]) + new_col
        def move_word_left(self):
            i = self.caret
            s = self.text
            while i > 0 and s[i - 1].isspace():
                i -= 1
            while i > 0 and not s[i - 1].isspace():
                i -= 1
            self.caret = i
        def move_word_right(self):
            n = len(self.text)
            i = self.caret
            s = self.text
            while i < n and not s[i].isspace():
                i += 1
            while i < n and s[i].isspace():
                i += 1
            self.caret = i
        def delete_word_left(self):
            j = self.caret
            s = self.text
            i = j
            while i > 0 and s[i - 1].isspace():
                i -= 1
            while i > 0 and not s[i - 1].isspace():
                i -= 1
            self.text = s[:i] + s[j:]
            self.caret = i
        def delete_word_right(self):
            n = len(self.text)
            if self.caret >= n:
                return
            s = self.text
            i = self.caret
            j = i
            while j < n and s[j].isspace():
                j += 1
            while j < n and not s[j].isspace():
                j += 1
            self.text = s[:i] + s[j:]
        def delete_to_line_end(self):
            s = self.text
            i = self.caret
            nl = s.find('\n', i)
            end = nl if nl != -1 else len(s)
            self.text = s[:i] + s[end:]

# Selection editor fallback
SELECTION_EDITOR_FALLBACK = False
try:
    from modules.selection_editor import SelectEditor  # type: ignore
except Exception:
    SELECTION_EDITOR_FALLBACK = True
    @dataclass
    class SelectEditor:
        text: str
        caret: int
        sel_start: int
        sel_end: int

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
            self.text = self.text[:a] + self.text[b:]
            self.caret = a
            self.clear_selection()
            self._clamp_all()

        def insert(self, s: str) -> None:
            if not s:
                return
            if self.has_selection():
                self._delete_range(int(self.sel_start), int(self.sel_end))
            a = int(self.caret)
            self.text = self.text[:a] + s + self.text[a:]
            self.caret = a + len(s)
            self.clear_selection()
            self._clamp_all()

        def backspace(self) -> None:
            if self.has_selection():
                self._delete_range(int(self.sel_start), int(self.sel_end))
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
                self._delete_range(int(self.sel_start), int(self.sel_end))
                return
            if self.caret >= len(self.text):
                return
            a = int(self.caret)
            self.text = self.text[:a] + self.text[a + 1:]
            self.clear_selection()
            self._clamp_all()

# Unified chat input helpers (cursor-aware editing + blink tracking)
class _LocalChatInputFallback:
    """Inline chat input helper used until module/chat_input becomes available."""

    CURSOR_BLINK_INTERVAL = 0.55

    @staticmethod
    def _snapshot(state):
        text = getattr(state, 'input_buffer', '') or ''
        caret = max(0, min(len(text), int(getattr(state, 'input_caret', len(text)))))
        sel_start = max(0, min(len(text), int(getattr(state, 'input_sel_start', caret))))
        sel_end = max(0, min(len(text), int(getattr(state, 'input_sel_end', caret))))
        return text, caret, sel_start, sel_end

    @staticmethod
    def _apply(state, text, caret, sel_start=None, sel_end=None):
        state.input_buffer = text
        state.input_caret = caret
        state.input_sel_start = caret if sel_start is None else sel_start
        state.input_sel_end = caret if sel_end is None else sel_end
        try:
            state.input_cursor_visible = True
            state.input_cursor_last_toggle = time.time()
        except Exception:
            pass

    @classmethod
    def insert_text(cls, state, data: str) -> None:
        if not data:
            return
        text, caret, sel_start, sel_end = cls._snapshot(state)
        normalized = (data or '').replace('\r\n', '\n').replace('\r', '\n')
        sel = SelectEditor(text, caret, sel_start, sel_end)
        sel.insert(normalized)
        cls._apply(state, sel.text, sel.caret, sel.sel_start, sel.sel_end)

    @classmethod
    def insert_newline(cls, state) -> None:
        cls.insert_text(state, '\n')

    @classmethod
    def backspace(cls, state) -> None:
        text, caret, sel_start, sel_end = cls._snapshot(state)
        sel = SelectEditor(text, caret, sel_start, sel_end)
        sel.backspace()
        cls._apply(state, sel.text, sel.caret, sel.sel_start, sel.sel_end)

    @classmethod
    def delete_forward(cls, state) -> None:
        text, caret, sel_start, sel_end = cls._snapshot(state)
        sel = SelectEditor(text, caret, sel_start, sel_end)
        sel.delete()
        cls._apply(state, sel.text, sel.caret, sel.sel_start, sel.sel_end)

    @classmethod
    def set_text(cls, state, text: str, caret_at_end: bool = True) -> None:
        text = text or ''
        caret = len(text) if caret_at_end else 0
        cls._apply(state, text, caret)

    @classmethod
    def clear_selection(cls, state) -> None:
        text, caret, *_ = cls._snapshot(state)
        cls._apply(state, text, caret, caret, caret)

    @classmethod
    def _text_editor(cls, state):
        text, caret, *_ = cls._snapshot(state)
        return TextEditor(text, caret)

    @classmethod
    def move_left(cls, state):
        ed = cls._text_editor(state)
        ed.move_left()
        cls._apply(state, ed.text, ed.caret)

    @classmethod
    def move_right(cls, state):
        ed = cls._text_editor(state)
        ed.move_right()
        cls._apply(state, ed.text, ed.caret)

    @classmethod
    def move_up(cls, state):
        ed = cls._text_editor(state)
        ed.move_up()
        cls._apply(state, ed.text, ed.caret)

    @classmethod
    def move_down(cls, state):
        ed = cls._text_editor(state)
        ed.move_down()
        cls._apply(state, ed.text, ed.caret)

    @classmethod
    def move_word_left(cls, state):
        ed = cls._text_editor(state)
        ed.move_word_left()
        cls._apply(state, ed.text, ed.caret)

    @classmethod
    def move_word_right(cls, state):
        ed = cls._text_editor(state)
        ed.move_word_right()
        cls._apply(state, ed.text, ed.caret)

    @classmethod
    def delete_word_left(cls, state):
        ed = cls._text_editor(state)
        ed.delete_word_left()
        cls._apply(state, ed.text, ed.caret)

    @classmethod
    def delete_word_right(cls, state):
        ed = cls._text_editor(state)
        ed.delete_word_right()
        cls._apply(state, ed.text, ed.caret)

    @classmethod
    def delete_to_line_end(cls, state):
        ed = cls._text_editor(state)
        ed.delete_to_line_end()
        cls._apply(state, ed.text, ed.caret)

    @classmethod
    def move_line_start(cls, state):
        text, caret, *_ = cls._snapshot(state)
        pos = text.rfind('\n', 0, caret)
        caret = (pos + 1) if pos >= 0 else 0
        cls._apply(state, text, caret)

    @classmethod
    def move_line_end(cls, state):
        text, caret, *_ = cls._snapshot(state)
        pos = text.find('\n', caret)
        caret = pos if pos >= 0 else len(text)
        cls._apply(state, text, caret)

    @classmethod
    def ensure_cursor_tick(cls, state, now=None, force_visible=False):
        if now is None:
            now = time.time()
        if force_visible:
            state.input_cursor_visible = True
            state.input_cursor_last_toggle = now
            return
        last = getattr(state, 'input_cursor_last_toggle', 0.0) or 0.0
        visible = getattr(state, 'input_cursor_visible', True)
        if now - last >= cls.CURSOR_BLINK_INTERVAL:
            state.input_cursor_visible = not visible
            state.input_cursor_last_toggle = now


class _ChatInputProxy:
    _mod = None
    _fallback = _LocalChatInputFallback

    def _load(self):
        if _ChatInputProxy._mod is not None:
            return _ChatInputProxy._mod
        try:
            from modules import chat_input as _mod  # type: ignore
        except Exception as exc:
            logging.getLogger('client').warning("modules. import failed: %s", exc)
            return _ChatInputProxy._fallback
        _ChatInputProxy._mod = _mod
        return _mod

    def __getattr__(self, name):
        mod = self._load()
        return getattr(mod, name)


chat_input = _ChatInputProxy()  # type: ignore

# File transfer helpers (shared module with safe fallbacks)
FILE_TRANSFER_FALLBACK = False
try:
    from modules.file_transfer import extract_path_candidate, file_meta_for, iter_base64_chunks, DEFAULT_CHUNK_SIZE, progress_percent, sanitize_remote_filename  # type: ignore
except Exception:
    FILE_TRANSFER_FALLBACK = True
    # Precompiled regexes for fallback to avoid per-call compilation
    _FT_TRAIL_CHARS = set(list('.,;:!?✓…') + [')', ']', '}', '»', '”', '’'])
    _FT_ABS_PAT = re.compile(r'''(?:^|\s|[\[\(\{"'«»“”„])((?:[A-Za-z]:)?[/\\][^\s\]\)"'«»“”„]+?\.(?:[A-Za-z0-9]{1,6}))''')
    _FT_REL_PAT = re.compile(r'''(?:^|\s|[\[\(\{"'«»“”„])((?:\./|\.\./)?(?:[A-Za-z0-9._-]+[/\\])+[A-Za-z0-9._-]+\.(?:[A-Za-z0-9]{1,6}))''')
    _FT_NAME_PAT = re.compile(r"([A-Za-z0-9._-]+\.(?:png|jpe?g|gif|webp|bmp|pdf|txt|zip|tar|gz|7z|mp4|mov|mkv))", re.IGNORECASE)
    _FT_GENERAL_PAT = re.compile(r'''(?:^|\s|[\[\(\{"'«»“”„])((?:[A-Za-z]:)?[/\\][^\s\]\)"'«»“”„]+)''')
    def sanitize_remote_filename(name: str, *, default: str = "file") -> str:
        try:
            s = str(name or "")
        except Exception:
            s = ""
        s = s.strip().replace("\\", "/").split("/")[-1].strip()
        if not s or s in (".", ".."):
            s = default
        try:
            s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        except Exception:
            s = default
        if not s or s in (".", ".."):
            s = default
        try:
            max_len = 128
            if len(s) > max_len:
                root, ext = os.path.splitext(s)
                ext = ext[:16]
                s = (root[: max(1, max_len - len(ext))] + ext)[:max_len]
        except Exception:
            pass
        return s
    def extract_path_candidate(text: str) -> Optional[str]:
        """Fallback extractor: reuse shared regexes to avoid heavy work per keystroke."""
        if not text:
            return None
        s = (text or '').strip()
        if not s:
            return None
        def _sanitize(p: str) -> str:
            if not isinstance(p, str):
                return p
            t = p.strip()
            if len(t) >= 2 and t[0] in "\"'«“„””" and t[-1] in "\"'»“„””":
                t = t[1:-1].strip()
            while len(t) > 1 and t[-1] in _FT_TRAIL_CHARS:
                t = t[:-1]
                t = t.strip()
            return t
        if (_FT_ABS_PAT.search(s) and _FT_ABS_PAT.findall(s)):
            hits = _FT_ABS_PAT.findall(s)
            if hits:
                return _sanitize(hits[-1])
        hits = _FT_REL_PAT.findall(s)
        if hits:
            return _sanitize(hits[-1])
        hits = _FT_NAME_PAT.findall(s)
        if hits:
            return _sanitize(hits[-1])
        hits = _FT_GENERAL_PAT.findall(s)
        if hits:
            return _sanitize(hits[-1])
        parts = [p for p in s.replace('\n', ' ').split(' ') if p]
        for tok in reversed(parts):
            if (os.sep in tok) or tok.startswith('/') or tok.startswith('./') or tok.startswith('../'):
                return _sanitize(tok)
        return None
    def file_meta_for(path):
        try:
            p = Path(path).expanduser()
            st = p.stat()
            if not p.is_file():
                return None
            return type('FileMeta', (), {'path': p, 'name': p.name, 'size': int(st.st_size)})
        except Exception:
            return None
    def iter_base64_chunks(p: Path, chunk_size: int = 32*1024):
        import base64 as _b64
        seq = 0
        with open(p, 'rb') as f:
            while True:
                buf = f.read(max(1, int(chunk_size)))
                if not buf:
                    break
                yield seq, _b64.b64encode(buf).decode('ascii')
                seq += 1
    DEFAULT_CHUNK_SIZE = 32 * 1024
    def progress_percent(done: int, total: int) -> int:
        try:
            if total <= 0:
                return 0
            return max(0, min(100, int((max(0, done)/float(total))*100)))
        except Exception:
            return 0

# Menu/modal helper (fallback if external module missing)
MODALS_MODULE_FALLBACK = False
try:
    from modules.ui_modals import build_menu_modal_lines  # type: ignore
except Exception:
    MODALS_MODULE_FALLBACK = True
    def build_menu_modal_lines(title: str, options, selected_index: int = 0, subtitle_lines=None, footer: str = "Enter — выбрать | Esc — закрыть | ↑/↓ — выбор"):
        lines = [f" {title} "]
        if subtitle_lines:
            for s in subtitle_lines:
                s = str(s or '').strip()
                if s:
                    lines.append(s)
        idx = max(0, int(selected_index))
        opts = list(options or [])
        for i, opt in enumerate(opts):
            prefix = "> " if i == idx else "  "
            lines.append(f"{prefix}{opt}")
        if footer:
            lines.append(footer)
        return lines

# Two‑pane file browser helpers (fallback switch)
try:
    from modules.file_browser import (
        FileBrowserState as FBState,
        init_browser as fb_init,
        handle_key as fb_handle,
        compute_window as fb_window,
        row_for as fb_row,
        list_dir_opts as fb_list,
    )  # type: ignore
    FILE_BROWSER_FALLBACK = False
except Exception:
    FILE_BROWSER_FALLBACK = True

# Multiline editor view and slash/file suggestions (optional modules)
SUGGEST_FALLBACK = False
try:
    from modules.multiline_editor import EditorView  # type: ignore
    from modules.slash_commands import suggest as slash_suggest  # type: ignore
    from modules.file_suggest import get_file_system_suggestions  # type: ignore
except Exception:
    SUGGEST_FALLBACK = True
    class EditorView:  # minimal stub
        def __init__(self, text: str, caret: int, width: int):
            self._text, self._caret, self._width = text, caret, max(1, width)
        def get_row(self) -> int:
            pre = (self._text or '')[:max(0, min(len(self._text or ''), int(self._caret)))]
            return pre.count('\n')
        def get_col(self) -> int:
            pre = (self._text or '')[:max(0, min(len(self._text or ''), int(self._caret)))]
            last = pre.split('\n')[-1] if pre else ''
            return len(last) % max(1, self._width)
        def is_cursor_at_last_row(self) -> bool:
            return True
    def slash_suggest(prefix: str, limit: int = 10):
        cmds = ['/file']
        return [type('Cmd', (), {'name': c, 'description': ''}) for c in cmds if c.startswith(prefix or '')][:limit]
    def get_file_system_suggestions(token: str, cwd=None, limit: int = 20):
        return []

CLIENT_VERSION = "0.4.1674"
_VER_PART_RE = re.compile(r"\d+")


def _parse_version_tuple(ver: str) -> Optional[Tuple[int, ...]]:
    """Best-effort parse for dotted versions like '0.4.1387' (also accepts 'v0.4.1387', '0.4.1387/srv')."""
    try:
        if not isinstance(ver, str):
            ver = str(ver)
        ver = ver.strip()
    except Exception:
        return None
    if not ver:
        return None
    parts = _VER_PART_RE.findall(ver)
    if not parts:
        return None
    try:
        return tuple(int(p) for p in parts)
    except Exception:
        return None


def _cmp_versions(a: str, b: str) -> Optional[int]:
    """Return -1/0/1 when both versions are parseable, else None."""
    ta = _parse_version_tuple(a)
    tb = _parse_version_tuple(b)
    if not ta or not tb:
        return None
    max_len = max(len(ta), len(tb))
    ta = ta + (0,) * (max_len - len(ta))
    tb = tb + (0,) * (max_len - len(tb))
    if ta < tb:
        return -1
    if ta > tb:
        return 1
    return 0

def _is_ephemeral() -> bool:
    try:
        v = os.environ.get('EPHEMERAL')
        if not v:
            return False
        return str(v).strip().lower() in ('1', 'true', 'yes', 'on')
    except Exception:
        return False

def _word_ops_enabled() -> bool:
    """Enable word-wise jump/delete operations.

    Defaults to disabled on macOS (darwin) to avoid terminal mapping quirks,
    can be enabled via CLIENT_WORD_OPS=1. Enabled by default on other OSes.
    """
    try:
        val = os.environ.get('CLIENT_WORD_OPS')
        if val is not None:
            return str(val).strip().lower() in ('1','true','yes','on')
        import sys as _sys
        return _sys.platform != 'darwin'
    except Exception:
        return True


def format_search_id(query: str) -> str:
    """Auto-format digit-only search queries into groups of 3 with dashes."""
    try:
        if any((not ch.isdigit()) and ch != '-' for ch in query):
            return query
        stripped = ''.join(ch for ch in query if ch.isdigit())[:9]
    except Exception:
        return query
    if not stripped:
        return query
    groups = [stripped[i:i+3] for i in range(0, len(stripped), 3)]
    return '-'.join(groups)


def _digits_only(val: str) -> str:
    """Return only digit characters from a string."""
    return "".join(ch for ch in val if ch.isdigit())


def normalize_search_token(val: str) -> str:
    """
    Normalize search token for lookups: handles -> @handle; digits -> dashed groups.

    Used in live search and group creation to allow prefix digit queries without строго заданных '-'.
    """
    try:
        if any(c.isalpha() for c in val) or val.startswith('@'):
            nh = normalize_handle(val)
            return nh or val
        d = _digits_only(val)
        if d:
            return format_search_id(d)
    except Exception:
        pass
    return val


def format_member_tokens(text: str) -> str:
    """Format member tokens: digits -> grouped with dashes; keep handles as-is."""
    try:
        import re as _re
        tokens = [t for t in _re.split(r"[\s,]+", text) if t]
        fmt = []
        for t in tokens:
            d = _digits_only(t)
            if d:
                fmt.append(format_search_id(d))
            else:
                fmt.append(t)
        return " ".join(fmt)
    except Exception:
        return text


def match_token_query(token: str, query: str) -> bool:
    """
    Match user-entered token to a search query (handles and digit prefixes).

    - For digits: prefix match ignoring separators (123-45 matches 123 or 1234).
    - For handles: normalized @handle equality.
    """
    try:
        td = _digits_only(token)
        qd = _digits_only(query)
        if td and qd:
            return td.startswith(qd) or qd.startswith(td)
        if any(c.isalpha() for c in token) or token.startswith('@') or any(c.isalpha() for c in query):
            try:
                return normalize_handle(token) == normalize_handle(query)
            except Exception:
                return token == query
    except Exception:
        pass
    return token == query


def member_labels(ids: List[str], state, owner_id: Optional[str] = None) -> List[str]:
    """Return display labels for member IDs using cached profiles."""
    labels: List[str] = []
    for uid in ids:
        try:
            prof = (state.profiles or {}).get(uid) or {}
        except Exception:
            prof = {}
        label = (prof.get('display_name') or '').strip() or (prof.get('handle') or '').strip() or uid
        if owner_id and uid == owner_id:
            label += " (владелец)"
        labels.append(label)
    return labels


def open_members_view(state, net, peer: str) -> None:
    """Open read-only members list for group/board."""
    try:
        if peer in getattr(state, 'groups', {}):
            g = state.groups.get(peer) or {}
            members = list(g.get('members') or [])
            owner_id = str(g.get('owner_id') or '')
            state.members_view_mode = True
            state.members_view_target = peer
            state.members_view_entries = member_labels(members, state, owner_id=owner_id)
            name = str(g.get('name') or peer)
            state.members_view_title = f"Участники чата: {name}"
            try:
                net.send({"type": "group_info", "group_id": peer})
            except Exception:
                pass
        elif peer in getattr(state, 'boards', {}):
            b = (getattr(state, 'boards', {}) or {}).get(peer) or {}
            members = list(b.get('members') or [])
            owner_id = str(b.get('owner_id') or '')
            state.members_view_mode = True
            state.members_view_target = peer
            state.members_view_entries = member_labels(members, state, owner_id=owner_id)
            name = str(b.get('name') or peer)
            state.members_view_title = f"Участники доски: {name}"
            try:
                net.send({"type": "board_info", "board_id": peer})
            except Exception:
                pass
    except Exception:
        try:
            state.status = "Не удалось получить список участников"
        except Exception:
            pass
# Цветовые пары и инициализация цветов
CP = {'enabled': False}

# Default update URL for production host (HTTPS)
DEFAULT_UPDATE_URL = "https://yagodka.org:17778"
EXPECTED_ROOT_DIRS = {"bin", "modules", "config", "scripts", "var"}
FORBIDDEN_ROOT_FILES = {"client.py", "bootstrap.py", "pubkey.txt", "schema.json", "version.json"}
REQUIRED_VAR_SUBDIRS = {"files", "history", "log", "update", "users"}

def _is_within_root(root: Path, target: Path) -> bool:
    """Return True if target (after resolving symlinks) stays within root."""
    try:
        root_real = os.path.realpath(str(root))
        target_real = os.path.realpath(str(target))
        return os.path.commonpath([root_real, target_real]) == root_real
    except Exception:
        return False

def _logs_dir() -> Path:
    return LOG_DIR

def _users_dir() -> Path:
    return USERS_DIR

def ensure_storage_dirs() -> None:
    try:
        _logs_dir().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        _users_dir().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        FILES_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def ensure_group_members_authorized(state, member_ids: List[str], net) -> bool:
    """Client-side precheck for group creation.

    - Require at least one participant
    - If известны друзья (roster), подскажем, что по умолчанию можно добавлять только друзей
    """
    try:
        valid = [m for m in member_ids if m]
    except Exception:
        valid = []
    if valid:
        # If we know friends roster and none of ids belong to friends, hint the policy early.
        try:
            friends_set = set(getattr(state, 'friends', {}).keys()) | set(getattr(state, 'roster_friends', {}).keys())
        except Exception:
            friends_set = set()
        # Only check ID-like tokens here; handles are resolved elsewhere
        def _id_like(tok: str) -> bool:
            try:
                import re as _re2
                return bool(_re2.match(r"^(?:\d{3}-\d{2}|\d{3}(?:-\d{3})+)$", tok))
            except Exception:
                return False
        only_ids = [m for m in valid if isinstance(m, str) and _id_like(m) and not m.startswith('@')]
        if friends_set and only_ids and not any((m in friends_set) for m in only_ids):
            try:
                state.modal_message = "Добавьте хотя бы одного друга (по умолчанию — только друзья)"
                state.status = "Требуется друг в участниках"
                state.group_create_mode = True
                state.group_create_field = 1
                state.group_verify_mode = False
            except Exception:
                pass
            return False
        return True
    # Keep modal open and ask to add at least one participant
    try:
        state.modal_message = "Добавьте хотя бы одного участника"
        state.status = "Введите участников"
        state.group_create_mode = True
        state.group_create_field = 1
        state.group_verify_mode = False
    except Exception:
        pass


def _fix_structure() -> bool:
    """
    Ensure client layout is sane before updates.
    Removes forbidden files, runtime dirs, and recreates required folders.
    """
    root = CLIENT_DIR
    ok = True
    try:
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        present_dirs = {p.name for p in root.iterdir() if p.is_dir()}
        for d in EXPECTED_ROOT_DIRS:
            if d not in present_dirs:
                try:
                    (root / d).mkdir(parents=True, exist_ok=True)
                except Exception:
                    ok = False
        for fname in FORBIDDEN_ROOT_FILES:
            # Разрешаем плоскую раскладку: если нет bin/client.py — не удаляем client.py в корне
            if fname == "client.py" and not (root / "bin" / "client.py").exists():
                continue
            try:
                (root / fname).unlink()
            except FileNotFoundError:
                pass
            except Exception:
                ok = False
        # Remove stray runtime/modules duplicates
        for stray in (root / "runtime", root / "var" / "runtime", root / "modules 2"):
            try:
                if stray.exists():
                    import shutil
                    shutil.rmtree(stray)
            except Exception:
                ok = False
        var_dir = root / "var"
        for sub in REQUIRED_VAR_SUBDIRS:
            try:
                (var_dir / sub).mkdir(parents=True, exist_ok=True)
            except Exception:
                ok = False
    except Exception:
        ok = False
    return ok

def _is_board_owner(state, bid: str) -> bool:
    # Client no longer enforces board ownership; server is authoritative.
    # Keep helper for potential UI hints but do not block actions on it.
    try:
        b = (getattr(state, 'boards', {}) or {}).get(str(bid)) or {}
        owner = str(b.get('owner_id') or '')
        return bool(owner) and (owner == str(getattr(state, 'self_id', '') or ''))
    except Exception:
        return False

def init_colors():
    try:
        if not curses.has_colors():
            return
        curses.start_color()
        try:
            curses.use_default_colors()
        except Exception:
            pass
        # Пары: 1 header (white on blue), 2 title (yellow), 3 selected (black on green)
        # 4 unread (yellow), 5 error (red), 6 success (green), 7 warn (yellow), 8 divider (blue)
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.init_pair(4, curses.COLOR_YELLOW, -1)
        curses.init_pair(5, curses.COLOR_RED, -1)
        curses.init_pair(6, curses.COLOR_GREEN, -1)
        curses.init_pair(7, curses.COLOR_YELLOW, -1)
        curses.init_pair(8, curses.COLOR_BLUE, -1)
        CP.update({
            'enabled': True,
            'header': curses.color_pair(1),
            'title': curses.color_pair(2) | curses.A_BOLD,
            'selected': curses.color_pair(3) | curses.A_BOLD,
            'unread': curses.color_pair(4) | curses.A_BOLD,
            'error': curses.color_pair(5) | curses.A_BOLD,
            'success': curses.color_pair(6) | curses.A_BOLD,
            'warn': curses.color_pair(7),
            'div': curses.color_pair(8),
        })
    except Exception:
        CP['enabled'] = False


def _get_update_base_url() -> Optional[str]:
    def _check_https(url: str) -> Optional[str]:
        insecure = str(os.environ.get('ALLOW_INSECURE_DEV', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
        if url.lower().startswith('https://'):
            return url
        if insecure:
            logging.getLogger('client').warning('Insecure UPDATE_URL used (ALLOW_INSECURE_DEV=1): %s', url)
            return url
        logging.getLogger('client').error('UPDATE_URL must use HTTPS')
        return None

    base_url = os.environ.get('UPDATE_URL')
    if base_url:
        return _check_https(base_url.strip())
    # Config-driven update URL has higher priority than inferring from SERVER_ADDR
    cfg = _load_config_full()
    cfg_url = cfg.get('update_url')
    if cfg_url:
        return _check_https(str(cfg_url).strip())
    # If nothing configured, prefer known production URL
    if DEFAULT_UPDATE_URL:
        return DEFAULT_UPDATE_URL
    return None


def _fetch_url(url: str, timeout: float = 6.0) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "yagodka-client"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        logging.getLogger('client').exception("Update fetch failed: %s", url)
        return None


def _parse_pubkey_env() -> Optional[bytes]:
    """Get update public key from env, pinned config, or embedded default.

    Order:
    1) UPDATE_PUBKEY env (hex/base64)
    2) pinned in client_config.json as 'update_pubkey'
    3) embedded constant EMBEDDED_UPDATE_PUBKEY_HEX (optional)
    """
    # 1) env
    try:
        pk = os.environ.get('UPDATE_PUBKEY')
        if pk:
            s = pk.strip()
            try:
                b = bytes.fromhex(s)
                if len(b) == 32:
                    return b
            except Exception:
                pass
            try:
                b = base64.b64decode(s, validate=True)
                if len(b) == 32:
                    return b
            except Exception:
                pass
    except Exception:
        pass
    # 2) pinned in config
    try:
        cfg = _load_config_full()
        txt = cfg.get('update_pubkey')
        if isinstance(txt, str) and txt.strip():
            s = txt.strip()
            try:
                b = bytes.fromhex(s)
                if len(b) == 32:
                    return b
            except Exception:
                pass
            try:
                b = base64.b64decode(s, validate=True)
                if len(b) == 32:
                    return b
            except Exception:
                pass
    except Exception:
        pass
    # 3) embedded
    try:
        EMBEDDED_UPDATE_PUBKEY_HEX = globals().get('EMBEDDED_UPDATE_PUBKEY_HEX')  # type: ignore
        if isinstance(EMBEDDED_UPDATE_PUBKEY_HEX, str) and EMBEDDED_UPDATE_PUBKEY_HEX:
            b = bytes.fromhex(EMBEDDED_UPDATE_PUBKEY_HEX.strip())
            if len(b) == 32:
                return b
    except Exception:
        pass
    return None

def _store_pinned_pubkey(pub: bytes) -> bool:
    if _is_ephemeral():
        return False
    try:
        data = _load_config_full()
        data['update_pubkey'] = pub.hex()
        with open(_config_path(), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        logging.getLogger('client').info('Pinned update pubkey: %s', hashlib.sha256(pub).hexdigest()[:16])
        return True
    except Exception:
        logging.getLogger('client').exception('Failed to pin update pubkey')
        return False

def _attempt_fetch_server_pubkey(base_url: str, timeout: float = 6.0) -> Optional[bytes]:
    try:
        b = _fetch_url(base_url.rstrip('/') + '/pubkey.txt', timeout=timeout)
        if not b:
            return None
        s = b.strip()
        # accept hex or base64
        try:
            pk = bytes.fromhex(s.decode('ascii'))
            if len(pk) == 32:
                return pk
        except Exception:
            pass
        try:
            pk = base64.b64decode(s, validate=True)
            if len(pk) == 32:
                return pk
        except Exception:
            pass
        return None
    except Exception:
        return None


def _verify_ed25519_signature(message: bytes, signature: bytes, pubkey: bytes):
    """Verify Ed25519 signature using available backends.

    Tries cryptography, then ed25519 (pure Python). Returns None if no backend.
    """
    # cryptography backend
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # type: ignore
        from cryptography.exceptions import InvalidSignature  # type: ignore
        try:
            pk = Ed25519PublicKey.from_public_bytes(pubkey)
            pk.verify(signature, message)
            return True
        except InvalidSignature:
            return False
        except Exception:
            pass
    except Exception:
        pass
    # ed25519 pure-python backend
    try:
        import ed25519  # type: ignore
        try:
            vk = ed25519.VerifyingKey(pubkey)
            vk.verify(signature, message)
            return True
        except ed25519.BadSignatureError:
            return False
        except Exception:
            pass
    except Exception:
        pass
    # Backend unavailable
    return None


def _decode_signature_blob(sig_b: bytes) -> Optional[bytes]:
    """Accept raw 64b, base64 or hex signature."""
    sig = None
    try:
        if len(sig_b) == 64:
            sig = sig_b
        else:
            sig_txt = sig_b.strip()
            try:
                sig = base64.b64decode(sig_txt, validate=True)
            except Exception:
                try:
                    sig = bytes.fromhex(sig_txt.decode('ascii'))
                except Exception:
                    sig = None
    except Exception:
        sig = None
    return sig


def _load_manifest(base_url: str, timeout: float = 6.0, forced: bool = False) -> Optional[dict]:
    """Fetch and verify dist/manifest.json with Ed25519 signature."""
    manifest_url = base_url.rstrip('/') + '/manifest.json'
    mani_b = _fetch_url(manifest_url, timeout=timeout)
    if not mani_b:
        return None
    pub = _parse_pubkey_env()
    if pub is None:
        logging.getLogger('client').error('UPDATE_PUBKEY is required for update signature verification')
        return None
    sig_b = _fetch_url(base_url.rstrip('/') + '/manifest.sig', timeout=timeout)
    if not sig_b:
        logging.getLogger('client').error('Update signature missing (manifest.sig)')
        return None
    sig = _decode_signature_blob(sig_b)
    if not sig or len(sig) != 64:
        logging.getLogger('client').error('Invalid manifest signature format')
        return None
    ver_ok = _verify_ed25519_signature(mani_b, sig, pub)
    if ver_ok is not True:
        logging.getLogger('client').error('Manifest signature verification failed (install ed25519 backend)')
        return None
    try:
        return json.loads(mani_b.decode('utf-8'))
    except Exception:
        return None


def get_server_version() -> Optional[str]:
    try:
        base = _get_update_base_url()
        if not base:
            return None
        mani = _load_manifest(base, timeout=4.0)
        if not isinstance(mani, dict):
            meta = _fetch_url(base.rstrip('/') + '/version.json', timeout=4.0)
            if not meta:
                return None
            try:
                data = json.loads(meta.decode('utf-8'))
                return str(data.get('version') or '') or None
            except Exception:
                return None
        latest = str(mani.get('version') or '')
        return latest or None
    except Exception:
        return None


def _safe_manifest_entries(manifest: dict) -> list[dict]:
    entries: list[dict] = []
    for item in manifest.get('files') or []:
        path_txt = str(item.get('path') or '').strip()
        sha = str(item.get('sha256') or '').strip()
        try:
            size = int(item.get('size') or 0)
        except Exception:
            size = 0
        p = Path(path_txt)
        if not path_txt or p.is_absolute() or '..' in p.parts:
            continue
        if len(sha) != 64:
            continue
        if size <= 0:
            continue
        entries.append({'path': path_txt, 'sha256': sha, 'size': size})
    return entries


def _hash_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _extract_client_release(tar_path: Path, root: Path) -> bool:
    """Extract client-release.tar.gz into root while keeping local var/ and config."""
    try:
        root = root.resolve()
        tmpdir = root / ".update_tmp_extract"
        if tmpdir.exists():
            # Refuse to operate on a symlink to avoid deleting/copying outside root.
            if tmpdir.is_symlink() or not tmpdir.is_dir():
                return False
            shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                safe_members: list[tarfile.TarInfo] = []
                for m in tf.getmembers():
                    name = str(m.name or "")
                    norm = name.replace("\\", "/")
                    p = Path(norm)
                    if not norm or p.is_absolute() or ".." in p.parts:
                        continue
                    # Only allow regular files and directories; drop links/devices/etc.
                    if not (m.isfile() or m.isdir()):
                        continue
                    safe_members.append(m)
                if not safe_members:
                    return False
                for m in safe_members:
                    norm = str(m.name or "").replace("\\", "/")
                    out_path = tmpdir / norm
                    if not _is_within_root(tmpdir, out_path.parent):
                        return False
                    if m.isdir():
                        out_path.mkdir(parents=True, exist_ok=True)
                        continue
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(m)
                    if src is None:
                        continue
                    with src:
                        with open(out_path, "wb") as dst_f:
                            shutil.copyfileobj(src, dst_f)
            entries = list(tmpdir.iterdir())
            if not entries:
                return False
            roots = {p.relative_to(tmpdir).parts[0] for p in entries}
            base = tmpdir
            if len(roots) == 1:
                candidate = tmpdir / next(iter(roots))
                if candidate.is_dir():
                    base = candidate
            # Ensure extracted release fully replaces managed dirs to avoid stale/extra files.
            preserved_cfg: Optional[bytes] = None
            cfg_rel = Path("config/client_config.json")
            cfg_path = root / cfg_rel
            try:
                if cfg_path.exists() and cfg_path.is_file():
                    preserved_cfg = cfg_path.read_bytes()
            except Exception:
                preserved_cfg = None
            try:
                managed_top = {p.name for p in base.iterdir()}
            except Exception:
                managed_top = set()
            for name in sorted(managed_top):
                if name == "var":
                    continue
                if name == "config":
                    if not (base / "config").exists():
                        continue
                    config_dir = root / "config"
                    if config_dir.exists():
                        if config_dir.is_symlink() or not config_dir.is_dir():
                            return False
                        try:
                            shutil.rmtree(config_dir, ignore_errors=False)
                        except Exception:
                            return False
                    continue
                target = root / name
                if not target.exists():
                    continue
                if target.is_symlink():
                    return False
                if target.is_dir():
                    try:
                        shutil.rmtree(target, ignore_errors=False)
                    except Exception:
                        return False
                else:
                    target.unlink(missing_ok=True)
            for f in base.rglob("*"):
                if f.is_dir():
                    continue
                if f.is_symlink():
                    return False
                rel = f.relative_to(base)
                if rel.parts and rel.parts[0] == "var":
                    continue
                if rel == cfg_rel and preserved_cfg is not None:
                    continue
                target = root / rel
                if not _is_within_root(root, target.parent):
                    return False
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
                try:
                    with os.fdopen(fd, "wb") as out_f, open(f, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
                    try:
                        os.chmod(tmp_name, 0o755 if target.name.endswith(".py") else 0o644)
                    except Exception:
                        pass
                    os.replace(tmp_name, target)
                finally:
                    try:
                        if os.path.exists(tmp_name):
                            os.unlink(tmp_name)
                    except Exception:
                        pass
            if preserved_cfg is not None:
                try:
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    cfg_path.write_bytes(preserved_cfg)
                except Exception:
                    return False
            return True
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        logging.getLogger('client').exception("Failed to extract client-release.tar.gz")
        return False


def _map_destination(rel: Path, base: Path) -> Path:
    """Map manifest path to target in client layout; fallback to flat layout for tests/standalone."""
    rel_posix = rel.as_posix()
    if rel_posix == "client.py":
        dest_bin = base / "bin" / "client.py"
        dest_flat = base / "client.py"
        if dest_bin.exists():
            return dest_bin
        if not dest_bin.parent.exists():
            return dest_flat
        # bin/ есть, но файла нет — поддержим плоскую раскладку (тест/standalone)
        return dest_flat
    if rel_posix == "schema.json":
        return base / "config" / "protocol" / "schema.json"
    if rel_posix == "bootstrap.py":
        return base / "scripts" / "bootstrap.py"
    if rel_posix == "pubkey.txt":
        return base / "config" / "pubkey.txt"
    if rel_posix == "version.json":
        return base / "var" / "update" / "version.json"
    if rel_posix == "client-release.tar.gz":
        return base / "var" / "update" / "client-release.tar.gz"
    return (base / rel)


def _manifest_diff(entries: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (to_download, extra) given manifest entries vs local files."""
    base = CLIENT_DIR.resolve()
    to_get: list[dict] = []
    expected_paths: set[Path] = set()
    for e in entries:
        rel = Path(e['path'])
        try:
            dest = _map_destination(rel, base).resolve()
        except Exception:
            continue
        if base not in dest.parents and base != dest:
            continue
        expected_paths.add(dest)
        if not dest.exists():
            to_get.append(e)
            continue
        local_hash = _hash_file(dest)
        if local_hash != e['sha256']:
            to_get.append(e)
    return to_get, []


def _manifest_root_hash(entries: list[dict]) -> str:
    try:
        lines = [f"{e['path']}:{e['sha256']}" for e in entries]
        txt = "\n".join(sorted(lines))
        h = hashlib.sha256()
        h.update(txt.encode('utf-8'))
        return h.hexdigest()
    except Exception:
        return ""


def _apply_manifest_update(manifest: dict, base_url: str, stdscr=None, forced: bool = False) -> Optional[str]:
    """Validate structure, download missing/mismatched files, verify, and restart if needed.

    Returns target version on success, None on failure.
    """
    if not _fix_structure():
        logging.getLogger('client').error("Client structure invalid; cannot proceed with update")
        return None
    entries = _safe_manifest_entries(manifest)
    if not entries:
        logging.getLogger('client').error('Manifest has no valid entries')
        return None
    # Root hash check (optional hardening)
    want_root = str(manifest.get('root_hash') or '')
    got_root = _manifest_root_hash(entries)
    if want_root and got_root and want_root != got_root:
        logging.getLogger('client').error('Manifest root hash mismatch')
        return None
    version = str(manifest.get('version') or '')
    to_get, extras = _manifest_diff(entries)
    base = CLIENT_DIR.resolve()
    update_dir = VAR_DIR / 'update'
    update_dir.mkdir(parents=True, exist_ok=True)
    tar_path: Optional[Path] = None

    def _progress(lines: list[str]):
        if stdscr:
            try:
                stdscr.clear()
            except Exception:
                pass
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
            try:
                stdscr.refresh()
            except Exception:
                pass
        else:
            logging.getLogger('client').info("update: %s", " | ".join(lines))

    if extras:
        logging.getLogger('client').info("Extra files not in manifest: %s", ", ".join(extras))

    if not to_get:
        return version or CLIENT_VERSION

    _progress(["Проверка обновлений…", f"Файлов к обновлению: {len(to_get)}"])
    staged: list[tuple[Path, Path]] = []  # (tmp, dest)
    for idx, entry in enumerate(to_get, 1):
        rel = Path(entry['path'])
        dest = _map_destination(rel, base).resolve()
        if base not in dest.parents and base != dest:
            continue
        url = base_url.rstrip('/') + '/' + rel.as_posix()
        _progress([f"Загрузка {idx}/{len(to_get)}", rel.as_posix()])
        data = _fetch_url(url, timeout=15.0)
        if not data:
            logging.getLogger('client').error("Failed to download %s", url)
            return None
        if len(data) != int(entry['size']):
            logging.getLogger('client').error("Size mismatch for %s", rel)
            return None
        h = hashlib.sha256()
        h.update(data)
        if h.hexdigest() != entry['sha256']:
            logging.getLogger('client').error("Hash mismatch for %s", rel)
            return None
        tmp_path = update_dir / (rel.as_posix().replace('/', '_') + ".tmp")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(data)
        except Exception as exc:
            logging.getLogger('client').error("Failed to stage %s: %s", rel, exc)
            return None
        staged.append((tmp_path, dest))
        if rel.as_posix() == "client-release.tar.gz":
            tar_path = dest

    # Apply staged files atomically
    changed = False
    for tmp_path, dest in staged:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            # Backup current file once if needed
            if dest.exists():
                try:
                    backup = dest.with_suffix(dest.suffix + ".bak")
                    if not backup.exists():
                        dest.replace(backup)
                except Exception:
                    pass
            tmp_path.replace(dest)
            if dest.suffix == ".py":
                try:
                    dest.chmod(0o755)
                except Exception:
                    pass
            changed = True
        except Exception as exc:
            logging.getLogger('client').error("Failed to install %s: %s", dest, exc)
            return None

    # Final verification of all manifest entries. If something is missing or mismatched,
    # try a one-shot re-download of just the bad entries to self-heal partial updates.
    def _verify(entries_to_check: list[dict]) -> list[dict]:
        bad: list[dict] = []
        for entry in entries_to_check:
            rel = Path(entry['path'])
            dest = _map_destination(rel, base).resolve()
            if base not in dest.parents and base != dest:
                logging.getLogger('client').error("Unsafe path in manifest: %s", rel)
                bad.append(entry)
                continue
            if not dest.exists():
                logging.getLogger('client').error("Missing file after update: %s", rel)
                bad.append(entry)
                continue
            if dest.stat().st_size != int(entry['size']):
                logging.getLogger('client').error("Size mismatch after update: %s", rel)
                bad.append(entry)
                continue
            local_hash = _hash_file(dest)
            if local_hash != entry['sha256']:
                logging.getLogger('client').error("Hash mismatch after update: %s", rel)
                bad.append(entry)
                continue
        return bad

    bad_entries = _verify(entries)
    if bad_entries:
        logging.getLogger('client').warning("Retrying download for %d missing/mismatched files", len(bad_entries))
        # Re-download only the bad ones
        staged_retry: list[tuple[Path, Path]] = []
        for entry in bad_entries:
            rel = Path(entry['path'])
            dest = _map_destination(rel, base).resolve()
            if base not in dest.parents and base != dest:
                continue
            data = _download_entry(base_url, entry)
            if not data:
                return None
            try:
                h = hashlib.sha256()
                h.update(data)
                if h.hexdigest() != entry['sha256']:
                    logging.getLogger('client').error("Hash mismatch (retry) for %s", rel)
                    return None
                tmp_path = update_dir / (rel.as_posix().replace('/', '_') + ".retry")
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_bytes(data)
                staged_retry.append((tmp_path, dest))
            except Exception as exc:
                logging.getLogger('client').error("Failed to stage %s on retry: %s", rel, exc)
                return None
        # Apply retried files
        for tmp_path, dest in staged_retry:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                tmp_path.replace(dest)
                if dest.suffix == ".py":
                    try:
                        dest.chmod(0o755)
                    except Exception:
                        pass
                changed = True
            except Exception as exc:
                logging.getLogger('client').error("Failed to install %s on retry: %s", dest, exc)
                return None
        bad_entries = _verify(entries)
        if bad_entries:
            logging.getLogger('client').error("Update failed: files still invalid after retry: %s", [e.get('path') for e in bad_entries])
            return None

    # Apply archive if present to refresh modules/config/bin layout
    if tar_path and tar_path.exists():
        if not _extract_client_release(tar_path, base):
            logging.getLogger('client').error("Failed to extract client-release.tar.gz")
            return None
        changed = True

    # Final cleanup to ensure forbidden files removed after update
    _fix_structure()

    # Restart if anything changed (non-ephemeral)
    if changed and not _is_ephemeral():
        try:
            curses.endwin()
        except Exception:
            pass
        restart_path: Optional[Path] = None
        for candidate in (
            (CLIENT_DIR / "bin" / "client.py"),
            Path(__file__),
            (CLIENT_DIR / "client.py"),
        ):
            try:
                if candidate.exists():
                    restart_path = candidate.resolve()
                    break
            except Exception:
                continue
        if restart_path is None:
            restart_path = Path(__file__).resolve()
        print(f"Обновлён клиент до версии {version or CLIENT_VERSION}. Перезапуск…")
        os.execv(sys.executable, [sys.executable, str(restart_path)])
    return version or CLIENT_VERSION


def _module_import_root() -> Path:
    """Choose actual import root: prefer module/ subdir if it exists, else flat modules/."""
    if MODULE_PKG_DIR.exists():
        return MODULE_PKG_DIR
    return MODULE_DIR


def _ensure_module_sys_path() -> None:
    """Ensure module/, module/<platform> and dll/ are on sys.path."""
    root = _module_import_root()
    try:
        tag = _platform_tag()
        mod_os = root / tag
        if mod_os.exists() and str(mod_os) not in sys.path:
            sys.path.insert(0, str(mod_os))
    except Exception:
        pass
    for p in (root, CLIENT_DIR, DLL_DIR):
        try:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        except Exception:
            pass


def _platform_tag() -> str:
    try:
        import sys as _sys, platform as _plat
        p = _sys.platform
        if p.startswith('darwin'):
            return 'darwin'
        if p.startswith('linux'):
            return 'linux'
        if p.startswith('win'):
            return 'win'
        return _plat.system().strip().lower() or 'unknown'
    except Exception:
        return 'unknown'


def _draw_center_box(stdscr, lines: List[str], accent_attr: int = 0):
    h, w = stdscr.getmaxyx()
    box_w = min(max(len(max(lines, key=len, default='')) + 6, 40), max(40, w - 4))
    box_h = len(lines) + 4
    y0 = max(1, (h - box_h) // 2)
    x0 = max(2, (w - box_w) // 2)
    # Frame
    stdscr.hline(y0, x0, ord('-'), box_w, CP.get('div', 0))
    stdscr.hline(y0 + box_h - 1, x0, ord('-'), box_w, CP.get('div', 0))
    for yy in range(y0 + 1, y0 + box_h - 1):
        stdscr.addch(yy, x0, ord('|'), CP.get('div', 0))
        stdscr.addch(yy, x0 + box_w - 1, ord('|'), CP.get('div', 0))
    for i, line in enumerate(lines):
        try:
            if i == 0:
                # Center the first content line (title) within the box
                lx = x0 + 2 + max(0, (box_w - 4 - len(line)) // 2)
                stdscr.addnstr(y0 + 2 + i, lx, line, min(len(line), box_w - 4), accent_attr)
            else:
                stdscr.addnstr(y0 + 2 + i, x0 + 2, line.ljust(box_w - 4), box_w - 4, accent_attr)
        except Exception:
            pass
    # No refresh here; caller is responsible for a single end-of-frame refresh


def _prompt_update_available(stdscr, current: str, latest: str, *, title: str = "Обнаружено обновление") -> bool:
    """Ask user whether to start update now.

    Returns True if user chose to update (Ctrl+U or Enter/OK), False otherwise.
    """
    try:
        cur = str(current or "").strip() or "?"
        lat = str(latest or "").strip() or "?"
        lines = [title]
        if cur and lat and cur != lat:
            lines.append(f"{cur} → {lat}")
        elif lat:
            lines.append(f"Версия: {lat}")
        lines.extend(["", "Нажмите Ctrl+U или Enter (OK) — обновить", "Esc или любая клавиша — позже"])
        _draw_center_box(stdscr, lines, CP.get("title", curses.A_BOLD))
        try:
            stdscr.refresh()
        except Exception:
            pass
        while True:
            try:
                ch = stdscr.get_wch()
            except Exception:
                ch = None
            if isinstance(ch, str) and ch == "\x15":
                return True
            if isinstance(ch, str) and ch in ("\n", "\r"):
                return True
            if ch in (curses.KEY_ENTER, 10, 13):
                return True
            if (isinstance(ch, str) and ch in ("\x1b",)) or ch == 27:
                return False
            if ch is not None:
                return False
    except Exception:
        return False


def _ensure_update_pubkey_interactive(stdscr, base: str, *, forced: bool = False) -> bool:
    """Ensure UPDATE_PUBKEY is available; if not, try TOFU with user confirmation."""
    try:
        if _parse_pubkey_env() is not None:
            return True
        server_pk = _attempt_fetch_server_pubkey(base)
        if server_pk is None:
            globals()["LAST_UPDATE_ERROR"] = "Ключ обновлений не найден"
            return False
        if forced:
            if _is_ephemeral():
                os.environ["UPDATE_PUBKEY"] = server_pk.hex()
            else:
                _store_pinned_pubkey(server_pk)
            return True
        fp = hashlib.sha256(server_pk).hexdigest()[:16]
        _draw_center_box(
            stdscr,
            [
                "Первичная настройка подписи обновлений",
                f"Отпечаток ключа сервера: {fp}",
                "Доверять этому ключу? [Y] Да / [N] Нет",
            ],
            CP.get("title", curses.A_BOLD),
        )
        try:
            stdscr.refresh()
        except Exception:
            pass
        while True:
            try:
                ch = stdscr.get_wch()
            except Exception:
                ch = None
            if (isinstance(ch, str) and ch in ("y", "Y", "\n", "\r")) or ch in (curses.KEY_ENTER, 10, 13):
                _store_pinned_pubkey(server_pk)
                return True
            if (isinstance(ch, str) and ch in ("n", "N", "\x1b")) or ch == 27:
                globals()["LAST_UPDATE_ERROR"] = "Обновление отменено (ключ не подтверждён)"
                return False
    except Exception:
        return False


def startup_update_check(stdscr) -> Optional[str]:
    """Startup flow: check for updates and ask once (no auto-update/restart)."""
    try:
        base = _get_update_base_url()
        if not base:
            return None
        # Best-effort lightweight startup UI
        try:
            stdscr.clear()
        except Exception:
            pass
        _draw_center_box(stdscr, ["Старт клиента", "Проверка обновлений…"], CP.get("title", curses.A_BOLD))
        try:
            stdscr.refresh()
        except Exception:
            pass
        # Ensure pubkey (TOFU confirmation if needed)
        if not _ensure_update_pubkey_interactive(stdscr, base, forced=False):
            return None
        manifest = _load_manifest(base, forced=False)
        if not isinstance(manifest, dict):
            globals()["LAST_UPDATE_ERROR"] = "Не удалось загрузить/верифицировать манифест"
            return None
        latest = str(manifest.get("version") or "")
        entries = _safe_manifest_entries(manifest)
        if not entries:
            globals()["LAST_UPDATE_ERROR"] = "Манифест повреждён"
            return None
        to_get, _ = _manifest_diff(entries)
        if not to_get:
            return latest or CLIENT_VERSION
        if not _prompt_update_available(stdscr, CLIENT_VERSION, latest or CLIENT_VERSION):
            # user chose to continue without updating
            return latest or CLIENT_VERSION
        # User confirmed update now: apply and restart if needed.
        result = _apply_manifest_update(manifest, base, stdscr=stdscr, forced=True)
        if not result:
            globals()["LAST_UPDATE_ERROR"] = "Не удалось применить обновление"
            return latest or None
        return result
    except Exception:
        logging.getLogger("client").exception("Startup update check failed")
        return None
    finally:
        try:
            stdscr.erase()
        except Exception:
            pass


def interactive_update_check(stdscr, forced: bool = False, *, confirm: bool = True) -> Optional[str]:
    try:
        base = _get_update_base_url()
        if not base:
            return None
        def _step(lines: list[str], sleep: float = 0.0, accent=CP.get('title', curses.A_BOLD)):
            try:
                stdscr.clear()
            except Exception:
                pass
            _draw_center_box(stdscr, ["Старт клиента", "Проверка обновлений…", *lines], accent)
            try:
                stdscr.refresh()
            except Exception:
                pass
            if sleep > 0:
                time.sleep(sleep)
        _step(["Подключение к серверу обновлений…"], 0.2)
        if not _ensure_update_pubkey_interactive(stdscr, base, forced=forced):
            err = str(globals().get("LAST_UPDATE_ERROR") or "Ключ обновлений не найден")
            _step([err, "Обновление недоступно", "Нажмите любую клавишу…"], accent=CP.get('error', curses.A_BOLD))
            try:
                stdscr.getch()
            except Exception:
                pass
            return None
        _step(["Загрузка манифеста…"], 0.1)
        manifest = _load_manifest(base, forced=forced)
        if not isinstance(manifest, dict):
            _step(["Не удалось загрузить/верифицировать манифест", "Нажмите любую клавишу…"], accent=CP.get('error', curses.A_BOLD))
            globals()['LAST_UPDATE_ERROR'] = "Не удалось загрузить/верифицировать манифест"
            try:
                stdscr.getch()
            except Exception:
                pass
            return None
        latest = str(manifest.get('version') or '')
        entries = _safe_manifest_entries(manifest)
        if not entries:
            _step(["Манифест пуст или повреждён", "Нажмите любую клавишу…"], accent=CP.get('error', curses.A_BOLD))
            globals()['LAST_UPDATE_ERROR'] = "Манифест повреждён"
            try:
                stdscr.getch()
            except Exception:
                pass
            return None
        to_get, _ = _manifest_diff(entries)
        if not to_get:
            return latest or CLIENT_VERSION
        if confirm and latest and latest != CLIENT_VERSION and not forced:
            _draw_center_box(
                stdscr,
                [
                    f"Доступно обновление клиента: {CLIENT_VERSION} → {latest}",
                    "Обновить сейчас? [Y] Да / [N] Нет",
                ],
                CP.get('title', curses.A_BOLD),
            )
            while True:
                ch = stdscr.getch()
                if ch in (ord('y'), ord('Y'), 10, 13):
                    break
                if ch in (ord('n'), ord('N'), 27):
                    return latest
        result = _apply_manifest_update(manifest, base, stdscr=stdscr, forced=forced)
        if not result:
            _step(["Не удалось применить обновление", "Нажмите любую клавишу…"], accent=CP.get('error', curses.A_BOLD))
            globals()['LAST_UPDATE_ERROR'] = "Не удалось применить обновление"
            try:
                stdscr.getch()
            except Exception:
                pass
        return result
    except Exception:
        logging.getLogger('client').exception("Interactive update failed")
        return None
    finally:
        # Ensure updater screen does not linger under subsequent UI (auth/main).
        try:
            stdscr.erase()
        except Exception:
            pass

def _profile_suffix() -> str:
    p = os.environ.get('CLIENT_PROFILE', '').strip().lower()
    if not p:
        return ""
    try:
        import re as _re
        if not _re.fullmatch(r"[a-z0-9_-]{1,32}", p or ""):
            return ""
    except Exception:
        return ""
    return f".{p}"


def _config_path() -> Path:
    return CONFIG_DIR / f"client_config{_profile_suffix()}.json"


def _history_path() -> Path:
    return HISTORY_DIR / f"client_history{_profile_suffix()}.jsonl"


def _history_index_path() -> Path:
    return HISTORY_DIR / f"client_history_index{_profile_suffix()}.json"


def load_history_index() -> Dict[str, int]:
    if _is_ephemeral():
        return {}
    p = _history_index_path()
    if p.exists():
        try:
            return json.load(open(p, 'r', encoding='utf-8'))
        except Exception:
            logging.getLogger('client').exception("Failed to read history index")
    return {}


def save_history_index(idx: Dict[str, int]) -> None:
    if _is_ephemeral():
        return
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        json.dump(idx, open(_history_index_path(), 'w', encoding='utf-8'))
    except Exception:
        logging.getLogger('client').exception("Failed to write history index")


def append_history_record(rec: dict) -> None:
    if _is_ephemeral():
        return
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(rec, ensure_ascii=False)
        with open(_history_path(), 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        logging.getLogger('client').exception("Failed to append history record")

def _write_user_history_line(peer: str, direction: str, text: str, ts: float) -> None:
    if _is_ephemeral():
        return
    try:
        d = _users_dir()
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        fn = d / f"{peer}.txt"
        t = time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))
        prefix = 'in ' if direction == 'in' else 'out'
        with open(fn, 'a', encoding='utf-8') as f:
            f.write(f"[{t}] {prefix}: {text}\n")
    except Exception:
        logging.getLogger('client').exception("Failed to write per-user history for %s", peer)

def _touch_user_history(peer: str) -> None:
    if _is_ephemeral():
        return
    try:
        d = _users_dir()
        d.mkdir(parents=True, exist_ok=True)
        fn = d / f"{peer}.txt"
        if not fn.exists():
            fn.touch()
    except Exception:
        logging.getLogger('client').exception("Failed to touch user history file for %s", peer)

def purge_local_history_for_peer(state: 'ClientState', peer: str) -> None:
    """Remove all cached local DM history with given peer (both directions).

    Keeps group history intact. Updates history index accordingly.
    """
    try:
        # Drop in-memory conversation + counters
        try:
            if peer in state.conversations:
                del state.conversations[peer]
        except Exception:
            pass
        try:
            state.unread.pop(peer, None)
        except Exception:
            pass
        try:
            if peer in state.history_last_ids:
                state.history_last_ids.pop(peer, None)
                save_history_index(state.history_last_ids)
        except Exception:
            pass
        # Remove per-user txt history file as well
        try:
            hp_users = _users_dir() / f"{peer}.txt"
            if hp_users.exists():
                try:
                    hp_users.unlink()
                except Exception:
                    pass
        except Exception:
            pass
        if _is_ephemeral():
            return
        hp = _history_path()
        if not hp.exists():
            return
        tmp = hp.with_suffix('.tmp')
        with open(hp, 'r', encoding='utf-8') as fin, open(tmp, 'w', encoding='utf-8') as fout:
            for line in fin:
                try:
                    rec = json.loads(line.strip())
                except Exception:
                    # Keep unknown lines
                    fout.write(line)
                    continue
                if not isinstance(rec, dict):
                    fout.write(line)
                    continue
                # Keep non-DM (rooms) and DMs not involving this peer
                room = rec.get('room')
                if isinstance(room, str) and room:
                    fout.write(line)
                    continue
                frm = rec.get('from')
                to = rec.get('to')
                if frm == peer or to == peer:
                    # Skip records involving this peer
                    continue
                fout.write(line)
        try:
            os.replace(tmp, hp)
        except Exception:
            # Fallback: remove tmp on failure
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        logging.getLogger('client').exception("Failed to purge local history for %s", peer)


def load_local_history(state: 'ClientState') -> None:
    p = _history_path()
    if _is_ephemeral() or not p.exists():
        state.history_last_ids = load_history_index()
        state.history_loaded = True
        return
    try:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                # Determine channel
                room = rec.get('room')
                to = rec.get('to')
                frm = rec.get('from')
                msg_id = rec.get('id')
                text = rec.get('text', '')
                ts = float(rec.get('ts') or time.time())
                if isinstance(room, str) and room:
                    # Group message
                    chan = room
                    m = ChatMessage('in' if frm != state.self_id else 'out', text, ts, sender=frm, msg_id=msg_id)
                else:
                    # Private: choose peer id
                    peer = to if frm == state.self_id else frm
                    if not peer:
                        continue
                    chan = str(peer)
                    direction = 'out' if frm == state.self_id else 'in'
                    m = ChatMessage(direction, text, ts, sender=frm, msg_id=msg_id)
                lst = state.conversations.setdefault(chan, [])
                lst.append(m)
                # Track last ids per channel
                if isinstance(msg_id, int):
                    state.history_last_ids[chan] = max(state.history_last_ids.get(chan, 0), int(msg_id))
        # Also load index (for channels that might be missing ids)
        idx = load_history_index()
        for k, v in idx.items():
            if k not in state.history_last_ids:
                state.history_last_ids[k] = int(v)
    except Exception:
        logging.getLogger('client').exception("Failed to load local history")
    state.history_loaded = True


def _load_config_full() -> dict:
    cfg_path = _config_path()
    if cfg_path.exists():
        try:
            return json.load(open(cfg_path, 'r', encoding='utf-8'))
        except Exception:
            logging.getLogger('client').exception("Failed to load config")
            return {}
    return {}


def load_config() -> Optional[Tuple[str, int, bool]]:
    cfg_path = _config_path()
    if cfg_path.exists():
        try:
            data = _load_config_full()
            host = data.get('host', '127.0.0.1')
            port = int(data.get('port', 7777))
            tls = bool(data.get('tls', False))
            logging.getLogger('client').debug("Loaded config: %s:%s", host, port)
            return host, port, tls
        except Exception:
            logging.getLogger('client').exception("Failed to load config")
            return None
    return None


def save_config(host: str, port: int, *, tls: Optional[bool] = None) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = _load_config_full()
        data['host'] = host
        data['port'] = port
        if tls is not None:
            data['tls'] = bool(tls)
        json.dump(data, open(_config_path(), 'w', encoding='utf-8'))
        logging.getLogger('client').debug("Saved config: %s:%s", host, port)
    except Exception:
        logging.getLogger('client').exception("Failed to save config")


def save_update_url(url: str) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = _load_config_full()
        data['update_url'] = url
        json.dump(data, open(_config_path(), 'w', encoding='utf-8'))
        logging.getLogger('client').debug("Saved update URL: %s", url)
    except Exception:
        logging.getLogger('client').exception("Failed to save update url")


def _prompt(text: str, default: Optional[str] = None) -> str:
    prompt = f"{text}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    try:
        val = input(prompt).strip()
    except EOFError:
        val = ''
    return val or (default or '')


# ===== File browser preferences (persisted in client_config.json) =====
def _get_fb_prefs() -> dict:
    try:
        data = _load_config_full()
        prefs = {
            'fb_show_hidden0': bool(data.get('fb_show_hidden0', False)),
            'fb_show_hidden1': bool(data.get('fb_show_hidden1', False)),
            'fb_sort0': str(data.get('fb_sort0', 'name')),
            'fb_sort1': str(data.get('fb_sort1', 'name')),
            'fb_dirs_first0': bool(data.get('fb_dirs_first0', True)),
            'fb_dirs_first1': bool(data.get('fb_dirs_first1', True)),
            'fb_reverse0': bool(data.get('fb_reverse0', False)),
            'fb_reverse1': bool(data.get('fb_reverse1', False)),
            'fb_view0': data.get('fb_view0', None),
            'fb_view1': data.get('fb_view1', None),
            'fb_path0': data.get('fb_path0'),
            'fb_path1': data.get('fb_path1'),
            'fb_side': data.get('fb_side', 0),
        }
        return prefs
    except Exception:
        return {
            'fb_show_hidden0': False, 'fb_show_hidden1': False,
            'fb_sort0': 'name', 'fb_sort1': 'name',
            'fb_dirs_first0': True, 'fb_dirs_first1': True,
            'fb_reverse0': False, 'fb_reverse1': False,
            'fb_view0': None, 'fb_view1': None,
            'fb_path0': None, 'fb_path1': None,
            'fb_side': 0,
        }

def _save_fb_prefs(st) -> None:
    try:
        data = _load_config_full()
        data['fb_show_hidden0'] = bool(getattr(st, 'show_hidden0', True))
        data['fb_show_hidden1'] = bool(getattr(st, 'show_hidden1', True))
        data['fb_sort0'] = str(getattr(st, 'sort0', 'name'))
        data['fb_sort1'] = str(getattr(st, 'sort1', 'name'))
        data['fb_dirs_first0'] = bool(getattr(st, 'dirs_first0', False))
        data['fb_dirs_first1'] = bool(getattr(st, 'dirs_first1', False))
        data['fb_reverse0'] = bool(getattr(st, 'reverse0', False))
        data['fb_reverse1'] = bool(getattr(st, 'reverse1', False))
        data['fb_view0'] = getattr(st, 'view0', None)
        data['fb_view1'] = getattr(st, 'view1', None)
        data['fb_path0'] = str(getattr(st, 'path0', ''))
        data['fb_path1'] = str(getattr(st, 'path1', ''))
        data['fb_side'] = int(getattr(st, 'side', 0))
        json.dump(data, open(_config_path(), 'w', encoding='utf-8'))
    except Exception:
        logging.getLogger('client').exception('Failed to save file browser prefs')

def _save_fb_prefs_values(**vals) -> None:
    try:
        data = _load_config_full()
        for k, v in vals.items():
            data[k] = v
        json.dump(data, open(_config_path(), 'w', encoding='utf-8'))
    except Exception:
        logging.getLogger('client').exception('Failed to save fb prefs (fallback)')


# ===== File manager (F7) — simple one-pane picker =====
_FM_TIME_SORT_KEYS = {
    'mtime', 'modified',
    'ctime', 'changed', 'added',
    'created',
}


@dataclass
class FileManagerState:
    path: str
    show_hidden: bool = False
    sort: str = 'name'
    dirs_first: bool = True
    reverse: bool = False
    view: Optional[str] = None  # None|'dirs'|'files'
    # Derived listing (filtered view)
    items: List[Tuple[str, bool]] = field(default_factory=list)      # filtered
    all_items: List[Tuple[str, bool]] = field(default_factory=list)  # unfiltered
    err: Optional[str] = None
    index: int = 0
    scroll: int = 0  # topmost visible index in items
    # Metadata cache for list rendering (name -> (size_str, mtime_str))
    meta_base: str = ""
    meta: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    # Input modes: browse|filter|goto
    mode: str = 'browse'
    filter_text: str = ''
    goto_text: str = ''
    # Used to restore selection when going up
    last_dir: Optional[str] = None


def _fm_norm_path(p: str) -> str:
    try:
        # Use absolute() (abspath) instead of resolve() to avoid per-component fs calls.
        # The file manager doesn't need symlink resolution; it just needs a stable absolute path.
        return str(Path(p or '.').expanduser().absolute())
    except Exception:
        try:
            return str(Path('.').absolute())
        except Exception:
            return str(Path('.'))


def _fm_is_time_sort(sort: str) -> bool:
    try:
        return str(sort or '').strip().lower() in _FM_TIME_SORT_KEYS
    except Exception:
        return False


def _fm_list_dir_opts(
    path: str,
    *,
    show_hidden: bool = False,
    sort: str = 'name',
    dirs_first: bool = True,
    reverse: bool = False,
    view: Optional[str] = None,
) -> Tuple[List[Tuple[str, bool]], Optional[str]]:
    """List a directory for the F7 file manager.

    Returns (items, err). Always includes '..' when parent exists.
    """
    base = _fm_norm_path(path)
    p = Path(base)
    err: Optional[str] = None
    items: List[Tuple[str, bool]] = []
    out: List[Tuple[str, bool]] = []
    try:
        with os.scandir(p) as it:
            for ent in it:
                try:
                    name = ent.name
                except Exception:
                    continue
                if name in ('.',):
                    continue
                if (not show_hidden) and name.startswith('.'):
                    continue
                is_dir = False
                try:
                    is_dir = bool(ent.is_dir(follow_symlinks=False))
                except Exception:
                    is_dir = False
                if view == 'dirs' and (not is_dir):
                    continue
                if view == 'files' and is_dir:
                    continue
                out.append((name, is_dir))
    except PermissionError:
        return ([('..', True)] if p.parent and p.parent != p else [], "Permission denied")
    except FileNotFoundError:
        return ([('..', True)] if p.parent and p.parent != p else [], "Not found")
    except Exception as e:
        return ([('..', True)] if p.parent and p.parent != p else [], str(e))

    if _fm_is_time_sort(sort):
        s = str(sort or '').strip().lower()

        def time_key(entry: Tuple[str, bool]) -> float:
            try:
                full = p / entry[0]
                st = full.stat()
                if s in ('mtime', 'modified'):
                    return float(st.st_mtime)
                if s in ('created',):
                    ts = getattr(st, 'st_birthtime', None)
                    return float(ts if ts is not None else st.st_ctime)
                return float(st.st_ctime)
            except Exception:
                return 0.0

        out.sort(key=time_key, reverse=bool(reverse))
    else:
        out.sort(key=lambda e: e[0].lower(), reverse=bool(reverse))

    if dirs_first:
        out = [e for e in out if e[1]] + [e for e in out if not e[1]]

    # Parent row
    try:
        if p.parent and p.parent != p:
            items.append(('..', True))
    except Exception:
        pass
    items.extend(out)
    return items, err


def _fm_filter_items(items: List[Tuple[str, bool]], filter_text: str) -> List[Tuple[str, bool]]:
    parts = [p for p in (filter_text or '').strip().lower().split() if p]
    if not parts:
        return items
    out: List[Tuple[str, bool]] = []
    for name, is_dir in items:
        if name == '..':
            out.append((name, is_dir))
            continue
        low = (name or '').lower()
        if all(p in low for p in parts):
            out.append((name, is_dir))
    return out


def _fm_relist(fm: FileManagerState, *, prefer_name: Optional[str] = None, rescan: bool = True) -> None:
    try:
        # Normalize path once per relist (avoid repeated resolve() in draw loop).
        try:
            norm = _fm_norm_path(fm.path)
            if norm and norm != fm.path:
                fm.path = norm
        except Exception:
            pass
        cur_name = prefer_name
        if cur_name is None and fm.items:
            cur_name = fm.items[max(0, min(int(fm.index), len(fm.items) - 1))][0]
        if rescan:
            # Reset per-directory metadata cache when path changes.
            try:
                base_key = str(fm.path or "")
                if getattr(fm, "meta_base", "") != base_key:
                    fm.meta_base = base_key
                    fm.meta.clear()
            except Exception:
                pass
            items, err = _fm_list_dir_opts(
                fm.path,
                show_hidden=bool(fm.show_hidden),
                sort=str(fm.sort or 'name'),
                dirs_first=bool(fm.dirs_first),
                reverse=bool(fm.reverse),
                view=fm.view,
            )
            fm.all_items = items
            fm.err = err
        # Apply filter over cached directory listing (avoid hitting the FS on each keystroke).
        fm.items = _fm_filter_items(getattr(fm, "all_items", []) or [], fm.filter_text)
        if cur_name:
            try:
                names = [n for (n, _) in fm.items]
                if cur_name in names:
                    fm.index = int(names.index(cur_name))
                else:
                    fm.index = min(int(fm.index), max(0, len(fm.items) - 1))
            except Exception:
                fm.index = min(int(fm.index), max(0, len(fm.items) - 1))
        else:
            fm.index = min(int(fm.index), max(0, len(fm.items) - 1))
        if fm.index < 0:
            fm.index = 0
    except Exception:
        fm.all_items = []
        fm.items = []
        fm.err = "Ошибка чтения каталога"
        fm.index = 0
        fm.scroll = 0


def _fm_window_start(idx: int, total: int, rows: int) -> int:
    rows = max(1, int(rows))
    total = max(0, int(total))
    if total <= rows:
        return 0
    idx = max(0, min(int(idx), total - 1))
    start = idx - rows // 2
    start = max(0, min(start, total - rows))
    return start


def _fm_human_size(bytes_val: int) -> str:
    try:
        units = ['B', 'K', 'M', 'G', 'T', 'P']
        b = float(bytes_val)
        for u in units:
            if b < 1024.0:
                return f"{b:.0f}{u}"
            b /= 1024.0
        return f"{b:.0f}E"
    except Exception:
        return str(bytes_val)


def _fm_fmt_mtime(ts: float) -> str:
    try:
        return time.strftime('%b %e %H:%M', time.localtime(ts))
    except Exception:
        return ''


def _fm_ensure_visible(fm: FileManagerState, rows: int) -> None:
    """Ensure fm.index is within bounds and visible within fm.scroll..fm.scroll+rows-1."""
    try:
        rows = max(1, int(rows))
    except Exception:
        rows = 1
    total = len(getattr(fm, 'items', []) or [])
    if total <= 0:
        fm.index = 0
        fm.scroll = 0
        return
    fm.index = max(0, min(int(getattr(fm, 'index', 0)), total - 1))
    max_scroll = max(0, total - rows)
    fm.scroll = max(0, min(int(getattr(fm, 'scroll', 0)), max_scroll))
    if fm.index < fm.scroll:
        fm.scroll = fm.index
    elif fm.index >= (fm.scroll + rows):
        fm.scroll = fm.index - rows + 1
    fm.scroll = max(0, min(int(fm.scroll), max_scroll))


def _fm_file_meta(fm: FileManagerState, base: Path, name: str) -> Tuple[str, str]:
    """Return (size_str, mtime_str) for a file entry, with caching."""
    try:
        base_key = str(getattr(fm, "meta_base", "") or "")
        cur_key = str(base)
        if base_key != cur_key:
            fm.meta_base = cur_key
            fm.meta.clear()
    except Exception:
        pass
    try:
        cached = fm.meta.get(name)
        if cached is not None:
            return cached
    except Exception:
        cached = None
    size_s = ''
    mtime_s = ''
    try:
        st = (base / name).stat()
        size_s = _fm_human_size(int(st.st_size))
        mtime_s = _fm_fmt_mtime(float(st.st_mtime))
    except Exception:
        pass
    try:
        fm.meta[name] = (size_s, mtime_s)
    except Exception:
        pass
    return size_s, mtime_s


def _draw_file_manager_modal(stdscr, state: "ClientState", *, top_line: int = 1) -> None:
    """Draw the F7 file manager overlay.

    This is intentionally lightweight (no chat/history rendering behind it).
    """
    try:
        h, w = stdscr.getmaxyx()
    except Exception:
        return
    try:
        fm = getattr(state, 'file_browser_state', None)
        if not isinstance(fm, FileManagerState):
            prefs = _get_fb_prefs()
            start = _fm_norm_path('.')
            try:
                p0 = prefs.get('fb_path0')
                if p0 and os.path.isdir(str(p0)):
                    start = _fm_norm_path(str(p0))
                else:
                    home = _fm_norm_path('~')
                    if os.path.isdir(home):
                        start = home
            except Exception:
                pass
            fm = FileManagerState(
                path=start,
                show_hidden=bool(prefs.get('fb_show_hidden0', False)),
                sort=str(prefs.get('fb_sort0', 'name')),
                dirs_first=bool(prefs.get('fb_dirs_first0', True)),
                reverse=bool(prefs.get('fb_reverse0', False)),
                view=prefs.get('fb_view0', None),
            )
            _fm_relist(fm)
            state.file_browser_state = fm
    except Exception:
        return

    header_y = int(top_line)
    input_y = header_y + 1
    list_top = header_y + 2
    hint_y = max(0, h - 1)
    list_rows = max(1, hint_y - list_top)

    total = len(fm.items)
    _fm_ensure_visible(fm, list_rows)
    total = len(fm.items)
    idx = int(getattr(fm, 'index', 0)) if total else 0
    start = int(getattr(fm, 'scroll', 0)) if total else 0
    end = min(total, start + list_rows)

    # Header line: keep right-side flags visible, truncate path from the left.
    try:
        mode_lab = {
            'browse': 'Файлы',
            'filter': 'Файлы (поиск)',
            'goto': 'Файлы (путь)',
        }.get(str(fm.mode or 'browse'), 'Файлы')
    except Exception:
        mode_lab = 'Файлы'
    try:
        sort_key = str(fm.sort or 'name').strip().lower()
        sort_lab = 'имя' if sort_key == 'name' else 'дата'
        sort_order = '↓' if bool(fm.reverse) else '↑'
    except Exception:
        sort_lab = 'имя'
        sort_order = '↑'
    flags = f"скрытые:{'Вкл' if fm.show_hidden else 'Выкл'}  сорт:{sort_lab}{sort_order}"
    avail = max(0, w - display_width(flags) - 3)
    path_disp = right_truncate_to_width(str(fm.path or ''), max(0, avail - display_width(mode_lab) - 3))
    head = f" {mode_lab}: {path_disp}"
    head = pad_to_width(head, avail) + "  " + flags
    try:
        stdscr.addnstr(header_y, 0, pad_to_width(head, w), w, CP.get('header', 0) or curses.A_REVERSE)
    except Exception:
        pass

    # Input line (filter / goto)
    if fm.mode == 'goto':
        prompt = 'Путь: '
        val = fm.goto_text or ''
    else:
        prompt = 'Поиск: '
        val = fm.filter_text or ''
    if fm.mode in ('filter', 'goto'):
        line = f" {prompt}{val}▌"
        attr = CP.get('selected', curses.A_REVERSE)
    else:
        line = f" {prompt}{val}" if val else f" {prompt}/ (нажмите / для поиска)"
        attr = CP.get('div', 0)
    try:
        stdscr.addnstr(input_y, 0, pad_to_width(line, w), w, attr)
    except Exception:
        pass

    # List rows (fast: show names only; details shown in the hint bar for selected item).
    name_w = max(10, w)
    base = Path(getattr(fm, 'meta_base', '') or (fm.path or '.'))

    y = list_top
    if total == 0:
        try:
            stdscr.addnstr(y, 0, pad_to_width(" (пусто) ", w), w, CP.get('div', 0))
        except Exception:
            pass
        y += 1
    for row_i in range(start, end):
        try:
            name, is_dir = fm.items[row_i]
        except Exception:
            continue
        selected = (row_i == idx)
        marker = '>' if selected else ' '
        if name == '..':
            label = ".. (вверх)"
            is_dir = True
        else:
            label = (name or '') + ('/' if is_dir else '')
        row = pad_to_width(f"{marker}{'/' if is_dir else ' '} {label}", name_w)
        try:
            stdscr.addnstr(y, 0, pad_to_width(row, w), w, CP.get('selected', curses.A_REVERSE) if selected else CP.get('div', 0))
        except Exception:
            pass
        y += 1
    while y < hint_y:
        try:
            stdscr.addnstr(y, 0, ' ' * w, w, CP.get('div', 0))
        except Exception:
            pass
        y += 1

    # Hint bar (includes selected item details to avoid per-row stat calls).
    err = (fm.err or '').strip()
    pos = "0/0"
    sel_info = ""
    try:
        if total > 0:
            pos = f"{idx + 1}/{total}"
            sname, sdir = fm.items[idx]
            if sname == '..':
                sel_info = ".. (вверх)"
            elif sdir:
                sel_info = f"{sname}/"
            else:
                size_s, date_s = _fm_file_meta(fm, base, sname)
                meta_parts = [p for p in (size_s, date_s) if p]
                if meta_parts:
                    sel_info = f"{sname} ({', '.join(meta_parts)})"
                else:
                    sel_info = str(sname or '')
    except Exception:
        pass
    hint = "Enter — открыть/выбрать | Backspace — вверх | / — поиск | Ctrl+L — путь | H — скрытые | S — сорт | R — порядок | D — папки | V — вид | F5 — обновить | Esc — закрыть"
    if sel_info:
        hint = f"{pos} | {sel_info} | {hint}"
    else:
        hint = f"{pos} | {hint}"
    if err:
        hint = f"{err} | {hint}"
    try:
        stdscr.addnstr(hint_y, 0, pad_to_width(' ' + hint + ' ', w), w, CP.get('header', 0) or curses.A_REVERSE)
    except Exception:
        pass


def get_saved_id() -> Optional[str]:
    data = _load_config_full()
    cid = data.get('client_id')
    if cid:
        logging.getLogger('client').debug("Using saved client_id: %s", cid)
    return cid


def set_saved_id(client_id: str) -> None:
    try:
        data = _load_config_full()
        data['client_id'] = client_id
        json.dump(data, open(_config_path(), 'w', encoding='utf-8'))
        logging.getLogger('client').info("Persisted client_id: %s", client_id)
    except Exception:
        logging.getLogger('client').exception("Failed to save client_id")


def _parse_tls_mode() -> str:
    raw = str(os.environ.get('SERVER_TLS') or '').strip().lower()
    if raw in ('0', 'off', 'false', 'no', 'disable', 'disabled'):
        return 'off'
    if raw in ('1', 'on', 'true', 'yes', 'require', 'required'):
        return 'on'
    return 'auto'


def _tls_verify_enabled() -> bool:
    raw = os.environ.get('SERVER_TLS_VERIFY')
    if raw is None:
        return True
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


def _tls_ca_file() -> Optional[str]:
    p = str(os.environ.get('SERVER_TLS_CA_FILE') or '').strip()
    return p or None


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except Exception:
        return False


def _tls_server_hostname_for(host: str) -> Optional[str]:
    override = str(os.environ.get('SERVER_TLS_SNI') or '').strip()
    if override:
        try:
            if re.fullmatch(r"[A-Za-z0-9.-]{1,255}", override):
                return override
        except Exception:
            pass
    h = (host or '').strip()
    if h and not _is_ip_literal(h) and h.lower() not in ('localhost',):
        return h
    try:
        base = _get_update_base_url()
        if base:
            u = urlparse(base)
            if u.hostname:
                return str(u.hostname)
    except Exception:
        pass
    return h or None


def _make_tls_context() -> ssl.SSLContext:
    verify = _tls_verify_enabled()
    ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    cafile = _tls_ca_file()
    if cafile:
        ctx.load_verify_locations(cafile=cafile)
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    try:
        if hasattr(ssl, "TLSVersion"):
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2  # type: ignore[attr-defined]
    except Exception:
        pass
    return ctx


def _candidate_tls_ports(base_port: int) -> List[int]:
    raw = str(os.environ.get('SERVER_TLS_PORT') or '').strip()
    if raw:
        try:
            p = int(raw)
            if 1 <= p <= 65535:
                return [p]
        except Exception:
            pass
    # Default migration port: 7777 -> 7778 (avoids sending TLS handshakes to plaintext port)
    if int(base_port) == 7777:
        return [7778]
    # Unknown port: assume it might already be TLS
    return [int(base_port)]


def _probe_best_endpoint(host: str, port: int, timeout_per_try: float) -> Optional[Tuple[str, int, bool]]:
    tls_mode = _parse_tls_mode()
    if tls_mode == 'off':
        if try_connect(host, port, timeout_per_try, use_tls=False):
            return host, port, False
        return None
    tls_ports = []
    try:
        tls_ports = list(dict.fromkeys(_candidate_tls_ports(int(port))))
    except Exception:
        tls_ports = []
    tls_timeout = max(1.0, float(timeout_per_try))
    for p in tls_ports:
        if try_connect(host, int(p), tls_timeout, use_tls=True):
            return host, int(p), True
    # If TLS is required, do not attempt plaintext
    if tls_mode == 'on':
        if int(port) not in set(tls_ports):
            if try_connect(host, int(port), tls_timeout, use_tls=True):
                return host, int(port), True
        return None
    # AUTO: try plaintext on base port, then TLS on base port as a fallback (TLS-only cutover)
    if try_connect(host, int(port), float(timeout_per_try), use_tls=False):
        return host, int(port), False
    if int(port) not in set(tls_ports):
        if try_connect(host, int(port), tls_timeout, use_tls=True):
            return host, int(port), True
    return None


def discover_server(timeout_per_try: float = 0.5) -> Optional[Tuple[str, int, bool]]:
    # Highest priority: explicit env
    env_addr = os.environ.get('SERVER_ADDR')
    if env_addr:
        try:
            if ':' in env_addr:
                host, p = env_addr.rsplit(':', 1)
                port = int(p)
            else:
                host, port = env_addr, 7777
            logging.getLogger('client').info("Using SERVER_ADDR %s:%s", host, port)
            ep = _probe_best_endpoint(host, port, timeout_per_try)
            if ep:
                return ep
        except Exception:
            logging.getLogger('client').exception("Invalid SERVER_ADDR")

    env_host = os.environ.get('SERVER_HOST')
    env_port = os.environ.get('SERVER_PORT')
    if env_host:
        try:
            port = int(env_port or '7777')
            logging.getLogger('client').info("Trying SERVER_HOST %s:%s", env_host, port)
            ep = _probe_best_endpoint(env_host, port, timeout_per_try)
            if ep:
                return ep
        except Exception:
            logging.getLogger('client').exception("Invalid SERVER_HOST/PORT")

    # Try config first
    cfg = load_config()
    if cfg:
        host, port, tls = cfg
        logging.getLogger('client').info("Trying saved server %s:%s (tls=%s)", host, port, int(bool(tls)))
        if tls:
            if try_connect(host, port, max(1.0, timeout_per_try), use_tls=True):
                return host, port, True
        else:
            ep = _probe_best_endpoint(host, port, timeout_per_try)
            if ep:
                return ep

    # Try defaults (prod hosts only; localhost не используем как fallback)
    candidates = [
        ("yagodka.org", 7777),
        ("168.222.252.108", 7777),
    ]
    for host, port in candidates:
        logging.getLogger('client').info("Probing server %s:%s", host, port)
        ep = _probe_best_endpoint(host, port, timeout_per_try)
        if ep:
            return ep
    return None


def try_connect(host: str, port: int, timeout: float, *, use_tls: bool = False) -> bool:
    try:
        log = logging.getLogger('client.net')
        sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        log.debug("Connecting to %s:%s (tls=%s) timeout=%s", host, port, int(bool(use_tls)), timeout)
        sock.connect((host, port))
        if use_tls:
            ctx = _make_tls_context()
            server_hostname = None
            if _tls_verify_enabled():
                server_hostname = _tls_server_hostname_for(host) or host
            sock = ctx.wrap_socket(sock, server_hostname=server_hostname)
        # Read one line expecting welcome
        f = sock.makefile('rb')
        line = f.readline()
        try:
            f.close()
        except Exception:
            pass
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        sock.close()
        if not line:
            log.warning("Probe got empty response")
            return False
        try:
            msg = json.loads(line.decode('utf-8'))
            ok = msg.get('type') == T.WELCOME
            log.debug("Probe welcome: %s", msg)
            return ok
        except Exception:
            log.exception("Probe response not JSON")
            return False
    except Exception:
        logging.getLogger('client.net').exception("Probe connect failed")
        return False


@dataclass
class ChatMessage:
    direction: str  # 'in' or 'out'
    text: str
    ts: float
    sender: Optional[str] = None
    msg_id: Optional[int] = None
    broadcast: bool = False
    status: Optional[str] = None  # 'sent'|'queued'|'delivered'|'read'


@dataclass
class ClientState:
    self_id: Optional[str] = None
    self_id_lucky: bool = False
    self_id_digits: Optional[int] = None
    # Online contacts (legacy server snapshot)
    contacts: List[str] = field(default_factory=list)
    # Rich roster from server: friend id -> {online: bool, last_seen_at: Optional[str], unread: int}
    roster_friends: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # Full directory of user IDs provided by server (excluding self)
    directory: Set[str] = field(default_factory=set)
    selected_index: int = 0
    conversations: Dict[str, List[ChatMessage]] = field(default_factory=dict)
    input_buffer: str = ""
    input_caret: int = 0
    input_sel_start: int = 0
    input_sel_end: int = 0
    input_cursor_visible: bool = True
    input_cursor_last_toggle: float = 0.0
    history_scroll: int = 0
    # Scroll offset for the left contacts list (topmost visible row index)
    contacts_scroll: int = 0
    # Last computed visible height of the left contacts pane
    last_left_h: int = 0
    # Last computed width of the left contacts pane
    last_left_w: int = 0
    # Cached rows for the left contacts pane (performance): rebuild only when inputs change.
    _contact_rows_rev: int = 0
    _contact_rows_cache_rev: int = -1
    _contact_rows_cache: List[str] = field(default_factory=list)
    # File browser modal state
    file_browser_mode: bool = False
    # Two-pane file browser state (MC-like)
    file_browser_side: int = 0  # 0=left, 1=right
    file_browser_path0: str = ''
    file_browser_path1: str = ''
    file_browser_items0: List[Tuple[str, bool]] = field(default_factory=list)  # (name, is_dir)
    file_browser_items1: List[Tuple[str, bool]] = field(default_factory=list)
    file_browser_index0: int = 0
    file_browser_index1: int = 0
    # Pure state holder (modules.)
    file_browser_state: object = None
    # File browser view options menu
    file_browser_view_mode: bool = False
    file_browser_view_index: int = 0
    # File browser top menu (MC-like)
    file_browser_menu_mode: bool = False
    file_browser_menu_top: int = 0  # 0=тумблер активной панели, 1=сортировка/вид, 2=вторая панель
    file_browser_menu_index: int = 0
    # Geometry of top menu labels for precise mouse hit-testing: [(x_start,x_end), ...]
    file_browser_menu_pos: List[Tuple[int, int]] = field(default_factory=list)
    # Main top hotkeys hit-testing: list of (name, x_start, x_end) for 'F1'..'F7'
    main_hotkeys_pos: List[Tuple[str, int, int]] = field(default_factory=list)
    # Fallback preferences (when FILE_BROWSER_FALLBACK is True)
    file_browser_show_hidden0: bool = True
    file_browser_show_hidden1: bool = True
    file_browser_sort0: str = 'name'
    file_browser_sort1: str = 'name'
    file_browser_dirs_first0: bool = False
    file_browser_dirs_first1: bool = False
    file_browser_reverse0: bool = False
    file_browser_reverse1: bool = False
    file_browser_view0: Optional[str] = None   # None|'dirs'|'files'
    file_browser_view1: Optional[str] = None
    file_browser_settings_mode: bool = False
    file_browser_settings_side: int = 0
    file_browser_settings_mode: bool = False
    file_browser_settings_side: int = 0
    status: str = "Connecting..."
    authed: bool = False
    auth_pw: str = ""
    last_submit_pw: str = ""
    # Последний запрос авторизации (для повторной отправки при переподключении)
    last_auth_id: str = ""
    login_pending_since: float = 0.0
    login_retry_count: int = 0
    # Версия сервера
    server_version: Optional[str] = None
    # Client integrity vs server
    last_local_sha: Optional[str] = None
    last_server_sha: Optional[str] = None
    last_integrity_size_ok: bool = False
    last_integrity_hash_ok: bool = False
    # Auth UI
    auth_mode: str = "login"  # "login" or "register"
    login_field: int = 0  # login: 0=id,1=pw ; register: 0=pw1,1=pw2
    id_input: str = ""
    pw1: str = ""
    pw2: str = ""
    login_msg: str = ""
    # Notifications / непрочитанные
    unread: Dict[str, int] = field(default_factory=dict)
    # Друзья (авторизованные контакты)
    friends: Dict[str, bool] = field(default_factory=dict)
    # Поиск
    search_mode: bool = False
    search_query: str = ""
    search_results: List[dict] = field(default_factory=list)
    # Запросы авторизации
    pending_requests: List[str] = field(default_factory=list)
    authz_prompt_from: Optional[str] = None
    # Исходящие запросы авторизации (ожидают решения собеседника)
    pending_out: Set[str] = field(default_factory=set)
    # Преференции уведомлений
    muted: Set[str] = field(default_factory=set)
    blocked: Set[str] = field(default_factory=set)
    # Заглушённые групповые чаты (локально; серверных предпочтений для групп нет)
    group_muted: Set[str] = field(default_factory=set)
    # Заглушённые доски (локально)
    board_muted: Set[str] = field(default_factory=set)
    # Peers that have blocked the current user
    blocked_by: Set[str] = field(default_factory=set)
    # Скрытые заблокированные контакты (не показывать в левом списке)
    hidden_blocked: Set[str] = field(default_factory=set)
    # Меню действий
    action_menu_mode: bool = False
    action_menu_peer: Optional[str] = None
    action_menu_options: List[str] = field(default_factory=list)
    action_menu_index: int = 0
    # Просмотр участников (группы/доски)
    members_view_mode: bool = False
    members_view_title: str = ""
    members_view_entries: List[str] = field(default_factory=list)
    members_view_target: Optional[str] = None
    # Простая модалка для уведомлений (центр экрана)
    modal_message: Optional[str] = None
    # Update notification modal (manual update; no auto-download/restart)
    update_prompt_mode: bool = False
    update_prompt_latest: Optional[str] = None
    update_prompt_reason: Optional[str] = None
    # Don't re-open the same update modal again after dismissing it (per session)
    update_prompt_dismissed_latest: Optional[str] = None
    # Профиль (модальное окно)
    profile_mode: bool = False
    profile_field: int = 0  # 0=name, 1=handle
    profile_name_input: str = ""
    profile_handle_input: str = ""
    # Профили контактов: id -> {display_name, handle}
    profiles: Dict[str, Dict[str, Optional[str]]] = field(default_factory=dict)
    # Просмотр карточки выбранного пользователя
    profile_view_mode: bool = False
    profile_view_id: Optional[str] = None
    # Статусы онлайн/оффлайн
    statuses: Dict[str, bool] = field(default_factory=dict)
    # Помощь
    help_mode: bool = False
    # Selection state (reserved for future copy/selection UX)
    select_active: bool = False
    sel_anchor_y: int = -1
    sel_anchor_x: int = -1
    sel_cur_y: int = -1
    sel_cur_x: int = -1
    # Last drawn history geometry + lines snapshot for selection mapping
    last_hist_y: int = 0
    last_hist_x: int = 0
    last_hist_h: int = 0
    last_hist_w: int = 0
    # Полное число строк истории после разбиения по ширине (для скролла)
    last_history_lines_count: int = 0
    last_lines: List[str] = field(default_factory=list)
    last_start: int = 0
    # История: локальный индекс последних id по каналам
    history_last_ids: Dict[str, int] = field(default_factory=dict)
    history_loaded: bool = False
    # Mouse capture mode (wheel + in-app selection). Always enabled.
    mouse_enabled: bool = True
    # Prefer raw SGR/X10 mouse parsing instead of curses.KEY_MOUSE/getmouse().
    # Helps on some macOS curses builds where KEY_MOUSE mapping is flaky.
    mouse_raw: bool = False
    # ===== Debug overlay (F12): last key/mouse events and raw sequences =====
    debug_mode: bool = False
    debug_lines: List[str] = field(default_factory=list)
    debug_last_key: str = ""
    debug_last_seq: str = ""
    debug_last_mouse: str = ""
    # Mouse diagnostics
    mouse_events_total: int = 0
    mouse_last_seen_ts: float = 0.0
    # tmux mouse diagnostics (when running under tmux)
    tmux_mouse: str = ""
    # Incremental ESC sequence buffer (SGR/X10 across frames)
    esc_seq_buf: str = ""
    esc_seq_started_at: float = 0.0
    # Optional VI-style navigation (J/K). Disabled by default.
    vi_keys: bool = False
    # Search flow: action modal for found user
    search_action_mode: bool = False
    search_action_peer: Optional[str] = None
    search_action_options: List[str] = field(default_factory=list)
    search_action_index: int = 0
    search_action_step: str = "choose"  # choose|waiting|accepted|declined
    # Groups: group_id -> {name, owner_id, members}
    groups: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # Boards: board_id -> {name, owner_id, handle, members?}
    boards: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # Pending group join requests for groups owned by the current user: gid -> set(user_id)
    group_join_requests: Dict[str, Set[str]] = field(default_factory=dict)
    # Group create modal state
    group_create_mode: bool = False
    group_create_field: int = 0
    group_name_input: str = ""
    group_members_input: str = ""
    # Board create modal state
    board_create_mode: bool = False
    board_create_field: int = 0  # 0=name, 1=handle
    board_name_input: str = ""
    board_handle_input: str = ""
    # Group pre-validation flow (resolve @handles -> ids via search)
    group_verify_mode: bool = False
    group_verify_tokens: List[str] = field(default_factory=list)
    group_verify_map: Dict[str, Optional[str]] = field(default_factory=dict)
    group_verify_pending: Set[str] = field(default_factory=set)
    # Track last group creation intent to invite non-friends
    last_group_create_name: Optional[str] = None
    last_group_create_intended: Set[str] = field(default_factory=set)
    last_group_create_gid: Optional[str] = None
    # Group manage modal
    group_manage_mode: bool = False
    group_manage_gid: Optional[str] = None
    group_manage_field: int = 0  # 0=name, 1=members (readonly)
    group_manage_name_input: str = ""
    group_manage_member_count: int = 0
    # Board manage modal
    board_manage_mode: bool = False
    board_manage_bid: Optional[str] = None
    board_manage_field: int = 0  # 0=name, 1=handle
    board_manage_name_input: str = ""
    board_manage_handle_input: str = ""
    # Input history for Up/Down browsing
    input_history: List[str] = field(default_factory=list)
    input_history_index: int = -1  # -1: not browsing; otherwise index in input_history
    # Suggestions/typeahead
    suggest_mode: bool = False
    suggest_kind: str = ""  # 'slash' | 'file'
    suggest_items: List[str] = field(default_factory=list)
    suggest_index: int = 0
    suggest_start: int = 0
    suggest_end: int = 0
    # Debounce for /search requests
    last_search_sent: float = 0.0
    board_manage_member_count: int = 0
    # Board participant management
    board_member_add_mode: bool = False
    board_member_add_bid: Optional[str] = None
    board_member_add_input: str = ""
    board_member_remove_mode: bool = False
    board_member_remove_bid: Optional[str] = None
    board_member_remove_options: List[str] = field(default_factory=list)
    board_member_remove_index: int = 0
    # Board invite prompt (incoming)
    board_invite_mode: bool = False
    board_invite_bid: Optional[str] = None
    board_invite_name: str = ""
    board_invite_from: Optional[str] = None
    board_invite_index: int = 0
    # Pending board invites to display under "Ожидают авторизацию": bid -> {name, from}
    board_pending_invites: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Track pending invites to differentiate consensual join vs forced add
    board_recent_invites: Set[str] = field(default_factory=set)
    # Pending group invites: gid -> {name, from}
    group_pending_invites: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Consent modal for unexpected add
    board_added_consent_mode: bool = False
    board_added_bid: Optional[str] = None
    board_added_index: int = 0
    # Known boards set to detect new boards from snapshot events
    known_boards: Set[str] = field(default_factory=set)
    # Mark that initial boards snapshot has been processed to avoid consent modal on first load
    boards_initialized: bool = False
    # Group participant management
    group_member_add_mode: bool = False
    group_member_add_gid: Optional[str] = None
    group_member_add_input: str = ""
    group_member_remove_mode: bool = False
    group_member_remove_gid: Optional[str] = None
    group_member_remove_options: List[str] = field(default_factory=list)
    group_member_remove_index: int = 0
    # Authorization UX helpers
    lock_selection_peer: Optional[str] = None  # keep selection anchored to this peer after auth until ack
    suppress_auto_menu: bool = False          # prevent auto-open of action menu for next pending contact
    # Remember peers for which incoming auth menu was dismissed via Esc to avoid re-opening on navigation.
    authz_menu_snoozed: Set[str] = field(default_factory=set)
    # History probes: peers for which we sent a server-side history check after re-auth
    history_probe_peers: Set[str] = field(default_factory=set)
    # Outgoing authorization requests that should show persistent overlay in the chat
    authz_out_pending: Set[str] = field(default_factory=set)
    # Live search (F3) status
    search_live_id: Optional[str] = None
    search_live_ok: bool = False
    # Cursor stability: remember last applied hardware cursor to avoid flicker
    cursor_last_y: int = -1
    cursor_last_x: int = -1
    cursor_last_vis: int = 0
    # ===== Файлы: подтверждение отправки при пути в тексте =====
    file_confirm_mode: bool = False
    file_confirm_path: Optional[str] = None
    file_confirm_target: Optional[str] = None
    file_confirm_text_full: str = ""
    file_confirm_index: int = 0  # 0=Да, 1=Нет, 2=Отмена
    file_confirm_prev_text: str = ""
    file_confirm_prev_caret: int = 0
    # ===== Файлы: прогресс скачивания =====
    file_progress_mode: bool = False
    file_progress_name: str = ""
    file_progress_pct: int = 0
    file_progress_file_id: Optional[str] = None
    # ===== Файлы: индексы офферов по каналам (для /ok<ID>) =====
    file_offer_counters: Dict[str, int] = field(default_factory=dict)  # chan -> next int id
    # ===== UI: ESC exit guard (double-press to exit)
    last_esc_ts: float = 0.0
    # File browser: last click info (for simulating double-click on macOS)
    fb_last_click_ts: float = 0.0
    fb_last_click_side: int = -1
    fb_last_click_row: int = -1
    file_offer_map: Dict[str, Dict[int, str]] = field(default_factory=dict)  # chan -> {num: fid}
    file_offer_rev: Dict[str, Tuple[str, int]] = field(default_factory=dict)  # fid -> (chan, num)
    # ===== Файлы: модалка при конфликте имён (заменить?) =====
    file_exists_mode: bool = False
    file_exists_fid: Optional[str] = None
    file_exists_name: str = ""
    file_exists_target: str = ""
    file_exists_index: int = 0  # 0=Заменить, 1=Оставить
    # File transfer (send)
    file_send_path: Optional[str] = None
    file_send_name: Optional[str] = None
    file_send_size: int = 0
    file_send_to: Optional[str] = None
    file_send_room: Optional[str] = None
    file_send_file_id: Optional[str] = None
    file_send_seq: int = 0
    file_send_fp: Optional[object] = None
    file_send_bytes: int = 0
    # File transfer (receive)
    incoming_file_offers: Dict[str, dict] = field(default_factory=dict)  # file_id -> meta
    incoming_by_peer: Dict[str, List[str]] = field(default_factory=dict)  # peer/room -> [file_id]
    file_recv_open: Dict[str, dict] = field(default_factory=dict)  # file_id -> {fp, name, size, from, received, path}

def _compute_wheel_masks() -> Tuple[int, int]:
    """Return (wheel_up_mask, wheel_down_mask) for curses KEY_MOUSE bstate.

    Some macOS builds of curses expose only BUTTON4_* (no BUTTON5_*). In that
    case derive BUTTON5_* masks by shifting BUTTON4_* masks by 6 bits (x64),
    which is how ncurses encodes successive button groups.
    """
    def _sum(names: List[str]) -> int:
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
        # Derive BUTTON5_* by shifting BUTTON4_* bit group by 6 (multiply by 64)
        down = (up << 6)
    return up, down

def _wheel_direction_from_bstate(bstate: int, wheel_up: int, wheel_down: int) -> Optional[str]:
    """Return 'up'/'down' if bstate looks like a wheel event."""
    if bstate & wheel_up:
        return 'up'
    if bstate & wheel_down:
        return 'down'
    # Fallback: BUTTON4/BUTTON5 masks when wheel_* == 0 on some ncurses builds
    b4_mask = (
        getattr(curses, 'BUTTON4_PRESSED', 0)
        | getattr(curses, 'BUTTON4_CLICKED', 0)
        | getattr(curses, 'BUTTON4_RELEASED', 0)
        | getattr(curses, 'BUTTON4_DOUBLE_CLICKED', 0)
        | getattr(curses, 'BUTTON4_TRIPLE_CLICKED', 0)
    )
    b5_mask = (
        getattr(curses, 'BUTTON5_PRESSED', 0)
        | getattr(curses, 'BUTTON5_CLICKED', 0)
        | getattr(curses, 'BUTTON5_RELEASED', 0)
        | getattr(curses, 'BUTTON5_DOUBLE_CLICKED', 0)
        | getattr(curses, 'BUTTON5_TRIPLE_CLICKED', 0)
    )
    if bstate & b4_mask:
        return 'up'
    if bstate & b5_mask:
        return 'down'
    # Last resort: SGR-like bit pattern (0x40 set for wheel)
    if bstate & 0x40:
        return 'down' if (bstate & 1) else 'up'
    return None

def _handle_wheel_scroll(state, stdscr, mx: int, my: int, direction: str) -> None:
    """Apply wheel scroll to contacts or history based on pointer position."""
    try:
        h, w = stdscr.getmaxyx()
        left_w = max(20, min(30, w // 4))
    except Exception:
        left_w = 20
    # Если есть открытая история (selected_id), крутим её; контакты трогаем только при Ctrl
    ctrl_pressed = bool(getattr(state, 'mouse_ctrl_pressed', False))
    try:
        # Сбрасываем флаг, чтобы модификатор не «прилипал» на последующие колёсики
        state.mouse_ctrl_pressed = False
    except Exception:
        pass
    if not ctrl_pressed:
        # Плавный шаг, чтобы не дёргать UI и не сбрасывать статус-бар
        step = 1
        if direction == 'up':
            state.history_scroll += step
        else:
            state.history_scroll = max(0, state.history_scroll - step)
        _clamp_history_scroll(state)
        try:
            state.history_dirty = True
        except Exception:
            pass
        return
    # Ctrl+wheel — прокрутка списка контактов
    rows = build_contact_rows(state)
    vis = int(getattr(state, 'last_left_h', 10) or 10)
    max_start = max(0, len(rows) - max(0, vis))
    cs = max(0, int(getattr(state, 'contacts_scroll', 0)))
    delta = 3 if direction == 'down' else -3
    cs = max(0, min(max_start, cs + delta))
    state.contacts_scroll = cs
    clamp_selection(state, prefer='down', rows=rows)

def _clamp_history_scroll(state: object) -> None:
    """Clamp history_scroll to the latest rendered range and request redraw."""
    try:
        total = int(getattr(state, 'last_history_lines_count', 0))
        hist_h = int(getattr(state, 'last_hist_h', 0))
        max_scroll = max(0, total - max(1, hist_h))
    except Exception:
        max_scroll = 0
    try:
        hs = int(getattr(state, 'history_scroll', 0))
    except Exception:
        hs = 0
    hs = max(0, hs)
    if hs > max_scroll:
        hs = max_scroll
    try:
        state.history_scroll = hs
        state.history_dirty = True
        state._hist_draw_state = None
    except Exception:
        pass


def _maybe_send_message_read(state: object, net: object, peer: Optional[str]) -> None:
    """Send message_read only when it can actually change state (unread>0).

    This prevents flooding the server with DB-heavy mark_read/unread_counts calls
    when the user just navigates the contact list.
    """
    if not peer:
        return
    try:
        if is_separator(peer):
            return
    except Exception:
        pass
    try:
        if peer in getattr(state, 'groups', {}):
            return
    except Exception:
        pass
    try:
        if peer in getattr(state, 'boards', {}):
            return
    except Exception:
        pass
    try:
        unread_map = getattr(state, 'unread', {}) or {}
        unread = int(unread_map.get(str(peer), 0) or 0)
    except Exception:
        unread = 0
    if unread <= 0:
        return
    try:
        # Optimistically clear locally so UI updates immediately.
        unread_map[str(peer)] = 0
        setattr(state, 'unread', unread_map)
    except Exception:
        try:
            state.unread[str(peer)] = 0  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        net.send({"type": "message_read", "peer": str(peer)})
    except Exception:
        pass


_SGR_MOUSE_RE1 = re.compile(r"\x1b\[<(?P<b>\d+);(?P<x>\d+);(?P<y>\d+)(?P<t>[Mm])")
_SGR_MOUSE_RE2 = re.compile(r"\x1b\[(?P<b>\d+);(?P<x>\d+);(?P<y>\d+)(?P<t>[Mm])")
_X10_MOUSE_RE = re.compile(r"\x1b\[M(?P<b>.)(?P<x>.)(?P<y>.)")


def _mouse_mask(names: Tuple[str, ...]) -> int:
    mask = 0
    for name in names:
        mask |= getattr(curses, name, 0)
    return mask


def _mouse_button_masks(button: int) -> Tuple[int, int]:
    mapping = {
        0: (('BUTTON1_PRESSED', 'BUTTON1_CLICKED', 'BUTTON1_DOUBLE_CLICKED'),
            ('BUTTON1_RELEASED', 'BUTTON1_CLICKED', 'BUTTON1_DOUBLE_CLICKED')),
        1: (('BUTTON2_PRESSED', 'BUTTON2_CLICKED', 'BUTTON2_DOUBLE_CLICKED'),
            ('BUTTON2_RELEASED', 'BUTTON2_CLICKED', 'BUTTON2_DOUBLE_CLICKED')),
        2: (('BUTTON3_PRESSED', 'BUTTON3_CLICKED', 'BUTTON3_DOUBLE_CLICKED'),
            ('BUTTON3_RELEASED', 'BUTTON3_CLICKED', 'BUTTON3_DOUBLE_CLICKED')),
    }
    press_names, release_names = mapping.get(button, mapping[0])
    return _mouse_mask(press_names), _mouse_mask(release_names)


def _parse_sgr_mouse(seq: str) -> Optional[Tuple[int, int, int]]:
    """
    Convert an SGR/X10 mouse escape sequence to (x, y, bstate) usable with curses.ungetmouse.
    Returns None if the sequence is not a mouse event.
    """
    # Two common encodings:
    # - SGR 1006: CSI < b ; x ; y M/m   (b is decoded button code)
    # - URXVT 1015: CSI b ; x ; y M/m  (b is X10-style and includes +32 offset)
    m1 = _SGR_MOUSE_RE1.match(seq)
    m2 = None if m1 else _SGR_MOUSE_RE2.match(seq)
    if m1 or m2:
        m = m1 or m2
        try:
            cb_raw = int(m.group('b'))
            mx = max(0, int(m.group('x')) - 1)
            my = max(0, int(m.group('y')) - 1)
            t = m.group('t')
        except Exception:
            return None
        # Decode 1015's +32 offset so bit tests (wheel/motion/modifiers) are correct.
        cb = cb_raw if m1 else (cb_raw - 32)
        if cb < 0:
            return None
    else:
        m2 = _X10_MOUSE_RE.match(seq)
        if not m2:
            return None
        cb = ord(m2.group('b')) - 32
        mx = max(0, ord(m2.group('x')) - 32 - 1)
        my = max(0, ord(m2.group('y')) - 32 - 1)
        t = 'M'

    wheel_up, wheel_down = _compute_wheel_masks()
    bstate = 0

    if cb & 0x40:
        # Wheel event (bit 6 set): map to BUTTON4/BUTTON5 so downstream wheel logic fires
        if (cb & 1) == 0:
            bstate |= wheel_up
        else:
            bstate |= wheel_down
    else:
        btn = cb & 3
        if btn == 3:
            # Many terminals (notably URXVT 1015) encode button release as btn=3 (often with trailing 'M').
            # Treat non-motion releases as a left-button release so contact clicks work reliably.
            # If motion bit is set, this is likely "hover" and should be ignored.
            if cb & 0x20:
                return None
            _press_mask, release_mask = _mouse_button_masks(0)
            bstate |= release_mask or _press_mask
        else:
            press_mask, release_mask = _mouse_button_masks(btn)
            if t == 'm':
                bstate |= release_mask or press_mask
            else:
                bstate |= press_mask or release_mask

    if bstate == 0:
        return None
    return mx, my, bstate


class NetworkClient(threading.Thread):
    def __init__(self, host: str, port: int, incoming: queue.Queue, *, use_tls: bool = False):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.use_tls = bool(use_tls)
        self.incoming = incoming
        self.outgoing: "queue.Queue[dict]" = queue.Queue()
        self.stop_event = threading.Event()
        self.sock: Optional[socket.socket] = None
        self.file_r = None
        self.file_w = None
        self._prebuffer: bytes = b""
        self.sent_count = 0
        self.recv_count = 0
        self._tls_ctx: Optional[ssl.SSLContext] = None
        self._tls_server_hostname: Optional[str] = None
        try:
            self._max_frame_bytes = int(os.environ.get("MSG_MAX_BYTES", "65536"))
        except Exception:
            self._max_frame_bytes = 65536
        if self.use_tls:
            try:
                self._tls_ctx = _make_tls_context()
                if _tls_verify_enabled():
                    self._tls_server_hostname = _tls_server_hostname_for(self.host) or self.host
            except Exception:
                logging.getLogger('client.net').exception("Failed to init TLS context; falling back to plaintext")
                self.use_tls = False
                self._tls_ctx = None
                self._tls_server_hostname = None

    def run(self):
        log = logging.getLogger('client.net')
        backoff = 1.0
        was_connected = False
        while not self.stop_event.is_set():
            try:
                # Signal attempt
                self.incoming.put({"type": "net_status", "status": "reconnecting"})
                raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                log.info("Connecting to %s://%s:%s", "tls" if self.use_tls else "tcp", self.host, self.port)
                raw_sock.settimeout(5.0)
                raw_sock.connect((self.host, self.port))
                if self.use_tls and self._tls_ctx is not None:
                    tls_sock = self._tls_ctx.wrap_socket(
                        raw_sock,
                        server_hostname=(self._tls_server_hostname if _tls_verify_enabled() else None),
                        do_handshake_on_connect=False,
                    )
                    tls_sock.settimeout(5.0)
                    tls_sock.do_handshake()
                    tls_sock.setblocking(False)
                    self.sock = tls_sock
                else:
                    raw_sock.setblocking(False)
                    self.sock = raw_sock

                # Expect welcome
                welcome = self._read_line_blocking(timeout=5.0)
                if not welcome:
                    log.error("No welcome from server")
                    raise RuntimeError("No welcome from server")
                try:
                    msg = json.loads(welcome.decode('utf-8'))
                    if msg.get('type') == T.WELCOME:
                        log.info("Welcome received: %s", msg)
                        self.incoming.put(msg)
                        self.incoming.put({"type": "net_status", "status": "connected"})
                        was_connected = True
                    else:
                        log.error("Unexpected first message: %s", msg)
                        raise RuntimeError("Unexpected first message")
                except Exception as e:
                    log.exception("Invalid welcome parse")
                    raise RuntimeError("Invalid welcome") from e

                last_ping = time.time()
                # Main loop: read + write
                buffer = self._prebuffer or b""
                out_buf = b""
                while not self.stop_event.is_set():
                    try:
                        pending = 0
                        try:
                            if hasattr(self.sock, "pending"):
                                pending = int(getattr(self.sock, "pending")())  # type: ignore[misc]
                        except Exception:
                            pending = 0
                        r_watch = [] if pending > 0 else [self.sock]
                        w_watch = [self.sock] if out_buf else []
                        rlist, wlist, _ = select.select(r_watch, w_watch, [], 0.1)
                    except Exception:
                        rlist, wlist, pending = ([], [], 0)
                    if pending > 0 or rlist:
                        try:
                            data = self.sock.recv(4096)
                        except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
                            data = b""
                        except Exception:
                            log.exception("Recv failed")
                            break
                        if data:
                            buffer += data
                            try:
                                if self._max_frame_bytes and len(buffer) > self._max_frame_bytes:
                                    log.error("Incoming frame exceeds max size (%s bytes); closing", self._max_frame_bytes)
                                    raise RuntimeError("frame_too_large")
                            except Exception:
                                raise
                            while b"\n" in buffer:
                                line, buffer = buffer.split(b"\n", 1)
                                if not line:
                                    continue
                                try:
                                    if self._max_frame_bytes and len(line) > self._max_frame_bytes:
                                        log.error("Incoming line exceeds max size (%s bytes); closing", self._max_frame_bytes)
                                        raise RuntimeError("frame_too_large")
                                except Exception:
                                    raise
                                try:
                                    msg = json.loads(line.decode('utf-8'))
                                    self.recv_count += 1
                                    log.debug("Incoming[%s]: %s", self.recv_count, msg)
                                    self.incoming.put(msg)
                                    try:
                                        self.incoming.put({"type": "debug_log", "dir": "in", "payload": msg, "ts": time.time()})
                                    except Exception:
                                        pass
                                except Exception:
                                    log.exception("Invalid JSON from server")
                                    pass
                        else:
                            # socket closed by server
                            log.warning("Socket closed by server")
                            break
                    # Outgoing: enqueue into out_buf, then flush when socket is writable
                    try:
                        outgoing = self.outgoing.get_nowait()
                    except queue.Empty:
                        outgoing = None
                    if outgoing is not None:
                        try:
                            payload = (json.dumps(outgoing, ensure_ascii=False) + "\n").encode('utf-8')
                            self.sent_count += 1
                            log.debug("Outgoing[%s]: %s", self.sent_count, outgoing)
                            out_buf += payload
                        except Exception:
                            log.exception("Encode outgoing failed")
                            break
                    if out_buf and wlist:
                        try:
                            sent = self.sock.send(out_buf)
                            if sent > 0:
                                out_buf = out_buf[sent:]
                        except (BlockingIOError, ssl.SSLWantWriteError, ssl.SSLWantReadError):
                            pass
                        except Exception:
                            log.exception("Send failed")
                            break

                    # Ping every 20s (optional)
                    if time.time() - last_ping > 20:
                        try:
                            self.send({"type": T.PING})
                        except Exception:
                            log.exception("Ping queue failed")
                            pass
                        last_ping = time.time()

                # Loop will reconnect
                raise RuntimeError("Connection loop ended")
            except Exception:
                log.exception("Network error; will retry")
            finally:
                try:
                    if self.file_r:
                        self.file_r.close()
                except Exception:
                    pass
                try:
                    if self.file_w:
                        self.file_w.close()
                except Exception:
                    pass
                try:
                    if self.sock:
                        self.sock.close()
                except Exception:
                    pass
                self.file_r = self.file_w = None
                self.sock = None
                self._prebuffer = b""
                if was_connected:
                    self.incoming.put({"type": "net_status", "status": "disconnected"})
                # Avoid 'break' inside finally (SyntaxWarning in newer Python);
                # only sleep/backoff when we are not stopping.
                if not self.stop_event.is_set():
                    time.sleep(min(5.0, backoff))
                    backoff = min(5.0, backoff * 1.5)
        # Exit
        self.incoming.put({"type": "disconnected"})

    def _read_line_blocking(self, timeout: float = 5.0) -> Optional[bytes]:
        log = logging.getLogger('client.net')
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline and not self.stop_event.is_set():
            remaining = max(0.0, deadline - time.time())
            try:
                pending = 0
                try:
                    if hasattr(self.sock, "pending"):
                        pending = int(getattr(self.sock, "pending")())  # type: ignore[misc]
                except Exception:
                    pending = 0
                if pending <= 0:
                    rlist, _, _ = select.select([self.sock], [], [], remaining)
                    if not rlist:
                        continue
            except Exception:
                pass
            try:
                chunk = self.sock.recv(4096)
            except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
                continue
            except Exception:
                break
            if not chunk:
                # closed by peer
                break
            buf += chunk
            try:
                if self._max_frame_bytes and len(buf) > self._max_frame_bytes:
                    log.error("Incoming frame exceeds max size (%s bytes); closing", self._max_frame_bytes)
                    return None
            except Exception:
                return None
            if b"\n" in buf:
                line, rest = buf.split(b"\n", 1)
                self._prebuffer = rest
                try:
                    if self._max_frame_bytes and len(line) > self._max_frame_bytes:
                        log.error("Incoming line exceeds max size (%s bytes); closing", self._max_frame_bytes)
                        return None
                except Exception:
                    return None
                return line
        log.debug("_read_line_blocking timeout or closed; got=%r", buf[:200])
        return buf if buf else None

    def send(self, payload: dict):
        _dbg(f"[net send enqueue] {payload}")
        self.outgoing.put(payload)
        try:
            # also log to incoming queue for debug overlay
            self.incoming.put({"type": "debug_log", "dir": "out", "payload": payload, "ts": time.time()})
        except Exception:
            pass

    def stop(self):
        self.stop_event.set()


def wrap_text(text: str, width: int) -> List[str]:
    if width <= 0:
        return [text]
    lines: List[str] = []
    for raw_line in text.splitlines() or [""]:
        s = raw_line
        while len(s) > width:
            # try to break at last space within width
            cut = s.rfind(' ', 0, width)
            if cut == -1:
                cut = width
            lines.append(s[:cut])
            s = s[cut:].lstrip()
        lines.append(s)
    return lines


def request_history_if_needed(state, net, chan: Optional[str], force: bool = False) -> None:
    """Fetch history for channel/peer.

    - force=True: always request using last known id (default 0).
    - force=False: request only when нет локальной переписки.
    """
    if not chan or not isinstance(chan, str) or chan.startswith('BINV:') or chan.startswith('GINV:') or chan.startswith('JOIN:'):
        return
    try:
        if chan in getattr(state, 'history_fetching', set()):
            return
        if not force:
            conv = state.conversations.get(chan, [])
            if conv:
                return
        # Room vs peer
        try:
            since = int(getattr(state, 'history_last_ids', {}).get(chan, 0))
        except Exception:
            since = 0
        if (chan in getattr(state, 'groups', {})) or (chan in getattr(state, 'boards', {})) or chan.startswith('b-'):
            net.send({"type": "history", "room": chan, "since_id": since})
        else:
            net.send({"type": "history", "peer": chan, "since_id": since})
        try:
            state.history_fetching.add(chan)
        except Exception:
            pass
    except Exception:
        pass


def apply_format_to_input(state, kind: str, link_text: str = "", link_url: str = "") -> None:
    """Apply formatting to selection/word in the input buffer and update caret."""
    try:
        text = getattr(state, 'input_buffer', '') or ''
        caret = int(getattr(state, 'input_caret', len(text)))
        sel_start = int(getattr(state, 'input_sel_start', caret))
        sel_end = int(getattr(state, 'input_sel_end', caret))
    except Exception:
        text = getattr(state, 'input_buffer', '') or ''
        caret = len(text)
        sel_start = sel_end = caret
    res = apply_text_format(kind, text, caret, sel_start, sel_end, link_text=link_text, link_url=link_url)
    try:
        new_text = res.text
        new_caret = int(getattr(res, 'caret', len(new_text)))
    except Exception:
        new_text = text
        new_caret = caret
    try:
        state.input_buffer = new_text
        state.input_caret = new_caret
        state.input_sel_start = new_caret
        state.input_sel_end = new_caret
    except Exception:
        pass


def copy_to_clipboard(text: str) -> bool:
    try:
        # Try macOS pbcopy
        if shutil.which('pbcopy'):
            p = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
            p.communicate(text.encode('utf-8'))
            return p.returncode == 0
        # Try Wayland wl-copy
        if shutil.which('wl-copy'):
            p = subprocess.Popen(['wl-copy'], stdin=subprocess.PIPE)
            p.communicate(text.encode('utf-8'))
            return p.returncode == 0
        # Try xclip
        if shutil.which('xclip'):
            p = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
            p.communicate(text.encode('utf-8'))
            return p.returncode == 0
        # Try xsel
        if shutil.which('xsel'):
            p = subprocess.Popen(['xsel', '--clipboard', '--input'], stdin=subprocess.PIPE)
            p.communicate(text.encode('utf-8'))
            return p.returncode == 0
    except Exception:
        logging.getLogger('client').exception("Clipboard copy failed")
    return False


def capture_screen_text(stdscr) -> str:
    """Capture the currently visible curses screen as plain text."""
    try:
        h, w = stdscr.getmaxyx()
    except Exception:
        return ""
    encoding = locale.getpreferredencoding(False) or "utf-8"
    lines: List[str] = []
    for y in range(max(0, int(h))):
        raw = b""
        try:
            raw = stdscr.instr(y, 0, max(1, int(w)))
        except Exception:
            try:
                raw = stdscr.instr(y, 0, max(1, int(w) - 1))
            except Exception:
                try:
                    raw = stdscr.instr(y, 0)
                except Exception:
                    raw = b""
        try:
            if isinstance(raw, bytes):
                s = raw.decode(encoding, errors="replace")
            else:
                s = str(raw)
        except Exception:
            try:
                s = str(raw)
            except Exception:
                s = ""
        s = s.replace("\x00", "")
        lines.append(s.rstrip())
    return "\n".join(lines).rstrip()


def copy_screen_to_clipboard(stdscr) -> Tuple[bool, Optional[Path]]:
    """Copy full screen dump to clipboard; fallback to saving a text file."""
    text = capture_screen_text(stdscr)
    if not text:
        return False, None
    if copy_to_clipboard(text):
        return True, None
    if _is_ephemeral():
        return False, None
    try:
        ts = time.strftime("%Y%m%d-%H%M%S")
        p = _logs_dir() / f"screen{_profile_suffix()}.{ts}.txt"
        p.write_text(text + "\n", encoding="utf-8")
        return False, p
    except Exception:
        logging.getLogger('client').exception("Failed to save screen dump")
        return False, None


def extract_selection_text(state: object, end_y: Optional[int] = None, end_x: Optional[int] = None) -> str:
    """Extract selected text from the last drawn history window.

    Uses state.sel_anchor_*, state.sel_cur_* and state.last_hist_*.
    Selection is clamped to the visible history area.
    """
    try:
        hist_y = int(getattr(state, 'last_hist_y', 0))
        hist_x = int(getattr(state, 'last_hist_x', 0))
        hist_h = int(getattr(state, 'last_hist_h', 0))
        hist_w = int(getattr(state, 'last_hist_w', 0))
    except Exception:
        return ""
    if hist_h <= 0 or hist_w <= 0:
        return ""
    try:
        ay = int(getattr(state, 'sel_anchor_y', hist_y))
        ax = int(getattr(state, 'sel_anchor_x', hist_x))
    except Exception:
        ay, ax = hist_y, hist_x
    try:
        cy = int(end_y if end_y is not None else getattr(state, 'sel_cur_y', ay))
        cx = int(end_x if end_x is not None else getattr(state, 'sel_cur_x', ax))
    except Exception:
        cy, cx = ay, ax
    y0, y1 = sorted([ay, cy])
    x0, x1 = sorted([ax, cx])
    # Clamp to history area
    y0 = max(y0, hist_y)
    y1 = min(y1, hist_y + hist_h - 1)
    x0 = max(x0, hist_x)
    x1 = min(x1, hist_x + hist_w - 1)
    if y0 > y1 or x0 > x1:
        return ""
    # Single cell -> whole line (matches mouse behavior)
    if y0 == y1 and x0 == x1:
        x0 = hist_x
        x1 = hist_x + hist_w - 1
    try:
        last_lines = list(getattr(state, 'last_lines', []) or [])
    except Exception:
        last_lines = []
    selection_lines: List[str] = []
    for sy in range(y0, y1 + 1):
        idx = sy - hist_y
        if 0 <= idx < len(last_lines):
            line_obj = last_lines[idx]
            line = "".join(line_obj) if isinstance(line_obj, list) else str(line_obj)
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
    return "\n".join(selection_lines).rstrip("\n")


# Tokens used to render separators inside the contacts list
SEP1 = "::SEP1::"       # search results header marker
SEP2 = "::SEP2::"       # offline header marker
SEP3 = "::SEP3::"       # boundary before 'unauthorized incoming' section
SEP4 = "::SEP4::"       # boundary before 'waiting outgoing' section
SEP5 = "::SEP5::"       # boundary before 'groups' section
SEP6 = "::SEP6::"       # header before 'online' friends section
SEP7 = "::SEP7::"       # header before 'blocked' section
SEP8 = "::SEP8::"       # header before 'group join requests' section
SEP9 = "::SEP9::"       # header before 'boards' section


def _touch_contact_rows(state: ClientState) -> None:
    """Invalidate cached contact rows (left pane)."""
    try:
        state._contact_rows_rev = int(getattr(state, '_contact_rows_rev', 0)) + 1
    except Exception:
        pass
    # Keep authz snooze set bounded to currently pending incoming requests.
    try:
        snoozed = getattr(state, "authz_menu_snoozed", None)
        pending = set(getattr(state, "pending_requests", []) or [])
        if isinstance(snoozed, set):
            snoozed.intersection_update(pending)
    except Exception:
        pass


def build_contact_rows(state: ClientState) -> List[str]:
    """Build rows for the left contacts panel.

    Normal mode order:
    - SEP5, groups...
    - SEP6, online friends...
    - SEP2, offline friends...
    - SEP3, pending incoming...
    - SEP4, pending outgoing...
    In search mode: IDs only.
    """
    # Hot path: avoid rebuilding/sorting the entire contacts list on every UI tick.
    try:
        rev = int(getattr(state, '_contact_rows_rev', 0))
        cache_rev = int(getattr(state, '_contact_rows_cache_rev', -1))
        cached = getattr(state, '_contact_rows_cache', None)
        if cache_rev == rev and isinstance(cached, list):
            return cached
    except Exception:
        pass
    # Only show authorized contacts (friends) and groups, plus unauthorized sections.
    # When search is active, prepend a "Поиск" group with results but keep the regular list visible.
    rows: List[str] = []
    search_ids: List[str] = []
    if state.search_mode:
        try:
            search_ids = [str(item.get('id')) for item in (state.search_results or []) if item and item.get('id')]
        except Exception:
            search_ids = []
        if search_ids:
            rows += [SEP1] + search_ids
    # Compute blocked union (both directions)
    try:
        blocked_set = set(state.blocked) | set(getattr(state, 'blocked_by', set()))
    except Exception:
        blocked_set = set()
    # Do not include self id
    try:
        if state.self_id:
            blocked_set.discard(state.self_id)
    except Exception:
        pass

    # Boards (if any)
    try:
        if getattr(state, 'boards', {}):
            rows += [SEP9] + sorted(list(state.boards.keys()))
    except Exception:
        pass
    # Groups (if any)
    try:
        if state.groups:
            rows += [SEP5] + sorted(list(state.groups.keys()))
    except Exception:
        pass
    # Pending group join requests (owner only; entries as JOIN:<gid>:<uid>)
    try:
        gj = getattr(state, 'group_join_requests', {}) or {}
        join_tokens: List[str] = []
        for gid, reqs in gj.items():
            for uid in list(reqs or set()):
                join_tokens.append(f"JOIN:{gid}:{uid}")
        if join_tokens:
            rows += [SEP8] + join_tokens
    except Exception:
        pass
    # Primary source: server-provided friends roster
    if state.roster_friends:
        exclude = set(search_ids) if state.search_mode else set()
        online = sorted([fid for fid, info in state.roster_friends.items() if info.get('online') and fid != state.self_id and fid not in exclude and fid not in blocked_set])
        offline = sorted([fid for fid, info in state.roster_friends.items() if not info.get('online') and fid != state.self_id and fid not in exclude and fid not in blocked_set])
        if online:
            rows += [SEP6] + online
        if offline:
            rows += [SEP2] + offline
    else:
        # Strict fallback: show only known friends (if we have them); otherwise don't show online snapshot of everyone
        friends_ids = set(state.friends.keys())
        if state.search_mode:
            friends_ids -= set(search_ids)
        if friends_ids:
            friends_ids = friends_ids - blocked_set
            online_list = sorted([fid for fid in friends_ids if fid != state.self_id and state.statuses.get(fid)])
            offline_list = sorted([fid for fid in friends_ids if fid != state.self_id and not state.statuses.get(fid)])
            if online_list:
                rows += [SEP6] + online_list
            if offline_list:
                rows += [SEP2] + offline_list
    # Append unauthorized users group (pending in/out), excluding already friends
    try:
        pending_in = set(state.pending_requests)
    except Exception:
        pending_in = set()
    try:
        pending_out = set(state.pending_out)
    except Exception:
        pending_out = set()
    # Если контакт одновременно в входящих и исходящих — отдаём приоритет входящим
    # (чтобы можно было сразу авторизовать). Исключаем пересечение из исходящих.
    try:
        pending_out = pending_out - pending_in
    except Exception:
        pass
    # Incoming requests (Неавторизованные)
    unauth_in = sorted(list(pending_in - set(state.roster_friends.keys()) - {state.self_id or ''} - blocked_set))
    if state.search_mode:
        unauth_in = [x for x in unauth_in if x not in set(search_ids)]
    if unauth_in:
        rows += [SEP3] + unauth_in
    # Outgoing requests (Ждут авторизации) + pending board/group invites
    unauth_out = sorted(list(pending_out - set(state.roster_friends.keys()) - {state.self_id or ''} - blocked_set))
    if state.search_mode:
        unauth_out = [x for x in unauth_out if x not in set(search_ids)]
    try:
        binv_tokens = [f"BINV:{bid}" for bid in sorted((getattr(state, 'board_pending_invites', {}) or {}).keys())]
    except Exception:
        binv_tokens = []
    try:
        ginv_tokens = [f"GINV:{gid}" for gid in sorted((getattr(state, 'group_pending_invites', {}) or {}).keys())]
    except Exception:
        ginv_tokens = []
    if unauth_out or binv_tokens or ginv_tokens:
        rows += [SEP4] + unauth_out + binv_tokens + ginv_tokens

    # Blocked section (both blocked and blocked_by)
    try:
        hidden_blk = set(getattr(state, 'hidden_blocked', set()))
    except Exception:
        hidden_blk = set()
    blocked_list = sorted([x for x in list(blocked_set) if x not in hidden_blk])
    if state.search_mode:
        blocked_list = [x for x in blocked_list if x not in set(search_ids)]
    if blocked_list:
        rows += [SEP7] + blocked_list
    try:
        state._contact_rows_cache = rows
        state._contact_rows_cache_rev = int(getattr(state, '_contact_rows_rev', 0))
    except Exception:
        pass
    return rows


def is_separator(token: Optional[str]) -> bool:
    return token in (SEP1, SEP2, SEP3, SEP4, SEP5, SEP6, SEP7, SEP8, SEP9)


def clamp_selection(state: ClientState, prefer: str = 'down', rows: Optional[List[str]] = None) -> None:
    """Ensure selected_index points to a valid (non-separator) row within bounds.

    prefer: 'down' or 'up' — which direction to prefer when skipping separators.
    """
    if KEYLOG_ENABLED:
        _dbg(f"[clamp_selection] selected_index={getattr(state,'selected_index',None)} prefer={prefer}")
    if rows is None:
        rows = build_contact_rows(state)
    if not rows:
        state.selected_index = 0
        return
    # Bound
    if state.selected_index >= len(rows):
        state.selected_index = max(0, len(rows) - 1)
    if state.selected_index < 0:
        state.selected_index = 0
    # Skip separators
    if is_separator(rows[state.selected_index]):
        if prefer == 'up':
            # try move up first
            i = state.selected_index
            while i >= 0 and is_separator(rows[i]):
                i -= 1
            if i >= 0:
                state.selected_index = i
            else:
                # then move down
                i = state.selected_index
                while i < len(rows) and is_separator(rows[i]):
                    i += 1
                state.selected_index = min(i, len(rows) - 1)
        else:
            # try move down first (default behavior)
            i = state.selected_index
            while i < len(rows) and is_separator(rows[i]):
                i += 1
            if i < len(rows):
                state.selected_index = i
            else:
                # move up
                i = state.selected_index
                while i >= 0 and is_separator(rows[i]):
                    i -= 1
                state.selected_index = max(0, i)


def current_selected_id(state: ClientState, rows: Optional[List[str]] = None) -> Optional[str]:
    if rows is None:
        rows = build_contact_rows(state)
    if not rows:
        return None
    if state.selected_index < 0 or state.selected_index >= len(rows):
        return None
    tok = rows[state.selected_index]
    if is_separator(tok):
        return None
    return tok

def open_actions_menu_for_selection(state: ClientState, net) -> None:
    """Open the same actions menu as on KEY_LEFT for the current selection.

    This helper is used by both keyboard (LEFT) and mouse right-click handlers.
    """
    # Close conflicting overlays
    try:
        state.search_action_mode = False
        state.profile_view_mode = False
        state.profile_mode = False
        state.modal_message = None
        state.help_mode = False
    except Exception:
        pass
    sel = current_selected_id(state)
    # If selection is on a separator (None) and there are pending requests only, pick first pending for menu
    try:
        if (sel is None) and state.pending_requests:
            rows = build_contact_rows(state)
            for tok in rows:
                if (not is_separator(tok)) and (tok in state.pending_requests):
                    sel = tok
                    break
    except Exception:
        pass
    # JOIN token (group join request)
    if sel and isinstance(sel, str) and sel.startswith('JOIN:'):
        try:
            _, gid, rid = sel.split(':', 2)
        except Exception:
            gid, rid = '', ''
        options = ["Принять в чат", "Отклонить", "Профиль пользователя", "Профиль чата"]
        state.action_menu_mode = True
        state.action_menu_peer = sel
        state.action_menu_options = options
        state.action_menu_index = 0
        return
    # Group actions
    if sel and (sel in state.groups):
        g = state.groups.get(sel) or {}
        is_owner = (str(g.get('owner_id') or '') == str(state.self_id or ''))
        options: List[str] = []
        try:
            members = set(g.get('members') or [])
            is_member = (str(state.self_id or '') in members) if members else True
        except Exception:
            is_member = True
        if is_member:
            options.append("Отправить файл")
        # Просмотр участников доступен всем (и членам, и владельцу)
        options.append("Участники")
        if is_owner:
            options.append("Добавить участника")
            options.append("Удалить участника")
            options.append("Рассформировать чат")
            try:
                net.send({"type": "group_info", "group_id": sel})
            except Exception:
                pass
        else:
            options.append("Профиль чата")
        try:
            muted = sel in getattr(state, 'group_muted', set())
            options.append("Включить уведомления" if muted else "Заглушить чат")
        except Exception:
            options.append("Заглушить чат")
        if options:
            state.action_menu_mode = True
            state.action_menu_peer = sel
            state.action_menu_options = options
            state.action_menu_index = 0
        return
    # Board actions
    if sel and (sel in getattr(state, 'boards', {})):
        b = (getattr(state, 'boards', {}) or {}).get(sel) or {}
        is_owner = (str(b.get('owner_id') or '') == str(state.self_id or ''))
        options: List[str] = []
        if is_owner:
            options.append("Отправить файл")
        if not is_owner:
            try:
                muted = sel in getattr(state, 'board_muted', set())
                options.append("Включить уведомления" if muted else "Заглушить доску")
            except Exception:
                options.append("Заглушить доску")
        options.append("Участники")
        options.append("Добавить участника")
        options.append("Удалить участника")
        if is_owner:
            options.append("Рассформировать доску")
        else:
            options.append("Покинуть доску")
        state.action_menu_mode = True
        state.action_menu_peer = sel
        state.action_menu_options = options
        state.action_menu_index = 0
        return
    # Direct/user actions
    if sel:
        options: List[str] = []
        # FILE allowed for friends (server authoritative anyway)
        options.append("Отправить файл")
        options.append("Удалить из контактов")
        options.append("Очистить чат")
        try:
            options.append("Снять заглушку" if sel in state.muted else "Заглушить")
        except Exception:
            options.append("Заглушить")
        try:
            options.append("Разблокировать" if sel in state.blocked else "Заблокировать")
        except Exception:
            options.append("Заблокировать")
        # Authorization related
        try:
            if sel in (state.pending_requests or []):
                options.append("Авторизовать")
                options.append("Отклонить")
                options.append("Заблокировать")
            elif sel in (state.pending_out or set()):
                options.append("Отменить запрос")
                options.append("Заблокировать")
            else:
                options.append("Запросить авторизацию")
        except Exception:
            pass
        state.action_menu_mode = True
        state.action_menu_peer = sel
        state.action_menu_options = options
        state.action_menu_index = 0
        return

def _is_auth_actions_menu(state: ClientState) -> bool:
    try:
        if not state.action_menu_mode:
            return False
        opts = set(state.action_menu_options or [])
        return ('Авторизовать' in opts) and ('Отклонить' in opts)
    except Exception:
        return False


_AUTHZ_MENU_OPTIONS = ["Авторизовать", "Отклонить", "Заблокировать", "Профиль пользователя"]


def _maybe_open_authz_actions_menu(state: ClientState, peer: Optional[str]) -> bool:
    try:
        if not peer:
            return False
        if not isinstance(peer, str):
            return False
        if getattr(state, "search_mode", False):
            return False
        if peer not in (getattr(state, "pending_requests", []) or []):
            return False
        if peer in (getattr(state, "authz_menu_snoozed", set()) or set()):
            return False
        state.action_menu_mode = True
        state.action_menu_peer = peer
        state.action_menu_options = list(_AUTHZ_MENU_OPTIONS)
        state.action_menu_index = 0
        return True
    except Exception:
        return False


def draw_ui(stdscr, state: ClientState):
    # Avoid full-screen erase every frame to reduce flicker; clear specific panes instead.
    # Full erase is still used on explicit resize/full-redraw requests.
    try:
        if getattr(state, '_force_full_redraw', False):
            stdscr.erase()
            state._force_full_redraw = False
    except Exception:
        pass
    try:
        h, w = stdscr.getmaxyx()
    except Exception:
        h, w = (0, 0)
    if DEBUG_LOG_ENABLED:
        try:
            sel = current_selected_id(state)
            conv_len = len(state.conversations.get(sel, [])) if sel else 0
            sig = (sel, conv_len, bool(state.search_mode), bool(getattr(state, 'action_menu_mode', False) or getattr(state, 'search_action_mode', False)))
            global _LAST_DRAW_SIG  # type: ignore
            if sig != _LAST_DRAW_SIG:
                _dbg(f"[draw_ui] sel={sel} conv_len={conv_len} search_mode={state.search_mode} modal={getattr(state,'action_menu_mode',False) or getattr(state,'search_action_mode',False)}")
                _LAST_DRAW_SIG = sig
        except Exception:
            pass
    # Begin cursor frame: hide, then request positions via CURSOR.want
    try:
        CURSOR.begin(stdscr)
    except Exception:
        pass
    # Fast path: F7 file manager is an exclusive modal — avoid rendering the whole chat UI behind it.
    try:
        if (
            getattr(state, 'file_browser_mode', False)
            and bool(getattr(state, 'authed', False))
            and not (
                state.search_action_mode
                or state.action_menu_mode
                or state.profile_mode
                or state.profile_view_mode
                or state.modal_message
                or state.help_mode
                or state.members_view_mode
            )
        ):
            try:
                title = " [F7] Файлы  Enter — открыть/выбрать | Esc — закрыть "
                stdscr.addnstr(0, 0, pad_to_width(title, w), w, CP.get('header', 0) or curses.A_REVERSE)
            except Exception:
                pass
            _draw_file_manager_modal(stdscr, state, top_line=1)
            try:
                CURSOR.apply(stdscr)
            except Exception:
                pass
            stdscr.refresh()
            return
    except Exception:
        pass
    # Blink the caret roughly once every 2 seconds with a short off phase
    # Visible ~1.75s, hidden ~0.25s to avoid perceptible jitter
    try:
        _t = time.time() % 2.0
        blink_on = (_t < 1.75)
        # On macOS rely on terminal's own blinking cursor style (DECSCUSR) — keep visible
        import sys as _sys
        if _sys.platform.startswith('darwin'):
            blink_on = True
    except Exception:
        blink_on = True
    # Track desired hardware cursor for this frame (overlays and chat input)
    hw_cursor = {"vis": 0, "y": None, "x": None}
    # Outer border
    try:
        # vertical borders
        for y in range(0, h):
            stdscr.addch(y, 0, ord('|'), CP.get('div', 0))
            stdscr.addch(y, max(0, w - 1), ord('|'), CP.get('div', 0))
        # bottom border will be replaced by footer status bar
    except Exception:
        pass

    if not state.authed:
        # Login/Register screen
        tabs = [" Вход ", " Регистрация "]
        active = 0 if state.auth_mode == 'login' else 1
        tab_line = ""
        for i, t in enumerate(tabs):
            if i == active:
                tab_line += f"[{t}]  "
            else:
                tab_line += f" {t}   "
        title = f" {tab_line}| TAB/←/→ — переключить вкладку "
        stdscr.addnstr(0, 0, title.ljust(w), w, CP.get('header', 0) or curses.A_REVERSE)
        mid_y = h // 2 - 4
        box_w = min(64, w - 4)
        x = max(2, (w - box_w) // 2)

        def mask(s: str) -> str:
            return '*' * len(s)

        # Draw form frame
        box_h = 8
        # top border
        stdscr.hline(mid_y - 1, x, ord('-'), box_w, CP.get('div', 0))
        # bottom border
        stdscr.hline(mid_y + box_h, x, ord('-'), box_w, CP.get('div', 0))
        # sides
        for yy in range(mid_y, mid_y + box_h):
            stdscr.addch(yy, x - 1, ord('|'))
            stdscr.addch(yy, x + box_w, ord('|'))

        if state.auth_mode == 'login':
            lbl0 = "ID/@логин: "
            lbl1 = "Пароль: "
            prefix0 = "> " if state.login_field == 0 else "  "
            prefix1 = "> " if state.login_field == 1 else "  "
            line0 = (prefix0 + lbl0 + state.id_input)
            line1 = (prefix1 + lbl1 + mask(state.pw1))
            stdscr.addnstr(mid_y, x, line0[: box_w], box_w)
            stdscr.addnstr(mid_y + 2, x, line1[: box_w], box_w)
            hint = "Enter — далее / войти | ESC — выход"
        else:
            lbl1 = "Пароль: "
            lbl2 = "Подтвердите: "
            prefix0 = "> " if state.login_field == 0 else "  "
            prefix1 = "> " if state.login_field == 1 else "  "
            line0 = (prefix0 + lbl1 + mask(state.pw1))
            line1 = (prefix1 + lbl2 + mask(state.pw2))
            stdscr.addnstr(mid_y, x, line0[: box_w], box_w)
            stdscr.addnstr(mid_y + 2, x, line1[: box_w], box_w)
            hint = "Enter — далее / зарегистрироваться | ESC — выход"

        stdscr.addnstr(mid_y + 5, x, hint[: box_w], box_w, CP.get('warn', curses.A_BOLD))
        if state.login_msg:
            stdscr.addnstr(mid_y + 7, x, state.login_msg[: box_w], box_w, CP.get('error', curses.A_BOLD))
        # Caret on the active input field (hardware cursor via controller)
        try:
            # Clamp caret within content box and avoid last column to prevent wrap
            max_off = max(0, min(box_w - 2, (w - 2) - x))
            if state.auth_mode == 'login':
                if state.login_field == 0:
                    cy, cx = (mid_y, x + min(len(line0), max_off))
                else:
                    cy, cx = (mid_y + 2, x + min(len(line1), max_off))
            else:
                if state.login_field == 0:
                    cy, cx = (mid_y, x + min(len(line0), max_off))
                else:
                    cy, cx = (mid_y + 2, x + min(len(line1), max_off))
            try:
                CURSOR.want(cy, cx, 2)
            except Exception:
                pass
        except Exception:
            pass
        # Help overlay also works on auth screen
        if state.help_mode:
            try:
                help_lines = [
                    "Подсказка клавиш:",
                    "F1 — помощь (закрыть: F1/ESC/Enter)",
                    "F2 — профиль",
                    "F3 — поиск (введите ID/@логин, Enter)",
                    "F4 — копировать экран (в буфер)",
                    "Ctrl+J — новая строка (до 4)",
                    "Мышь: ЛКМ×2 — слово, ЛКМ×3 — строка, два клика — диапазон",
                    "Tab/←/→ — переключить вкладку",
                    "Enter — далее | ESC — выход",
                ]
                _draw_center_box(stdscr, help_lines, CP.get('title', curses.A_BOLD))
            except Exception:
                pass
        try:
            CURSOR.apply(stdscr)
        except Exception:
            pass
        stdscr.refresh()
        return

    # (No separate hw_cursor tracking; CURSOR controller is authoritative)

    # Layout sizes
    left_w = max(20, min(30, w // 4))
    # Compute history pane width for wrapping before deciding input height
    hist_y = 2
    hist_x = left_w + 2
    hist_w = w - hist_x - 2
    # Determine how many input lines to show (up to 8), based on wrapping
    # Use split by '\n' to preserve trailing empty lines (so a freshly inserted newline adds a visible empty row)
    raw_input_lines = state.input_buffer.split('\n')
    wrapped_input: List[str] = []
    for idx, raw in enumerate(raw_input_lines):
        chunks = wrap_text(raw, max(1, hist_w - 2))  # reserve 2 cols for "> " or indent
        for j, c in enumerate(chunks):
            prefix = "> " if (idx == 0 and j == 0) else "  "
            wrapped_input.append((prefix + c)[: max(0, hist_w)])
    visible_input = min(8, max(1, len(wrapped_input)))
    toolbar_h = 1  # formatting toolbar
    status_h = 1  # one-line status under the input content
    footer_h = 1  # bottom footer (podval)
    sep_h = 1     # dotted separator above footer
    input_h = 1 + visible_input + toolbar_h + status_h  # divider + content + toolbar + status
    history_h = h - input_h - 2 - sep_h - footer_h  # minus header, contacts title, separator, footer
    if history_h < 1:
        history_h = 1

    # Заголовок с суммой непрочитанного (не учитывать заглушённых)
    unread_total = sum(v for k, v in state.unread.items() if v and k not in state.muted)
    notif = f" | Непроч. ЛС: {unread_total}" if unread_total > 0 else ""
    ver_txt = f" v{CLIENT_VERSION}/srv {state.server_version or '?'} "
    header_attr = CP.get('header', 0) or curses.A_REVERSE
    id_attr = header_attr
    id_txt = state.self_id or '—'
    if getattr(state, 'self_id_lucky', False):
        id_txt = f"{id_txt} ★"
        try:
            id_attr = CP.get('title', curses.A_BOLD) or curses.A_BOLD
        except Exception:
            id_attr = header_attr
    # Clear header line
    try:
        stdscr.addnstr(0, 0, " " * w, w, header_attr)
    except Exception:
        pass
    prefix = " Ваш ID: "
    stdscr.addnstr(0, 0, prefix[:w], min(len(prefix), w), header_attr)
    xcur = min(len(prefix), w)
    if xcur < w:
        chunk = id_txt[: w - xcur]
        stdscr.addnstr(0, xcur, chunk, len(chunk), id_attr)
        xcur += len(chunk)
    rest = f" {ver_txt}|  {state.status}{notif} "
    if xcur < w and rest:
        stdscr.addnstr(0, xcur, rest[: w - xcur], w - xcur, header_attr)
    # Hotkeys hint on the right
    buttons_def = [
        ('F1', 'помощь'),
        ('F2', 'профиль'),
        ('F3', 'поиск'),
        ('F4', 'скрин'),
        ('F5', 'чат'),
        ('F6', 'доска'),
        ('F7', 'файлы'),
    ]
    buttons_attr = CP.get('header', 0) or curses.A_REVERSE
    buttons_hotspots: List[Tuple[str, int, int]] = []
    # Align buttons to the right edge
    total_len = sum(len(f"[ {name} {label} ]") for name, label in buttons_def) + (len(buttons_def) - 1)
    hx = max(0, w - total_len - 1)
    xcur = hx
    for idx, (name, label) in enumerate(buttons_def):
        if xcur >= w:
            break
        text = f"[ {name} {label} ]"
        remaining = w - xcur
        if remaining <= 0:
            break
        chunk = text[:remaining]
        if chunk:
            stdscr.addnstr(0, xcur, chunk, len(chunk), buttons_attr)
            start = xcur
            end = xcur + len(chunk)
            buttons_hotspots.append((name, start, end))
            xcur = end
        if idx < len(buttons_def) - 1 and xcur < w - 1:
            stdscr.addnstr(0, xcur, " ", 1, buttons_attr)
            xcur += 1
    # Compute hotkey hit-boxes for mouse clicks on the top hints
    try:
        state.main_hotkeys_pos = buttons_hotspots
    except Exception:
        pass

    # Decide if any overlay is actually active (only those that render UI). Auto-clear broken/late flags.
    try:
        if state.action_menu_mode and not state.action_menu_options:
            state.action_menu_mode = False
        if state.search_action_mode and not state.search_action_peer:
            state.search_action_mode = False
            state.search_mode = False
            state.search_query = ""
            state.search_results = []
            _touch_contact_rows(state)
        # If поисковый оверлей без вариантов/поиска — тушим его, чтобы не затирал историю
        if state.search_action_mode and (not state.search_mode) and (not state.search_action_options):
            state.search_action_mode = False
            state.search_action_peer = None
            state.search_action_step = 'choose'
            state.search_query = ""
            state.search_results = []
            _touch_contact_rows(state)
        if getattr(state, 'suggest_mode', False) and not getattr(state, 'suggest_items', []):
            state.suggest_mode = False
        # If мы уже вышли из режима поиска/действий, гасим застрявший search_action_mode,
        # чтобы оверлей не затирал историю.
        if state.search_action_mode and (not state.search_mode) and (not state.search_results):
            state.search_action_mode = False
            state.search_action_peer = None
            state.search_action_options = []
            state.search_action_step = 'choose'
    except Exception:
        pass
    def _overlay_rendering(st) -> bool:
        """Return True when modules.modal_std reports an active modal."""
        try:
            from modules.modal_std import modal_active as _modal_active  # type: ignore
            val = bool(_modal_active(st))
            if DEBUG_LOG_ENABLED:
                _dbg(f"[modal_active] val={val} search_mode={getattr(st,'search_mode',False)} "
                     f"search_action_mode={getattr(st,'search_action_mode',False)} help={getattr(st,'help_mode',False)}")
            return val
        except Exception:
            return False

    overlay_active = _overlay_rendering(state)
    # Если modal_std не отразил вспомогательные моды — принудительно считаем оверлей активным
    try:
        if (
            getattr(state, 'profile_mode', False)
            or getattr(state, 'profile_view_mode', False)
            or getattr(state, 'group_create_mode', False)
            or getattr(state, 'board_create_mode', False)
            or getattr(state, 'board_manage_mode', False)
            or getattr(state, 'search_action_mode', False)
            or getattr(state, 'format_link_mode', False)
            or getattr(state, 'update_prompt_mode', False)
        ):
            overlay_active = True
        # F1 help не считаем модалкой для скрима
        if getattr(state, 'help_mode', False):
            overlay_active = False
    except Exception:
        pass
    modal_now = overlay_active

    # Draw an opaque scrim over чат/историю (правая колонка), оставляя контакты видимыми
    if modal_now:
        try:
            state.history_dirty = True
        except Exception:
            pass
        try:
            bg_attr = CP.get('div', 0)
            chat_x = left_w + 1
            chat_w = max(0, w - chat_x - 1)
            if chat_w > 0:
                for y in range(1, max(1, h - 1)):
                    stdscr.addnstr(y, chat_x, " " * chat_w, chat_w, bg_attr)
                # восстановить вертикальный разделитель
                try:
                    stdscr.vline(0, left_w, ord('|'), h, CP.get('div', 0))
                except curses.error:
                    pass
        except Exception:
            pass

    # Vertical divider (hide under modals)
    if not modal_now:
        for y in range(1, h):
            try:
                stdscr.addch(y, left_w, ord('|'), CP.get('div', 0))
            except curses.error:
                pass

    # Контакты или поиск — заголовок
    if state.search_mode:
        base = "Поиск: "
        title = f"{base}{state.search_query}" if state.search_query else f"{base}введите ID и Enter"
    else:
        title = "Контакты"
    # Заголовок контактов должен оставаться видимым всегда
    title_w = max(0, left_w - 1)
    stdscr.addnstr(1, 1, pad_to_width(title, title_w), title_w, CP.get('title', curses.A_BOLD))

    # Helper to build label from profile
    def label_for(uid: str) -> str:
        # Pending board invite token (BINV:<bid>) → blinking bell + board name
        if isinstance(uid, str) and uid.startswith('BINV:'):
            try:
                bid = uid.split(':', 1)[1]
            except Exception:
                bid = uid
            meta = (getattr(state, 'board_pending_invites', {}) or {}).get(bid, {})
            name = str(meta.get('name') or bid)
            try:
                blink_on = int(time.time() * 2) % 2 == 0
            except Exception:
                blink_on = True
            bell = "🔔 " if blink_on else "   "
            return f"{bell}{name}"
        if isinstance(uid, str) and uid.startswith('GINV:'):
            try:
                gid = uid.split(':', 1)[1]
            except Exception:
                gid = uid
            meta = (getattr(state, 'group_pending_invites', {}) or {}).get(gid, {})
            name = str(meta.get('name') or gid)
            try:
                blink_on = int(time.time() * 2) % 2 == 0
            except Exception:
                blink_on = True
            bell = "🔔 " if blink_on else "   "
            return f"{bell}{name}"
        # Group label
        if isinstance(uid, str) and uid.startswith('JOIN:'):
            try:
                _, gid, rid = uid.split(':', 2)
            except Exception:
                return uid
            g = state.groups.get(gid) or {}
            gname = str(g.get('name') or gid)
            prof = state.profiles.get(rid) or {}
            dn = (prof.get('display_name') or '').strip()
            hh = (prof.get('handle') or '').strip()
            who = dn or hh or rid
            return f"⏳ Запрос в чат {gname} от {who}"
        if uid in getattr(state, 'boards', {}):
            b = state.boards.get(uid) or {}
            name = str(b.get('name') or uid)
            handle = str(b.get('handle') or '')
            suffix = f" [{handle[1:]}]" if handle and handle.startswith('@') else (f" [{handle}]" if handle else '')
            return f"# {name}{suffix}"
        if uid in state.groups:
            g = state.groups.get(uid) or {}
            name = str(g.get('name') or uid)
            handle = str(g.get('handle') or '')
            # Показываем только имя чата и, при наличии, [login] (без '@'). ID скрываем
            suffix = f" [{handle[1:]}]" if handle and handle.startswith('@') else (f" [{handle}]" if handle else '')
            return f"# {name}{suffix}"
        prof = state.profiles.get(uid) or {}
        dn = (prof.get('display_name') or '').strip()
        hh = (prof.get('handle') or '').strip()
        # Source of truth for presence: statuses map
        online = bool(state.statuses.get(uid))
        icon = status_icon(online)
        if dn:
            base = f"{icon} {dn}"
            if hh:
                base += f" ({hh})"
            return f"{base} [{uid}]"
        if hh:
            return f"{icon} {hh} [{uid}]"
        return f"{icon} {uid}"

    # Ensure trailing ID fits into the left column by trimming the prefix with ellipsis
    # Use shared fit_contact_label imported at module level (with fallback)

    # Список контактов: чаты + друзья онлайн/оффлайн + секции "Неавторизованные"
    rows = build_contact_rows(state)
    clamp_selection(state, rows=rows)
    contacts_start_y = 2
    # Оставляем 2 последних строки экрана под пунктирный разделитель и футер,
    # чтобы список контактов не перекрывался и не терял нижние элементы.
    visible_rows = max(0, h - contacts_start_y - 2)
    # Clear full contacts area each frame to avoid artifacts and footer flicker on fast navigation.
    try:
        blank_left = " " * max(0, left_w - 1)
        for i in range(max(0, visible_rows)):
            stdscr.addnstr(contacts_start_y + i, 1, blank_left, left_w - 1)
    except Exception:
        pass
    # Запомним последний рассчитанный размер видимой области для клавиш/колёсика
    try:
        state.last_left_h = visible_rows  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        state.last_left_w = int(left_w)  # type: ignore[attr-defined]
    except Exception:
        pass
    total_rows = len(rows)
    # Ограничить смещение прокрутки в допустимые пределы
    try:
        cs = max(0, int(getattr(state, 'contacts_scroll', 0)))
    except Exception:
        cs = 0
    max_start = max(0, total_rows - max(0, visible_rows))
    cs = max(0, min(cs, max_start))
    # Поддержать инвариант «выбранная строка видна»
    if total_rows > 0 and visible_rows > 0:
        if state.selected_index < cs:
            cs = state.selected_index
        elif state.selected_index >= cs + visible_rows:
            cs = state.selected_index - visible_rows + 1
        cs = max(0, min(cs, max_start))
    # Избежать ситуации, когда последний видимый элемент — это разделитель,
    # а реальные элементы находятся ниже и «прячутся».
    if total_rows > 0 and visible_rows > 0:
        try:
            end_idx = min(total_rows - 1, cs + max(1, visible_rows) - 1)
            # Если на последней видимой строке сепаратор и ниже есть элементы — сдвинем окно вниз
            safety = 0
            while end_idx < (total_rows - 1) and is_separator(rows[end_idx]) and cs < max_start:
                cs = min(cs + 1, max_start)
                end_idx = min(total_rows - 1, cs + max(1, visible_rows) - 1)
                safety += 1
                if safety > 8:
                    break
        except Exception:
            pass
    # Сохранить
    try:
        state.contacts_scroll = cs
    except Exception:
        pass
    start_idx = cs
    max_rows = min(total_rows - start_idx, max(0, visible_rows))
    # If modal overlays are active, we normally blank the contacts area to avoid leaks.
    # Exception: keep contacts visible during debug overlay (F12), so we can observe selection live.
    try:
        from modules.modal_std import modal_active as _modal_active  # type: ignore
    except Exception:
        def _modal_active(state):  # type: ignore
            return False
    # Оставляем контакты видимыми даже при модалках (скрываем только правую часть)
    modal_contacts = False
    if modal_contacts:
        try:
            state.history_dirty = True
        except Exception:
            pass
        try:
            for i in range(max(0, visible_rows)):
                stdscr.addnstr(contacts_start_y + i, 1, " " * (left_w - 1), left_w - 1)
        except Exception:
            pass
    else:
        offline_header_drawn = False
        unauth_header_drawn = False
        waiting_header_drawn = False
        online_header_drawn = False
        # Cache a simple hit-map from screen Y -> row index for mouse click selection.
        # This keeps click handling consistent with what's actually drawn (incl. scroll window).
        contacts_y_map: Dict[int, int] = {}
        for i in range(max_rows):
            idx = start_idx + i
            if idx < 0 or idx >= total_rows:
                break
            tok = rows[idx]
            try:
                contacts_y_map[int(contacts_start_y + i)] = int(idx)
            except Exception:
                pass
            if is_separator(tok):
                label = None
                attr_line = CP.get('div', 0)
                if tok == SEP1:
                    label = " Поиск "
                elif tok == SEP2 and not offline_header_drawn:
                    label = " Оффлайн "
                    attr_line = CP.get('error', curses.A_BOLD)
                    offline_header_drawn = True
                elif tok == SEP9:
                    label = " Доски "
                elif tok == SEP5:
                    label = " Чаты "
                elif tok == SEP6 and not online_header_drawn:
                    label = " Онлайн "
                    attr_line = CP.get('success', curses.A_BOLD)
                    online_header_drawn = True
                elif tok == SEP8:
                    label = " Запросы в чат "
                    attr_line = CP.get('warn', curses.A_BOLD)
                elif tok == SEP7:
                    label = " Заблокированные "
                elif tok == SEP3 and not unauth_header_drawn:
                    label = " Неавторизованные "
                    attr_line = CP.get('warn', curses.A_BOLD)
                    unauth_header_drawn = True
                elif tok == SEP4 and not waiting_header_drawn:
                    label = " Ожидают авторизации "
                    waiting_header_drawn = True
                if label is not None:
                    try:
                        width = left_w - 2
                        dash = max(0, (width - len(label)) // 2)
                        line = ("=" * dash) + label + ("=" * (width - dash - len(label)))
                        stdscr.addnstr(contacts_start_y + i, 1, line[: width], width, attr_line)
                    except curses.error:
                        pass
                continue
            attr = curses.A_NORMAL
            if idx == state.selected_index:
                attr = CP.get('selected', curses.A_REVERSE)
            label = label_for(tok)
            try:
                blocked_union = set(state.blocked) | set(getattr(state, 'blocked_by', set()))
            except Exception:
                blocked_union = set()
            if tok in blocked_union:
                if label.startswith('● ') or label.startswith('○ ') or label.startswith('★ '):
                    label = label[2:]
            if (not state.search_mode) and not (isinstance(tok, str) and tok.startswith('JOIN:')):
                cnt = int(state.unread.get(tok, 0))
                muted_group = tok in getattr(state, 'group_muted', set())
                muted_dm = tok in state.muted
                if cnt > 0 and not (muted_group or muted_dm):
                    label = f"{label} ({cnt})"
                    if idx != state.selected_index:
                        attr = CP.get('unread', curses.A_BOLD)
            if not state.search_mode:
                if tok in state.muted or tok in getattr(state, 'group_muted', set()):
                    label = "🔕 " + label
                if (state.friends.get(tok) or tok in state.roster_friends):
                    label = f"★ {label}"
                else:
                    try:
                        pin = tok in set(state.pending_requests)
                    except Exception:
                        pin = False
                    pout = tok in set(getattr(state, 'pending_out', set()))
                    if pin:
                        try:
                            blink_on = int(time.time() * 2) % 2 == 0
                        except Exception:
                            blink_on = True
                        prefix = "🔔 " if blink_on else "   "
                        label = f"{prefix}{label}"
                        if idx != state.selected_index:
                            attr = CP.get('warn', curses.A_BOLD)
                    elif pout:
                        label = f"⏳ {label}"
                if tok in state.blocked:
                    if idx == state.selected_index:
                        label = "⛔ " + label
                    else:
                        label = "⛔ " + label
                    if idx != state.selected_index:
                        attr = CP.get('error', curses.A_BOLD)
            if state.search_mode:
                res = next((r for r in state.search_results if r.get('id') == tok), None)
                if res:
                    if res.get('friend'):
                        label = f"★ {label}"
                    if res.get('online'):
                        label = f"{label} [онлайн]"
            label = fit_contact_label(label, left_w - 2)
            stdscr.addnstr(contacts_start_y + i, 1, pad_to_width(label, left_w - 2), left_w - 2, attr)
            if (
                not is_separator(tok)
                and tok not in getattr(state, 'boards', {})
                and tok not in state.groups
                and tok not in blocked_union
                and not (isinstance(tok, str) and (tok.startswith('BINV:') or tok.startswith('JOIN:')))
            ):
                try:
                    is_on = bool(state.statuses.get(tok))
                    icon = status_icon(is_on)
                    color_attr = CP.get('success', curses.A_BOLD) if is_on else CP.get('error', curses.A_BOLD)
                    stdscr.addch(contacts_start_y + i, 1, ord(icon), color_attr)
                except Exception:
                        pass
        try:
            state.contacts_y_map = contacts_y_map  # type: ignore[attr-defined]
        except Exception:
            pass

    # Заголовок области чата
    chat_title_x = left_w + 2
    # Актуальный выбор
    sel_contact = current_selected_id(state, rows=rows)
    if state.search_mode:
        chat_title = " Режим поиска (Enter — найти) "
    else:
        if sel_contact:
            who_label = label_for(sel_contact)
        else:
            who_label = '—'
        chat_title = f" Чат с: {who_label} "
    # Не мутируем счётчики/состояние в draw-функции: чтение/ack делается в обработчиках событий.
    # Hide chat title under modal overlays to comply with unified modal standard
    if (not _modal_active(state)) or getattr(state, 'help_mode', False):
        chat_w = max(0, w - chat_title_x - 1)
        stdscr.addnstr(1, chat_title_x, pad_to_width(chat_title, chat_w), chat_w, CP.get('title', curses.A_BOLD))
    else:
        try:
            stdscr.addnstr(1, chat_title_x, " " * (w - chat_title_x - 1), w - chat_title_x - 1)
        except Exception:
            pass

    # History box area (uses hist_y/x/w computed above)
    hist_h = max(1, history_h)

    lines: List[str] = []  # plain text lines (for selection/copy)
    line_attrs: List[List[int]] = []  # per-line per-char attrs
    # Map global line index -> color attribute for single checkmark suffix (kept for compatibility)
    check_meta: Dict[int, int] = {}
    # Decide if any modal overlay is active (affects history redraw/blanking)
    # Для любых модалок чат скрываем, чтобы не просвечивал под прозрачными окнами.
    modal_for_history = bool(overlay_active)
    # Справка и поиск: контакты оставляем, чат скрываем
    try:
        if getattr(state, 'help_mode', False) or getattr(state, 'search_mode', False):
            modal_for_history = True
        # В debug режиме принудительно показываем историю (лог) — не считаем модалкой
        if getattr(state, 'debug_mode', False):
            modal_for_history = False
    except Exception:
        pass
    hist_was_blank = bool(getattr(state, '_hist_blank', False))
    # Skip heavy history redraw if nothing changed (selection, scroll, size, search mode)
    redraw_history = True
    cache_sig = None  # signature for heavy history wrapping (excludes scroll)
    try:
        if getattr(state, 'debug_mode', False):
            logs = list(getattr(state, 'debug_lines', []) or [])
            tail = logs[-300:] if len(logs) > 300 else logs
            has_mouse = "?"
            try:
                has_mouse = "1" if curses.has_mouse() else "0"
            except Exception:
                has_mouse = "?"
            try:
                update_url = os.environ.get('UPDATE_URL', '') or ''
            except Exception:
                update_url = ''
            hint = ""
            try:
                if os.environ.get('TMUX'):
                    hint = "TIP: tmux может блокировать клики — попробуйте: tmux set -g mouse on (или CLIENT_TMUX_AUTO_MOUSE=1)"
            except Exception:
                hint = ""
            last_key = getattr(state, 'debug_last_key', '') or "last_key: <none>"
            last_seq = repr(getattr(state, 'debug_last_seq', '') or "") or "''"
            last_mouse = getattr(state, 'debug_last_mouse', '') or "last_mouse: <none>"
            armed = "1" if bool(globals().get("__MOUSE_ARMED__")) else "0"
            try:
                tty_info = f"tty_in={'1' if sys.stdin.isatty() else '0'} tty_out={'1' if sys.stdout.isatty() else '0'}"
            except Exception:
                tty_info = "tty_in=? tty_out=?"
            try:
                tmux_mouse = getattr(state, 'tmux_mouse', '') or ''
            except Exception:
                tmux_mouse = ''
            try:
                me = int(getattr(state, 'mouse_events_total', 0))
                mts = float(getattr(state, 'mouse_last_seen_ts', 0.0))
                ago = (time.time() - mts) if mts else 0.0
                mouse_diag = f"mouse_events={me} last_seen={ago:.1f}s_ago" if me else "mouse_events=0"
            except Exception:
                mouse_diag = "mouse_events=?"
            header = [
                "DEBUG (F12)",
                f"TERM={os.environ.get('TERM','')} TMUX={'1' if os.environ.get('TMUX') else '0'} platform={sys.platform} mouse={'ON' if state.mouse_enabled else 'OFF'} armed={armed} mode={'raw' if getattr(state, 'mouse_raw', False) else 'curses'} has_mouse={has_mouse} {tty_info}",
                f"UPDATE_URL={update_url} tmux_mouse={tmux_mouse} {mouse_diag}",
                hint,
                last_key,
                f"last_seq={last_seq}",
                last_mouse,
                "",
            ]
            hdr = [ln for ln in header if ln]
            body = tail if tail else ["(debug) нет событий — нажмите клавиши, затем F12 выключит режим"]
            # Pin header at the top so environment/last events are always visible.
            room = max(0, hist_h - len(hdr))
            dbg_lines = (hdr[:hist_h] if room <= 0 else (hdr + body[-room:]))
            lines = [(ln, CP.get('title', curses.A_BOLD) if i == 0 else curses.A_NORMAL) for i, ln in enumerate(dbg_lines)]
            state._hist_draw_state = None
            redraw_history = True
        else:
            conv_len = 0
            last_ts = 0.0
            last_status = None
            sel_id = sel_contact
            if (not state.search_mode) and sel_id:
                conv = state.conversations.get(sel_id, [])
                conv_len = len(conv) if conv else 0
                if conv_len:
                    last_ts = getattr(conv[-1], 'ts', 0.0) or 0.0
                    last_status = getattr(conv[-1], 'status', None)
            # Signature for cached wrapped history lines (do not include scroll/height)
            try:
                peer_kind = 'user'
                if sel_id and (sel_id in getattr(state, 'boards', {}) or str(sel_id).startswith('b-')):
                    peer_kind = 'board'
                elif sel_id and (sel_id in getattr(state, 'groups', {})):
                    peer_kind = 'group'
            except Exception:
                peer_kind = 'user'
            cache_sig = (sel_id, conv_len, last_ts, last_status, hist_w, peer_kind)
            cur_hist_sig = (
                sel_id,
                conv_len,
                last_ts,
                last_status,
                state.history_scroll,
                hist_w,
                hist_h,
                state.search_mode,
                modal_for_history,
            )
            prev_hist_sig = getattr(state, '_hist_draw_state', None)
            if prev_hist_sig == cur_hist_sig:
                redraw_history = False
            else:
                state._hist_draw_state = cur_hist_sig
    except Exception:
        redraw_history = True
    if DEBUG_LOG_ENABLED and redraw_history:
        try:
            sel = sel_contact
            conv = state.conversations.get(sel, [])
            sig = (sel, len(conv) if conv else 0, state.history_scroll, modal_for_history)
            global _LAST_HISTORY_SIG  # type: ignore
            if sig != _LAST_HISTORY_SIG:
                _dbg(f"[history] sel={sel} conv_len={len(conv) if conv else 0} redraw={redraw_history} modal_for_history={modal_for_history}")
                _LAST_HISTORY_SIG = sig
        except Exception:
            pass
    # Force redraw if history was previously blanked by an overlay
    try:
        if getattr(state, 'history_dirty', False):
            redraw_history = True
    except Exception:
        pass
    # Selection highlight requires a redraw even if content didn't change.
    try:
        if getattr(state, 'select_active', False):
            redraw_history = True
    except Exception:
        pass
    # Do not rely on hist_was_blank anymore to prevent flicker
    hist_was_blank = False
    # В режиме debug_mode показываем только debug_lines; обычные сообщения не рисуем
    if redraw_history and (not modal_for_history) and (not getattr(state, 'debug_mode', False)):
        # Fast-path: reuse cached wrapped history when only scrolling/selection changed.
        use_cache = False
        try:
            prev_sig = getattr(state, '_hist_cache_sig', None)
            if cache_sig is not None and prev_sig == cache_sig:
                cached_lines = getattr(state, '_hist_cache_lines', None)
                cached_attrs = getattr(state, '_hist_cache_attrs', None)
                cached_meta = getattr(state, '_hist_cache_check_meta', None)
                if isinstance(cached_lines, list) and isinstance(cached_attrs, list) and isinstance(cached_meta, dict):
                    # Avoid O(n) copies on every scroll; cached structures are treated as immutable.
                    lines = cached_lines
                    line_attrs = cached_attrs
                    check_meta = cached_meta
                    use_cache = True
        except Exception:
            use_cache = False
        if not use_cache:
            def compact_identity(uid: str) -> str:
                prof = state.profiles.get(uid) or {}
                name = (prof.get('display_name') or '').strip()
                handle = (prof.get('handle') or '').strip()
                if name:
                    return name
                if handle:
                    return f"({handle})"
                return f"[{uid}]"

            def icon_for(uid: str) -> str:
                try:
                    return status_icon(bool(state.statuses.get(uid)))
                except Exception:
                    return status_icon(False)

            msgs = state.conversations.get(sel_contact, [])
            last_key: Optional[str] = None
            for m in msgs:
                if (sel_contact in state.groups) or (sel_contact in getattr(state, 'boards', {})):
                    sender_id = str(m.sender or '?')
                    key = f"out:self" if m.direction == 'out' else f"in:{sender_id}"
                else:
                    key = f"out:self" if m.direction == 'out' else f"in:{sel_contact}"

                new_group = (key != last_key)
                last_key = key

                t = time.strftime('%H:%M', time.localtime(m.ts))
                if new_group:
                    if m.direction == 'out':
                        who_disp = 'Вы'
                        prefix = f"{t} {who_disp}: "
                    else:
                        if sel_contact in state.groups:
                            uid = str(m.sender or '')
                        else:
                            uid = str(sel_contact)
                        ident = compact_identity(uid)
                        display_ident = ident
                        prefix = f"{t} {icon_for(uid)} {display_ident}: "
                else:
                    prefix = f"{t}: "

                def _formatted_chars(raw: str):
                    s = raw or ""
                    out: List[Tuple[str, int]] = []
                    strong_attr = (CP.get('title', curses.A_BOLD) or curses.A_BOLD)
                    link_attr = (CP.get('title', curses.A_UNDERLINE) or curses.A_UNDERLINE) | curses.A_BOLD

                    while s:
                        if s.startswith("**"):
                            end = s.find("**", 2)
                            if end != -1:
                                for ch in s[2:end]:
                                    out.append((ch, strong_attr))
                                s = s[end + 2:]
                                continue
                        if s.startswith("["):
                            mid = s.find("](")
                            end = s.find(")", mid + 2) if mid != -1 else -1
                            if mid != -1 and end != -1:
                                label = s[1:mid]
                                for ch in label:
                                    out.append((ch, link_attr))
                                s = s[end + 1:]
                                continue
                        out.append((s[0], curses.A_NORMAL))
                        s = s[1:]
                    return out

                ZERO_WIDTH = {"\u200d", "\u200c", "\ufeff", "\u200b", "\u200e", "\u200f", "\ufe0e", "\ufe0f"}

                def _is_zero_width(ch: str) -> bool:
                    if ch in ZERO_WIDTH:
                        return True
                    try:
                        if unicodedata.category(ch) == "Cf":
                            return True
                    except Exception:
                        pass
                    return False

                def _wcw(ch: str) -> int:
                    # Avoid per-char imports (wcwidth may be absent). Use shared display_width() cache.
                    try:
                        if _is_zero_width(ch) or unicodedata.combining(ch):
                            return 0
                    except Exception:
                        pass
                    try:
                        return int(display_width(ch))
                    except Exception:
                        return 1

                def _cells_from_string(s: str, attr: int) -> List[Tuple[str, int, int]]:
                    cells: List[Tuple[str, int, int]] = []
                    for ch in s:
                        if _is_zero_width(ch):
                            if cells:
                                txt, a, w = cells[-1]
                                cells[-1] = (txt + ch, a, w)
                            continue
                        if unicodedata.combining(ch):
                            if cells:
                                txt, a, w = cells[-1]
                                cells[-1] = (txt + ch, a, w)
                            continue
                        w = max(1, _wcw(ch))
                        cells.append((ch, attr, w))
                    return cells

                def _flush_line(cells: List[Tuple[str, int, int]]) -> None:
                    if not cells:
                        return
                    render_chars: List[str] = []
                    attrs: List[int] = []
                    for txt, attr, w in cells:
                        render_chars.append(txt)
                        attrs.extend([attr] * max(1, w))
                        if w > 1:
                            render_chars.extend([" "] * (w - 1))
                    lines.append("".join(render_chars))
                    line_attrs.append(attrs)

                suffix = ''
                suffix_color = None

                content_chars = _formatted_chars(m.text)
                def _strip_bold(attr: int) -> int:
                    try:
                        return attr & ~(curses.A_BOLD | curses.A_STANDOUT)
                    except Exception:
                        return curses.A_NORMAL
                base_attr = _strip_bold(CP.get('success', curses.A_NORMAL)) if m.direction == 'out' else curses.A_NORMAL
                # Build wrapped lines char-by-char with width awareness (combining marks stay inside same cell)
                cur_line_cells: List[Tuple[str, int, int]] = _cells_from_string(prefix, base_attr)
                cur_cols = sum(c[2] for c in cur_line_cells)
                pad_cells = _cells_from_string(" " * len(prefix), base_attr)
                pad_cols = sum(c[2] for c in pad_cells)

                for ch, attr in content_chars:
                    # Preserve explicit newlines inside messages without letting control chars leak into curses
                    # rendering (printing '\n' would move the cursor and corrupt the layout).
                    if ch in ('\n', '\r'):
                        _flush_line(cur_line_cells)
                        cur_line_cells = list(pad_cells)
                        cur_cols = pad_cols
                        continue
                    if ch == '\t':
                        # Expand tab to spaces (simple, predictable in TUI).
                        for _ in range(4):
                            if (cur_cols + 1) > hist_w and cur_line_cells:
                                _flush_line(cur_line_cells)
                                cur_line_cells = list(pad_cells)
                                cur_cols = pad_cols
                            cur_line_cells.append((' ', base_attr | attr, 1))
                            cur_cols += 1
                        continue
                    try:
                        # Replace other ASCII control chars with a visible placeholder.
                        if ch and ord(ch) < 32:
                            ch = '␀'
                    except Exception:
                        pass
                    if _is_zero_width(ch):
                        if cur_line_cells:
                            txt, a, w = cur_line_cells[-1]
                            cur_line_cells[-1] = (txt + ch, a | attr, w)
                        continue
                    if unicodedata.combining(ch):
                        if cur_line_cells:
                            txt, a, w = cur_line_cells[-1]
                            cur_line_cells[-1] = (txt + ch, a | attr, w)
                        continue
                    w = max(1, _wcw(ch))
                    if (cur_cols + w) > hist_w and cur_line_cells:
                        _flush_line(cur_line_cells)
                        cur_line_cells = list(pad_cells)
                        cur_cols = pad_cols
                    cur_line_cells.append((ch, base_attr | attr, w))
                    cur_cols += w
                    if cur_cols >= hist_w:
                        _flush_line(cur_line_cells)
                        cur_line_cells = list(pad_cells)
                        cur_cols = pad_cols

                if suffix:
                    for ch in suffix:
                        if unicodedata.combining(ch):
                            if cur_line_cells:
                                txt, a, w = cur_line_cells[-1]
                                cur_line_cells[-1] = (txt + ch, a, w)
                            continue
                        w = max(1, _wcw(ch))
                        if (cur_cols + w) > hist_w and cur_line_cells:
                            _flush_line(cur_line_cells)
                            cur_line_cells = list(pad_cells)
                            cur_cols = pad_cols
                        cur_line_cells.append((ch, suffix_color or base_attr, w))
                        cur_cols += w
                    if suffix_color is not None:
                        check_meta[len(lines)] = int(suffix_color)
                if cur_line_cells:
                    _flush_line(cur_line_cells)

            # Persist cache for future scroll-only redraws
            try:
                if cache_sig is not None:
                    state._hist_cache_sig = cache_sig
                    state._hist_cache_lines = list(lines)
                    state._hist_cache_attrs = list(line_attrs)
                    state._hist_cache_check_meta = dict(check_meta)
            except Exception:
                pass

    # Apply scrolling with clamping to available range.
    # IMPORTANT: do not overwrite last_history_lines_count when we didn't rebuild `lines`
    # (otherwise first scroll gets clamped to 0 and feels like a 1s "lag" until key repeat).
    try:
        if redraw_history and (not modal_for_history):
            state.last_history_lines_count = len(lines)
    except Exception:
        pass
    # Compute visible window and avoid explicit blanking to prevent flicker
    if redraw_history:
        max_scroll = max(0, len(lines) - hist_h)
        if state.history_scroll > max_scroll:
            state.history_scroll = max_scroll
        start = max(0, len(lines) - hist_h - state.history_scroll)
        end = max(0, min(len(lines), start + hist_h))
        if DEBUG_LOG_ENABLED:
            _dbg(f"[history] redraw lines={len(lines)} start={start} end={end} scroll={state.history_scroll} hist_h={hist_h} modal_for_history={modal_for_history}")
    else:
        start = int(getattr(state, 'last_start', 0))
        end = start + hist_h
    # Decide whether to suppress chat rendering when a modal/overlay is active
    menu_active = modal_for_history
    # Last guard: if overlays are off but menu_active somehow true, reset history_dirty to re-render
    if menu_active and (not modal_for_history):
        try:
            state.history_dirty = True
        except Exception:
            pass
    # If any modal is active, additionally blank the chat title line to avoid leaks
    if menu_active:
        try:
            state.history_dirty = True
        except Exception:
            pass
        try:
            right_w = max(0, w - (left_w + 2) - 1)
            if right_w > 0:
                stdscr.addnstr(1, left_w + 2, " " * right_w, right_w)
        except Exception:
            pass
    # Save geometry and snapshot for selection mapping
    state.last_hist_y, state.last_hist_x, state.last_hist_h, state.last_hist_w = hist_y, hist_x, hist_h, hist_w
    state.last_start = start
    # Clear also the 1-col gap between divider and history (x=left_w+1), otherwise artifacts remain there.
    chat_clear_x = max(0, hist_x - 1)
    chat_clear_w = max(0, w - chat_clear_x - 1)
    selected_attr = CP.get('selected', curses.A_REVERSE)
    base_blank_attr = curses.A_NORMAL

    def _draw_attr_runs(row_y: int, text_line: str, attrs_cells: List[int], sel_range: Optional[Tuple[int, int]]) -> None:
        # Render the row by batching consecutive cells with the same attribute.
        try:
            if len(text_line) < hist_w:
                text_line = text_line + (" " * (hist_w - len(text_line)))
            else:
                text_line = text_line[:hist_w]
        except Exception:
            text_line = (str(text_line or "") + (" " * hist_w))[:hist_w]
        # Skip placeholder cells inserted after wide characters to avoid breaking glyphs and shifting.
        # History lines are built with "wide_char + spaces" to keep column indexing stable; we must not print
        # those placeholder spaces in batched addnstr calls.
        skip_cell = [False] * hist_w
        try:
            for k in range(max(0, hist_w - 1)):
                ch = text_line[k]
                if ch and unicodedata.east_asian_width(ch) in ("W", "F"):
                    skip_cell[k + 1] = True
        except Exception:
            skip_cell = [False] * hist_w
        try:
            if len(attrs_cells) < hist_w:
                attrs_cells = list(attrs_cells) + [base_blank_attr] * (hist_w - len(attrs_cells))
            else:
                attrs_cells = list(attrs_cells[:hist_w])
        except Exception:
            attrs_cells = [base_blank_attr] * hist_w

        col = 0
        while col < hist_w:
            if skip_cell[col]:
                col += 1
                continue
            in_sel = bool(sel_range and sel_range[0] <= col < sel_range[1])
            cur_attr = selected_attr if in_sel else attrs_cells[col]
            end_col = col + 1
            while end_col < hist_w:
                in_sel2 = bool(sel_range and sel_range[0] <= end_col < sel_range[1])
                a2 = selected_attr if in_sel2 else attrs_cells[end_col]
                if a2 != cur_attr:
                    break
                end_col += 1
            try:
                chunk = "".join(text_line[k] for k in range(col, end_col) if not skip_cell[k])
                if chunk:
                    stdscr.addnstr(row_y, hist_x + col, chunk, end_col - col, cur_attr)
            except Exception:
                pass
            col = end_col
    # Очищаем область истории только когда реально перерисовываем, чтобы избежать мигания
    if redraw_history or menu_active:
        blank_row = " " * chat_clear_w
        for j in range(hist_h):
            try:
                stdscr.addnstr(hist_y + j, chat_clear_x, blank_row, chat_clear_w)
            except Exception:
                pass

    if redraw_history or menu_active:
        # Always preserve actual lines; do not blank history to avoid disappearing chat
        def _line_text(idx: int) -> str:
            if idx >= len(lines) or idx < 0:
                return ""
            li = lines[idx]
            try:
                if isinstance(li, list):
                    return "".join(li)
                if isinstance(li, tuple):
                    return str(li[0])
                return str(li)
            except Exception:
                return ""

        def _cache_line(i: int):
            if i >= len(lines):
                return ""
            v = lines[i]
            if isinstance(v, tuple):
                return str(v[0])
            return v
        state.last_lines = [_cache_line(i) for i in range(start, end)]
        state.last_line_attrs = [line_attrs[i] if i < len(line_attrs) else [] for i in range(start, end)]
        try:
            state._hist_blank = False
        except Exception:
            pass
        idx = 0
        for i in range(start, end):
            if menu_active:
                idx += 1
                continue
            line_obj = lines[i] if i < len(lines) else ""
            attrs_line = line_attrs[i] if i < len(line_attrs) else []

            row_attr = None
            if isinstance(line_obj, tuple) and len(line_obj) >= 2:
                try:
                    text_line = str(line_obj[0] or "")
                except Exception:
                    text_line = ""
                try:
                    row_attr = int(line_obj[1])
                except Exception:
                    row_attr = base_blank_attr
                attrs_cells = [row_attr] * hist_w
            else:
                if isinstance(line_obj, list):
                    try:
                        text_line = "".join(line_obj)
                    except Exception:
                        text_line = str(line_obj)
                else:
                    text_line = str(line_obj or "")
                attrs_cells = list(attrs_line) if isinstance(attrs_line, list) else []

            sel_range = None
            if state.select_active:
                try:
                    ay, ax = state.sel_anchor_y, state.sel_anchor_x
                    cy, cx = state.sel_cur_y, state.sel_cur_x
                    y0, y1 = sorted([ay, cy])
                    x0, x1 = sorted([ax, cx])
                    row_y = hist_y + idx
                    if (row_y >= y0) and (row_y <= y1):
                        sel_start = hist_x if row_y > y0 else max(hist_x, x0)
                        sel_end = (hist_x + hist_w - 1) if row_y < y1 else min(hist_x + hist_w - 1, x1)
                        if sel_start <= sel_end:
                            sel_range = (max(0, sel_start - hist_x), min(hist_w, sel_end - hist_x + 1))
                except Exception:
                    sel_range = None

            _draw_attr_runs(hist_y + idx, text_line, attrs_cells, sel_range)

            try:
                if (not state.select_active) and (i in check_meta):
                    suffix = " ✓"
                    start_col = max(0, hist_w - len(suffix))
                    if start_col < hist_w:
                        color_attr = check_meta.get(i) or curses.A_BOLD
                        stdscr.addnstr(hist_y + idx, hist_x + start_col, suffix[: max(0, hist_w - start_col)], max(0, hist_w - start_col), color_attr)
            except Exception:
                pass
            idx += 1
    else:
        # No redraw requested — repaint cached lines with cached attrs
        blank_row = " " * chat_clear_w
        for j in range(hist_h):
            try:
                stdscr.addnstr(hist_y + j, chat_clear_x, blank_row, chat_clear_w)
            except Exception:
                pass
        cached_lines = getattr(state, 'last_lines', [])
        cached_attrs = getattr(state, 'last_line_attrs', [])
        for j in range(min(hist_h, len(cached_lines))):
            line_obj = cached_lines[j] if j < len(cached_lines) else ""
            attrs_line = cached_attrs[j] if j < len(cached_attrs) else []
            if isinstance(line_obj, tuple) and len(line_obj) >= 2:
                try:
                    text_line = str(line_obj[0] or "")
                except Exception:
                    text_line = ""
                try:
                    row_attr = int(line_obj[1])
                except Exception:
                    row_attr = base_blank_attr
                attrs_cells = [row_attr] * hist_w
            else:
                if isinstance(line_obj, list):
                    try:
                        text_line = "".join(line_obj)
                    except Exception:
                        text_line = str(line_obj)
                else:
                    text_line = str(line_obj or "")
                attrs_cells = list(attrs_line) if isinstance(attrs_line, list) else []
            _draw_attr_runs(hist_y + j, text_line, attrs_cells, sel_range=None)
    # Mark history clean after successful redraw (skip when menu blanks)
    if redraw_history and (not menu_active):
        try:
            state.history_dirty = False
        except Exception:
            pass

    # Input area
    input_y = hist_y + hist_h
    # Divider above input box
    stdscr.hline(input_y, left_w + 1, ord('-'), w - left_w - 2, CP.get('div', 0))
    # Clear input area fully to avoid artifacts when line count shrinks
    blank = " " * chat_clear_w
    # Do not clear separator/footer rows
    sep_y = max(0, h - 2)
    for yy in range(input_y + 1, sep_y):
        try:
            stdscr.addnstr(yy, chat_clear_x, blank, chat_clear_w)
        except Exception:
            pass
    # Draw last visible_input lines of wrapped input only when no overlays are active
    input_overlays_active = bool(
        state.search_mode or state.search_action_mode or state.action_menu_mode
        or getattr(state, 'group_create_mode', False) or state.group_manage_mode
        or state.modal_message or getattr(state, 'board_create_mode', False) or getattr(state, 'board_manage_mode', False)
        or getattr(state, 'file_confirm_mode', False) or getattr(state, 'file_progress_mode', False)
        or getattr(state, 'file_browser_mode', False)
        or getattr(state, 'board_invite_mode', False) or getattr(state, 'board_added_consent_mode', False)
        or getattr(state, 'board_member_add_mode', False) or getattr(state, 'board_member_remove_mode', False)
        or getattr(state, 'group_member_add_mode', False) or getattr(state, 'group_member_remove_mode', False)
        or getattr(state, 'format_link_mode', False)
    )
    if not input_overlays_active:
        # Compute caret-aware start row so that caret stays in view
        try:
            width = max(1, hist_w - 2)
            text = state.input_buffer
            caret = max(0, min(len(text), int(getattr(state, 'input_caret', 0))))
            # total display rows for full text
            def _rows_for(t: str) -> int:
                rows = 0
                for raw in t.split('\n') or [""]:
                    chunks = wrap_text(raw, width)
                    rows += max(1, len(chunks))
                return rows
            pre = text[:caret]
            pre_rows = 0
            for raw in pre.split('\n')[:-1]:
                pre_rows += max(1, len(wrap_text(raw, width)))
            cur_line = pre.split('\n')[-1] if pre else ""
            cur_row_offset = len(wrap_text(cur_line, width)) - 1 if cur_line else 0
            total_rows = _rows_for(text)
            display_row = pre_rows + cur_row_offset
            start_row = min(max(0, display_row - visible_input + 1), max(0, total_rows - visible_input))
        except Exception:
            start_row = max(0, len(wrapped_input) - visible_input)
        last_lines = wrapped_input[start_row : start_row + visible_input]
        for i, line in enumerate(last_lines):
            try:
                stdscr.addnstr(input_y + 1 + i, hist_x, line[: hist_w], hist_w)
            except Exception:
                pass
        # Formatting toolbar (Tab to focus, ←/→ to choose, Enter to apply)
        try:
            parts: List[str] = []
            for idx, (code, title, _kind) in enumerate(FORMAT_ACTIONS):
                active = bool(getattr(state, 'format_toolbar_mode', False) and getattr(state, 'format_toolbar_index', 0) == idx)
                parts.append(f"[{code} {title}]" if active else f"{code} {title}")
            bar = " | ".join(parts)
            hint = "Tab — панель/ввод"
            line = bar
            if len(line) + len(hint) + 3 < hist_w:
                line = f"{bar}   {hint}"
            ty = min(input_y + 1 + visible_input, sep_y - 1)
            attr = CP.get('title', curses.A_BOLD) if getattr(state, 'format_toolbar_mode', False) else (CP.get('warn', curses.A_DIM) or curses.A_DIM)
            stdscr.addnstr(ty, hist_x, line[: hist_w], hist_w, attr)
        except Exception:
            pass
        # Status line with typed char count (kept within input box, not overlapping footer)
        try:
            status_text = f"Символов: {len(state.input_buffer)}"
            sy = min(input_y + 1 + visible_input + 1, sep_y - 1)
            stdscr.addnstr(sy, hist_x, status_text[: hist_w], hist_w, CP.get('warn', curses.A_DIM))
        except Exception:
            pass
    # Chat input caret (hardware): do not position when any overlay/menu is active
    overlays_active = bool(
        state.search_mode or state.search_action_mode or state.action_menu_mode
        or getattr(state, 'group_create_mode', False) or state.group_manage_mode
        or state.modal_message
        or getattr(state, 'file_confirm_mode', False) or getattr(state, 'file_progress_mode', False)
        or getattr(state, 'file_browser_mode', False)
        or getattr(state, 'board_invite_mode', False) or getattr(state, 'board_added_consent_mode', False)
        or getattr(state, 'board_member_add_mode', False) or getattr(state, 'board_member_remove_mode', False)
        or getattr(state, 'group_member_add_mode', False) or getattr(state, 'group_member_remove_mode', False)
        or getattr(state, 'format_link_mode', False)
    )
    now_ts = time.time()
    if overlays_active:
        state.input_cursor_visible = False
        state.input_cursor_last_toggle = now_ts
    else:
        chat_input.ensure_cursor_tick(state, now=now_ts)
    try:
        if overlays_active:
            raise Exception("skip_chat_caret")
        # Compute caret position based on input_caret (logical lines, wrapping by width)
        width = max(1, hist_w - 2)
        text = state.input_buffer
        caret = max(0, min(len(text), int(getattr(state, 'input_caret', 0))))
        # total display rows for full text
        def _rows_for(t: str) -> int:
            rows = 0
            for raw in t.split('\n') or [""]:
                chunks = wrap_text(raw, width)
                rows += max(1, len(chunks))
            return rows
        # rows before caret and col in current row
        pre = text[:caret]
        # rows before caret
        pre_rows = 0
        for raw in pre.split('\n')[:-1]:
            pre_rows += max(1, len(wrap_text(raw, width)))
        # current line col
        cur_line = pre.split('\n')[-1] if pre else ""
        cur_row_offset = len(wrap_text(cur_line, width)) - 1 if cur_line else 0
        total_rows = _rows_for(text)
        display_row = pre_rows + cur_row_offset
        # Keep caret in view rather than always showing the last lines
        start_row = min(max(0, display_row - visible_input + 1), max(0, total_rows - visible_input))
        caret_row_in_box = display_row - start_row
        caret_row_in_box = max(0, min(visible_input - 1, caret_row_in_box))
        # caret column within current wrapped line
        col_in_line = len(cur_line) % width
        caret_col = min(2 + col_in_line, max(0, hist_w - 2))
        caret_x = hist_x + caret_col
        caret_y = min(input_y + 1 + caret_row_in_box, sep_y - 2)
        if caret_col < max(0, hist_w - 1):
            blink_flag = bool(getattr(state, 'input_cursor_visible', True))
            CURSOR.want(caret_y, caret_x, 2 if blink_flag else 0)
    except Exception:
        pass

    # Dotted separator line above footer
    try:
        pattern = ("· ") * ((w // 2) + 1)
        stdscr.addnstr(sep_y, 0, pattern[: w], w, CP.get('div', 0))
    except Exception:
        pass
    # Search overlay window (only when no other modal overlays active)
    if state.search_mode and not (state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or state.help_mode):
        title = "Введите @логин или ID"
        prompt = f"> Поиск: {state.search_query}"
        status_line = ""
        status_attr = 0
        try:
            from modules.ui_utils import search_status_line  # type: ignore
        except Exception:
            def search_status_line(q: str, ok: bool, rid: Optional[str] = None):
                try:
                    q = (q or '').strip()
                    if not q:
                        return '', ''
                    qcore = q[1:] if q.startswith('@') else q
                    if len(qcore) < 3:
                        return "Минимум 3 символа для живого поиска", 'warn'
                    if ok and (rid or '').strip():
                        return f"Найден: {rid} — Enter/A: выбрать", 'success'
                    return "Не найден — исправьте ID/@логин", 'error'
                except Exception:
                    return '', ''
        msg_txt, cat = search_status_line(state.search_query or '', bool(state.search_live_ok), state.search_live_id)
        status_line = msg_txt
        if cat == 'warn':
            status_attr = CP.get('warn', curses.A_BOLD) or curses.A_BOLD
        elif cat == 'success':
            status_attr = CP.get('success', curses.A_BOLD) or curses.A_BOLD
        elif cat == 'error':
            status_attr = CP.get('error', curses.A_BOLD) or curses.A_BOLD
        else:
            status_attr = 0
        overlay_lines = [
            title,
            prompt,
            status_line,
            "Esc — закрыть",
        ]
        try:
            _draw_center_box(stdscr, overlay_lines, status_attr if status_line else (CP.get('title', curses.A_BOLD)))
        except Exception:
            pass

    # ===== Подсказки (slash-команды / файловая навигация) =====
    if getattr(state, 'suggest_mode', False) and not (
        state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or state.help_mode
        or getattr(state, 'group_create_mode', False) or state.group_manage_mode or getattr(state, 'board_create_mode', False) or getattr(state, 'board_manage_mode', False)
    ):
        try:
            items = list(getattr(state, 'suggest_items', []) or [])
            idx = max(0, min(len(items) - 1, int(getattr(state, 'suggest_index', 0))))
            kind = getattr(state, 'suggest_kind', '')
            title = "Команды" if kind == 'slash' else "Файлы"
            lines = [f" {title} "]
            if not items:
                lines += ["(нет вариантов)", "Esc — закрыть"]
            else:
                for i, it in enumerate(items[:10]):
                    prefix = "> " if i == idx else "  "
                    if kind == 'slash':
                        nm = str(getattr(it, 'name', None) or str(it))
                        ds = str(getattr(it, 'description', '') or '')
                        # Align names to 12 chars for a neat two-column look
                        label = f"{nm.ljust(12)} {ds}".rstrip()
                    else:
                        label = str(getattr(it, 'name', None) or str(it))
                    lines.append(prefix + label)
                lines.append("Tab/Enter — выбрать | Esc — закрыть | ↑/↓ — выбор")
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass
    # Вставка ссылки — модальное окно
    if getattr(state, 'format_link_mode', False):
        try:
            t = str(getattr(state, 'format_link_text', '') or '')
            u = str(getattr(state, 'format_link_url', '') or '')
            field = int(getattr(state, 'format_link_field', 0))
            lines = [
                " Вставить ссылку ",
                f"{'>' if field == 0 else ' '} Текст: {t or '(необязательно)'}",
                f"{'>' if field == 1 else ' '} URL:   {u or ''}",
                "Enter — вставить | Tab/↑/↓ — поле | Esc — отмена",
            ]
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # Footer status bar at bottom (подвал) с агрегированными счётчиками
    try:
        foot_y = max(0, h - 1)
        # Counters: online/offline among friends, pending in/out
        try:
            if state.roster_friends:
                online_cnt = sum(1 for fid, info in state.roster_friends.items() if info.get('online'))
                offline_cnt = max(0, len(state.roster_friends) - online_cnt)
            else:
                friend_ids = set(state.friends.keys())
                online_cnt = sum(1 for fid in friend_ids if state.statuses.get(fid))
                offline_cnt = max(0, len(friend_ids) - online_cnt)
        except Exception:
            online_cnt = 0
            offline_cnt = 0
        try:
            wait_in = len(state.pending_requests)
        except Exception:
            wait_in = 0
        try:
            wait_out = len(state.pending_out)
        except Exception:
            wait_out = 0
        summary = f"Онлайн: {online_cnt} | Оффлайн: {offline_cnt} | Ожидают: {wait_in} | Отправлено: {wait_out}"
        status_part = (state.status or '').strip()
        # Add contextual hints for search/search-action modes
        hint_part = ""
        try:
            if state.search_action_mode:
                hint_part = " | Подсказки: Enter — выбрать, Esc — закрыть, A — запрос"
            elif state.search_mode:
                hint_part = " | Подсказки: Enter — найти, Esc — закрыть, A — запрос, Y/N — принять/откл."
        except Exception:
            pass
        footer = (f" {summary} — {status_part}{hint_part} " if status_part else f" {summary}{hint_part} ")
        stdscr.addnstr(foot_y, 0, footer.ljust(w), w, CP.get('header', 0) or curses.A_REVERSE)
    except Exception:
        pass

    

    # Profile modal window (skip if another modal overlay is active)
    if state.profile_mode and not (state.search_action_mode or state.action_menu_mode or state.modal_message):
        dn = state.profile_name_input or ""
        hh = state.profile_handle_input or ""
        pointer0 = "> " if state.profile_field == 0 else "  "
        pointer1 = "> " if state.profile_field == 1 else "  "
        id_txt = state.self_id or '—'
        if getattr(state, 'self_id_lucky', False):
            id_txt = f"{id_txt} ★"
        lines = [
            " Профиль ",
            f"{pointer0}Отображаемое имя: {dn}",
            f"{pointer1}Логин (@):        {hh}",
            f"Ваш ID: {id_txt}",
            "Enter — сохранить | ↑/↓ — поле | Esc — закрыть",
        ]
        try:
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
            # No hardware cursor in profile overlay
            try:
                h2, w2 = stdscr.getmaxyx()
                box_w = min(max(len(max(lines, key=len)) + 6, 40), max(40, w2 - 4))
                box_h = len(lines) + 4
                y0 = max(1, (h2 - box_h) // 2)
                x0 = max(2, (w2 - box_w) // 2)
                if state.profile_field == 0:
                    base = f"{pointer0}Отображаемое имя: "
                    cy = y0 + 2 + 1
                    cx = x0 + 2 + len(base) + len(dn)
                else:
                    base = f"{pointer1}Логин (@):        "
                    cy = y0 + 2 + 2
                    cx = x0 + 2 + len(base) + len(hh)
                # (cursor hidden)
            except Exception:
                pass
        except Exception:
            pass

    # File manager modal (F7) — simple one‑pane picker
    if getattr(state, 'file_browser_mode', False) and not (
        state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or state.help_mode or state.members_view_mode
    ):
        _draw_file_manager_modal(stdscr, state, top_line=1)

    # Chat (group) create modal (exclusive)
    if getattr(state, 'group_create_mode', False) and not (state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message):
        try:
            name = getattr(state, 'group_name_input', '')
            members = getattr(state, 'group_members_input', '')
            field = int(getattr(state, 'group_create_field', 0))
            p0 = "> " if field == 0 else "  "
            p1 = "> " if field == 1 else "  "
            # Build placeholder lines; a status line for live validation follows below
            lines = [
                " Создать чат ",
                f"{p0}Название: {name}",
                f"{p1}Участники (ID/@логины, через запятую):",
                f"{members}",
                "Проверка участников:",
                "Enter — создать | Tab/↑/↓ — поле | Esc — закрыть",
            ]
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
            # No hardware cursor in create chat overlay
            try:
                h, w = stdscr.getmaxyx()
                box_w = min(max(len(max(lines, key=len)) + 6, 40), max(40, w - 4))
                box_h = len(lines) + 4
                y0 = max(1, (h - box_h) // 2)
                x0 = max(2, (w - box_w) // 2)
                if field == 0:
                    base = f"{p0}Название: "
                    cy = y0 + 2 + 1
                    cx = x0 + 2 + len(base) + len(name)
                else:
                    cy = y0 + 2 + 3
                    cx = x0 + 2 + len(members)
                # (cursor hidden)
                # Draw live validation line with colored tokens
                try:
                    # Tokens: split by comma/space
                    import re as _re
                    tokens = [t for t in _re.split(r"[\s,]+", members) if t]
                    # compute start position of the validation line (after members input line)
                    vy = y0 + 2 + 4
                    vx = x0 + 2
                    maxw = box_w - 4
                    # Clear line
                    try:
                        stdscr.addnstr(vy, vx, " " * maxw, maxw)
                    except Exception:
                        pass
                    # Compose colored segments
                    cursor_x = vx
                    def _draw_seg(text: str, attr: int):
                        nonlocal cursor_x
                        if cursor_x >= vx + maxw:
                            return
                        room = (vx + maxw) - cursor_x
                        s = text[: max(0, room)]
                        try:
                            stdscr.addnstr(vy, cursor_x, s, len(s), attr)
                        except Exception:
                            pass
                        cursor_x += len(s)
                    # Prefix label
                    _draw_seg("Статусы: ", CP.get('title', curses.A_BOLD) or curses.A_BOLD)
                    for i, t in enumerate(tokens):
                        if i > 0:
                            _draw_seg(", ", curses.A_NORMAL)
                        status = None
                        try:
                            if t in getattr(state, 'group_verify_pending', set()):
                                status = 'pending'
                            elif (getattr(state, 'group_verify_map', {}) or {}).get(t):
                                status = 'ok'
                            else:
                                status = 'bad'
                        except Exception:
                            status = 'bad'
                        attr = curses.A_NORMAL
                        if status == 'ok':
                            attr = CP.get('success', curses.A_BOLD) or curses.A_BOLD
                        elif status == 'pending':
                            attr = CP.get('warn', curses.A_BOLD) or curses.A_BOLD
                        else:
                            attr = CP.get('error', curses.A_BOLD) or curses.A_BOLD
                        _draw_seg(t, attr)
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

    # Board create modal (exclusive)
    if getattr(state, 'board_create_mode', False) and not (state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or getattr(state, 'group_create_mode', False)):
        try:
            name = getattr(state, 'board_name_input', '')
            handle = getattr(state, 'board_handle_input', '')
            field = int(getattr(state, 'board_create_field', 0))
            p0 = "> " if field == 0 else "  "
            p1 = "> " if field == 1 else "  "
            lines = [
                " Создать доску ",
                f"{p0}Название: {name}",
                f"{p1}Логин (@): {handle}",
                "Enter — создать | Tab/↑/↓ — поле | Esc — закрыть",
            ]
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
            try:
                h2, w2 = stdscr.getmaxyx()
                box_w = min(max(len(max(lines, key=len)) + 6, 40), max(40, w2 - 4))
                box_h = len(lines) + 4
                y0 = max(1, (h2 - box_h) // 2)
                x0 = max(2, (w2 - box_w) // 2)
                if field == 0:
                    base = f"{p0}Название: "
                    cy = y0 + 2 + 1
                    cx = x0 + 2 + len(base) + len(name)
                else:
                    base = f"{p1}Логин (@): "
                    cy = y0 + 2 + 2
                    cx = x0 + 2 + len(base) + len(handle)
                # (cursor hidden)
            except Exception:
                pass
        except Exception:
            pass

    # Group add participants modal
    if getattr(state, 'group_member_add_mode', False) and not (
        state.search_action_mode
        or state.profile_mode
        or state.profile_view_mode
        or state.modal_message
        or state.group_manage_mode
        or getattr(state, 'group_create_mode', False)
    ):
        try:
            gid = state.group_member_add_gid or ''
            g = state.groups.get(gid, {}) if gid else {}
            name = str(g.get('name') or gid)
            handle = str(g.get('handle') or '')
            prompt = state.group_member_add_input
            add_lines = [
                " Добавить участников ",
                f"  Чат: {name}",
                f"  Логин: {handle}" if handle else "",
                "  Участники (ID/@логины, через запятую):",
                f"> {prompt}",
                "Enter — добавить | Esc — отмена",
            ]
            add_lines = [ln for ln in add_lines if ln]
            _draw_center_box(stdscr, add_lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # Group remove participants modal
    if getattr(state, 'group_member_remove_mode', False) and not (
        state.search_action_mode
        or state.profile_mode
        or state.profile_view_mode
        or state.modal_message
        or state.group_manage_mode
        or getattr(state, 'group_create_mode', False)
    ):
        try:
            gid = state.group_member_remove_gid or ''
            g = state.groups.get(gid, {}) if gid else {}
            name = str(g.get('name') or gid)
            opts = list(state.group_member_remove_options or [])
            idx = min(state.group_member_remove_index, max(0, len(opts) - 1))
            rem_lines = [
                " Участники чата ",
                f"  Чат: {name}",
                "",
            ]
            if not opts:
                rem_lines.extend(["  Нет участников для удаления", "Esc — закрыть"])
            else:
                rem_lines.append("  Выберите участника для удаления:")
                for i, uid in enumerate(opts):
                    prof = state.profiles.get(uid) or {}
                    disp = (prof.get('display_name') or '').strip()
                    handle = (prof.get('handle') or '').strip()
                    label = disp or handle or uid
                    prefix = "→ " if i == idx else "  "
                    rem_lines.append(f"{prefix}{label}")
                rem_lines.append("")
                rem_lines.append("Enter — удалить | Esc — отмена")
            _draw_center_box(stdscr, rem_lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # Chat (group) manage modal (exclusive)
    if state.group_manage_mode and not (state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message):
        try:
            gid = state.group_manage_gid or ''
            g = state.groups.get(gid, {}) if gid else {}
            name = state.group_manage_name_input or str(g.get('name') or '')
            count = int(getattr(state, 'group_manage_member_count', 0))
            field = int(getattr(state, 'group_manage_field', 0))
            p0 = "> " if field == 0 else "  "
            p1 = "> " if field == 1 else "  "
            # Resolve owner identity for display
            owner_id = str(g.get('owner_id') or '')
            try:
                prof = state.profiles.get(owner_id) or {}
                dn = (prof.get('display_name') or '').strip()
                hh = (prof.get('handle') or '').strip()
                if dn:
                    owner_disp = dn
                elif hh:
                    owner_disp = f"({hh})"
                elif owner_id:
                    owner_disp = f"[{owner_id}]"
                else:
                    owner_disp = '—'
            except Exception:
                owner_disp = f"[{owner_id}]" if owner_id else '—'
            # field 0 = name (editable), field 1 = handle (editable)
            handle = state.group_manage_handle_input or str(g.get('handle') or '')
            lines = [
                " Чат ",
                f"{p0}Название: {name}",
                f"{p1}Логин: {handle}",
                f"  Создатель: {owner_disp}",
                f"  Участников: {count}",
                "Enter — сохранить | Tab/↑/↓ — поле | Esc — закрыть",
            ]
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
            # No hardware cursor in manage chat overlay
            try:
                h, w = stdscr.getmaxyx()
                box_w = min(max(len(max(lines, key=len)) + 6, 40), max(40, w - 4))
                box_h = len(lines) + 4
                y0 = max(1, (h - box_h) // 2)
                x0 = max(2, (w - box_w) // 2)
                if field == 0:
                    base = f"{p0}Название: "
                    cy = y0 + 2 + 1
                    cx = x0 + 2 + len(base) + len(name)
                elif field == 1:
                    base = f"{p1}Логин: "
                    cy = y0 + 2 + 2
                    cx = x0 + 2 + len(base) + len(handle)
                # (cursor hidden)
                # Inline валидация логина для чата: ^@[a-z0-9_]{3,16}$
                if field == 1:
                    try:
                        from modules.profile import normalize_handle  # type: ignore
                    except Exception:
                        def normalize_handle(v: str) -> str:
                            import re as _re2
                            v = (v or '').strip().lower()
                            if not v:
                                return ''
                            if not v.startswith('@'):
                                v = '@' + v
                            base = _re2.sub(r"[^a-z0-9_]", "", v[1:])
                            return '@' + base
                    raw = state.group_manage_handle_input or ''
                    cand = normalize_handle(raw)
                    import re as _re
                    valid = bool(_re.match(r"^@[a-z0-9_]{3,32}$", cand))
                    if not valid:
                        msg = "Ошибка в логине. Используйте корректный ввод."
                        try:
                            stdscr.addnstr(y0 + box_h - 2, x0 + 2, msg[: box_w - 4], box_w - 4, CP.get('error', curses.A_BOLD))
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception:
            pass

    # Board add participants modal
    if getattr(state, 'board_member_add_mode', False) and not (
        state.search_action_mode
        or state.profile_mode
        or state.profile_view_mode
        or state.modal_message
        or state.board_manage_mode
        or getattr(state, 'group_create_mode', False)
    ):
        try:
            # Blank chat/history area behind the modal to avoid showing conversation under overlay
            try:
                blank = " " * hist_w
                for yy in range(hist_y, hist_y + max(1, history_h)):
                    stdscr.addnstr(yy, hist_x, blank, hist_w)
            except Exception:
                pass
            bid = state.board_member_add_bid or ''
            b = (getattr(state, 'boards', {}) or {}).get(bid, {}) if bid else {}
            name = str(b.get('name') or bid)
            handle = str(b.get('handle') or '')
            prompt = state.board_member_add_input
            add_lines = [
                " Добавить участников в доску ",
                f"  Доска: {name}",
                f"  Логин: {handle}" if handle else "",
                "  Участники (ID/@логины, через запятую):",
                f"> {prompt}",
                "Enter — пригласить | Esc — отмена",
            ]
            add_lines = [ln for ln in add_lines if ln]
            _draw_center_box(stdscr, add_lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # Board remove participants modal
    if getattr(state, 'board_member_remove_mode', False) and not (
        state.search_action_mode
        or state.profile_mode
        or state.profile_view_mode
        or state.modal_message
        or state.board_manage_mode
        or getattr(state, 'group_create_mode', False)
    ):
        try:
            bid = state.board_member_remove_bid or ''
            b = (getattr(state, 'boards', {}) or {}).get(bid, {}) if bid else {}
            name = str(b.get('name') or bid)
            opts = list(state.board_member_remove_options or [])
            idx = min(state.board_member_remove_index, max(0, len(opts) - 1))
            rem_lines = [
                " Участники доски ",
                f"  Доска: {name}",
                "",
            ]
            if not opts:
                rem_lines.extend(["  Нет участников для удаления", "Esc — закрыть"])
            else:
                rem_lines.append("  Выберите участника для удаления:")
                for i, uid in enumerate(opts):
                    prof = state.profiles.get(uid) or {}
                    disp = (prof.get('display_name') or '').strip()
                    handle = (prof.get('handle') or '').strip()
                    label = disp or handle or uid
                    prefix = "→ " if i == idx else "  "
                    rem_lines.append(f"{prefix}{label}")
                rem_lines.append("")
                rem_lines.append("Enter — удалить | Esc — отмена")
            _draw_center_box(stdscr, rem_lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # Board invite modal (incoming) — make it top-priority (draw regardless of other overlays)
    # Also show it when a pending invite token (BINV:<bid>) is selected, even if board_invite_mode is not yet set.
    _want_board_invite_modal = bool(getattr(state, 'board_invite_mode', False))
    _sel_tok = None
    try:
        _sel_tok = current_selected_id(state)
        if (not _want_board_invite_modal) and isinstance(_sel_tok, str) and _sel_tok.startswith('BINV:'):
            _want_board_invite_modal = True
            try:
                bid = _sel_tok.split(':', 1)[1]
                meta = (getattr(state, 'board_pending_invites', {}) or {}).get(bid, {})
                # Pre-fill transient fields for display only; actual state will be set upon key handling
                if not getattr(state, 'board_invite_bid', None):
                    state.board_invite_bid = bid
                if not getattr(state, 'board_invite_name', ''):
                    state.board_invite_name = str(meta.get('name') or bid)
                if not getattr(state, 'board_invite_from', None):
                    state.board_invite_from = str(meta.get('from') or '')
            except Exception:
                pass
    except Exception:
        pass
    if _want_board_invite_modal:
        try:
            name = state.board_invite_name or (state.board_invite_bid or '')
            who = state.board_invite_from or ''
            subtitle = [f"  Доска: {name}", f"  От: {who}" if who else ""]
            options = ["Принять приглашение", "Отклонить"]
            lines = build_menu_modal_lines("Приглашение в доску", options, int(getattr(state, 'board_invite_index', 0)), subtitle_lines=[ln for ln in subtitle if ln])
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # Group invite modal (incoming)
    if getattr(state, 'group_invite_mode', False):
        try:
            name = state.group_invite_name or (state.group_invite_gid or '')
            who = state.group_invite_from or ''
            subtitle = [f"  Группа: {name}", f"  От: {who}" if who else ""]
            options = ["Авторизовать", "Отклонить", "Заблокировать"]
            lines = build_menu_modal_lines("Приглашение в группу", options, int(getattr(state, 'group_invite_index', 0)), subtitle_lines=[ln for ln in subtitle if ln])
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # Board added consent modal (unexpected add) — unified menu style
    if getattr(state, 'board_added_consent_mode', False) and not (
        state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message
    ):
        try:
            bid = state.board_added_bid or ''
            b = (getattr(state, 'boards', {}) or {}).get(bid, {}) if bid else {}
            name = str(b.get('name') or bid)
            subtitle = [f"  Доска: {name}"]
            options = ["Остаться в доске", "Покинуть доску"]
            lines = build_menu_modal_lines("Вы добавлены в доску", options, int(getattr(state, 'board_added_index', 0)), subtitle_lines=subtitle)
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # Board manage modal (exclusive)
    if getattr(state, 'board_manage_mode', False) and not (state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message):
        try:
            bid = state.board_manage_bid or ''
            b = (getattr(state, 'boards', {}) or {}).get(bid, {}) if bid else {}
            name = state.board_manage_name_input or str(b.get('name') or '')
            count = int(getattr(state, 'board_manage_member_count', 0))
            field = int(getattr(state, 'board_manage_field', 0))
            p0 = "> " if field == 0 else "  "
            p1 = "> " if field == 1 else "  "
            owner_id = str(b.get('owner_id') or '')
            try:
                prof = state.profiles.get(owner_id) or {}
                dn = (prof.get('display_name') or '').strip()
                hh = (prof.get('handle') or '').strip()
                if dn:
                    owner_disp = dn
                elif hh:
                    owner_disp = f"({hh})"
                elif owner_id:
                    owner_disp = f"[{owner_id}]"
                else:
                    owner_disp = '—'
            except Exception:
                owner_disp = f"[{owner_id}]" if owner_id else '—'
            handle = state.board_manage_handle_input or str(b.get('handle') or '')
            lines = [
                " Доска ",
                f"{p0}Название: {name}",
                f"{p1}Логин: {handle}",
                f"  Создатель: {owner_disp}",
                f"  Участников: {count}",
                "Enter — сохранить | Tab/↑/↓ — поле | Esc — закрыть",
            ]
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
            try:
                h3, w3 = stdscr.getmaxyx()
                box_w = min(max(len(max(lines, key=len)) + 6, 40), max(40, w3 - 4))
                box_h = len(lines) + 4
                y0 = max(1, (h3 - box_h) // 2)
                x0 = max(2, (w3 - box_w) // 2)
                if field == 0:
                    base = f"{p0}Название: "
                    cy = y0 + 2 + 1
                    cx = x0 + 2 + len(base) + len(name)
                elif field == 1:
                    base = f"{p1}Логин: "
                    cy = y0 + 2 + 2
                    cx = x0 + 2 + len(base) + len(handle)
                # cursor hidden
            except Exception:
                pass
        except Exception:
            pass

    # Search action modal (exclusive)
    if state.search_action_mode and state.search_action_peer and not (state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message):
        try:
            peer = state.search_action_peer
            if state.search_action_step == 'choose':
                lines = [f"Найден: {peer}", "Выберите действие:"]
                for i, opt in enumerate(state.search_action_options):
                    prefix = "> " if i == state.search_action_index else "  "
                    lines.append(prefix + opt)
                lines.append("Enter — выбрать | Esc — закрыть | A — запрос")
                _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
            elif state.search_action_step == 'waiting':
                _draw_center_box(stdscr, [f"Запрос авторизации отправлен: {peer}", "Ожидаем подтверждения…", "Esc — закрыть"], CP.get('warn', curses.A_BOLD))
            elif state.search_action_step == 'accepted':
                _draw_center_box(stdscr, [f"Авторизация принята: {peer}", "Нажмите Enter"], CP.get('success', curses.A_BOLD))
            elif state.search_action_step == 'declined':
                _draw_center_box(stdscr, [f"Запрос отклонён: {peer}", "Нажмите Enter"], CP.get('error', curses.A_BOLD))
        except Exception:
            pass

    # Profile view card for selected user (skip if another modal overlay is active)
    if state.profile_view_mode and not (state.search_action_mode or state.action_menu_mode or state.modal_message):
        pid = state.profile_view_id or "—"
        prof = state.profiles.get(pid) or {}
        dn = prof.get('display_name') or '—'
        hh = prof.get('handle') or '—'
        cv = prof.get('client_version') or '—'
        lines = [
            " Карточка пользователя ",
            f"ID: {pid}",
            f"Имя: {dn}",
            f"Логин: {hh}",
            f"Клиент: v{cv}",
            "Esc/←/→ — закрыть",
        ]
        try:
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # ===== Модалка подтверждения отправки файла (путь в тексте) =====
    if getattr(state, 'file_confirm_mode', False) and not (
        state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or getattr(state, 'group_create_mode', False) or state.group_manage_mode or getattr(state, 'board_create_mode', False) or getattr(state, 'board_manage_mode', False)
    ):
        try:
            pth = state.file_confirm_path or ''
            tgt = state.file_confirm_target or ''
            opts = ["Да", "Нет", "Отмена"]
            idx = max(0, min(2, int(getattr(state, 'file_confirm_index', 0))))
            lines = [
                " Отправить файл? ",
                f"  Путь: {pth}",
                (f"  Кому: {tgt}" if tgt else ""),
                "",
            ]
            for i, opt in enumerate(opts):
                prefix = "> " if i == idx else "  "
                lines.append(prefix + opt)
            lines.append("Enter — выбрать | Esc — закрыть | ↑/↓ — выбор")
            _draw_center_box(stdscr, [ln for ln in lines if ln], CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # ===== Прогресс скачивания файла (текст‑бар) =====
    if getattr(state, 'file_progress_mode', False) and not (
        state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message
    ):
        try:
            nm = state.file_progress_name or ''
            pct = max(0, min(100, int(getattr(state, 'file_progress_pct', 0))))
            # Сформируем простую шкалу 30 символов
            width = 30
            fill = int((pct * width) / 100)
            bar = '[' + ('#' * fill) + ('-' * (width - fill)) + f'] {pct}%'
            _draw_center_box(stdscr, [" Скачивание файла ", f"  {nm}", "", bar, "Esc — скрыть окно"], CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # ===== Модалка: файл уже существует (заменить?) =====
    if getattr(state, 'file_exists_mode', False) and not (
        state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message
    ):
        try:
            nm = state.file_exists_name or ''
            tgt = state.file_exists_target or ''
            idx = max(0, min(1, int(getattr(state, 'file_exists_index', 0))))
            opts = ["Заменить", "Оставить"]
            lines = [
                " Файл уже существует ",
                f"  {nm}",
                f"  Путь: {tgt}",
                "",
                f"{'> ' if idx == 0 else '  '}{opts[0]}",
                f"{'> ' if idx == 1 else '  '}{opts[1]}",
                "Enter — выбрать | Esc — оставить",
            ]
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # If selected contact is blocked, show persistent overlay in chat area to indicate state (only if no modal overlays)
    try:
        if (not state.search_mode) and sel_contact and (sel_contact not in state.groups) and not (state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or state.help_mode):
            if sel_contact in state.blocked:
                _draw_center_box(stdscr, ["⛔ Вы заблокировали этот аккаунт", "Нажмите ← для действий (Разблокировать)"] , CP.get('error', curses.A_BOLD))
            elif sel_contact in getattr(state, 'blocked_by', set()):
                _draw_center_box(stdscr, ["⛔ Этот аккаунт заблокировал вас", "Общение недоступно"], CP.get('error', curses.A_BOLD))
    except Exception:
        pass

    # When chat notifications are disabled (muted), do not show additional overlays.

    # Persistent overlay for outgoing authorization request in current chat
    if (not state.search_mode) and sel_contact and (sel_contact in getattr(state, 'authz_out_pending', set())) and not (state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or state.help_mode or getattr(state, 'group_create_mode', False) or state.group_manage_mode or getattr(state, 'group_member_add_mode', False) or getattr(state, 'group_member_remove_mode', False)):
        try:
            _draw_center_box(stdscr, [
                " Запрос авторизации отправлен ",
                f"ID: {sel_contact}",
                "Ожидаем подтверждения…",
                "Esc — отменить запрос",
            ], CP.get('warn', curses.A_BOLD))
        except Exception:
            pass

    # (Incoming auth simple prompt removed; use only actions menu overlay)

    # Actions menu overlay (context actions on contact, exclusive)
    if state.action_menu_mode and state.action_menu_options and state.action_menu_peer and not (state.search_action_mode or state.profile_mode or state.profile_view_mode or state.modal_message or state.members_view_mode):
        try:
            # Blank chat/history area behind the modal to avoid showing conversation under overlay
            try:
                blank = " " * hist_w
                for yy in range(hist_y, hist_y + max(1, history_h)):
                    stdscr.addnstr(yy, hist_x, blank, hist_w)
            except Exception:
                pass
            # Normalize options to ensure "Участники" доступен для групп/досок даже если список собран старой логикой
            peer_raw = state.action_menu_peer
            opts_norm = list(state.action_menu_options or [])
            try:
                if (peer_raw in getattr(state, 'groups', {})) and ("Участники" not in opts_norm):
                    # после "Отправить файл" если он есть
                    try:
                        idx = opts_norm.index("Отправить файл") + 1
                    except ValueError:
                        idx = 0
                    opts_norm.insert(idx, "Участники")
                if (peer_raw in getattr(state, 'boards', {})) and ("Участники" not in opts_norm):
                    try:
                        idx = opts_norm.index("Отправить файл") + 1
                    except ValueError:
                        idx = 0
                    opts_norm.insert(idx, "Участники")
                state.action_menu_options = opts_norm
            except Exception:
                pass
            display_peer = peer_raw
            try:
                if isinstance(display_peer, str) and display_peer.startswith('BINV:'):
                    # Resolve board invite token to board name for nicer header
                    bid = display_peer.split(':', 1)[1]
                    meta = (getattr(state, 'board_pending_invites', {}) or {}).get(bid, {})
                    display_peer = str(meta.get('name') or bid)
                elif display_peer in state.groups:
                    g = state.groups.get(display_peer) or {}
                    nm = str(g.get('name') or display_peer)
                    display_peer = nm
                elif display_peer in getattr(state, 'boards', {}):
                    b = (getattr(state, 'boards', {}) or {}).get(display_peer) or {}
                    nm = str(b.get('name') or display_peer)
                    display_peer = nm
            except Exception:
                pass
            lines: List[str] = [f"Действия: {display_peer}"]
            for i, opt in enumerate(state.action_menu_options):
                prefix = "> " if i == state.action_menu_index else "  "
                lines.append(prefix + opt)
            if _is_auth_actions_menu(state):
                lines.append("Enter — выбрать | Esc — закрыть | Tab — выбор")
            else:
                lines.append("Enter — выбрать | Esc — закрыть | ↑/↓ — выбор")
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # Members view overlay (read-only list)
    if state.members_view_mode:
        try:
            try:
                blank = " " * hist_w
                for yy in range(hist_y, hist_y + max(1, history_h)):
                    stdscr.addnstr(yy, hist_x, blank, hist_w)
            except Exception:
                pass
            entries = list(state.members_view_entries or [])
            title = state.members_view_title or "Участники"
            lines = [title, f"Всего: {len(entries)}", ""]
            if entries:
                for lbl in entries:
                    lines.append(f"  {lbl}")
            else:
                lines.append("  Список недоступен")
            lines.append("")
            lines.append("Esc/Enter — закрыть")
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass

    # Help overlay (draw last; не скрывает чат/контакты)
    if state.help_mode:
        try:
            try:
                state.history_dirty = True
            except Exception:
                pass
            lines = [
                "Подсказка клавиш:",
                "F1 — помощь (закрыть: F1/ESC/Enter)",
                "F2 — профиль",
                "F3 — поиск (введите ID/@логин, Enter)",
                "F4 — копировать экран (в буфер)",
                "F5 — создать чат",
                "F6 — создать доску",
                "Ctrl+J — новая строка (до 4)",
                "Ctrl+U — обновить клиент",
                "Поле ввода: ←/→ — по символу; ↑/↓ — по строкам (при пустой строке — навигация по списку)",
                "В профиле: ↑/↓ — следующий/предыдущий профиль",
                "Список: стрелки — выбор; ← — меню действий; Enter — выполнить",
                "Приглашение в доску: наведите на 🔔 в ‘Ожидают авторизацию’ и нажмите Enter",
                "F12 — отладка (C — копировать, S — сохранить)",
            ]
            _draw_center_box(stdscr, lines, CP.get('title', curses.A_BOLD))
        except Exception:
            pass
    # Update prompt (manual; does not auto-update)
    if getattr(state, "update_prompt_mode", False):
        try:
            latest = str(getattr(state, "update_prompt_latest", "") or "").strip()
            reason = str(getattr(state, "update_prompt_reason", "") or "").strip()
            title = "Обнаружено обновление клиента"
            if reason:
                title = f"{title}: {reason}"
            lines = [title]
            if latest and latest != CLIENT_VERSION:
                lines.append(f"{CLIENT_VERSION} → {latest}")
            elif latest:
                lines.append(f"Версия: {latest}")
            else:
                lines.append(f"Текущая версия: {CLIENT_VERSION}")
            lines.extend(["", "Ctrl+U или Enter (OK) — обновить", "Esc или любая клавиша — позже"])
            _draw_center_box(stdscr, lines, CP.get("title", curses.A_BOLD))
        except Exception:
            pass
    # Block/modal overlay (simple one-line message, top-most)
    if state.modal_message:
        try:
            _draw_center_box(stdscr, [state.modal_message, "Нажмите любую клавишу…"], CP.get('title', curses.A_BOLD))
        except Exception:
            pass
    # Apply hardware cursor once per frame (after overlays)
    try:
        CURSOR.apply(stdscr)
    except Exception:
        pass
    stdscr.refresh()


def main(stdscr):
    # Logger setup for client: log to file by default to avoid curses conflicts
    log_level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger = logging.getLogger('client')
    logger.setLevel(log_level)
    # Clear handlers only for our named loggers to avoid duplicate logs on reruns
    lg_client = logging.getLogger('client')
    lg_net = logging.getLogger('client.net')
    lg_client.handlers = []
    lg_net.handlers = []
    lg_client.setLevel(log_level)
    lg_net.setLevel(log_level)
    # Use single handler on parent logger; allow child to propagate up to avoid duplicates
    # Configure dedicated debug logger to file only (avoid spamming TUI)
    try:
        DEBUG_LOGGER.handlers = []
        DEBUG_LOGGER.propagate = False
        if DEBUG_LOG_ENABLED:
            dbg_path = _logs_dir() / f"client-debug{_profile_suffix()}.log"
            dbg_handler = logging.handlers.RotatingFileHandler(dbg_path, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
            dbg_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S'))
            DEBUG_LOGGER.addHandler(dbg_handler)
            DEBUG_LOGGER.setLevel(logging.DEBUG)
            DEBUG_LOGGER.debug("[debug] logger initialized")
    except Exception:
        pass
    # Ensure directories for logs and user histories exist
    ensure_storage_dirs()
    default_log = _logs_dir() / f"client{_profile_suffix()}.log"
    log_path = os.environ.get('CLIENT_LOG_FILE', str(default_log))
    fh = None
    if not _is_ephemeral():
        fh = logging.handlers.RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
    log_json = str(os.environ.get('LOG_JSON', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            try:
                payload = {
                    'ts': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(record.created)),
                    'level': record.levelname,
                    'logger': record.name,
                    'msg': record.getMessage(),
                }
                if record.exc_info:
                    payload['exc'] = self.formatException(record.exc_info)
                return json.dumps(payload, ensure_ascii=False)
            except Exception:
                return f"{record.levelname} {record.name}: {record.getMessage()}"
    if log_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S')
    if fh is not None:
        fh.setFormatter(formatter)
        lg_client.addHandler(fh)
        lg_net.propagate = True
    # Do not spam stderr while TUI is active; allow only when explicitly forced
    if os.environ.get('CLIENT_STDERR_TUI') or _is_ephemeral():
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(formatter)
        logging.getLogger('client').addHandler(sh)
        logging.getLogger('client.net').addHandler(sh)
    logger.info("Client starting with LOG_LEVEL=%s", log_level_name)

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    # Reduce latency for ESC-prefixed sequences (arrow keys/Alt) on some terminals/SSH links.
    try:
        curses.set_escdelay(25)
    except Exception:
        pass
    try:
        stdscr.scrollok(False)
    except Exception:
        pass
    # Disable CR->LF input translation to distinguish Enter (\r) from Ctrl+J (\n)
    try:
        curses.nonl()
    except Exception:
        pass
    # Mouse capture enabled by default (wheel routed into history UI)
    env_mouse = True
    try:
        v = os.environ.get('CLIENT_MOUSE')
        if v is not None:
            env_mouse = str(v).strip().lower() in ('1', 'true', 'yes', 'on')
    except Exception:
        env_mouse = True
    # Prefer raw SGR mouse on macOS to avoid flaky KEY_MOUSE mapping
    try:
        _plat = sys.platform
    except Exception:
        _plat = ''
    prefer_sgr = (os.environ.get('CLIENT_MOUSE_SGR') or ('darwin' if _plat.startswith('darwin') else '')).strip().lower() in ('1','true','yes','on','darwin')
    def _set_bracketed_paste(enabled: bool) -> None:
        """Keep bracketed paste enabled regardless of mouse capture mode."""
        try:
            _term_write("\x1b[?2004h" if enabled else "\x1b[?2004l", tmux_passthrough=True)
        except Exception:
            pass

    def _apply_mouse(enabled: bool) -> None:
        global __MOUSE_ARMED__
        try:
            if enabled:
                # Ask curses for mouse events; we still keep terminal "any-motion" disabled (1003l)
                # to avoid event storms while retaining wheel/click support.
                try:
                    mask = int(getattr(curses, 'ALL_MOUSE_EVENTS', 0) or 0)
                    # Some curses builds behave better when REPORT_MOUSE_POSITION is requested,
                    # even if the terminal side keeps 1003 disabled.
                    mask |= int(getattr(curses, 'REPORT_MOUSE_POSITION', 0) or 0)
                    curses.mousemask(mask or 0)
                    curses.mouseinterval(0)
                except Exception:
                    pass
                try:
                    # Enable robust reporting in the terminal:
                    # - 1000: basic clicks
                    # - 1002: button events (many terminals require it for wheel/release)
                    # - 1006/1015: extended coordinates encodings (SGR preferred)
                    # Keep 1003 (any-motion) disabled to avoid event storms.
                    # Also force-disable alternate scroll mode (1007), which can make the wheel look like ↑/↓ keys
                    # after screen mode changes (seen on some terminals after overlays).
                    _term_write("\x1b[?1003l\x1b[?1007l\x1b[?1000h\x1b[?1002h\x1b[?1006h\x1b[?1015h\x1b[?1004h", tmux_passthrough=True)
                except Exception:
                    pass
                __MOUSE_ARMED__ = True
            else:
                try:
                    curses.mousemask(0)
                    curses.mouseinterval(0)
                except Exception:
                    pass
                try:
                    _term_write("\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1015l\x1b[?1006l\x1b[?1007l\x1b[?1004l", tmux_passthrough=True)
                except Exception:
                    pass
                __MOUSE_ARMED__ = False
            # Always keep bracketed paste active during the TUI so pastes remain lossless
            _set_bracketed_paste(True)
        except Exception:
            pass

    def _maybe_enable_tmux_mouse(state: "ClientState") -> None:
        """Best-effort: if running under tmux, ensure tmux forwards mouse events to this app."""
        global __TMUX_MOUSE_PREV__, __TMUX_MOUSE_AUTO_ENABLED__
        try:
            if not os.environ.get('TMUX'):
                return
            v = str(os.environ.get('CLIENT_TMUX_AUTO_MOUSE', '1')).strip().lower()
            if v in ('0', 'no', 'false', 'off'):
                return
            prev = None
            try:
                r = subprocess.run(
                    ['tmux', 'show', '-gv', 'mouse'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=0.25,
                    check=False,
                )
                prev = (r.stdout or '').strip()
            except Exception:
                prev = None
            prev_norm = (prev or '').strip().lower()
            if prev_norm in ('', 'off', '0', 'false', 'no'):
                try:
                    subprocess.run(
                        ['tmux', 'set', '-g', 'mouse', 'on'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=0.25,
                        check=False,
                    )
                    __TMUX_MOUSE_PREV__ = prev
                    __TMUX_MOUSE_AUTO_ENABLED__ = True
                    try:
                        state.tmux_mouse = 'on'
                    except Exception:
                        pass
                    try:
                        if getattr(state, 'debug_mode', False):
                            state.debug_lines.append("[tmux] включено: set -g mouse on")
                            if len(state.debug_lines) > 300:
                                del state.debug_lines[:len(state.debug_lines) - 300]
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                try:
                    state.tmux_mouse = prev_norm
                except Exception:
                    pass
        except Exception:
            pass
    # Цвета
    init_colors()
    # Проверка обновлений (до подключения): только уведомление, без авто-апдейта во время работы
    srv_ver = startup_update_check(stdscr)
    # Ensure module paths are available locally
    try:
        _ensure_module_sys_path()
    except Exception:
        logging.getLogger('client').exception('module path setup failed')

    # Discover/connect (non-blocking; if not found, still run and wait)
    discovered = discover_server()
    if discovered:
        host, port, use_tls = discovered
        save_config(host, port, tls=use_tls)
    else:
        # Нет доступного сервера: используем явный SERVER_ADDR или прод по умолчанию,
        # но не сохраняем и не переключаемся на localhost.
        logging.getLogger('client').warning("Server not found; will wait for it")
        env_addr = os.environ.get('SERVER_ADDR', 'yagodka.org:7777')
        try:
            if ':' in env_addr:
                host, p = env_addr.rsplit(':', 1)
                port = int(p)
            else:
                host, port = env_addr, 7777
        except Exception:
            host, port = ('yagodka.org', 7777)
        # Best-effort guess (real endpoint will be probed on next start)
        try:
            use_tls = bool(_parse_tls_mode() == 'on' or int(port) != 7777)
        except Exception:
            use_tls = False

    scheme = "tls" if use_tls else "tcp"
    state = ClientState(status=f"Connecting to {scheme}://{host}:{port} ...")
    # Если интерактивный апдейтер сохранил сообщение об ошибке — покажем его как модалку,
    # чтобы оно не исчезало при переходе в основную UI.
    try:
        msg = globals().get('LAST_UPDATE_ERROR')
        if isinstance(msg, str) and msg:
            state.modal_message = msg
    except Exception:
        pass
    state.mouse_enabled = env_mouse
    state.mouse_raw = bool(prefer_sgr)
    # Arm mouse tracking for the main UI (some terminals reset modes during startup screens).
    try:
        _apply_mouse(state.mouse_enabled)
    except Exception:
        pass
    try:
        if state.mouse_enabled:
            _maybe_enable_tmux_mouse(state)
    except Exception:
        pass
    # Enable VI keys via env (CLIENT_VI_KEYS=1)
    try:
        state.vi_keys = os.environ.get('CLIENT_VI_KEYS', '0').strip().lower() in ('1', 'true', 'yes', 'on')
    except Exception:
        state.vi_keys = False
    # Сохраняем известную версию сервера (если смогли прочитать)
    try:
        if srv_ver:
            state.status = f"Сервер v{srv_ver} — подключение…"
            # Сохраним версию для отображения позже
            state.server_version = srv_ver  # type: ignore[attr-defined]
    except Exception:
        pass
    incoming: "queue.Queue[dict]" = queue.Queue()
    net = NetworkClient(host, port, incoming, use_tls=use_tls)
    net.start()

    def trigger_hotkey(name: str) -> None:
        nonlocal state
        _log_action(f"hotkey {name} pressed")
        if name == 'F1':
            state.help_mode = not state.help_mode
            state.status = "Справка открыта" if state.help_mode else ""
        elif name == 'F2':
            if state.profile_mode:
                state.profile_mode = False
                state.status = "Профиль закрыт"
                _log_action("profile edit closed")
            else:
                state.profile_mode = True
                state.profile_field = 0
                state.status = "Профиль: редактирование"
                state.profile_name_input = ""
                state.profile_handle_input = ""
                try:
                    net.send({"type": "profile_get"})
                    _log_action("profile_get sent")
                except Exception:
                    pass
        elif name == 'F3':
            # Close conflicting overlays/modals
            try:
                state.profile_mode = False
                state.profile_view_mode = False
                state.action_menu_mode = False
                state.search_action_mode = False
                state.modal_message = None
            except Exception:
                pass
            state.search_mode = True
            state.search_query = ""
            state.search_results = []
            state.search_live_id = None
            state.search_live_ok = False
            state.selected_index = 0
            state.status = "Режим поиска: введите ID/@логин и Enter"
            _touch_contact_rows(state)
            _log_action("search mode ON")
        elif name == 'F4':
            _log_action("screen dump (F4)")
            copied, saved_path = copy_screen_to_clipboard(stdscr)
            if copied:
                state.status = "Экран скопирован в буфер"
            elif saved_path:
                state.status = f"Буфер недоступен; сохранено: {saved_path}"
            else:
                state.status = "Копирование экрана не удалось"
        elif name == 'F5':
            try:
                state.search_action_mode = False
                state.search_action_peer = None
                state.search_action_options = []
                state.search_mode = False
                state.profile_view_mode = False
                state.profile_mode = False
                state.modal_message = None
                state.group_create_mode = True  # type: ignore[attr-defined]
                state.group_create_field = 0    # type: ignore[attr-defined]
                state.group_name_input = getattr(state, 'group_name_input', '') or ""  # type: ignore[attr-defined]
                state.group_members_input = getattr(state, 'group_members_input', '') or ""  # type: ignore[attr-defined]
                state.status = "Создание чата: введите имя и участников"
                _touch_contact_rows(state)
                _log_action("group create modal open")
            except Exception:
                pass
        elif name == 'F6':
            try:
                state.search_action_mode = False
                state.search_action_peer = None
                state.search_action_options = []
                state.search_mode = False
                state.profile_view_mode = False
                state.profile_mode = False
                state.modal_message = None
                state.board_create_mode = True
                state.board_create_field = 0
                state.board_name_input = getattr(state, 'board_name_input', '') or ""
                state.board_handle_input = getattr(state, 'board_handle_input', '') or ""
                state.status = "Создание доски: введите имя и логин"
                _touch_contact_rows(state)
                _log_action("board create modal open")
            except Exception:
                pass
        elif name == 'F7':
            try:
                _log_action("file browser open")
                start_file_browser(state)  # type: ignore[name-defined]
            except Exception:
                close_file_browser(state)  # type: ignore[name-defined]

    def handle_hotkey_click(mx: int, my: int) -> bool:
        if my != 0:
            return False
        try:
            for name, xs, xe in getattr(state, 'main_hotkeys_pos', []) or []:
                if xs <= mx < xe:
                    trigger_hotkey(name)
                    return True
        except Exception:
            pass
        return False

    last_redraw = 0.0
    need_redraw = True
    running = True
    modal_now = False  # ensure defined for early debug logging
    # Throttle redraw rate and main loop frequency to avoid pegging CPU on noisy terminals.
    try:
        max_fps = float(os.environ.get('CLIENT_MAX_FPS', '30') or '30')
        if max_fps <= 0:
            max_fps = 30.0
    except Exception:
        max_fps = 30.0
    min_redraw_interval = 1.0 / max_fps if max_fps > 0 else 0.0
    try:
        min_tick = float(os.environ.get('CLIENT_MIN_TICK', '0.001') or '0.001')
        if min_tick < 0:
            min_tick = 0.0
    except Exception:
        min_tick = 0.001
    last_tick = time.time()
    while running:
        now = time.time()
        try:
            dt = now - float(last_tick)
            if dt < min_tick:
                time.sleep(max(0.0, min_tick - dt))
                now = time.time()
        except Exception:
            now = time.time()
        last_tick = now
        # Reset "Входим..." if нет ответа слишком долго
        try:
            if (not state.authed) and getattr(state, 'login_pending_since', 0.0):
                if (now - float(state.login_pending_since)) > 10.0:
                    state.login_msg = "Нет ответа от сервера, попробуйте снова"
                    state.login_pending_since = 0.0
        except Exception:
            pass
        # Compute current modal flag for logging/input handling
        try:
            from modules.modal_std import modal_active as _modal_active_loop  # type: ignore
            modal_now = bool(_modal_active_loop(state))
        except Exception:
            modal_now = False
        overlay_active = modal_now
        # If selection is on a pending board invite token, open the consent modal
        try:
            sel_tok = current_selected_id(state)
            if isinstance(sel_tok, str) and sel_tok.startswith('BINV:') and not getattr(state, 'board_invite_mode', False):
                bid = sel_tok.split(':', 1)[1]
                meta = (getattr(state, 'board_pending_invites', {}) or {}).get(bid, {})
                state.board_invite_mode = True
                state.board_invite_bid = bid
                state.board_invite_name = str(meta.get('name') or bid)
                state.board_invite_from = str(meta.get('from') or '')
                state.board_invite_index = 0
        except Exception:
            pass
        # Re-apply mouse after resume (SIGCONT)
        try:
            if globals().get('__RESUME_MOUSE__'):
                globals()['__RESUME_MOUSE__'] = False
                _apply_mouse(state.mouse_enabled)
        except Exception:
            pass
        # Draw only when needed or on periodic idle ticks; slower when idle, faster after events
        idle_interval = 0.15
        if need_redraw:
            if (now - last_redraw) >= max(0.0, min_redraw_interval):
                draw_ui(stdscr, state)
                last_redraw = now
                need_redraw = False
        elif (now - last_redraw) > idle_interval:
            draw_ui(stdscr, state)
            last_redraw = now
            need_redraw = False
        # Background upload pump: send one chunk per frame if pending
        try:
            if state.file_send_file_id and state.file_send_fp and state.file_send_size > 0:
                buf = state.file_send_fp.read(DEFAULT_CHUNK_SIZE)
                if buf:
                    b64 = base64.b64encode(buf).decode('ascii')
                    net.send({
                        "type": getattr(T, 'FILE_CHUNK', 'file_chunk'),
                        "file_id": state.file_send_file_id,
                        "seq": state.file_send_seq,
                        "data": b64,
                    })
                    state.file_send_seq += 1
                    state.file_send_bytes += len(buf)
                    pct = 0
                    try:
                        pct = int((state.file_send_bytes / float(state.file_send_size)) * 100)
                    except Exception:
                        pct = progress_percent(state.file_send_bytes, state.file_send_size)
                    state.status = f"Отправка на сервер: {state.file_send_name} ({pct}%)"
                    need_redraw = True
                else:
                    try:
                        state.file_send_fp.close()
                    except Exception:
                        pass
                    state.file_send_fp = None
                    net.send({"type": getattr(T, 'FILE_UPLOAD_COMPLETE', 'file_upload_complete'), "file_id": state.file_send_file_id})
                    try:
                        chan = (state.file_send_room or state.file_send_to)
                        if chan:
                            nm = state.file_send_name or 'файл'
                            state.conversations.setdefault(chan, []).append(ChatMessage('sys', f"Файл [{nm}] доступен для скачивания", time.time()))
                    except Exception:
                        pass
                    state.status = f"Файл загружен на сервер: {state.file_send_name}"
                    need_redraw = True
        except Exception:
            pass

        # Process incoming network messages
        try:
            msg = incoming.get_nowait()
        except queue.Empty:
            msg = None
        if msg is not None:
            mtype = msg.get('type')
            if mtype == "debug_log":
                try:
                    direction = msg.get('dir', '')
                    payload = msg.get('payload')
                    ts = msg.get('ts') or time.time()
                    ts_str = time.strftime('%H:%M:%S', time.localtime(ts))
                    state.debug_lines.append(f"[{ts_str}] {direction}: {payload}")
                    if len(state.debug_lines) > 300:
                        del state.debug_lines[: len(state.debug_lines) - 300]
                except Exception:
                    pass
                need_redraw = True
                continue
            if mtype == T.WELCOME:
                saved = get_saved_id()
                state.contacts = list(msg.get('contacts', []))
                state.contacts.sort()
                # track statuses for known online contacts
                for cid in state.contacts:
                    if cid:
                        state.statuses[cid] = True
                # Capture server version if provided by server
                try:
                    sv = msg.get('server_version')
                    if isinstance(sv, str) and sv:
                        state.server_version = sv
                except Exception:
                    pass
                if state.authed and state.self_id and state.auth_pw:
                    # Auto-re-auth on reconnect
                    state.status = "Переподключение..."
                    logging.getLogger('client').info("Re-auth as %s", state.self_id)
                    net.send({"type": "auth", "id": state.self_id, "password": state.auth_pw})
                else:
                    state.id_input = saved or ""
                    if saved:
                        state.auth_mode = 'login'
                        state.login_field = 1  # курсор на пароль
                        state.status = f"Введите пароль для {saved}"
                    else:
                        state.auth_mode = 'register'
                        state.login_field = 0  # курсор на пароль
                        state.status = "Регистрация: введите пароль и подтверждение"
                    logging.getLogger('client').info("Handshake OK. saved_id=%s", saved)
                if not state.server_version:
                    sv = get_server_version()
                    if sv:
                        state.server_version = sv
                clamp_selection(state)
            elif mtype == 'net_status':
                st = msg.get('status')
                try:
                    state.net_status = st
                except Exception:
                    pass
                if st == 'connected':
                    if not state.authed:
                        state.status = "Связь с сервером установлена"
                        state.login_msg = "Соединение установлено — введите пароль" if (state.login_msg or "") else ""
                        try:
                            state.login_pending_since = 0.0
                        except Exception:
                            pass
                elif st == 'reconnecting':
                    state.status = "Ожидание сервера..."
                    if not state.authed:
                        state.login_msg = "Ожидание сервера..."
                        try:
                            state.login_pending_since = 0.0
                        except Exception:
                            pass
                elif st == 'disconnected':
                    state.status = "Соединение потеряно. Переподключение..."
                    if not state.authed:
                        state.login_msg = "Нет соединения с сервером"
                        try:
                            state.login_pending_since = 0.0
                        except Exception:
                            pass
            elif mtype == getattr(T, 'UPDATE_REQUIRED', 'update_required') or mtype == 'update_required':
                # Server indicates our client is outdated → prompt user (no auto-update/restart)
                try:
                    latest = str(msg.get('latest') or '').strip()
                except Exception:
                    latest = ""
                try:
                    key = latest or "<unknown>"
                    dismissed = str(getattr(state, "update_prompt_dismissed_latest", "") or "").strip()
                    if dismissed and dismissed == key:
                        if latest:
                            state.status = f"Доступно обновление до v{latest} (Ctrl+U)"
                        continue
                    show = True
                    if latest:
                        cmp = _cmp_versions(CLIENT_VERSION, latest)
                        if cmp is None:
                            show = (latest != CLIENT_VERSION)
                        else:
                            # Show prompt only when server's latest is newer (never prompt a downgrade).
                            show = (cmp == -1)
                    if show:
                        state.update_prompt_latest = latest or state.update_prompt_latest
                        state.update_prompt_reason = "сервер требует обновление"
                        state.update_prompt_mode = True
                        if latest:
                            state.status = f"Обнаружено обновление до v{latest} (Ctrl+U)"
                except Exception:
                    pass
                continue
            elif mtype == getattr(T, 'FILE_OFFER', 'file_offer'):
                # Входящий оффер: добавим строку в соответствующий канал
                try:
                    fid = str(msg.get('file_id') or '')
                    if fid:
                        raw_name = str(msg.get('name') or '')
                        safe_name = sanitize_remote_filename(raw_name)
                        meta = {
                            'from': str(msg.get('from') or ''),
                            'name': safe_name,
                            'size': int(msg.get('size') or 0),
                            'room': msg.get('room'),
                        }
                        state.incoming_file_offers[fid] = meta
                        # Для room-оффера — пишем в room; иначе — в ЛС с отправителем
                        key = (str(meta.get('room') or '') or meta['from'])
                        # Присвоим числовой ID для текущего канала
                        try:
                            n = int(state.file_offer_counters.get(key, 0)) + 1
                        except Exception:
                            n = 1
                        state.file_offer_counters[key] = n
                        mp = state.file_offer_map.get(key, {})
                        mp[n] = fid
                        state.file_offer_map[key] = mp
                        state.file_offer_rev[fid] = (key, n)
                        if key:
                            arr = list(state.incoming_by_peer.get(key, []))
                            arr.append(fid)
                            state.incoming_by_peer[key] = arr
                        label = meta['name'] or 'файл'
                        from_id = meta['from'] or '—'
                        text_line = f"Пользователь {from_id} передает файл [{label}] [ID:{n}] — примите /ok{n}"
                        chan = key
                        if chan:
                            state.conversations.setdefault(chan, []).append(ChatMessage('sys', text_line, time.time()))
                        state.status = f"Входящий файл: {label}"
                except Exception:
                    pass
            elif mtype == getattr(T, 'FILE_ACCEPT_NOTICE', 'file_accept_notice'):
                try:
                    fid = str(msg.get('file_id') or '')
                    peer = str(msg.get('peer') or '')
                    room = msg.get('room')
                    chan = (str(room) if room else peer)
                    if chan:
                        state.conversations.setdefault(chan, []).append(ChatMessage('sys', f"Получатель {peer} начал скачивание файла [{fid}]", time.time()))
                except Exception:
                    pass
            elif mtype == getattr(T, 'FILE_RECEIVED', 'file_received'):
                try:
                    fid = str(msg.get('file_id') or '')
                    peer = str(msg.get('peer') or '')
                    room = msg.get('room')
                    chan = (str(room) if room else peer)
                    if chan:
                        state.conversations.setdefault(chan, []).append(ChatMessage('sys', f"Получатель {peer} получил файл [{fid}]", time.time()))
                except Exception:
                    pass
            elif mtype == getattr(T, 'FILE_OFFER_RESULT', 'file_offer_result'):
                if bool(msg.get('ok')):
                    fid = str(msg.get('file_id') or '')
                    if fid and state.file_send_path and (state.file_send_file_id is None):
                        state.file_send_file_id = fid
                        try:
                            p = Path(state.file_send_path).expanduser()
                            state.file_send_fp = open(p, 'rb')
                            state.file_send_seq = 0
                            state.file_send_bytes = 0
                            state.status = f"Загрузка на сервер: {state.file_send_name} (0%)"
                        except Exception:
                            state.status = "Не удалось открыть файл для чтения"
                            state.file_send_file_id = None
                else:
                    state.status = f"Отправка отклонена: {msg.get('reason') or ''}"
                    # Reset
                    try:
                        if state.file_send_fp:
                            state.file_send_fp.close()
                    except Exception:
                        pass
                    state.file_send_fp = None
                    state.file_send_path = None
                    state.file_send_name = None
                    state.file_send_size = 0
                    state.file_send_to = None
                    state.file_send_room = None
                    state.file_send_file_id = None
                    state.file_send_seq = 0
                    state.file_send_bytes = 0
            elif mtype == getattr(T, 'FILE_DOWNLOAD_BEGIN', 'file_download_begin'):
                try:
                    fid = str(msg.get('file_id') or '')
                    name = sanitize_remote_filename(str(msg.get('name') or ''))
                    size = int(msg.get('size') or 0)
                    src = str(msg.get('from') or '')
                    base_dir = FILES_DIR / (src or 'unknown')
                    base_dir.mkdir(parents=True, exist_ok=True)
                    target = base_dir / name
                    conflict = target.exists()
                    if conflict:
                        tmp = target.with_name(target.name + '.part')
                        fp = open(tmp, 'wb')
                        state.file_recv_open[fid] = {'fp': fp, 'name': name, 'size': size, 'from': src, 'received': 0, 'path': str(target), 'conflict': True, 'tmp': str(tmp), 'choice': None}
                        # Откроем модалку выбора
                        state.file_exists_mode = True
                        state.file_exists_fid = fid
                        state.file_exists_name = name
                        state.file_exists_target = str(target)
                        state.file_exists_index = 0
                    else:
                        fp = open(target, 'wb')
                        state.file_recv_open[fid] = {'fp': fp, 'name': name, 'size': size, 'from': src, 'received': 0, 'path': str(target), 'conflict': False}
                    # Прогресс‑бар скачивания
                    state.file_progress_mode = True
                    state.file_progress_name = name
                    state.file_progress_pct = 0
                    state.file_progress_file_id = fid
                    state.status = f"Прием файла: {name} (0%)"
                except Exception:
                    state.status = "Не удалось подготовить прием файла"
            elif mtype == getattr(T, 'FILE_CHUNK', 'file_chunk'):
                try:
                    fid = str(msg.get('file_id') or '')
                    ent = state.file_recv_open.get(fid) or {}
                    fp = ent.get('fp')
                    if fp is None:
                        pass
                    else:
                        data_b64 = msg.get('data')
                        if isinstance(data_b64, str):
                            try:
                                blob = base64.b64decode(data_b64.encode('ascii'), validate=True)
                            except Exception:
                                blob = base64.b64decode(data_b64.encode('ascii'))
                            fp.write(blob)
                            ent['received'] = int(ent.get('received') or 0) + len(blob)
                    pct = progress_percent(ent['received'], int(ent.get('size') or 0))
                    state.file_progress_pct = pct
                    state.status = f"Скачивание: {ent.get('name')} ({pct}%)"
                except Exception:
                    pass
            elif mtype == getattr(T, 'FILE_DOWNLOAD_COMPLETE', 'file_download_complete'):
                try:
                    fid = str(msg.get('file_id') or '')
                    ent = state.file_recv_open.pop(fid, None)
                    if ent and ent.get('fp'):
                        try:
                            ent['fp'].close()
                        except Exception:
                            pass
                        # Разрешим конфликт, если был
                        try:
                            if ent.get('conflict'):
                                choice = None
                                if state.file_exists_fid == fid:
                                    # Примем решение пользователя
                                    choice = 'replace' if int(getattr(state, 'file_exists_index', 0)) == 0 else 'keep'
                                    state.file_exists_mode = False
                                    state.file_exists_fid = None
                                else:
                                    choice = ent.get('choice') or 'keep'
                                if choice == 'replace':
                                    # Переместим .part на целевой путь, перепишем
                                    try:
                                        Path(ent.get('path')).unlink(missing_ok=True)
                                    except Exception:
                                        pass
                                    Path(ent.get('tmp')).rename(ent.get('path'))
                                    state.status = f"Файл сохранен: {ent.get('path')}"
                                else:
                                    # Отменим скачивание: удалим .part
                                    try:
                                        Path(ent.get('tmp')).unlink(missing_ok=True)
                                    except Exception:
                                        pass
                                    state.status = f"Скачивание отменено, файл уже существует: {ent.get('path')}"
                            else:
                                state.status = f"Файл сохранен: {ent.get('path')}"
                        except Exception:
                            pass
                    # Скрываем прогресс‑бар
                    if getattr(state, 'file_progress_file_id', None) == fid:
                        state.file_progress_mode = False
                        state.file_progress_file_id = None
                except Exception:
                    pass
            elif mtype == getattr(T, 'FILE_ERROR', 'file_error'):
                try:
                    state.status = f"Ошибка файла: {msg.get('reason') or ''}"
                except Exception:
                    pass
            elif mtype == T.CONTACT_JOINED:
                cid = msg.get('id')
                if cid and cid != state.self_id and cid not in state.contacts:
                    state.contacts.append(cid)
                    state.contacts.sort()
                    logging.getLogger('client').info("Contact joined: %s", cid)
                    # Fetch profile for the new contact
                    try:
                        net.send({"type": "profile_get", "id": cid})
                    except Exception:
                        pass
                if cid:
                    state.statuses[cid] = True
            elif mtype == T.CONTACT_LEFT:
                cid = msg.get('id')
                if cid in state.contacts:
                    # adjust selection if needed
                    idx = state.contacts.index(cid)
                    state.contacts.remove(cid)
                    clamp_selection(state)
                    logging.getLogger('client').info("Contact left: %s", cid)
                if cid:
                    state.statuses[cid] = False
            elif mtype == T.CONTACTS:
                # Snapshot reconciliation
                contacts = list(msg.get('contacts', []))
                if state.self_id:
                    contacts = [c for c in contacts if c != state.self_id]
                contacts.sort()
                state.contacts = contacts
                # Update statuses for online contacts
                for cid in contacts:
                    state.statuses[cid] = True
                # keep selection within bounds
                clamp_selection(state)
                logging.getLogger('client').debug("Contacts snapshot applied: %d", len(state.contacts))
                # Best-effort: request missing profiles for contacts
                for cid in state.contacts[:50]:
                    if cid not in state.profiles:
                        try:
                            net.send({"type": "profile_get", "id": cid})
                        except Exception:
                            pass
                # Update statuses map using snapshot
                online_set = set(state.contacts)
                for cid in online_set:
                    state.statuses[cid] = True
                # Mark previously known as offline if absent
                for known in list(state.statuses.keys()):
                    if known not in online_set:
                        state.statuses[known] = False
            elif mtype == 'roster_full':
                # Full server-driven roster with presence and unread counts
                try:
                    items = list(msg.get('friends', []))
                    newmap: Dict[str, Dict[str, object]] = {}
                    for it in items:
                        fid = str(it.get('id'))
                        if not fid:
                            continue
                        online = bool(it.get('online'))
                        newmap[fid] = {
                            'online': online,
                            'last_seen_at': it.get('last_seen_at'),
                            'unread': int(it.get('unread') or 0),
                        }
                        state.statuses[fid] = online
                    state.roster_friends = newmap
                    _touch_contact_rows(state)
                    # Any friend present here is no longer pending: remove persistent auth overlay
                    try:
                        for fid in newmap.keys():
                            state.authz_out_pending.discard(str(fid))
                    except Exception:
                        pass
                    # Special case: if there is exactly one pending and no other contacts, open menu for it
                    try:
                        rows = build_contact_rows(state)
                        tokens = [t for t in rows if not is_separator(t)]
                        if len(tokens) == 1 and tokens[0] in state.pending_requests and (not state.action_menu_mode):
                            peer = tokens[0]
                            state.action_menu_mode = True
                            state.action_menu_peer = peer
                            state.action_menu_options = ["Авторизовать", "Отклонить", "Заблокировать", "Профиль пользователя"]
                            state.action_menu_index = 0
                    except Exception:
                        pass
                    # sync friends dict for existing UI markers
                    state.friends = {fid: True for fid in state.roster_friends.keys()}
                    # Sync pending in/out if provided
                    try:
                        pin = [str(x) for x in (msg.get('pending_in', []) or [])]
                        state.pending_requests = pin
                        # Не открываем автоматическое меню — пользователь выберет контакт вручную
                    except Exception:
                        state.pending_requests = list(getattr(state, 'pending_requests', []))
                    try:
                        pout = list(msg.get('pending_out', []))
                        state.pending_out = set(str(x) for x in pout)
                    except Exception:
                        state.pending_out = set()
                    _touch_contact_rows(state)
                    # update unread counters from roster
                    for fid, info in state.roster_friends.items():
                        cnt = int(info.get('unread') or 0)
                        if cnt > 0:
                            state.unread[fid] = cnt
                    # Request profiles for friends
                    try:
                        for fid in list(state.roster_friends.keys())[:200]:
                            if fid not in state.profiles:
                                net.send({"type": "profile_get", "id": fid})
                    except Exception:
                        pass
                except Exception:
                    logging.getLogger('client').exception('Failed to apply roster_full')
                clamp_selection(state)
                # Request history for friends/groups/boards (full fetch when нет локального кеша)
                try:
                    for fid in list(state.roster_friends.keys())[:200]:
                        last_id = int(state.history_last_ids.get(fid, 0))
                        net.send({"type": "history", "peer": fid, "since_id": last_id})
                except Exception:
                    pass
                try:
                    for gid, ginfo in (state.groups or {}).items():
                        request_history_if_needed(state, net, gid, force=True)
                except Exception:
                    pass
                try:
                    for bid in list(getattr(state, 'boards', {}) or {}):
                        request_history_if_needed(state, net, bid, force=True)
                except Exception:
                    pass
            elif mtype == 'history_result':
                rows = list(msg.get('rows', []))
                room = msg.get('room')
                peer = msg.get('peer')
                _dbg(f"[recv history_result] room={room} peer={peer} rows={len(rows)}")
                # clear pending flag
                try:
                    chan_id = room or peer
                    if chan_id in getattr(state, 'history_fetching', set()):
                        state.history_fetching.discard(chan_id)
                except Exception:
                    pass
                # Handle history probe: if re-auth probe and no rows, purge local cache
                try:
                    if isinstance(peer, str) and peer in state.history_probe_peers:
                        state.history_probe_peers.discard(peer)
                        if len(rows) == 0:
                            purge_local_history_for_peer(state, peer)
                except Exception:
                    pass
                # Channel selection: group room or private peer
                chan = (room if room else (str(peer) if peer else None))
                if not chan:
                    continue
                applied = 0
                conv = state.conversations.setdefault(chan, [])
                existing_ids = {m.msg_id for m in conv if getattr(m, 'msg_id', None) is not None}
                for r in rows:
                    rid = r.get('id')
                    frm = r.get('from')
                    to = r.get('to')
                    text = r.get('text', '')
                    ts = float(r.get('ts') or time.time())
                    delivered_flag = bool(r.get('delivered') or r.get('delivered_at'))
                    read_flag = bool(r.get('read') or r.get('read_at'))
                    new_status = None
                    if read_flag:
                        new_status = 'read'
                    elif delivered_flag:
                        new_status = 'delivered'
                    # Skip duplicates by msg_id if already present
                    if isinstance(rid, int) and rid in existing_ids:
                        continue
                    if room:
                        # group history row
                        m = ChatMessage('in' if frm != state.self_id else 'out', text, ts, sender=frm, msg_id=rid)
                        if m.direction == 'out' and isinstance(rid, int):
                            m.status = new_status or 'sent'
                        conv.append(m)
                        if isinstance(rid, int):
                            existing_ids.add(rid)
                        rec = {"id": rid, "from": frm, "text": text, "ts": ts, "room": room}
                    else:
                        direction = 'out' if frm == state.self_id else 'in'
                        # If this is our outgoing echo, try to update last pending message instead of duplicating
                        updated = False
                        if direction == 'out':
                            for m_old in reversed(conv):
                                if m_old.direction == 'out' and getattr(m_old, 'msg_id', None) is None:
                                    if m_old.text == text:
                                        m_old.msg_id = rid
                                        if new_status:
                                            # Upgrade status only if not downgrading read
                                            if m_old.status != 'read' or new_status == 'read':
                                                m_old.status = new_status
                                        updated = True
                                        if isinstance(rid, int):
                                            existing_ids.add(rid)
                                        break
                        if not updated:
                            m = ChatMessage(direction, text, ts, sender=frm, msg_id=rid)
                            if direction == 'out' and isinstance(rid, int):
                                m.status = new_status or 'sent'
                            conv.append(m)
                            if isinstance(rid, int):
                                existing_ids.add(rid)
                        rec = {"id": rid, "from": frm, "to": to, "text": text, "ts": ts}
                    try:
                        append_history_record(rec)
                    except Exception:
                        pass
                    if isinstance(rid, int):
                        state.history_last_ids[chan] = max(state.history_last_ids.get(chan, 0), int(rid))
                        applied += 1
                if applied:
                    save_history_index(state.history_last_ids)
                    state.status = f"История обновлена: {chan} +{applied}"
            elif mtype == 'unread_counts':
                try:
                    counts = dict(msg.get('counts', {}))
                    # normalize keys to str
                    state.unread = {str(k): int(v) for k, v in counts.items()}
                except Exception:
                    pass
            elif mtype == 'presence_update':
                try:
                    pid = str(msg.get('id'))
                    online = bool(msg.get('online'))
                    if pid:
                        state.statuses[pid] = online
                        touched = False
                        if pid in state.roster_friends:
                            state.roster_friends[pid]['online'] = online
                            touched = True
                        elif (not state.roster_friends) and (pid in getattr(state, 'friends', {})):
                            touched = True
                        if touched:
                            _touch_contact_rows(state)
                except Exception:
                    pass
            elif mtype == 'users':
                # Skip global directory enumeration to avoid heavy client-side state
                _dbg(f"[users] ignored directory broadcast ids={len(list(msg.get('ids', [])))}")
                continue
            elif mtype == 'friends':
                flist = list(msg.get('friends', []))
                state.friends = {f: True for f in flist}
                # Clear outgoing auth overlay for any id that is now a friend
                try:
                    for fid in flist:
                        try:
                            state.authz_out_pending.discard(str(fid))
                        except Exception:
                            pass
                except Exception:
                    pass
                # Обновим карту roster_friends, чтобы левая панель сразу отразила изменения,
                # даже если сервер прислал не 'roster_full', а лишь 'friends'.
                try:
                    newmap: Dict[str, Dict[str, object]] = {}
                    for fid in flist:
                        on = bool(state.statuses.get(fid))
                        newmap[str(fid)] = {
                            'online': on,
                            'last_seen_at': (state.roster_friends.get(str(fid), {}).get('last_seen_at') if state.roster_friends else None),
                            'unread': int(state.unread.get(str(fid), 0)),
                        }
                    state.roster_friends = newmap
                    _touch_contact_rows(state)
                except Exception:
                    # Fallback: если не удалось — хотя бы статусы
                    for f in flist:
                        if f not in state.statuses:
                            state.statuses[f] = False
                logging.getLogger('client').info("Friends list received: %d (roster updated)", len(flist))
            elif mtype == 'prefs':
                try:
                    state.muted = set(str(x) for x in (msg.get('muted') or []))
                    state.blocked = set(str(x) for x in (msg.get('blocked') or []))
                    state.blocked_by = set(str(x) for x in (msg.get('blocked_by') or []))
                except Exception:
                    state.muted = set()
                    state.blocked = set()
                    try:
                        state.blocked_by = set()
                    except Exception:
                        pass
                _touch_contact_rows(state)
                # Apply file‑browser prefs (if present) and persist locally
                try:
                    vals = {}
                    for k in ("fb_show_hidden0","fb_show_hidden1","fb_sort0","fb_sort1","fb_dirs_first0","fb_dirs_first1","fb_reverse0","fb_reverse1","fb_view0","fb_view1","fb_path0","fb_path1","fb_side"):
                        if k in msg:
                            vals[k] = msg[k]
                    if vals:
                        # Persist
                        _save_fb_prefs_values(**vals)
                        # Mirror into state for immediate use
                        try:
                            state.file_browser_show_hidden0 = bool(vals.get('fb_show_hidden0', getattr(state,'file_browser_show_hidden0', True)))
                            state.file_browser_show_hidden1 = bool(vals.get('fb_show_hidden1', getattr(state,'file_browser_show_hidden1', True)))
                            state.file_browser_sort0 = str(vals.get('fb_sort0', getattr(state,'file_browser_sort0','name')))
                            state.file_browser_sort1 = str(vals.get('fb_sort1', getattr(state,'file_browser_sort1','name')))
                            state.file_browser_dirs_first0 = bool(vals.get('fb_dirs_first0', getattr(state,'file_browser_dirs_first0', False)))
                            state.file_browser_dirs_first1 = bool(vals.get('fb_dirs_first1', getattr(state,'file_browser_dirs_first1', False)))
                            state.file_browser_reverse0 = bool(vals.get('fb_reverse0', getattr(state,'file_browser_reverse0', False)))
                            state.file_browser_reverse1 = bool(vals.get('fb_reverse1', getattr(state,'file_browser_reverse1', False)))
                            state.file_browser_view0 = vals.get('fb_view0', getattr(state,'file_browser_view0', None))
                            state.file_browser_view1 = vals.get('fb_view1', getattr(state,'file_browser_view1', None))
                            if 'fb_path0' in vals:
                                state.file_browser_path0 = str(vals.get('fb_path0') or state.file_browser_path0 or str(Path('.').resolve()))
                            if 'fb_path1' in vals:
                                state.file_browser_path1 = str(vals.get('fb_path1') or state.file_browser_path1 or str(Path('.').resolve()))
                            if 'fb_side' in vals:
                                try:
                                    state.file_browser_side = int(vals.get('fb_side'))
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # If the file manager modal is open, apply fresh prefs to its state immediately.
                        try:
                            if getattr(state, 'file_browser_mode', False):
                                fm = getattr(state, 'file_browser_state', None)
                                if isinstance(fm, FileManagerState):
                                    if 'fb_path0' in vals and vals.get('fb_path0'):
                                        fm.path = _fm_norm_path(str(vals.get('fb_path0')))
                                    if 'fb_show_hidden0' in vals:
                                        fm.show_hidden = bool(vals.get('fb_show_hidden0'))
                                    if 'fb_sort0' in vals:
                                        fm.sort = str(vals.get('fb_sort0') or 'name')
                                    if 'fb_reverse0' in vals:
                                        fm.reverse = bool(vals.get('fb_reverse0'))
                                    if 'fb_dirs_first0' in vals:
                                        fm.dirs_first = bool(vals.get('fb_dirs_first0'))
                                    if 'fb_view0' in vals:
                                        fm.view = vals.get('fb_view0', None)
                                    _fm_relist(fm)
                                    state.file_browser_state = fm
                        except Exception:
                            pass
                except Exception:
                    pass
            elif mtype == 'roster':
                # Populate friends, pending and online statuses in one shot
                try:
                    flist = list(msg.get('friends', []))
                    state.friends = {f: True for f in flist}
                    for f in flist:
                        pass
                except Exception:
                    pass
                try:
                    pin = [str(x) for x in (msg.get('pending_in', []) or [])]
                    state.pending_requests = pin
                    # Не всплываем автоприглашением; пользователь выберет контакт в списке
                    state.authz_prompt_from = None
                except Exception:
                    state.pending_requests = list(getattr(state, 'pending_requests', []))
                try:
                    pout = list(msg.get('pending_out', []))
                    # track outgoing pending to avoid duplicate requests
                    state.pending_out = set(str(x) for x in pout)
                except Exception:
                    pass
                try:
                    online = list(msg.get('online', []))
                    on_set = set(online)
                    for cid in on_set:
                        state.statuses[cid] = True
                    # Пометим всех известных друзей как оффлайн, если их нет в online
                    for f in list(state.friends.keys()):
                        if f not in on_set:
                            state.statuses[f] = False
                except Exception:
                    pass
                _touch_contact_rows(state)
            elif mtype == 'authz_pending':
                lst = list(msg.get('from', []))
                state.pending_requests.extend([x for x in lst if x not in state.pending_requests])
                _touch_contact_rows(state)
                # Без автоприглашения; подсветим в списке
                try:
                    rows = build_contact_rows(state)
                    tokens = [t for t in rows if not is_separator(t)]
                    if len(tokens) == 1 and (not state.action_menu_mode):
                        _maybe_open_authz_actions_menu(state, tokens[0])
                except Exception:
                    pass
                # Touch per-user file for each pending requester
                try:
                    for p in lst:
                        _touch_user_history(str(p))
                except Exception:
                    pass
            elif mtype == 'authz_request':
                who = msg.get('from')
                if who and who != state.self_id:
                    if who not in state.pending_requests and who != state.authz_prompt_from:
                        state.pending_requests.append(who)
                    # На входящий запрос гарантированно считаем этого пользователя входящим,
                    # даже если по каким-то причинам он оказался в исходящих (местный стейт/ранее отправляли)
                    try:
                        state.pending_out.discard(str(who))
                    except Exception:
                        pass
                    _touch_contact_rows(state)
                    state.status = f"Входящий запрос авторизации от {who}"
                    try:
                        _touch_user_history(str(who))
                    except Exception:
                        pass
                    # Special case: only one pending and no other contacts → open menu
                    try:
                        rows = build_contact_rows(state)
                        tokens = [t for t in rows if not is_separator(t)]
                        if len(tokens) == 1 and tokens[0] == who and (not state.action_menu_mode):
                            _maybe_open_authz_actions_menu(state, str(who))
                    except Exception:
                        pass
                elif mtype == 'authz_accepted':
                    who = msg.get('id')
                    if who:
                        state.friends[who] = True
                        try:
                            state.authz_out_pending.discard(str(who))
                        except Exception:
                            pass
                        # Если контакт был заблокирован ранее — автоматически снять блокировку при авторизации
                        try:
                            if who in state.blocked:
                                # Снимем локально и сообщим серверу
                                state.blocked.discard(who)
                                try:
                                    net.send({"type": "block_set", "peer": who, "value": False})
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    # Probe server-side history existence after re-auth; if none, we will purge local cache on result
                    try:
                        net.send({"type": "history", "peer": who, "since_id": 0, "limit": 1})
                        state.history_probe_peers.add(str(who))
                    except Exception:
                        pass
                    # Remove from incoming/outgoing queues
                    try:
                        state.pending_requests = [p for p in state.pending_requests if p != who]
                    except Exception:
                        pass
                    # Clear from pending_out if present
                    try:
                        if who in state.pending_out:
                            state.pending_out.discard(who)
                    except Exception:
                        pass
                    state.status = f"Авторизованы с {who}"
                    # Update search action modal if waiting for this peer
                    if state.search_action_mode and state.search_action_peer == who and state.search_action_step == 'waiting':
                        state.search_action_step = 'accepted'
                    # Exit search UI to immediately show updated contacts
                    try:
                        state.search_action_mode = False
                        state.search_action_step = 'choose'
                        state.search_mode = False
                        state.search_query = ''
                        state.search_results = []
                    except Exception:
                        pass
                    # Move selection to the new friend (avoid jumping to next pending)
                    _touch_contact_rows(state)
                    try:
                        rows = build_contact_rows(state)
                        if who in rows:
                            state.selected_index = rows.index(who)
                            clamp_selection(state)
                    except Exception:
                        pass
                    # If we were anchoring selection, release the lock and re-enable auto menu
                    try:
                        if getattr(state, 'lock_selection_peer', None) == who:
                            state.lock_selection_peer = None
                            state.suppress_auto_menu = False
                    except Exception:
                        pass
                    # Auto-add to the last created group, if this id was intended, and server supports it
                    try:
                        gid = getattr(state, 'last_group_create_gid', None)
                        intended = set(getattr(state, 'last_group_create_intended', set()))
                        if gid and who in intended:
                            net.send({"type": "group_add", "group_id": gid, "members": [who]})
                    except Exception:
                        pass
                elif mtype == 'authz_declined':
                    who = msg.get('id')
                    if who:
                        state.status = f"{who} отклонил запрос авторизации"
                        try:
                            state.authz_out_pending.discard(str(who))
                        except Exception:
                            pass
                    # Если это исходящий запрос (мы запрашивали) — убрать из pending_out
                    try:
                        state.pending_out.discard(str(who))
                    except Exception:
                        pass
                    # Если это входящий запрос (мы отклоняли) — убрать из pending_requests
                    try:
                        state.pending_requests = [p for p in state.pending_requests if p != who]
                    except Exception:
                        pass
                    _touch_contact_rows(state)
                    if state.search_action_mode and state.search_action_peer == who and state.search_action_step == 'waiting':
                        state.search_action_step = 'declined'
                    # Release selection lock if it was set for this peer
                    try:
                        if getattr(state, 'lock_selection_peer', None) == who:
                            state.lock_selection_peer = None
                            state.suppress_auto_menu = False
                    except Exception:
                        pass
            elif mtype == 'friend_remove_result':
                ok = bool(msg.get('ok'))
                peer = msg.get('peer')
                if ok and peer:
                    state.friends.pop(peer, None)
                    state.roster_friends.pop(peer, None)
                    _touch_contact_rows(state)
                    state.status = f"Удалён из контактов: {peer}"
                    try:
                        purge_local_history_for_peer(state, str(peer))
                    except Exception:
                        pass
            elif mtype == 'authz_cancel_result':
                ok = bool(msg.get('ok'))
                peer = msg.get('peer')
                if ok and peer:
                    try:
                        state.pending_out.discard(str(peer))
                    except Exception:
                        pass
                    state.pending_requests = [p for p in state.pending_requests if p != peer]
                    _touch_contact_rows(state)
                    state.status = f"Отменён запрос: {peer}"
                    try:
                        state.authz_out_pending.discard(str(peer))
                    except Exception:
                        pass
            elif mtype == getattr(T, 'AUTHZ_CANCELLED', 'authz_cancelled'):
                who = msg.get('id') or msg.get('from')
                if who:
                    state.pending_requests = [p for p in state.pending_requests if p != who]
                    try:
                        state.pending_out.discard(str(who))
                    except Exception:
                        pass
                    try:
                        state.authz_out_pending.discard(str(who))
                    except Exception:
                        pass
                    _touch_contact_rows(state)
                    state.status = f"{who} отменил запрос авторизации"
            elif mtype == 'mute_set_result':
                if bool(msg.get('ok')):
                    peer = str(msg.get('peer') or '')
                    val = bool(msg.get('value'))
                    if peer:
                        if val:
                            state.muted.add(peer)
                        else:
                            state.muted.discard(peer)
                        state.status = ("Заглушён: " if val else "Снята заглушка: ") + peer
            elif mtype == 'block_set_result':
                if bool(msg.get('ok')):
                    peer = str(msg.get('peer') or '')
                    val = bool(msg.get('value'))
                    if peer:
                        if val:
                            state.blocked.add(peer)
                        else:
                            try:
                                state.blocked.discard(peer)
                            except Exception:
                                pass
                            try:
                                # Показать его снова в списке (если ранее скрывали)
                                state.hidden_blocked.discard(peer)
                            except Exception:
                                pass
                            # Обеспечить отображение как авторизованного контакта без повторной авторизации
                            try:
                                state.friends[peer] = True
                            except Exception:
                                pass
                            try:
                                # Восстановить/добавить в roster для корректной отрисовки левой панели
                                on = bool(state.statuses.get(peer))
                                last_seen = None
                                try:
                                    last_seen = state.roster_friends.get(peer, {}).get('last_seen_at')
                                except Exception:
                                    last_seen = None
                                unread = int(state.unread.get(peer, 0))
                                state.roster_friends[peer] = {'online': on, 'last_seen_at': last_seen, 'unread': unread}
                            except Exception:
                                pass
                        state.status = ("Заблокирован: " if val else "Разблокирован: ") + peer
                        _touch_contact_rows(state)
            elif mtype == 'authz_request_result':
                ok = bool(msg.get('ok'))
                if ok:
                    to = str(msg.get('to') or '')
                    if to:
                        state.pending_out.add(to)
                        _touch_contact_rows(state)
                        state.status = f"Запрос авторизации отправлен: {to}"
                        try:
                            state.authz_out_pending.add(to)
                        except Exception:
                            pass
                        try:
                            _touch_user_history(to)
                        except Exception:
                            pass
                else:
                    reason = str(msg.get('reason') or '')
                    mapping = {
                        'not_found': 'пользователь не найден',
                        'self': 'нельзя отправить себе',
                        'already_friends': 'уже авторизованы',
                    }
                    state.status = f"Не удалось отправить запрос авторизации: {mapping.get(reason, reason)}"
            elif mtype == 'message_blocked':
                who = msg.get('to')
                reason = msg.get('reason')
                _dbg(f"[recv message_blocked] to={who} reason={reason}")
                if reason == 'not_authorized':
                    state.status = f"DM к {who} заблокирован: требуется авторизация"
                elif reason == 'blocked_by_recipient':
                    try:
                        if who:
                            state.blocked_by.add(str(who))
                    except Exception:
                        pass
                    _touch_contact_rows(state)
                    state.modal_message = "Аккаунт заблокировал вас"
                    state.status = f"Сообщение не доставлено {who}: блокировка"
                else:
                    state.status = f"Сообщение не доставлено {who}: {reason}"
            elif mtype == 'blocked_by_update':
                peer = str(msg.get('peer') or '')
                val = bool(msg.get('value'))
                if peer:
                    if val:
                        state.blocked_by.add(peer)
                    else:
                        try:
                            state.blocked_by.discard(peer)
                        except Exception:
                            pass
                    # Если собеседник снял блокировку, и мы ранее скрывали его из списка — вернём видимость,
                    # чтобы контакт снова появился (в блоке или в друзьях, в зависимости от нашего статуса).
                    try:
                        state.hidden_blocked.discard(peer)
                    except Exception:
                        pass
                    _touch_contact_rows(state)
                    state.status = ("Вы заблокированы пользователем: " if val else "Пользователь снял блокировку: ") + peer
            elif mtype == 'search_result':
                q = (msg.get('query') or '').strip()
                results = list(msg.get('results', []))
                # Ignore late search responses if пользователь уже вышел из поиска/оверлея
                if (not state.search_mode) and (not getattr(state, 'search_action_mode', False)) and (not getattr(state, 'group_create_mode', False)) and (not state.group_verify_mode):
                    continue
                # Intercept for group members validation (both live and submit flows)
                live_group_verify = False
                try:
                    # Treat as live verify if group create modal is open on members field and token is present
                    if getattr(state, 'group_create_mode', False) and int(getattr(state, 'group_create_field', 0)) == 1:
                        import re as _re
                        cur_members = getattr(state, 'group_members_input', '')
                        cur_tokens = [t for t in _re.split(r"[\s,]+", cur_members) if t]
                        if any(match_token_query(t, q) for t in cur_tokens):
                            live_group_verify = True
                except Exception:
                    pass
                def _token_for_query(tokens: List[str], qry: str) -> Optional[str]:
                    for t in tokens:
                        if match_token_query(t, qry):
                            return t
                    return None
                # While окно создания чата открыто — не открывать меню поиска/профиля,
                # а трактовать результаты как сигнал для валидации участников.
                if getattr(state, 'group_create_mode', False) or (state.group_verify_mode and _token_for_query(list(state.group_verify_tokens or []), q) is not None) or live_group_verify:
                    # Resolve exact match: handles are exact; IDs require exact id equality
                    rid = ''
                    q_digits = _digits_only(q)
                    if q.startswith('@'):
                        rid = str(results[0].get('id')) if results else ''
                    elif q_digits:
                        for it in results:
                            cand = str(it.get('id') or '')
                            if _digits_only(cand).startswith(q_digits):
                                rid = cand
                                break
                    if not rid and results:
                        rid = str(results[0].get('id') or '')
                    # Identify canonical token key to update (raw token the user typed)
                    try:
                        tokens_list = list(state.group_verify_tokens or [])
                    except Exception:
                        tokens_list = []
                    # Map only if we can associate q with текущим токеном; иначе просто подавить оверлей
                    key = _token_for_query(tokens_list, q)
                    if key is not None and rid:
                        state.group_verify_map[key] = rid
                        try:
                            state.directory.add(rid)
                        except Exception:
                            pass
                    elif key is not None:
                        state.group_verify_map[key] = None
                    if key is not None:
                        try:
                            if key in state.group_verify_pending:
                                state.group_verify_pending.discard(key)
                        except Exception:
                            pass
                    # If this was submit flow and all handled, decide next step
                    if state.group_verify_mode and not state.group_verify_pending:
                        unresolved = [t for t in state.group_verify_tokens if not state.group_verify_map.get(t)]
                        if unresolved:
                            state.status = "Не найдены: " + ", ".join(unresolved)
                            state.group_verify_mode = False
                            # Keep modal open on members field for correction
                            state.group_create_mode = True
                            state.group_create_field = 1
                        else:
                            name = (state.group_name_input or '').strip()
                            ids = [state.group_verify_map[t] for t in state.group_verify_tokens]
                            ids = [str(i) for i in ids if i]
                            if not ids:
                                state.group_verify_mode = False
                                state.modal_message = "Добавьте хотя бы одного участника"
                                state.status = "Введите участников"
                                continue
                            try:
                                state.last_group_create_name = name
                                state.last_group_create_intended = set(ids)
                            except Exception:
                                state.last_group_create_intended = set(ids)
                            if not ensure_group_members_authorized(state, ids, net):
                                state.group_verify_mode = False
                                continue
                            try:
                                net.send({"type": "group_create", "name": name, "members": ",".join(ids)})
                                state.group_create_mode = False
                                state.group_verify_mode = False
                                state.status = f"Создаём чат: {name}"
                            except Exception:
                                state.group_verify_mode = False
                                state.status = "Ошибка: не удалось отправить запрос на создание чата"
                    # Swallow into validation path; do not open search action modal
                else:
                    # Live search mode: do not open action modal automatically
                    # Only activate live results when запрос длиной ≥ 3 символов
                    q = str(msg.get('query') or '').strip()
                    if len(q) < 3:
                        state.search_results = []
                        state.search_live_id = None
                        state.search_live_ok = False
                        state.search_mode = True
                        _touch_contact_rows(state)
                    else:
                        # Update live flags for overlay rendering
                        try:
                            state.search_results = [x for x in results if isinstance(x, dict)]
                        except Exception:
                            state.search_results = []
                        if state.search_results:
                            rid = str(state.search_results[0].get('id') or '')
                            if rid:
                                state.search_live_id = rid
                                state.search_live_ok = True
                                try:
                                    state.directory.add(rid)
                                except Exception:
                                    pass
                            else:
                                state.search_live_id = None
                                state.search_live_ok = False
                        else:
                            state.search_live_id = None
                            state.search_live_ok = False
                        # Keep search overlay open
                        state.search_mode = True
                        # If search finds blocked contacts that were hidden, unhide them so they appear in the list
                        try:
                            hb = set(getattr(state, 'hidden_blocked', set()))
                            for it in list(state.search_results or []):
                                uid = str(it.get('id') or '')
                                if uid and uid in hb:
                                    hb.discard(uid)
                            state.hidden_blocked = hb
                        except Exception:
                            pass
                        _touch_contact_rows(state)
            elif mtype == 'profile':
                pid = msg.get('id')
                dn = msg.get('display_name') or ''
                h = msg.get('handle') or ''
                # Prefill modal if open and it's our profile
                if state.profile_mode and (not pid or pid == state.self_id):
                    state.profile_name_input = dn
                    state.profile_handle_input = h
                if pid:
                    entry = state.profiles.get(pid) or {}
                    entry.update({'display_name': dn or None, 'handle': h or None})
                    # Save reported client version if present
                    cv = msg.get('client_version')
                    if isinstance(cv, str) and cv.strip():
                        entry['client_version'] = cv.strip()
                    state.profiles[pid] = entry
                # Не засоряем общий чат служебными сообщениями о профилях; покажем только статус в хедере
                state.status = f"Профиль {pid}"
            elif mtype == 'profile_set_result':
                if msg.get('ok'):
                    dn = msg.get('display_name') or '—'
                    h = msg.get('handle') or '—'
                    state.status = f"Профиль обновлён: Имя={dn}, Логин={h}"
                    if state.profile_mode:
                        state.profile_mode = False
                else:
                    reason = str(msg.get('reason', 'error'))
                    mapping = {
                        'handle_invalid': 'Некорректный логин (@user, латиница/цифры/_ 3-32)',
                        'handle_taken': 'Логин уже занят',
                        'too_long': 'Имя слишком длинное',
                        'empty': 'Имя пусто',
                        'server_error': 'Ошибка сервера',
                        'unsupported': 'Операция не поддерживается',
                    }
                    state.status = f"Ошибка профиля: {mapping.get(reason, reason)}"
            elif mtype == 'profile_updated':
                pid = msg.get('id')
                dn = msg.get('display_name') or '—'
                h = msg.get('handle') or '—'
                state.status = f"{pid} обновил профиль: Имя={dn}, Логин={h}"
                if pid:
                    state.profiles[pid] = {'display_name': None if dn == '—' else dn, 'handle': None if h == '—' else h}
            elif mtype == 'message':
                from_id = msg.get('from')
                text = msg.get('text', '')
                ts = float(msg.get('ts', time.time()))
                mid = msg.get('id')
                room = msg.get('room')
                _dbg(f"[recv message] room={room} from={from_id} id={mid} text={text!r}")
                # Group message
                if isinstance(room, str) and room:
                    chan = room
                    conv = state.conversations.setdefault(chan, [])
                    conv.append(ChatMessage('in', text, ts, sender=from_id, msg_id=mid))
                    try:
                        append_history_record({"id": mid, "from": from_id, "text": text, "ts": ts, "room": chan})
                        if isinstance(mid, int):
                            state.history_last_ids[chan] = max(state.history_last_ids.get(chan, 0), int(mid))
                            save_history_index(state.history_last_ids)
                    except Exception:
                        pass
                    logging.getLogger('client').debug("Group %s from %s: %s", chan, from_id, text)
                    # Уведомления и счётчики для групп/досок
                    try:
                        current = current_selected_id(state)
                        if current != chan:
                            # Increment unread counter for the chat
                            state.unread[chan] = int(state.unread.get(chan, 0)) + 1
                            # Show status notification if chat is not muted (group or board)
                            muted_groups = set(getattr(state, 'group_muted', set()))
                            muted_boards = set(getattr(state, 'board_muted', set()))
                            if (chan not in muted_groups) and (chan not in muted_boards):
                                meta = state.groups.get(chan) or getattr(state, 'boards', {}).get(chan) or {}
                                nm = str(meta.get('name') or chan)
                                state.status = f"Новое сообщение в чате: {nm}"
                    except Exception:
                        pass
                else:
                    # Private message
                    conv = state.conversations.setdefault(from_id, [])
                    conv.append(ChatMessage('in', text, ts, sender=from_id, msg_id=mid))
                    try:
                        append_history_record({"id": mid, "from": from_id, "to": state.self_id, "text": text, "ts": ts})
                        if isinstance(mid, int):
                            state.history_last_ids[from_id] = max(state.history_last_ids.get(from_id, 0), int(mid))
                            save_history_index(state.history_last_ids)
                    except Exception:
                        pass
                    try:
                        _write_user_history_line(str(from_id), 'in', str(text), float(ts))
                    except Exception:
                        pass
                    logging.getLogger('client').debug("Msg from %s: %s", from_id, text)
                    # Notify if not currently viewing this private chat and not muted
                    current = current_selected_id(state)
                    if current != from_id:
                        state.unread[from_id] = state.unread.get(from_id, 0) + 1
                        if from_id not in state.muted:
                            state.status = f"Новое ЛС от {from_id}"
                    else:
                        # We are actively viewing this chat: mark as read on the server to keep
                        # DB unread_counts/read receipts consistent (even though we don't increment local unread).
                        try:
                            if from_id and isinstance(mid, int):
                                net.send({"type": "message_read", "peer": from_id, "up_to_id": int(mid)})
                        except Exception:
                            pass
            elif mtype == 'message_queued':
                # Direct messages only: update last outgoing to queued and set msg_id
                who = msg.get('to')
                mid = msg.get('id')
                try:
                    if who and (who not in state.groups):
                        conv = state.conversations.get(who, [])
                        for m in reversed(conv):
                            if m.direction == 'out':
                                m.status = 'queued'
                                if isinstance(mid, int):
                                    m.msg_id = mid
                                break
                        # If сервер не шлёт echo/delivered, подтянем историю вручную
                        # Throttle history refresh to avoid duplicate requests on rapid sends
                        now_ts = time.time()
                        try:
                            last_req = float(getattr(state, 'last_hist_req_ts', {}).get(who, 0.0))
                        except Exception:
                            last_req = 0.0
                        if now_ts - last_req > 0.3:
                            try:
                                since = int(state.history_last_ids.get(who, 0))
                            except Exception:
                                since = 0
                            try:
                                net.send({"type": "history", "peer": who, "since_id": since})
                                try:
                                    if not hasattr(state, 'last_hist_req_ts'):
                                        state.last_hist_req_ts = {}
                                    state.last_hist_req_ts[who] = now_ts  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                                _dbg(f"[action] history refresh after queued mid={mid} peer={who} since={since}")
                            except Exception:
                                pass
                except Exception:
                    pass
            elif mtype == 'message_delivered':
                # Direct messages only: mark as delivered and set msg_id
                who = msg.get('to')
                mid = msg.get('id')
                try:
                    if who and (who not in state.groups):
                        conv = state.conversations.get(who, [])
                        for m in reversed(conv):
                            if m.direction == 'out' and (m.msg_id == mid or m.msg_id is None):
                                m.status = 'delivered'
                                if isinstance(mid, int):
                                    m.msg_id = mid
                                break
                except Exception:
                    pass
            elif mtype == 'message_read_ack':
                # Direct messages only: mark all up to id as read
                peer = msg.get('peer')
                up_to = msg.get('up_to_id')
                try:
                    if peer and (peer not in state.groups):
                        conv = state.conversations.get(peer, [])
                        for m in conv:
                            if m.direction == 'out' and (up_to is None or (isinstance(m.msg_id, int) and isinstance(up_to, int) and m.msg_id <= up_to)):
                                m.status = 'read'
                except Exception:
                    pass
            elif mtype == 'chat_cleared':
                peer = str(msg.get('peer') or '')
                ok = bool(msg.get('ok', True))
                if peer:
                    try:
                        purge_local_history_for_peer(state, peer)
                    except Exception:
                        pass
                    # Also clear visible conversation
                    try:
                        state.conversations.pop(peer, None)
                    except Exception:
                        pass
                    # Reset unread and scroll
                    try:
                        state.unread.pop(peer, None)
                    except Exception:
                        pass
                state.status = (f"Чат очищен: {peer}" if ok else f"Не удалось очистить чат: {peer}")
            elif mtype == 'auth_ok':
                state.self_id = msg.get('id') or state.desired_id
                state.authed = True
                state.status = "Connected"
                logging.getLogger('client').info("Authenticated as %s", state.self_id)
                # Сброс ожидания логина/ретраев
                try:
                    state.login_pending_since = 0.0
                    state.login_retry_count = 0
                except Exception:
                    pass
                # Report our client version to the server
                try:
                    net.send({"type": "client_info", "version": CLIENT_VERSION})
                    # Запрос контактов (дублирует серверный снапшот, но ускоряет инициализацию UI)
                    net.send({"type": "list"})
                    # Запрос групп и досок для восстановления
                    net.send({"type": "group_list"})
                    net.send({"type": "board_list"})
                except Exception:
                    pass
                # After auth: send client integrity telemetry (local vs server dist)
                try:
                    # Compute local hash/size
                    path = Path(__file__).resolve()
                    local_size = path.stat().st_size
                    h = hashlib.sha256()
                    with open(path, 'rb') as f:
                        for chunk in iter(lambda: f.read(8192), b''):
                            h.update(chunk)
                    local_sha = h.hexdigest()
                except Exception:
                    local_size = -1
                    local_sha = None
                # Fetch server meta (requires pinned pubkey)
                try:
                    base = _get_update_base_url()
                    mani = _load_manifest(base) if base else None
                    want_sha = None
                    want_size = None
                    if isinstance(mani, dict):
                        for e in _safe_manifest_entries(mani):
                            if e.get('path') == 'client.py':
                                want_sha = str(e.get('sha256') or '') or None
                                try:
                                    want_size = int(e.get('size') or 0)
                                except Exception:
                                    want_size = None
                                break
                except Exception:
                    want_sha = None
                    want_size = None
                size_ok = (want_size is None) or (local_size == want_size)
                hash_ok = (want_sha is None) or (local_sha == want_sha)
                state.last_local_sha = local_sha
                state.last_server_sha = want_sha
                state.last_integrity_size_ok = bool(size_ok)
                state.last_integrity_hash_ok = bool(hash_ok)
                try:
                    net.send({
                        "type": "client_integrity",
                        "version": CLIENT_VERSION,
                        "local_sha": local_sha,
                        "server_sha": want_sha,
                        "size_ok": bool(size_ok),
                        "hash_ok": bool(hash_ok),
                    })
                except Exception:
                    pass
                if state.last_submit_pw:
                    state.auth_pw = state.last_submit_pw
                set_saved_id(state.self_id)
                # Initialize local history
                if not state.history_loaded:
                    load_local_history(state)
            elif mtype == 'register_ok':
                # Игнорируем selftest-регистрацию (только лог)
                if getattr(state, 'selftest_reg', False):
                    try:
                        state.debug_lines.append(f"[selftest:reg] register_ok игнорирован: {msg}")
                        if len(state.debug_lines) > 300:
                            del state.debug_lines[:len(state.debug_lines) - 300]
                    except Exception:
                        pass
                else:
                    state.self_id = msg.get('id')
                    try:
                        state.self_id_lucky = bool(msg.get('lucky'))
                    except Exception:
                        state.self_id_lucky = False
                    try:
                        dval = msg.get('digits')
                        state.self_id_digits = int(dval) if dval is not None else None
                    except Exception:
                        state.self_id_digits = None
                    state.authed = True
                    state.status = f"Registered as {state.self_id}"
                    if state.self_id_lucky:
                        state.login_msg = f"Ваш новый ID: {state.self_id} ★ (счастливый). Сохраните его."
                    else:
                        state.login_msg = f"Ваш новый ID: {state.self_id}. Сохраните его."
                    logging.getLogger('client').info("Registered as %s", state.self_id)
                    if state.last_submit_pw:
                        state.auth_pw = state.last_submit_pw
                    state.login_pending_since = 0.0
                    set_saved_id(state.self_id)
                    try:
                        net.send({"type": "group_list"})
                        net.send({"type": "board_list"})
                    except Exception:
                        pass
                # selftest флаг очищаем после register_ok
                try:
                    state.selftest_reg = False
                    state.selftest_reg_pw = None
                except Exception:
                    pass
            elif mtype == 'auth_fail':
                reason = msg.get('reason', 'unknown')
                mapping = {
                    'missing_id': 'Введите ID/@логин',
                    'no_such_user': 'Пользователь не найден',
                    'bad_password': 'Неверный пароль',
                    'rate_limited': 'Слишком много попыток. Подождите и попробуйте снова',
                    'already_connected': 'Пользователь уже онлайн',
                    'server_error': 'Ошибка сервера',
                }
                state.login_msg = f"Ошибка авторизации: {mapping.get(reason, reason)}"
                state.pw1 = ""
                state.pw2 = ""
                state.login_pending_since = 0.0
                state.login_retry_count = 0
                state.login_field = 0
                logging.getLogger('client').warning("Auth failed: %s", reason)
                # selftest рег: очистить флаг, но не трогать сохранённый ID/пароль
                try:
                    state.selftest_reg = False
                    state.selftest_reg_pw = None
                except Exception:
                    pass
            elif mtype == 'register_fail':
                if getattr(state, 'selftest_reg', False):
                    try:
                        state.debug_lines.append(f"[selftest:reg] register_fail: {msg}")
                        if len(state.debug_lines) > 300:
                            del state.debug_lines[:len(state.debug_lines) - 300]
                    except Exception:
                        pass
                    try:
                        state.selftest_reg = False
                        state.selftest_reg_pw = None
                    except Exception:
                        pass
                else:
                    reason = msg.get('reason', 'unknown')
                    mapping = {
                        'empty_password': 'Пароль не может быть пустым',
                        'password_too_short': 'Пароль слишком короткий',
                        'password_too_long': 'Пароль слишком длинный',
                        'rate_limited': 'Слишком много попыток. Подождите минуту',
                        'server_error': 'Ошибка сервера',
                    }
                    state.login_msg = f"Ошибка регистрации: {mapping.get(reason, reason)}"
                    state.pw1 = state.pw2 = ""
                    state.login_field = 0
                    logging.getLogger('client').warning("Register failed: %s", reason)
            elif mtype == 'group_create_result':
                ok = bool(msg.get('ok'))
                if ok:
                    g = dict(msg.get('group') or {})
                    gid = str(g.get('id') or '')
                    if gid:
                        try:
                            state.last_group_create_gid = gid
                        except Exception:
                            pass
                        state.groups[gid] = {"name": g.get('name'), "owner_id": g.get('owner_id'), "members": list(g.get('members') or [])}
                        state.status = f"Чат создан: {g.get('name')}"
                        _touch_contact_rows(state)
                        # Move selection to the group
                        try:
                            rows = build_contact_rows(state)
                            if gid in rows:
                                state.selected_index = rows.index(gid)
                                clamp_selection(state)
                        except Exception:
                            pass
                        # Notify/invite members per requested logic
                        try:
                            # Actual members returned by server (only friends + owner)
                            members_list = list(g.get('members') or [])
                            actual: Set[str] = set()
                            for mid in members_list:
                                if isinstance(mid, dict):
                                    smid = str(mid.get('id') or mid.get('user_id') or '')
                                else:
                                    smid = str(mid)
                                if smid:
                                    actual.add(smid)
                            name = str(g.get('name') or gid)
                            # DM to actual (friends), skip self
                            for smid in sorted(actual):
                                if not smid or smid == state.self_id:
                                    continue
                                try:
                                    net.send({"type": "send", "to": smid, "text": f"Вас добавили в чат: {name}"})
                                except Exception:
                                    pass
                            # Раньше: для отсутствующих (не добавленных) автоматически отправлялся запрос авторизации.
                            # Убрано, чтобы не требовать авторизацию для добавления в чат и не плодить оверлеи.
                        except Exception:
                            pass
                else:
                    reason = str(msg.get('reason') or '')
                    mapping = {
                        'empty_name': 'Введите название чата',
                        'name_too_long': 'Слишком длинное имя чата',
                        'no_members': 'Добавьте хотя бы одного друга (по умолчанию — только друзья)',
                        'rate_limited': 'Слишком часто',
                        'server_error': 'Ошибка сервера',
                    }
                    msg_txt = mapping.get(reason, reason)
                    state.modal_message = msg_txt
                    state.group_create_mode = True
                    state.group_create_field = 1
                    state.status = f"Ошибка создания чата: {msg_txt}"
            elif mtype == 'board_create_result':
                ok = bool(msg.get('ok'))
                if ok:
                    b = dict(msg.get('board') or {})
                    bid = str(b.get('id') or '')
                    if bid:
                        state.boards[bid] = {"name": b.get('name'), "owner_id": b.get('owner_id'), "handle": b.get('handle'), "members": list(b.get('members') or [])}
                        state.status = f"Доска создана: {b.get('name')}"
                        try:
                            rows = build_contact_rows(state)
                            if bid in rows:
                                state.selected_index = rows.index(bid)
                                clamp_selection(state)
                        except Exception:
                            pass
                        # Close create modal
                        state.board_create_mode = False
                else:
                    reason = str(msg.get('reason') or '')
                    mapping = {
                        'empty_name': 'Введите название доски',
                        'name_too_long': 'Слишком длинное имя',
                        'handle_invalid': 'Некорректный логин',
                        'handle_taken': 'Логин занят',
                        'rate_limited': 'Слишком часто',
                        'server_error': 'Ошибка сервера',
                    }
                    msg_txt = mapping.get(reason, reason)
                    state.modal_message = msg_txt
                    state.board_create_mode = True
                    state.board_create_field = 0
                    state.status = f"Ошибка создания доски: {msg_txt}"
            elif mtype == 'group_add_result':
                if bool(msg.get('ok')):
                    gid = str(msg.get('group_id') or '')
                    added = [str(x) for x in (msg.get('added') or []) if x]
                    if gid and gid in state.groups:
                        try:
                            members = list(state.groups[gid].get('members') or [])
                            merged = list(dict.fromkeys(members + added))
                            state.groups[gid]['members'] = merged
                            if state.group_manage_gid == gid:
                                state.group_manage_member_count = len(merged)
                        except Exception:
                            pass
                    if added:
                        state.status = "Добавлены участники: " + ", ".join(added)
                    else:
                        state.status = "Участники уже добавлены"
                    try:
                        if gid:
                            net.send({"type": "group_info", "group_id": gid})
                    except Exception:
                        pass
                else:
                    reason = str(msg.get('reason') or '')
                    mapping = {
                        'bad_gid': 'Неверный ID чата',
                        'not_found': 'Чат не найден',
                        'forbidden': 'Нет прав для добавления',
                        'server_error': 'Ошибка сервера',
                        'bad_args': 'Неверный формат запроса',
                        'no_members': 'Укажите участников',
                    }
                    state.status = "Не удалось добавить участников: " + mapping.get(reason, reason)
                    state.group_member_add_mode = True
            elif mtype == 'group_added':
                g = dict(msg.get('group') or {})
                gid = str(g.get('id') or '')
                if gid:
                    state.groups[gid] = {"name": g.get('name'), "owner_id": g.get('owner_id'), "members": list(g.get('members') or []), "handle": g.get('handle')}
                    state.status = f"Добавлен чат: {g.get('name')}"
                    try:
                        nm = str(g.get('name') or gid)
                        state.modal_message = f"Новый чат: {nm}"
                    except Exception:
                        pass
                    try:
                        state.group_pending_invites.pop(gid, None)
                    except Exception:
                        pass
                    # If there was a pending join request token for this gid from this user, clear it
                    try:
                        # Remove any JOIN:<gid>:<self_id> from pending map if present
                        gj = getattr(state, 'group_join_requests', {})
                        sid = str(state.self_id or '')
                        if sid and gid in gj:
                            if sid in gj[gid]:
                                gj[gid].discard(sid)
                                if not gj[gid]:
                                    gj.pop(gid, None)
                    except Exception:
                        pass
            elif mtype == 'group_remove_result':
                if bool(msg.get('ok')):
                    gid = str(msg.get('group_id') or '')
                    removed = [str(x) for x in (msg.get('removed') or []) if x]
                    if gid and gid in state.groups:
                        try:
                            members = [m for m in list(state.groups[gid].get('members') or []) if m not in removed]
                            state.groups[gid]['members'] = members
                            if state.group_manage_gid == gid:
                                state.group_manage_member_count = len(members)
                            if state.group_member_remove_gid == gid:
                                owner_id = str(state.groups[gid].get('owner_id') or '')
                                state.group_member_remove_options = [m for m in members if m and m != owner_id]
                                state.group_member_remove_index = 0
                        except Exception:
                            pass
                    if removed:
                        state.status = "Удалены участники: " + ", ".join(removed)
                    else:
                        state.status = "Участники не найдены"
                    try:
                        if gid:
                            net.send({"type": "group_info", "group_id": gid})
                    except Exception:
                        pass
                else:
                    reason = str(msg.get('reason') or '')
                    mapping = {
                        'bad_args': 'Неверный формат запроса',
                        'forbidden_or_not_found': 'Нет прав или чат не найден',
                        'no_members': 'Укажите участников',
                        'server_error': 'Ошибка сервера',
                    }
                    state.status = "Не удалось удалить участника: " + mapping.get(reason, reason)
                    state.group_member_remove_mode = True
            elif mtype == 'group_join_request':
                gid = str(msg.get('group_id') or '')
                who = str(msg.get('from') or '')
                if gid and who:
                    s = state.group_join_requests.get(gid)
                    if s is None:
                        s = set()
                        state.group_join_requests[gid] = s
                    s.add(who)
                    _touch_contact_rows(state)
                    # Prefetch profile for better labels
                    try:
                        if who and (who not in state.groups):
                            net.send({"type": "profile_get", "id": who})
                    except Exception:
                        pass
                    state.status = f"Заявка в чат: {who} → {gid}"
            elif mtype == 'group_join_response_result':
                if bool(msg.get('ok')):
                    gid = str(msg.get('group_id') or '')
                    peer = str(msg.get('peer') or '')
                    try:
                        if gid and peer and gid in state.group_join_requests:
                            state.group_join_requests.get(gid, set()).discard(peer)
                            if not state.group_join_requests.get(gid):
                                state.group_join_requests.pop(gid, None)
                    except Exception:
                        pass
                    _touch_contact_rows(state)
                    if msg.get('accept'):
                        state.status = f"Принят в чат: {peer}"
                    else:
                        state.status = f"Отклонён запрос в чат: {peer}"
            elif mtype == 'group_join_request_result':
                if bool(msg.get('ok')):
                    gid = str(msg.get('group_id') or '')
                    state.status = f"Заявка на вступление в чат отправлена: {gid}"
            elif mtype == 'group_info_result':
                if bool(msg.get('ok')):
                    g = dict(msg.get('group') or {})
                    gid = str(g.get('id') or '')
                    if gid:
                        # Update name from server
                        members = list(g.get('members') or [])
                        try:
                            members = list(g.get('members') or [])
                            if gid in state.groups:
                                state.groups[gid]['name'] = g.get('name') or state.groups[gid].get('name')
                                if g.get('owner_id'):
                                    state.groups[gid]['owner_id'] = g.get('owner_id')
                                if g.get('handle') is not None:
                                    state.groups[gid]['handle'] = g.get('handle')
                                state.groups[gid]['members'] = members
                            else:
                                state.groups[gid] = {"name": g.get('name'), "owner_id": g.get('owner_id'), "members": members, "handle": g.get('handle')}
                            if state.group_member_remove_gid == gid:
                                owner_id = str(state.groups[gid].get('owner_id') or '')
                                state.group_member_remove_options = [m for m in members if m and m != owner_id]
                                state.group_member_remove_index = 0
                        except Exception:
                            pass
                        try:
                            state.group_manage_member_count = len(members)
                        except Exception:
                            pass
                        # Request profile for owner for display if needed
                        try:
                            owner = str(g.get('owner_id') or '')
                            if owner and owner not in state.profiles:
                                net.send({"type": "profile_get", "id": owner})
                        except Exception:
                            pass
                        try:
                            if state.members_view_mode and state.members_view_target == gid:
                                state.members_view_entries = member_labels(members, state, owner_id=owner)
                                state.members_view_title = f"Участники чата: {g.get('name') or gid}"
                        except Exception:
                            pass
                        state.status = f"Чат: {g.get('name') or gid} (участников: {len(members)})"
                else:
                    state.status = f"Ошибка запроса чата: {msg.get('reason')}"
            elif mtype == 'group_rename_result':
                if bool(msg.get('ok')):
                    gid = str(msg.get('group_id') or '')
                    name = str(msg.get('name') or '')
                    if gid and name:
                        try:
                            if gid in state.groups:
                                state.groups[gid]['name'] = name
                            if state.group_manage_gid == gid:
                                state.group_manage_name_input = name
                        except Exception:
                            pass
                        state.status = f"Чат переименован: {name}"
                else:
                    state.status = f"Не удалось переименовать чат: {msg.get('reason')}"
            elif mtype == 'group_updated':
                g = dict(msg.get('group') or {})
                gid = str(g.get('id') or '')
                if gid:
                    try:
                        if gid in state.groups:
                            state.groups[gid]['name'] = g.get('name') or state.groups[gid].get('name')
                            if g.get('handle') is not None:
                                state.groups[gid]['handle'] = g.get('handle')
                            if g.get('members') is not None:
                                members = []
                                for mid in g.get('members') or []:
                                    if isinstance(mid, dict):
                                        smid = str(mid.get('id') or mid.get('user_id') or '')
                                    else:
                                        smid = str(mid)
                                    if smid:
                                        members.append(smid)
                                state.groups[gid]['members'] = members
                        else:
                            members = []
                            for mid in g.get('members') or []:
                                if isinstance(mid, dict):
                                    smid = str(mid.get('id') or mid.get('user_id') or '')
                                else:
                                    smid = str(mid)
                                if smid:
                                    members.append(smid)
                            state.groups[gid] = {"name": g.get('name'), "owner_id": g.get('owner_id'), "members": members, "handle": g.get('handle')}
                    except Exception:
                        pass
                    # Refresh selection UI if this group is selected
                    try:
                        rows = build_contact_rows(state)
                        if gid in rows:
                            clamp_selection(state)
                    except Exception:
                        pass
                    try:
                        request_history_if_needed(state, net, gid)
                    except Exception:
                        pass
            elif mtype == 'board_added':
                b = dict(msg.get('board') or {})
                bid = str(b.get('id') or '')
                if bid:
                    state.boards[bid] = {"name": b.get('name'), "owner_id": b.get('owner_id'), "handle": b.get('handle'), "members": list(b.get('members') or [])}
                    state.status = f"Новая доска: {b.get('name') or bid}"
                    _touch_contact_rows(state)
                    # Info toast only for consensual invite flow; otherwise show consent modal below
                    try:
                        if bid in getattr(state, 'board_recent_invites', set()):
                            nm = str(b.get('name') or bid)
                            state.modal_message = f"Новая доска: {nm}"
                        else:
                            # Ensure no blocking modal when we intend to show consent
                            if state.modal_message:
                                state.modal_message = None
                    except Exception:
                        pass
                    try:
                        state.known_boards.add(bid)
                    except Exception:
                        pass
                    try:
                        request_history_if_needed(state, net, bid, force=True)
                    except Exception:
                        pass
                    # If not from an invite, ask for consent
                    try:
                        if bid not in state.board_recent_invites:
                            state.board_added_consent_mode = True
                            state.board_added_bid = bid
                            state.board_added_index = 0
                            state.action_menu_mode = False
                            state.action_menu_options = []
                            state.action_menu_peer = None
                        else:
                            state.board_recent_invites.discard(bid)
                    except Exception:
                        pass
                    # Refresh selection UI if this board is now present
                    try:
                        rows = build_contact_rows(state)
                        if bid in rows:
                            clamp_selection(state)
                    except Exception:
                        pass
                    # Proactively refresh boards list to ensure full sync (covers rare race cases)
                    try:
                        net.send({"type": "board_list"})
                    except Exception:
                        pass
                    # Fetch full history for the board upon appearing (news feed semantics)
                    try:
                        net.send({"type": "history", "room": bid, "since_id": 0})
                    except Exception:
                        pass
                    # Remove from pending invites, if present
                    try:
                        state.board_pending_invites.pop(bid, None)
                    except Exception:
                        pass
                    _touch_contact_rows(state)
                    # Ensure any lingering action menu for this invite is closed
                    try:
                        if state.action_menu_mode:
                            peer = state.action_menu_peer
                            if peer == f"BINV:{bid}" or peer == bid:
                                state.action_menu_mode = False
                                state.action_menu_peer = None
                                state.action_menu_options = []
                                state.action_menu_index = 0
                    except Exception:
                        pass
            elif mtype == 'board_updated':
                b = dict(msg.get('board') or {})
                bid = str(b.get('id') or '')
                if bid and bid in state.boards:
                    try:
                        if b.get('name') is not None:
                            state.boards[bid]['name'] = b.get('name')
                        if b.get('handle') is not None:
                            state.boards[bid]['handle'] = b.get('handle')
                        if b.get('members') is not None:
                            mlist = []
                            for mid in b.get('members') or []:
                                mlist.append(str(mid))
                            state.boards[bid]['members'] = mlist
                            if getattr(state, 'board_manage_bid', None) == bid:
                                try:
                                    state.board_manage_member_count = len(mlist)
                                except Exception:
                                    pass
                    except Exception:
                        pass
            elif mtype == 'board_add_result':
                # Owner receives the result of adding members to a board; refresh info/lists
                if bool(msg.get('ok')):
                    bid = str(msg.get('board_id') or '')
                    try:
                        if bid:
                            # Request updated board members and refresh boards list for consistency
                            net.send({"type": "board_info", "board_id": bid})
                            net.send({"type": "board_list"})
                    except Exception:
                        pass
                    try:
                        added = list(msg.get('added') or [])
                        if added:
                            state.status = f"Добавлены в доску ({len(added)}): {bid}"
                        else:
                            state.status = f"Нет новых участников: {bid}"
                    except Exception:
                        state.status = f"Участники добавлены: {bid}"
                else:
                    reason = str(msg.get('reason') or 'error')
                    mapping = {
                        'bad_id': 'Неверный ID доски',
                        'forbidden_or_not_found': 'Нет прав или не найдена',
                        'no_members': 'Не указаны корректные ID участников',
                        'server_error': 'Ошибка сервера',
                    }
                    state.status = f"Не удалось добавить участников: {mapping.get(reason, reason)}"
            elif mtype == 'board_info_result':
                if msg.get('ok'):
                    b = dict(msg.get('board') or {})
                    bid = str(b.get('id') or '')
                    if bid:
                        try:
                            if bid not in state.boards:
                                state.boards[bid] = {}
                            state.boards[bid]['name'] = b.get('name')
                            state.boards[bid]['owner_id'] = b.get('owner_id')
                            state.boards[bid]['handle'] = b.get('handle')
                            mlist = [str(x) for x in (b.get('members') or [])]
                            state.boards[bid]['members'] = mlist
                            if getattr(state, 'board_manage_bid', None) == bid:
                                state.board_manage_member_count = len(mlist)
                            if getattr(state, 'board_member_remove_mode', False) and getattr(state, 'board_member_remove_bid', None) == bid:
                                owner_id = str(b.get('owner_id') or '')
                                opts = [m for m in mlist if m and m != owner_id]
                                state.board_member_remove_options = opts
                                state.board_member_remove_index = 0
                            if getattr(state, 'members_view_mode', False) and getattr(state, 'members_view_target', None) == bid:
                                state.members_view_entries = member_labels(mlist, state, owner_id=str(b.get('owner_id') or ''))
                                state.members_view_title = f"Участники доски: {b.get('name') or bid}"
                        except Exception:
                            pass
                else:
                    state.status = f"Не удалось получить информацию о доске: {msg.get('reason')}"
            elif mtype == 'group_invite':
                # Queue invitation for consent under "Ожидают авторизацию"
                try:
                    grp = dict(msg.get('group') or {})
                    gid = str(grp.get('id') or '')
                    if gid:
                        state.group_pending_invites[gid] = {
                            'name': str(grp.get('name') or gid),
                            'from': str(msg.get('from') or grp.get('owner_id') or ''),
                        }
                        state.status = f"Приглашение в чат: {state.group_pending_invites[gid]['name']}"
                        _touch_contact_rows(state)
                        try:
                            is_fresh = (not getattr(state, 'boards', {})) and (not state.groups) and (not state.roster_friends)
                        except Exception:
                            is_fresh = False
                        if is_fresh and (not state.action_menu_mode) and (not state.search_action_mode):
                            rows = build_contact_rows(state)
                            tok = f"GINV:{gid}"
                            if tok in rows:
                                state.selected_index = rows.index(tok)
                            state.action_menu_mode = True
                            state.action_menu_peer = tok
                            state.action_menu_options = ["Принять приглашение", "Отклонить"]
                            state.action_menu_index = 0
                except Exception:
                    pass
            elif mtype == 'group_invite_result':
                try:
                    gid = str(msg.get('group_id') or '')
                    if gid:
                        state.group_pending_invites.pop(gid, None)
                except Exception:
                    pass
                _touch_contact_rows(state)
                if msg.get('ok'):
                    state.status = "Приглашение в группу обработано"
            elif mtype == 'board_invite':
                # Queue invitation for user consent (no immediate modal). Visible under "Ожидают авторизацию".
                b = dict(msg.get('board') or {})
                bid = str(b.get('id') or '')
                if bid:
                    try:
                        # Remember this invite to differentiate consensual add later and to show in pending list
                        try:
                            state.board_recent_invites.add(bid)
                        except Exception:
                            pass
                        state.board_pending_invites[bid] = {
                            'name': str(b.get('name') or bid),
                            'from': str(msg.get('from') or ''),
                        }
                        state.status = f"Приглашение в доску: {state.board_pending_invites[bid]['name']}"
                        _touch_contact_rows(state)
                        # If this looks like a fresh account (no groups/boards/friends), auto-focus and open actions
                        try:
                            is_fresh = (not getattr(state, 'boards', {})) and (not state.groups) and (not state.roster_friends)
                        except Exception:
                            is_fresh = False
                        if is_fresh and (not state.action_menu_mode) and (not state.search_action_mode):
                            # Move selection to the invite token (first under SEP4) and open actions menu
                            try:
                                rows = build_contact_rows(state)
                                tok = f"BINV:{bid}"
                                if tok in rows:
                                    state.selected_index = rows.index(tok)
                                state.action_menu_mode = True
                                state.action_menu_peer = tok
                                state.action_menu_options = ["Принять приглашение", "Отклонить"]
                                state.action_menu_index = 0
                            except Exception:
                                pass
                    except Exception:
                        pass
            elif mtype == 'board_invite_response_result':
                bid = str(msg.get('board_id') or '')
                if msg.get('ok'):
                    try:
                        if bid:
                            state.board_pending_invites.pop(bid, None)
                    except Exception:
                        pass
                    try:
                        acc = msg.get('accept')
                        if acc is True:
                            state.status = f"Приглашение принято: {bid}"
                        elif acc is False:
                            state.status = f"Приглашение отклонено: {bid}"
                        else:
                            state.status = "Приглашение обработано"
                    except Exception:
                        state.status = "Приглашение обработано"
                else:
                    state.status = f"Не удалось обработать приглашение: {msg.get('reason')}"
                _touch_contact_rows(state)
            elif mtype == 'board_rename_result':
                if msg.get('ok'):
                    bid = str(msg.get('board_id') or '')
                    name = str(msg.get('name') or '')
                    if bid and bid in getattr(state, 'boards', {}):
                        try:
                            state.boards[bid]['name'] = name
                            if getattr(state, 'board_manage_bid', None) == bid:
                                state.board_manage_name_input = name
                        except Exception:
                            pass
                        state.status = f"Доска переименована: {name}"
                else:
                    state.status = f"Не удалось переименовать доску: {msg.get('reason')}"
            elif mtype == 'board_set_handle_result':
                if msg.get('ok'):
                    bid = str(msg.get('board_id') or '')
                    h = str(msg.get('handle') or '')
                    if bid and bid in getattr(state, 'boards', {}):
                        try:
                            state.boards[bid]['handle'] = h
                            if getattr(state, 'board_manage_bid', None) == bid:
                                state.board_manage_handle_input = h
                        except Exception:
                            pass
                        state.status = f"Логин доски обновлён: {h}"
            elif mtype == 'board_disband_result':
                if bool(msg.get('ok')):
                    bid = str(msg.get('board_id') or '')
                    if bid and bid in state.boards:
                        try:
                            del state.boards[bid]
                        except Exception:
                            pass
                        _touch_contact_rows(state)
                        try:
                            rows = build_contact_rows(state)
                            state.selected_index = min(state.selected_index, max(0, len(rows) - 1))
                            clamp_selection(state)
                        except Exception:
                            pass
                        state.status = f"Доска расформирована: {bid}"
            elif mtype == 'board_removed':
                bid = str(msg.get('board_id') or '')
                name = str(msg.get('name') or '')
                if bid:
                    try:
                        if bid in state.boards:
                            del state.boards[bid]
                    except Exception:
                        pass
                    try:
                        state.known_boards.discard(bid)
                    except Exception:
                        pass
                    _touch_contact_rows(state)
                    try:
                        rows = build_contact_rows(state)
                        state.selected_index = min(state.selected_index, max(0, len(rows) - 1))
                        clamp_selection(state)
                    except Exception:
                        pass
                    nm = name or bid
                    state.status = f"Доска удалена: {nm}"
                    # Close any lingering action menu
                    try:
                        state.action_menu_mode = False
                        state.action_menu_options = []
                        state.action_menu_peer = None
                    except Exception:
                        pass
            elif mtype == 'boards':
                try:
                    items = list(msg.get('boards') or [])
                    newmap: Dict[str, Dict[str, object]] = {}
                    new_ids: Set[str] = set()
                    for it in items:
                        bid = str(it.get('id') or '')
                        if bid:
                            newmap[bid] = {"name": it.get('name'), "owner_id": it.get('owner_id'), "handle": it.get('handle')}
                            new_ids.add(bid)
                    # Detect newly appeared boards (e.g., offline add where no board_added event was received)
                    try:
                        appeared = new_ids.difference(getattr(state, 'known_boards', set()))
                    except Exception:
                        appeared = set()
                    state.boards = newmap
                    _touch_contact_rows(state)
                    try:
                        state.known_boards = set(new_ids)
                    except Exception:
                        pass
                    try:
                        for bid in new_ids:
                            request_history_if_needed(state, net, bid, force=True)
                        for gid in getattr(state, 'groups', {}):
                            request_history_if_needed(state, net, gid, force=True)
                    except Exception:
                        pass
                    # For first newly appeared board not preceded by invite — ask for consent
                    try:
                        # Avoid showing consent modal on the very first boards snapshot after login
                        if getattr(state, 'boards_initialized', False):
                            bid = next((x for x in appeared if x not in state.board_recent_invites), None)
                            if bid:
                                state.board_added_consent_mode = True
                                state.board_added_bid = bid
                                state.board_added_index = 0
                                state.action_menu_mode = False
                                state.action_menu_options = []
                                state.action_menu_peer = None
                        # Mark snapshots as initialized
                        state.boards_initialized = True
                    except Exception:
                        pass
                except Exception:
                    pass
            elif mtype == 'board_leave_result':
                if bool(msg.get('ok')):
                    bid = str(msg.get('board_id') or '')
                    if bid and bid in getattr(state, 'boards', {}):
                        try:
                            del state.boards[bid]
                        except Exception:
                            pass
                        try:
                            state.known_boards.discard(bid)
                        except Exception:
                            pass
                        _touch_contact_rows(state)
                        try:
                            rows = build_contact_rows(state)
                            state.selected_index = min(state.selected_index, max(0, len(rows) - 1))
                            clamp_selection(state)
                        except Exception:
                            pass
                        state.status = f"Покинули доску: {bid}"
                        try:
                            state.action_menu_mode = False
                            state.action_menu_options = []
                            state.action_menu_peer = None
                        except Exception:
                            pass
                else:
                    reason = str(msg.get('reason') or 'error')
                    mapping = {
                        'not_found': 'Доска не найдена',
                        'owner_cannot_leave': 'Создатель не может покинуть',
                        'not_member': 'Вы не участник',
                        'bad_id': 'Неверный ID доски',
                        'server_error': 'Ошибка сервера',
                    }
                    state.status = f"Не удалось покинуть доску: {mapping.get(reason, reason)}"
            elif mtype == 'group_leave_result':
                if bool(msg.get('ok')):
                    gid = str(msg.get('group_id') or '')
                    if gid and gid in state.groups:
                        try:
                            del state.groups[gid]
                        except Exception:
                            pass
                        _touch_contact_rows(state)
                        try:
                            rows = build_contact_rows(state)
                            state.selected_index = min(state.selected_index, max(0, len(rows) - 1))
                            clamp_selection(state)
                        except Exception:
                            pass
                        state.status = f"Покинули чат: {gid}"
                        try:
                            state.action_menu_mode = False
                            state.action_menu_options = []
                            state.action_menu_peer = None
                        except Exception:
                            pass
                else:
                    reason = str(msg.get('reason') or 'error')
                    mapping = {
                        'not_found': 'Чат не найден',
                        'owner_cannot_leave': 'Создатель не может покинуть',
                        'not_member': 'Вы не участник',
                        'bad_gid': 'Неверный ID чата',
                        'server_error': 'Ошибка сервера',
                        'not_authenticated': 'Требуется вход',
                    }
                    state.status = f"Не удалось покинуть чат: {mapping.get(reason, reason)}"

            elif mtype == 'group_disband_result':
                if bool(msg.get('ok')):
                    gid = str(msg.get('group_id') or '')
                    if gid and gid in state.groups:
                        try:
                            del state.groups[gid]
                        except Exception:
                            pass
                        _touch_contact_rows(state)
                        # If currently selected, move selection safely
                        try:
                            rows = build_contact_rows(state)
                            state.selected_index = min(state.selected_index, max(0, len(rows) - 1))
                            clamp_selection(state)
                        except Exception:
                            pass
                        state.status = f"Чат расформирован: {gid}"
                else:
                    reason = str(msg.get('reason') or 'error')
                    mapping = {
                        'rate_limited': 'Слишком часто',
                        'bad_gid': 'Неверный ID чата',
                        'forbidden_or_not_found': 'Нет прав или не найдена',
                        'server_error': 'Ошибка сервера',
                    }
                    state.status = f"Не удалось расформировать: {mapping.get(reason, reason)}"
            elif mtype == 'group_removed':
                gid = str(msg.get('group_id') or '')
                name = str(msg.get('name') or '')
                if gid:
                    try:
                        if gid in state.groups:
                            del state.groups[gid]
                    except Exception:
                        pass
                    _touch_contact_rows(state)
                    try:
                        rows = build_contact_rows(state)
                        state.selected_index = min(state.selected_index, max(0, len(rows) - 1))
                        clamp_selection(state)
                    except Exception:
                        pass
                    nm = name or gid
                    state.status = f"Чат удалён: {nm}"
            elif mtype == 'groups':
                # full list refresh
                try:
                    items = list(msg.get('groups') or [])
                    newmap: Dict[str, Dict[str, object]] = {}
                    for it in items:
                        gid = str(it.get('id') or '')
                        if gid:
                            newmap[gid] = {"name": it.get('name'), "owner_id": it.get('owner_id'), "members": list(it.get('members') or [])}
                    state.groups = newmap
                    _touch_contact_rows(state)
                    try:
                        for gid in newmap.keys():
                            request_history_if_needed(state, net, gid, force=True)
                    except Exception:
                        pass
                except Exception:
                    pass
            elif mtype == 'error':
                em = str(msg.get('message') or '')
                state.status = f"Error: {em}"
                logging.getLogger('client').error("Server error: %s", em)
                # If board post was forbidden, undo last optimistic outbound line in current board
                try:
                    import time as _t
                    if em in ('board_post_forbidden', 'board_check_failed'):
                        sel = current_selected_id(state)
                        if sel and (sel in getattr(state, 'boards', {})):
                            conv = state.conversations.get(sel, [])
                            if conv:
                                now = _t.time()
                                for i in range(len(conv) - 1, -1, -1):
                                    m = conv[i]
                                    if getattr(m, 'direction', None) == 'out' and getattr(m, 'msg_id', None) in (None, 0):
                                        # Only rollback very recent optimistic sends (<5s)
                                        try:
                                            if (now - float(getattr(m, 'ts', now))) <= 5.0:
                                                del conv[i]
                                                break
                                        except Exception:
                                            del conv[i]
                                            break
                            # Optional: show a modal to inform the user
                            state.modal_message = "Публиковать может только владелец доски"
                except Exception:
                    pass
            elif mtype == 'disconnected':
                state.status = "Disconnected"
                running = False
                logging.getLogger('client').warning("Disconnected event received")
            # else: ignore
            need_redraw = True
            _dbg(f"[event] type={mtype} need_redraw={need_redraw} search_mode={state.search_mode} modal_active={modal_now}")

        # Handle input
        try:
            ch = stdscr.get_wch()
        except curses.error:
            ch = None
        if (ch is not None) and (KEYLOG_ENABLED or getattr(state, 'debug_mode', False)):
            try:
                _dbg(f"[key] { _key_repr(ch) } search_mode={state.search_mode} modal_active={overlay_active} {_state_sig(state)}")
                if isinstance(ch, str) and ch.startswith('\x1b'):
                    _dbg(f"[esc] raw={repr(ch)} len={len(ch)}")
            except Exception:
                pass
        # Assemble ESC sequences when they arrive piece-wise.
        # Handles SGR mouse (ESC [ < ... M/m) and legacy X10 mouse (ESC [ M b x y).
        if (isinstance(ch, str) and ch == '\x1b') or (not isinstance(ch, str) and ch == 27):
            try:
                buf = '\x1b'
                seq_complete = False
                x10_left = 0
                for _ in range(32):  # generous upper bound; typical sequences < 8 chars
                    try:
                        nxt = stdscr.get_wch()
                    except curses.error:
                        time.sleep(0.001)
                        continue
                    if isinstance(nxt, str):
                        buf += nxt
                        if x10_left > 0:
                            x10_left -= 1
                            if x10_left == 0:
                                seq_complete = True
                                break
                            continue
                        if buf == '\x1b[M':
                            # X10 mouse prefix; needs 3 more bytes (b, x, y)
                            x10_left = 3
                            continue
                        if buf in ('\x1b[I', '\x1b[O'):
                            seq_complete = True
                            break
                        if buf.startswith('\x1b['):
                            last = buf[-1]
                            if last in ('A', 'B', 'C', 'D'):
                                seq_complete = True
                                break
                            if last == '~' or last in ('H', 'F'):
                                seq_complete = True
                                break
                            if last in ('M', 'm'):
                                # Complete SGR mouse only (it contains '<' or ';').
                                if buf.startswith('\x1b[<') or (';' in buf):
                                    seq_complete = True
                                    break
                    else:
                        ch = nxt
                        break
                if seq_complete:
                    ch = buf
                else:
                    # return unread characters (except initial ESC) back to input buffer
                    for pending in reversed(buf[1:]):
                        try:
                            curses.ungetch(ord(pending))
                        except Exception:
                            break
                    ch = '\x1b'
            except Exception:
                pass
        # Optional key/mouse debugging for diagnostics (shown in F12 overlay).
        try:
            if ch is not None and (KEYLOG_ENABLED or getattr(state, 'debug_mode', False)):
                state.debug_last_key = f"KEY {repr(ch)} ({'str' if isinstance(ch,str) else 'int'})"
                if isinstance(ch, str) and ch.startswith('\x1b'):
                    state.debug_last_seq = ch
                if KEYLOG_ENABLED:
                    logging.getLogger('client').debug('KEY evt: %r (%s)', ch, ('str' if isinstance(ch, str) else f'int:{ch}'))
        except Exception:
            pass

        # On auth screen, treat ESC immediately as exit to avoid being masked by other handlers
        try:
            if (not state.authed) and (ch == 27 or (isinstance(ch, str) and ch == '\x1b')):
                running = False
                break
        except Exception:
            pass

        # Global: F4 copies full current screen into clipboard (and optionally saves a dump file).
        try:
            f4_seq = ('\x1bOS', '\x1b[14~')
            if ch == getattr(curses, 'KEY_F4', -9999) or (isinstance(ch, str) and ch in f4_seq):
                _log_action("screen dump key (F4)")
                copied, saved_path = copy_screen_to_clipboard(stdscr)
                if copied:
                    state.status = "Экран скопирован в буфер"
                elif saved_path:
                    state.status = f"Буфер недоступен; сохранено: {saved_path}"
                else:
                    state.status = "Копирование экрана не удалось"
                continue
        except Exception:
            pass

        # Terminal resize (KEY_RESIZE / code 410) — force full redraw to avoid blank history after curses clear
        try:
            if (not isinstance(ch, str)) and (ch == getattr(curses, 'KEY_RESIZE', 410) or ch == 410):
                need_redraw = True
                state.history_dirty = True
                state._hist_blank = False
                try:
                    state._force_full_redraw = True  # type: ignore[attr-defined]
                except Exception:
                    pass
                continue
        except Exception:
            pass

        # Read-only просмотр участников: закрыть по Esc/Enter, блокировать остальное
        if state.members_view_mode:
            if ch in ('\x1b',) or ch == 27 or ch in ('\n', '\r') or ch in (10, 13):
                state.members_view_mode = False
                state.members_view_target = None
                state.members_view_entries = []
                state.members_view_title = ""
            continue

        if ch is None:
            time.sleep(0.01)
            # Живой индикатор обнаруженного файла в поле ввода
            try:
                to = current_selected_id(state)
                if state.input_buffer and to and (not is_separator(to)) and (not state.modal_message):
                    txt = state.input_buffer
                    now_hint = time.time()
                    last_txt = getattr(state, "_input_hint_last_text", None)
                    last_ts = float(getattr(state, "_input_hint_last_ts", 0.0))
                    if txt != last_txt and (now_hint - last_ts) >= 0.2:
                        state._input_hint_last_text = txt
                        state._input_hint_last_ts = now_hint
                        cand = extract_path_candidate(txt)
                        state._input_hint_last_candidate = cand
                        meta = file_meta_for(cand) if cand else None
                        state._input_hint_last_meta_ok = bool(meta)
                        state._input_hint_last_meta_name = meta.name if meta else None
                    meta_ok = bool(getattr(state, "_input_hint_last_meta_ok", False))
                    meta_name = getattr(state, "_input_hint_last_meta_name", None)
                    if meta_ok and meta_name and not state.status:
                        state.status = f"Обнаружен файл: {meta_name}. Enter — подтвердить прикрепление"
                        need_redraw = True
                else:
                    # Сброс кеша, если ввода нет
                    state._input_hint_last_text = ""
                    state._input_hint_last_meta_ok = False
            except Exception:
                pass
            continue

        # Любое полученное событие ввода требует перерисовки интерфейса
        need_redraw = True

        # ===== Focus change and bracketed paste handling (macOS-friendly) =====
        try:
            if isinstance(ch, str):
                # Focus events: ESC [ I (focus in), ESC [ O (focus out)
                if ch in ('\x1b[I',):
                    try:
                        state.terminal_focused = True  # type: ignore[attr-defined]
                        # Re-enable mouse tracking on focus gain if it was enabled before
                        if bool(getattr(state, 'mouse_enabled', True)):
                            _apply_mouse(True)
                    except Exception:
                        pass
                    # do not consume other handlers — continue
                elif ch in ('\x1b[O',):
                    try:
                        state.terminal_focused = False  # type: ignore[attr-defined]
                    except Exception:
                        pass
                # Bracketed paste: ESC [ 200 ~ ... ESC [ 201 ~
                elif ch == '\x1b[200~':
                    paste_buf = ''
                    # Collect until end tag or until a safe bound
                    max_steps = 100000
                    steps = 0
                    while steps < max_steps:
                        steps += 1
                        try:
                            nxt = stdscr.get_wch()
                        except curses.error:
                            time.sleep(0.001)
                            continue
                        if isinstance(nxt, str):
                            paste_buf += nxt
                            # Detect end tag possibly split across calls
                            if paste_buf.endswith('\x1b[201~'):
                                paste_buf = paste_buf[:-len('\x1b[201~')]
                                break
                        else:
                            # Non-str — push back by reusing main loop on next frame
                            ch = nxt
                            break
                    # Apply paste into input editor at caret
                    try:
                        chat_input.insert_text(state, paste_buf)
                        state.status = f"Вставлено: {len(paste_buf)} символов"
                    except Exception:
                        pass
                    # consume event
                    continue
        except Exception:
            pass

        # Handle SGR/X10 wheel events directly (no KEY_MOUSE to avoid flicker)
        if isinstance(ch, str) and ch.startswith('\x1b[') and bool(getattr(state, 'mouse_enabled', False)):
            parsed_mouse = _parse_sgr_mouse(ch)
            if parsed_mouse:
                mx, my, bstate = parsed_mouse
                try:
                    if getattr(state, 'debug_mode', False):
                        state.debug_last_mouse = f"SGR_MOUSE x={mx} y={my} bstate={bstate}"
                except Exception:
                    pass
                try:
                    state.mouse_events_total = int(getattr(state, 'mouse_events_total', 0)) + 1
                    state.mouse_last_seen_ts = time.time()
                except Exception:
                    pass
                try:
                    wheel_up, wheel_down = _compute_wheel_masks()
                    direction = _wheel_direction_from_bstate(bstate, wheel_up, wheel_down)
                    if direction:
                        _handle_wheel_scroll(state, stdscr, mx, my, direction)
                        need_redraw = True
                        continue
                except Exception:
                    pass
                # Raw click selection for contacts list (for terminals where clicks arrive as SGR strings)
                try:
                    # Header hotkeys (top bar)
                    if int(my) == 0:
                        if handle_hotkey_click(int(mx), int(my)):
                            continue
                        continue
                except Exception:
                    pass
                try:
                    btn1_any = (
                        getattr(curses, 'BUTTON1_PRESSED', 0)
                        | getattr(curses, 'BUTTON1_CLICKED', 0)
                        | getattr(curses, 'BUTTON1_RELEASED', 0)
                        | getattr(curses, 'BUTTON1_DOUBLE_CLICKED', 0)
                    )
                    btn3_any = (
                        getattr(curses, 'BUTTON3_PRESSED', 0)
                        | getattr(curses, 'BUTTON3_CLICKED', 0)
                        | getattr(curses, 'BUTTON3_RELEASED', 0)
                        | getattr(curses, 'BUTTON3_DOUBLE_CLICKED', 0)
                    )
                    if not (bstate & (btn1_any | btn3_any)):
                        # Not a click-like event (ignore motion/noise)
                        pass
                    else:
                        # Determine left pane width to detect clicks on contacts.
                        try:
                            h, w = stdscr.getmaxyx()
                            left_w = int(getattr(state, 'last_left_w', 0) or 0) or max(20, min(30, w // 4))
                        except Exception:
                            left_w = 20
                        if int(mx) < int(left_w):
                            dbg = bool(getattr(state, 'debug_mode', False))
                            if bstate & btn3_any:
                                open_actions_menu_for_selection(state, net)
                                try:
                                    if dbg:
                                        state.debug_lines.append(f"[mouse] right_click x={mx} y={my} bstate={bstate}")
                                        if len(state.debug_lines) > 300:
                                            del state.debug_lines[:len(state.debug_lines) - 300]
                                except Exception:
                                    pass
                                need_redraw = True
                                continue
                            if bstate & btn1_any:
                                rows = build_contact_rows(state)
                                idx = None
                                tok = None
                                try:
                                    ymap = getattr(state, 'contacts_y_map', None)
                                    if isinstance(ymap, dict):
                                        idx = ymap.get(int(my))
                                except Exception:
                                    idx = None
                                cs = 0
                                vis_h = 0
                                max_rows2 = 0
                                if idx is None:
                                    start_y = 2  # contacts_start_y in draw_ui
                                    try:
                                        h, w = stdscr.getmaxyx()
                                        vis_h = max(0, h - start_y - 2)
                                    except Exception:
                                        vis_h = int(getattr(state, 'last_left_h', 10)) or 10
                                    cs = max(0, int(getattr(state, 'contacts_scroll', 0)))
                                    max_rows2 = min(max(0, len(rows) - cs), vis_h)
                                    if start_y <= int(my) < start_y + max_rows2:
                                        idx = cs + (int(my) - start_y)
                                try:
                                    if dbg:
                                        state.debug_lines.append(
                                            f"[mouse] btn1 x={mx} y={my} bstate={bstate} idx={idx} cs={cs} rows={len(rows)} vis_h={vis_h} max_rows={max_rows2}"
                                        )
                                        if len(state.debug_lines) > 300:
                                            del state.debug_lines[:len(state.debug_lines) - 300]
                                except Exception:
                                    pass
                                if isinstance(idx, int) and 0 <= idx < len(rows):
                                    try:
                                        tok = rows[idx]
                                    except Exception:
                                        tok = None
                                    state.selected_index = int(idx)
                                    clamp_selection(state, prefer='down', rows=rows)
                                    sel = current_selected_id(state, rows=rows)
                                    _maybe_send_message_read(state, net, sel)
                                    state.history_scroll = 0
                                    try:
                                        if dbg:
                                            state.debug_lines.append(f"[mouse] hit idx={idx} tok={repr(tok)}")
                                            state.debug_lines.append(f"[mouse] select {sel} idx={idx} x={mx} y={my}")
                                            if len(state.debug_lines) > 300:
                                                del state.debug_lines[:len(state.debug_lines) - 300]
                                    except Exception:
                                        pass
                                else:
                                    try:
                                        if dbg:
                                            state.debug_lines.append(f"[mouse] btn1 miss y={my} idx={idx}")
                                            if len(state.debug_lines) > 300:
                                                del state.debug_lines[:len(state.debug_lines) - 300]
                                    except Exception:
                                        pass
                                need_redraw = True
                                continue
                except Exception as e:
                    try:
                        if bool(getattr(state, 'debug_mode', False)):
                            state.debug_lines.append(f"[mouse] ERROR {type(e).__name__}: {e}")
                            if len(state.debug_lines) > 300:
                                del state.debug_lines[:len(state.debug_lines) - 300]
                    except Exception:
                        pass
        # Formatting toolbar focus/control
        if getattr(state, 'format_toolbar_mode', False):
            if ch in ('\t', curses.KEY_BTAB, '\x1b', 27):
                state.format_toolbar_mode = False
                state.status = ""
                continue
            if ch in (curses.KEY_LEFT, curses.KEY_UP):
                try:
                    n = len(FORMAT_ACTIONS)
                    state.format_toolbar_index = (int(getattr(state, 'format_toolbar_index', 0)) - 1) % n
                except Exception:
                    state.format_toolbar_index = 0
                continue
            if ch in (curses.KEY_RIGHT, curses.KEY_DOWN):
                try:
                    n = len(FORMAT_ACTIONS)
                    state.format_toolbar_index = (int(getattr(state, 'format_toolbar_index', 0)) + 1) % n
                except Exception:
                    state.format_toolbar_index = 0
                continue
            if ch in ('\n', '\r') or ch in (curses.KEY_ENTER, 10, 13):
                idx = max(0, min(len(FORMAT_ACTIONS) - 1, int(getattr(state, 'format_toolbar_index', 0))))
                _, _, kind = FORMAT_ACTIONS[idx]
                if kind == 'link':
                    state.format_toolbar_mode = False
                    state.format_link_mode = True
                    state.format_link_field = 0
                    state.format_link_text = ""
                    state.format_link_url = ""
                    state.status = "Введите текст и ссылку"
                else:
                    apply_format_to_input(state, kind)
                    state.format_toolbar_mode = False
                    state.status = "Формат применён"
                continue
            # Block other keys while toolbar active
            continue

        # Enter formatting toolbar with Tab when no overlays/suggestions wanted
        if ch in ('\t',) and state.authed and not (state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.help_mode or state.modal_message or getattr(state, 'group_create_mode', False) or state.group_manage_mode or getattr(state, 'board_create_mode', False) or getattr(state, 'board_manage_mode', False) or getattr(state, 'format_link_mode', False)):
            try:
                text = state.input_buffer or ''
            except Exception:
                text = ''
            looks_like_command = text.startswith('/')
            looks_like_path = ('/' in text or '\\' in text or text.startswith('~'))
            if not looks_like_command and not looks_like_path:
                state.suggest_mode = False
                state.format_toolbar_mode = True
                state.format_toolbar_index = 0
                state.status = "Панель форматирования: ←/→ выбрать, Enter применить, Tab — обратно"
                continue

        # ===== Подсказки: управление в режиме подсказок =====
        if getattr(state, 'suggest_mode', False):
            # Close
            if ch in ('\x1b',) or ch == 27:
                state.suggest_mode = False
                state.suggest_items = []
                state.suggest_index = 0
                continue
            # Navigate (skip when dropdown/view menus are active)
            if (not getattr(state, 'file_browser_menu_mode', False)) and (not getattr(state, 'file_browser_view_mode', False)) and (ch in ('k',) or ch == curses.KEY_UP):
                if state.suggest_items:
                    state.suggest_index = (state.suggest_index - 1) % len(state.suggest_items)
                continue
            if (not getattr(state, 'file_browser_menu_mode', False)) and (not getattr(state, 'file_browser_view_mode', False)) and (ch in ('j',) or ch == curses.KEY_DOWN):
                if state.suggest_items:
                    state.suggest_index = (state.suggest_index + 1) % len(state.suggest_items)
                continue
            # Accept
            if ch in ('\t',) or ch in ('\n', '\r') or ch in (curses.KEY_ENTER, 10, 13):
                try:
                    items = list(state.suggest_items or [])
                    if not items:
                        state.suggest_mode = False
                        continue
                    i = max(0, min(len(items) - 1, int(getattr(state, 'suggest_index', 0))))
                    sel = items[i]
                    repl = getattr(sel, 'name', None) or str(sel)
                    # Replace span in input_buffer
                    a = max(0, int(getattr(state, 'suggest_start', 0)))
                    b = max(a, int(getattr(state, 'suggest_end', a)))
                    state.input_buffer = (state.input_buffer[:a] + repl + state.input_buffer[b:])
                    state.input_caret = a + len(repl)
                except Exception:
                    pass
                state.suggest_mode = False
                continue
        # ===== Debug overlay close on ESC =====
        try:
            if getattr(state, 'debug_mode', False) and (ch in ('\x1b',) or ch == 27):
                state.debug_mode = False
                state.status = ''
                continue
        except Exception:
            pass
        # ===== Debug overlay actions (copy/save) отключены, чтобы не мешать вводу =====

        # Update prompt: allow manual update without auto-restart mid-work.
        if getattr(state, "update_prompt_mode", False):
            do_update = False
            if isinstance(ch, str) and ch == "\x15":  # Ctrl+U
                do_update = True
            if isinstance(ch, str) and ch in ("\n", "\r"):
                do_update = True
            if ch in (curses.KEY_ENTER, 10, 13):
                do_update = True
            if do_update:
                state.update_prompt_mode = False
                try:
                    state.update_prompt_reason = None
                except Exception:
                    pass
                latest = interactive_update_check(stdscr, confirm=False)
                if latest and latest == CLIENT_VERSION:
                    state.status = f"Клиент актуален (v{CLIENT_VERSION})"
                elif latest:
                    state.status = f"Клиент обновлён до v{latest}"
                else:
                    state.status = "Не удалось обновить клиент (проверьте сеть/подпись)"
                continue
            # Dismiss on Esc or any other key to avoid blocking work.
            if ch is not None:
                state.update_prompt_mode = False
                try:
                    lat = str(getattr(state, "update_prompt_latest", "") or "").strip()
                    state.update_prompt_dismissed_latest = lat or "<unknown>"
                    if lat and lat != CLIENT_VERSION:
                        state.status = f"Доступно обновление до v{lat} (Ctrl+U)"
                    else:
                        state.status = "Доступно обновление (Ctrl+U)"
                except Exception:
                    state.status = "Доступно обновление (Ctrl+U)"
                continue

        # Simple modal: for authed users block input until Enter/ESC; on auth screen keep modal but allow typing.
        if getattr(state, 'modal_message', None):
            if isinstance(ch, str) and ch in ('\n', '\r') or ch in (curses.KEY_ENTER, 10, 13, 27):
                state.modal_message = None
                state.status = ""
                if state.authed:
                    continue
            if state.authed:
                continue
        # Help overlay: close on Enter/ESC/F1 and block chat input
        if state.help_mode:
            if (isinstance(ch, str) and ch in ('\n', '\r')) or ch in (curses.KEY_ENTER, 13):
                state.help_mode = False
                state.status = ""
                continue
            if ch == curses.KEY_F1:
                state.help_mode = False
                state.status = ""
                continue
        # Modal for inserting link
        if getattr(state, 'format_link_mode', False):
            if ch in ('\x1b',) or ch == 27:
                state.format_link_mode = False
                state.status = "Вставка ссылки отменена"
                continue
            if ch in ('\t', curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT):
                state.format_link_field = 1 - int(getattr(state, 'format_link_field', 0))
                continue
            if ch in (curses.KEY_BACKSPACE, 127):
                if state.format_link_field == 0:
                    state.format_link_text = (state.format_link_text[:-1]) if state.format_link_text else ""
                else:
                    state.format_link_url = (state.format_link_url[:-1]) if state.format_link_url else ""
                continue
            if isinstance(ch, str) and ch in ('\n', '\r') or ch in (curses.KEY_ENTER, 10, 13):
                url = (state.format_link_url or "").strip()
                title = (state.format_link_text or "").strip()
                if not url:
                    state.status = "Введите ссылку"
                    continue
                apply_format_to_input(state, 'link', link_text=title, link_url=url)
                state.format_link_mode = False
                state.status = "Ссылка добавлена"
                continue
            if isinstance(ch, str) and ch and ch.isprintable():
                if state.format_link_field == 0:
                    state.format_link_text = (state.format_link_text or "") + ch
                else:
                    state.format_link_url = (state.format_link_url or "") + ch
                continue
        # Early handle: actions menu navigation (non-auth menus) should react to arrows/k/j immediately
        if state.action_menu_mode and state.action_menu_options and not _is_auth_actions_menu(state):
            # Arrows or vim-like keys
            if ch in ('k',) or ch in (curses.KEY_UP, curses.KEY_LEFT):
                try:
                    n = len(state.action_menu_options)
                    if n > 0:
                        state.action_menu_index = (state.action_menu_index - 1) % n
                except Exception:
                    pass
                continue
            if ch in ('j',) or ch in (curses.KEY_DOWN, curses.KEY_RIGHT):
                try:
                    n = len(state.action_menu_options)
                    if n > 0:
                        state.action_menu_index = (state.action_menu_index + 1) % n
                except Exception:
                    pass
                continue

        # Tab-completion: open suggestions for slash-commands or file paths
        if ch in ('\t',) and not (state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.help_mode or state.modal_message):
            try:
                text = state.input_buffer
                caret = max(0, min(len(text), int(getattr(state, 'input_caret', 0))))
                # Slash-commands at first token
                if text.startswith('/'):
                    # Span: from 0 to first space or caret
                    end = text.find(' ')
                    if end == -1 or caret <= end:
                        end = max(caret, end if end != -1 else caret)
                        items = slash_suggest(text[:end] or '/', limit=10)
                        if items:
                            state.suggest_mode = True
                            state.suggest_kind = 'slash'
                            state.suggest_items = items
                            state.suggest_index = 0
                            state.suggest_start = 0
                            state.suggest_end = end
                            continue
                # File token around caret
                # Determine token [a, b) around caret split by whitespace
                a = text.rfind(' ', 0, caret)
                a2 = text.rfind('\n', 0, caret)
                a = max(a, a2) + 1
                b = caret
                while b < len(text) and not text[b].isspace():
                    b += 1
                token = text[a:b]
                if token and (token.startswith('/') or token.startswith('~') or token.startswith('./') or token.startswith('../') or ('/' in token) or ('\\' in token)):
                    items = get_file_system_suggestions(token, cwd=str(Path('.').resolve()), limit=10)
                    if items:
                        state.suggest_mode = True
                        state.suggest_kind = 'file'
                        state.suggest_items = items
                        state.suggest_index = 0
                        state.suggest_start = a
                        state.suggest_end = b
                        continue
            except Exception:
                pass
        # File manager modal handling (F7)
        if getattr(state, 'file_browser_mode', False):
            fm = getattr(state, 'file_browser_state', None)
            if not isinstance(fm, FileManagerState):
                prefs = _get_fb_prefs()
                start = _fm_norm_path('.')
                try:
                    p0 = prefs.get('fb_path0')
                    if p0 and os.path.isdir(str(p0)):
                        start = _fm_norm_path(str(p0))
                    else:
                        home = _fm_norm_path('~')
                        if os.path.isdir(home):
                            start = home
                except Exception:
                    pass
                fm = FileManagerState(
                    path=start,
                    show_hidden=bool(prefs.get('fb_show_hidden0', False)),
                    sort=str(prefs.get('fb_sort0', 'name')),
                    dirs_first=bool(prefs.get('fb_dirs_first0', True)),
                    reverse=bool(prefs.get('fb_reverse0', False)),
                    view=prefs.get('fb_view0', None),
                )
                _fm_relist(fm)
                state.file_browser_state = fm

            def _persist_local() -> None:
                try:
                    _save_fb_prefs_values(
                        fb_show_hidden0=bool(fm.show_hidden),
                        fb_sort0=str(fm.sort or 'name'),
                        fb_dirs_first0=bool(fm.dirs_first),
                        fb_reverse0=bool(fm.reverse),
                        fb_view0=fm.view,
                        fb_path0=str(fm.path or ''),
                        fb_side=0,
                    )
                except Exception:
                    pass
                try:
                    state.file_browser_show_hidden0 = bool(fm.show_hidden)
                    state.file_browser_sort0 = str(fm.sort or 'name')
                    state.file_browser_dirs_first0 = bool(fm.dirs_first)
                    state.file_browser_reverse0 = bool(fm.reverse)
                    state.file_browser_view0 = fm.view
                    state.file_browser_path0 = str(fm.path or '')
                    state.file_browser_side = 0
                except Exception:
                    pass

            def _sync_prefs_to_server() -> None:
                try:
                    net.send({
                        "type": getattr(T, 'PREFS_SET', 'prefs_set'),
                        "values": {
                            "fb_show_hidden0": bool(fm.show_hidden),
                            "fb_sort0": str(fm.sort or 'name'),
                            "fb_dirs_first0": bool(fm.dirs_first),
                            "fb_reverse0": bool(fm.reverse),
                            "fb_view0": fm.view,
                        },
                    })
                except Exception:
                    pass

            def _list_geom() -> Tuple[int, int]:
                try:
                    h, _w = stdscr.getmaxyx()
                except Exception:
                    h = 24
                list_top = 3
                hint_y = h - 1
                return list_top, max(1, hint_y - list_top)

            def _clamp_index() -> None:
                n = len(fm.items)
                if n <= 0:
                    fm.index = 0
                else:
                    fm.index = max(0, min(int(fm.index), n - 1))

            def _ensure_visible() -> None:
                try:
                    _t, list_rows = _list_geom()
                    _fm_ensure_visible(fm, list_rows)
                except Exception:
                    pass

            def _go_up() -> None:
                base = Path(_fm_norm_path(fm.path))
                cur_name = base.name or None
                parent = base.parent if base.parent and base.parent != base else base
                fm.path = str(parent)
                fm.mode = 'browse'
                fm.filter_text = ''
                fm.goto_text = ''
                _fm_relist(fm, prefer_name=cur_name)
                _ensure_visible()

            def _activate_selected() -> None:
                if not fm.items:
                    return
                _clamp_index()
                name, is_dir = fm.items[int(fm.index)]
                base = Path(_fm_norm_path(fm.path))
                if is_dir or name == '..':
                    if name == '..':
                        _go_up()
                        return
                    try:
                        fm.path = _fm_norm_path(str(base / name))
                        fm.mode = 'browse'
                        fm.filter_text = ''
                        fm.goto_text = ''
                        fm.index = 0
                        fm.scroll = 0
                        _fm_relist(fm, prefer_name='')
                        _ensure_visible()
                    except Exception:
                        pass
                    return
                try:
                    sp = _fm_norm_path(str(base / name))
                except Exception:
                    sp = str(base / name)
                txt = f"/file {sp}"
                try:
                    state.input_buffer = txt
                    state.input_caret = len(txt)
                except Exception:
                    state.input_buffer = txt
                close_file_browser(state)  # type: ignore[name-defined]
                state.status = f"Выбран файл: {sp}"

            def _handle_mouse(mx: int, my: int, bstate: int) -> Optional[str]:
                list_top, list_rows = _list_geom()
                wheel_up, wheel_down = _compute_wheel_masks()
                d = _wheel_direction_from_bstate(bstate, wheel_up, wheel_down)
                if d:
                    step = 3
                    if d == 'up':
                        fm.index = max(0, int(fm.index) - step)
                    else:
                        fm.index = min(max(0, len(fm.items) - 1), int(fm.index) + step)
                    _fm_ensure_visible(fm, list_rows)
                    return 'handled'
                click_mask = (
                    getattr(curses, 'BUTTON1_PRESSED', 0)
                    | getattr(curses, 'BUTTON1_CLICKED', 0)
                    | getattr(curses, 'BUTTON1_RELEASED', 0)
                )
                if not (bstate & click_mask):
                    return None
                if not (list_top <= my < list_top + list_rows):
                    return 'handled'
                _fm_ensure_visible(fm, list_rows)
                start = int(getattr(fm, 'scroll', 0))
                new_idx = start + (my - list_top)
                if fm.items:
                    fm.index = max(0, min(len(fm.items) - 1, int(new_idx)))
                    _fm_ensure_visible(fm, list_rows)
                try:
                    now = time.time()
                    last_ts = float(getattr(state, 'fb_last_click_ts', 0.0))
                    last_row = int(getattr(state, 'fb_last_click_row', -1))
                    if last_row == int(fm.index) and (now - last_ts) <= 0.35:
                        state.fb_last_click_ts = 0.0
                        state.fb_last_click_row = -1
                        return 'activate'
                    state.fb_last_click_ts = now
                    state.fb_last_click_row = int(fm.index)
                    state.fb_last_click_side = 0
                except Exception:
                    pass
                if bstate & getattr(curses, 'BUTTON1_DOUBLE_CLICKED', 0):
                    return 'activate'
                return 'handled'

            # Raw SGR mouse sequences (macOS Terminal/iTerm2) fallback.
            if isinstance(ch, str) and ch.startswith('\x1b[<'):
                try:
                    import re as _re

                    m = _re.match(r"\x1b\[<(?P<b>\d+);(?P<x>\d+);(?P<y>\d+)(?P<t>[Mm])", ch)
                    if m:
                        Cb = int(m.group('b'))
                        mx = int(m.group('x')) - 1
                        my = int(m.group('y')) - 1
                        is_press = (m.group('t') == 'M')
                        wheel_up, wheel_down = _compute_wheel_masks()
                        bstate = 0
                        if (Cb & 0x40) and (Cb & 1) == 0:
                            bstate |= wheel_up
                        elif (Cb & 0x40) and (Cb & 1) == 1:
                            bstate |= wheel_down
                        else:
                            if is_press:
                                bstate |= getattr(curses, 'BUTTON1_PRESSED', 0) or getattr(curses, 'BUTTON1_CLICKED', 0)
                            else:
                                bstate |= getattr(curses, 'BUTTON1_RELEASED', 0) or getattr(curses, 'BUTTON1_CLICKED', 0)
                        res = _handle_mouse(mx, my, bstate)
                        if res == 'activate':
                            _activate_selected()
                        state.file_browser_state = fm
                        continue
                except Exception:
                    pass

            # Native mouse events
            if ch == curses.KEY_MOUSE:
                try:
                    _, mx, my, _, bstate = curses.getmouse()
                    res = _handle_mouse(int(mx), int(my), int(bstate))
                    if res == 'activate':
                        _activate_selected()
                    state.file_browser_state = fm
                except Exception:
                    pass
                continue

            # ESC: close or exit input modes.
            if ch in ('\x1b',) or ch == 27:
                if fm.mode == 'filter':
                    fm.mode = 'browse'
                    if fm.filter_text:
                        fm.filter_text = ''
                        _fm_relist(fm, rescan=False)
                        _ensure_visible()
                    state.file_browser_state = fm
                    continue
                if fm.mode == 'goto':
                    fm.mode = 'browse'
                    fm.goto_text = ''
                    state.file_browser_state = fm
                    continue
                close_file_browser(state)  # type: ignore[name-defined]
                continue

            # Ctrl+L: path prompt
            if (isinstance(ch, str) and ch == '\x0c') or ch == 12:
                fm.mode = 'goto'
                fm.goto_text = str(fm.path or '')
                state.file_browser_state = fm
                continue

            # Refresh (F5)
            if ch == curses.KEY_F5:
                _fm_relist(fm)
                _ensure_visible()
                state.file_browser_state = fm
                continue

            # Enter in goto mode
            if fm.mode == 'goto' and (ch in ('\n', '\r') or ch in (curses.KEY_ENTER, 10, 13)):
                try:
                    raw = (fm.goto_text or '').strip()
                    if not raw:
                        fm.mode = 'browse'
                        state.file_browser_state = fm
                        continue
                    cand = Path(raw).expanduser()
                    if not cand.is_absolute():
                        cand = (Path(_fm_norm_path(fm.path)) / cand).expanduser()
                    # Avoid resolve() here (can be slow on some FS); absolute path is enough.
                    cand = Path(_fm_norm_path(str(cand)))
                    if cand.is_dir():
                        fm.path = _fm_norm_path(str(cand))
                        fm.mode = 'browse'
                        fm.filter_text = ''
                        fm.goto_text = ''
                        fm.index = 0
                        fm.scroll = 0
                        _fm_relist(fm)
                        _ensure_visible()
                    elif cand.is_file():
                        fm.path = _fm_norm_path(str(cand.parent))
                        fm.mode = 'browse'
                        fm.filter_text = ''
                        fm.goto_text = ''
                        _fm_relist(fm, prefer_name=cand.name)
                        _ensure_visible()
                    else:
                        state.status = "Путь не найден"
                except Exception:
                    state.status = "Ошибка перехода по пути"
                state.file_browser_state = fm
                continue

            # Backspace: edit input mode or go up.
            if ch in (curses.KEY_BACKSPACE, 127, 8) or (isinstance(ch, str) and ch in ('\x7f', '\b')):
                if fm.mode == 'filter':
                    if fm.filter_text:
                        fm.filter_text = fm.filter_text[:-1]
                        _fm_relist(fm, rescan=False)
                        _ensure_visible()
                    else:
                        fm.mode = 'browse'
                    state.file_browser_state = fm
                    continue
                if fm.mode == 'goto':
                    if fm.goto_text:
                        fm.goto_text = fm.goto_text[:-1]
                    else:
                        fm.mode = 'browse'
                    state.file_browser_state = fm
                    continue
                _go_up()
                state.file_browser_state = fm
                continue

            # Start filter mode
            if isinstance(ch, str) and ch == '/':
                fm.mode = 'filter'
                fm.filter_text = ''
                _fm_relist(fm, rescan=False)
                _ensure_visible()
                state.file_browser_state = fm
                continue

            # Toggle hidden (H)
            if fm.mode == 'browse' and isinstance(ch, str) and ch.lower() == 'h':
                fm.show_hidden = not bool(fm.show_hidden)
                _fm_relist(fm)
                _ensure_visible()
                _persist_local()
                _sync_prefs_to_server()
                state.file_browser_state = fm
                continue

            # Toggle sort (S): name <-> modified (newest first)
            if fm.mode == 'browse' and isinstance(ch, str) and ch.lower() == 's':
                if str(fm.sort or 'name').strip().lower() == 'name':
                    fm.sort = 'modified'
                    fm.reverse = True
                else:
                    fm.sort = 'name'
                    fm.reverse = False
                _fm_relist(fm)
                _ensure_visible()
                _persist_local()
                _sync_prefs_to_server()
                state.file_browser_state = fm
                continue

            # Toggle reverse (R)
            if fm.mode == 'browse' and isinstance(ch, str) and ch.lower() == 'r':
                fm.reverse = not bool(fm.reverse)
                _fm_relist(fm)
                _ensure_visible()
                _persist_local()
                _sync_prefs_to_server()
                state.file_browser_state = fm
                continue

            # Toggle dirs-first (D)
            if fm.mode == 'browse' and isinstance(ch, str) and ch.lower() == 'd':
                fm.dirs_first = not bool(fm.dirs_first)
                _fm_relist(fm)
                _ensure_visible()
                _persist_local()
                _sync_prefs_to_server()
                state.file_browser_state = fm
                continue

            # Toggle view filter (V): all -> dirs -> files -> all
            if fm.mode == 'browse' and isinstance(ch, str) and ch.lower() == 'v':
                cur = fm.view
                fm.view = ('dirs' if cur is None else ('files' if cur == 'dirs' else None))
                _fm_relist(fm)
                _ensure_visible()
                _persist_local()
                _sync_prefs_to_server()
                state.file_browser_state = fm
                continue

            # Navigation keys
            if ch in (curses.KEY_UP,) or (isinstance(ch, str) and ch in ('k', 'K')):
                fm.index = max(0, int(fm.index) - 1)
                _ensure_visible()
                state.file_browser_state = fm
                continue
            if ch in (curses.KEY_DOWN,) or (isinstance(ch, str) and ch in ('j', 'J')):
                fm.index = min(max(0, len(fm.items) - 1), int(fm.index) + 1)
                _ensure_visible()
                state.file_browser_state = fm
                continue
            if ch in (curses.KEY_PPAGE,):
                _t, list_rows = _list_geom()
                fm.index = max(0, int(fm.index) - list_rows)
                _fm_ensure_visible(fm, list_rows)
                state.file_browser_state = fm
                continue
            if ch in (curses.KEY_NPAGE,):
                _t, list_rows = _list_geom()
                fm.index = min(max(0, len(fm.items) - 1), int(fm.index) + list_rows)
                _fm_ensure_visible(fm, list_rows)
                state.file_browser_state = fm
                continue
            if ch in (curses.KEY_HOME,):
                fm.index = 0
                _ensure_visible()
                state.file_browser_state = fm
                continue
            if ch in (curses.KEY_END,):
                fm.index = max(0, len(fm.items) - 1)
                _ensure_visible()
                state.file_browser_state = fm
                continue

            # Enter: open/select (also works while filtering)
            if ch in ('\n', '\r') or ch in (curses.KEY_ENTER, 10, 13):
                _activate_selected()
                state.file_browser_state = fm
                continue

            # Text input for filter/goto
            if isinstance(ch, str) and ch and ch.isprintable():
                if fm.mode == 'filter':
                    fm.filter_text += ch
                    _fm_relist(fm, rescan=False)
                    _ensure_visible()
                    state.file_browser_state = fm
                    continue
                if fm.mode == 'goto':
                    fm.goto_text += ch
                    state.file_browser_state = fm
                    continue

            # Block any other keys while modal is open
            continue

            import os as _os
            from pathlib import Path as _Path
            def _list_dir(p, for_side=None):
                """Fallback lister that honors saved prefs (show_hidden/view/dirs_first/reverse).

                - Only name-sorting is supported here (mtime sorts require extra stat and are handled by module path).
                - If for_side is None, uses the currently active pane in state.file_browser_side.
                """
                try:
                    base = str(_Path(p).expanduser().resolve())
                except Exception:
                    base = str(_Path('.').resolve())
                if for_side is None:
                    try:
                        for_side = int(getattr(state, 'file_browser_side', 0))
                    except Exception:
                        for_side = 0
                # Read flags
                show_hidden = bool(getattr(state, 'file_browser_show_hidden0', True)) if for_side == 0 else bool(getattr(state, 'file_browser_show_hidden1', True))
                view = getattr(state, 'file_browser_view0', None) if for_side == 0 else getattr(state, 'file_browser_view1', None)
                reverse = bool(getattr(state, 'file_browser_reverse0', False)) if for_side == 0 else bool(getattr(state, 'file_browser_reverse1', False))
                dirs_first = bool(getattr(state, 'file_browser_dirs_first0', False)) if for_side == 0 else bool(getattr(state, 'file_browser_dirs_first1', False))
                # List + filters
                try:
                    names = sorted(_os.listdir(base), key=lambda s: s.lower(), reverse=reverse)
                except Exception:
                    names = []
                items = []
                for name in names:
                    if (not show_hidden) and name.startswith('.') and name != '..':
                        continue
                    full = _os.path.join(base, name)
                    is_dir = bool(_os.path.isdir(full))
                    if view == 'dirs' and (not is_dir):
                        continue
                    if view == 'files' and is_dir:
                        continue
                    items.append((name, is_dir))
                if dirs_first and items:
                    items = [e for e in items if e[1]] + [e for e in items if not e[1]]
                # Parent row
                try:
                    parent = _os.path.dirname(base.rstrip(_os.sep)) or base
                    if parent and parent != base:
                        items = [('..', True)] + items
                except Exception:
                    pass
                return items
            def _list_dir_opts2(p, *, show_hidden=True, view=None, reverse=False, dirs_first=False):
                import os as __os
                from pathlib import Path as __Path
                try:
                    base = str(__Path(p).expanduser().resolve())
                except Exception:
                    base = str(__Path('.').resolve())
                names = []
                try:
                    names = sorted(__os.listdir(base), key=lambda s: s.lower(), reverse=bool(reverse))
                except Exception:
                    names = []
                items = []
                for name in names:
                    if (not show_hidden) and name.startswith('.') and name != '..':
                        continue
                    full = __os.path.join(base, name)
                    is_dir = bool(__os.path.isdir(full))
                    if view == 'dirs' and (not is_dir):
                        continue
                    if view == 'files' and is_dir:
                        continue
                    items.append((name, is_dir))
                if dirs_first and items:
                    items = [e for e in items if e[1]] + [e for e in items if not e[1]]
                try:
                    parent = __os.path.dirname(base.rstrip(__os.sep)) or base
                    if parent and parent != base:
                        items = [('..', True)] + items
                except Exception:
                    pass
                return items

            def _toggle_hidden(for_side: int) -> bool:
                """Toggle hidden-files visibility for pane side (0/1). Returns new flag."""
                try:
                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                        st = state.file_browser_state
                        cur = str(getattr(st, 'path0' if for_side == 0 else 'path1', str(_Path('.').resolve())))
                        val = not bool(getattr(st, 'show_hidden0' if for_side == 0 else 'show_hidden1', True))
                        if for_side == 0:
                            st.show_hidden0 = val
                            state.file_browser_show_hidden0 = val
                            st.items0 = fb_list(cur, show_hidden=val, sort=str(getattr(st, 'sort0', 'name')), dirs_first=bool(getattr(st, 'dirs_first0', False)), reverse=bool(getattr(st, 'reverse0', False)), view=getattr(st, 'view0', None))
                            st.index0 = 0 if not st.items0 else min(st.index0, len(st.items0) - 1)
                        else:
                            st.show_hidden1 = val
                            state.file_browser_show_hidden1 = val
                            st.items1 = fb_list(cur, show_hidden=val, sort=str(getattr(st, 'sort1', 'name')), dirs_first=bool(getattr(st, 'dirs_first1', False)), reverse=bool(getattr(st, 'reverse1', False)), view=getattr(st, 'view1', None))
                            st.index1 = 0 if not st.items1 else min(st.index1, len(st.items1) - 1)
                        state.file_browser_state = st
                        try:
                            _save_fb_prefs(st)
                            net.send({
                                "type": getattr(T, 'PREFS_SET', 'prefs_set'),
                                "values": {
                                    "fb_show_hidden0": bool(getattr(st, 'show_hidden0', True)),
                                    "fb_show_hidden1": bool(getattr(st, 'show_hidden1', True)),
                                    "fb_sort0": str(getattr(st, 'sort0', 'name')),
                                    "fb_sort1": str(getattr(st, 'sort1', 'name')),
                                    "fb_dirs_first0": bool(getattr(st, 'dirs_first0', False)),
                                    "fb_dirs_first1": bool(getattr(st, 'dirs_first1', False)),
                                    "fb_reverse0": bool(getattr(st, 'reverse0', False)),
                                    "fb_reverse1": bool(getattr(st, 'reverse1', False)),
                                    "fb_view0": getattr(st, 'view0', None),
                                    "fb_view1": getattr(st, 'view1', None),
                                }
                            })
                        except Exception:
                            pass
                        return val
                    else:
                        side = for_side
                        path = state.file_browser_path0 if side == 0 else state.file_browser_path1
                        flag = not bool(getattr(state, 'file_browser_show_hidden' + str(side), True))
                        if side == 0:
                            state.file_browser_show_hidden0 = flag
                        else:
                            state.file_browser_show_hidden1 = flag
                        items = _list_dir_opts2(
                            path,
                            show_hidden=bool(state.file_browser_show_hidden0 if side == 0 else state.file_browser_show_hidden1),
                            view=getattr(state, 'file_browser_view0' if side == 0 else 'file_browser_view1', None),
                            reverse=bool(state.file_browser_reverse0 if side == 0 else state.file_browser_reverse1),
                            dirs_first=bool(state.file_browser_dirs_first0 if side == 0 else state.file_browser_dirs_first1),
                        )
                        if side == 0:
                            state.file_browser_items0 = items; state.file_browser_index0 = 0
                        else:
                            state.file_browser_items1 = items; state.file_browser_index1 = 0
                        try:
                            _save_fb_prefs_values(
                                fb_show_hidden0=bool(getattr(state, 'file_browser_show_hidden0', True)),
                                fb_show_hidden1=bool(getattr(state, 'file_browser_show_hidden1', True)),
                                fb_sort0=str(getattr(state, 'file_browser_sort0', 'name')),
                                fb_sort1=str(getattr(state, 'file_browser_sort1', 'name')),
                                fb_dirs_first0=bool(getattr(state, 'file_browser_dirs_first0', False)),
                                fb_dirs_first1=bool(getattr(state, 'file_browser_dirs_first1', False)),
                                fb_reverse0=bool(getattr(state, 'file_browser_reverse0', False)),
                                fb_reverse1=bool(getattr(state, 'file_browser_reverse1', False)),
                                fb_view0=getattr(state, 'file_browser_view0', None),
                                fb_view1=getattr(state, 'file_browser_view1', None),
                                fb_path0=str(getattr(state, 'file_browser_path0', '') or ''),
                                fb_path1=str(getattr(state, 'file_browser_path1', '') or ''),
                                fb_side=int(getattr(state, 'file_browser_side', side)),
                            )
                            net.send({
                                "type": getattr(T, 'PREFS_SET', 'prefs_set'),
                                "values": {
                                    "fb_show_hidden0": bool(getattr(state, 'file_browser_show_hidden0', True)),
                                    "fb_show_hidden1": bool(getattr(state, 'file_browser_show_hidden1', True)),
                                    "fb_sort0": str(getattr(state, 'file_browser_sort0', 'name')),
                                    "fb_sort1": str(getattr(state, 'file_browser_sort1', 'name')),
                                    "fb_dirs_first0": bool(getattr(state, 'file_browser_dirs_first0', False)),
                                    "fb_dirs_first1": bool(getattr(state, 'file_browser_dirs_first1', False)),
                                    "fb_reverse0": bool(getattr(state, 'file_browser_reverse0', False)),
                                    "fb_reverse1": bool(getattr(state, 'file_browser_reverse1', False)),
                                    "fb_view0": getattr(state, 'file_browser_view0', None),
                                    "fb_view1": getattr(state, 'file_browser_view1', None),
                                }
                            })
                        except Exception:
                            pass
                        return flag
                except Exception:
                    return True
            # ESC — close
            if ch in ('\x1b',) or ch == 27:
                state.file_browser_view_mode = False
                state.file_browser_settings_mode = False
                close_file_browser(state)  # type: ignore[name-defined]
                state.status = ""
                continue
            # Settings modal (hidden files toggle)
            if getattr(state, 'file_browser_settings_mode', False):
                if ch in ('\x1b',) or ch == 27:
                    state.file_browser_settings_mode = False
                    continue
                if (
                    ch in ('\t', getattr(curses, 'KEY_BTAB', -9999), getattr(curses, 'KEY_STAB', -9999))
                    or ch in (curses.KEY_LEFT, curses.KEY_RIGHT, curses.KEY_UP, curses.KEY_DOWN)
                ):
                    try:
                        cur = int(getattr(state, 'file_browser_settings_side', int(getattr(state, 'file_browser_side', 0))))
                    except Exception:
                        cur = 0
                    state.file_browser_settings_side = 1 - cur
                    continue
                if ch in ('\n', '\r') or ch in (10, 13, curses.KEY_ENTER):
                    try:
                        side_sel = int(getattr(state, 'file_browser_settings_side', int(getattr(state, 'file_browser_side', 0))))
                    except Exception:
                        side_sel = 0
                    flag = _toggle_hidden(side_sel)
                    state.status = "Скрытые: " + ("Вкл" if flag else "Выкл")
                    state.file_browser_settings_mode = False
                    continue
                # Block other keys while settings open
                continue
            # Toggle hidden files for active pane (legacy shortcut, kept but not advertised)
            if isinstance(ch, str) and ch.lower() == 'h':
                try:
                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                        st = state.file_browser_state
                        side = int(getattr(st, 'side', 0))
                        if side == 0:
                            cur = str(getattr(st, 'path0', str(_Path('.').resolve())))
                            val = not bool(getattr(st, 'show_hidden0', True))
                            st.show_hidden0 = val
                            st.items0 = fb_list(cur, show_hidden=val, sort=str(getattr(st, 'sort0', 'name')), dirs_first=bool(getattr(st, 'dirs_first0', False)), reverse=bool(getattr(st, 'reverse0', False)))
                            st.index0 = 0 if not st.items0 else min(st.index0, len(st.items0)-1)
                        else:
                            cur = str(getattr(st, 'path1', str(_Path('.').resolve())))
                            val = not bool(getattr(st, 'show_hidden1', True))
                            st.show_hidden1 = val
                            st.items1 = fb_list(cur, show_hidden=val, sort=str(getattr(st, 'sort1', 'name')), dirs_first=bool(getattr(st, 'dirs_first1', False)), reverse=bool(getattr(st, 'reverse1', False)))
                            st.index1 = 0 if not st.items1 else min(st.index1, len(st.items1)-1)
                        state.status = "Скрытые: " + ("Вкл" if val else "Выкл")
                    else:
                        # fallback: rebuild items using persisted-like preferences
                        side = int(getattr(state, 'file_browser_side', 0))
                        path = state.file_browser_path0 if side == 0 else state.file_browser_path1
                        # Toggle state flag
                        flag = not bool(getattr(state, 'file_browser_show_hidden' + str(side), True))
                        if side == 0:
                            state.file_browser_show_hidden0 = flag
                        else:
                            state.file_browser_show_hidden1 = flag
                        # Apply list rebuild (name sort only)
                        def _list_dir_opts2(p, *, show_hidden=True, view=None, reverse=False, dirs_first=False):
                            import os as __os
                            from pathlib import Path as __Path
                            try:
                                base = str(__Path(p).expanduser().resolve())
                            except Exception:
                                base = str(__Path('.').resolve())
                            names = []
                            try:
                                names = sorted(__os.listdir(base), key=lambda s: s.lower(), reverse=bool(reverse))
                            except Exception:
                                names = []
                            items = []
                            for name in names:
                                if (not show_hidden) and name.startswith('.') and name != '..':
                                    continue
                                full = __os.path.join(base, name)
                                is_dir = bool(__os.path.isdir(full))
                                if view == 'dirs' and (not is_dir):
                                    continue
                                if view == 'files' and is_dir:
                                    continue
                                items.append((name, is_dir))
                            if dirs_first and items:
                                items = [e for e in items if e[1]] + [e for e in items if not e[1]]
                            try:
                                parent = __os.path.dirname(base.rstrip(__os.sep)) or base
                                if parent and parent != base:
                                    items = [('..', True)] + items
                            except Exception:
                                pass
                            return items
                        if side == 0:
                            items = _list_dir_opts2(path, show_hidden=bool(state.file_browser_show_hidden0), view=getattr(state, 'file_browser_view0', None), reverse=bool(state.file_browser_reverse0), dirs_first=bool(state.file_browser_dirs_first0))
                            state.file_browser_items0 = items; state.file_browser_index0 = 0
                        else:
                            items = _list_dir_opts2(path, show_hidden=bool(state.file_browser_show_hidden1), view=getattr(state, 'file_browser_view1', None), reverse=bool(state.file_browser_reverse1), dirs_first=bool(state.file_browser_dirs_first1))
                            state.file_browser_items1 = items; state.file_browser_index1 = 0
                        # Persist fallback flags
                        _save_fb_prefs_values(
                            fb_show_hidden0=bool(getattr(state, 'file_browser_show_hidden0', True)),
                            fb_show_hidden1=bool(getattr(state, 'file_browser_show_hidden1', True)),
                            fb_sort0=str(getattr(state, 'file_browser_sort0', 'name')),
                            fb_sort1=str(getattr(state, 'file_browser_sort1', 'name')),
                            fb_dirs_first0=bool(getattr(state, 'file_browser_dirs_first0', False)),
                            fb_dirs_first1=bool(getattr(state, 'file_browser_dirs_first1', False)),
                            fb_reverse0=bool(getattr(state, 'file_browser_reverse0', False)),
                            fb_reverse1=bool(getattr(state, 'file_browser_reverse1', False)),
                            fb_view0=getattr(state, 'file_browser_view0', None),
                            fb_view1=getattr(state, 'file_browser_view1', None),
                            fb_path0=str(getattr(state, 'file_browser_path0', '') or ''),
                            fb_path1=str(getattr(state, 'file_browser_path1', '') or ''),
                            fb_side=int(getattr(state, 'file_browser_side', side)),
                        )
                        state.status = "Скрытые: " + ("Вкл" if flag else "Выкл")
                    # Persist locally and sync to server
                    try:
                        if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                            st2 = state.file_browser_state
                            _save_fb_prefs(st2)
                            net.send({
                                "type": getattr(T,'PREFS_SET','prefs_set'),
                                "values": {
                                    "fb_show_hidden0": bool(getattr(st2,'show_hidden0',True)),
                                    "fb_show_hidden1": bool(getattr(st2,'show_hidden1',True)),
                                    "fb_sort0": str(getattr(st2,'sort0','name')),
                                    "fb_sort1": str(getattr(st2,'sort1','name')),
                                    "fb_dirs_first0": bool(getattr(st2,'dirs_first0',False)),
                                    "fb_dirs_first1": bool(getattr(st2,'dirs_first1',False)),
                                    "fb_reverse0": bool(getattr(st2,'reverse0',False)),
                                    "fb_reverse1": bool(getattr(st2,'reverse1',False)),
                                    "fb_view0": getattr(st2,'view0',None),
                                    "fb_view1": getattr(st2,'view1',None),
                                }
                            })
                        else:
                            _save_fb_prefs_values(
                                fb_show_hidden0=bool(getattr(state, 'file_browser_show_hidden0', True)),
                                fb_show_hidden1=bool(getattr(state, 'file_browser_show_hidden1', True)),
                                fb_sort0=str(getattr(state, 'file_browser_sort0', 'name')),
                                fb_sort1=str(getattr(state, 'file_browser_sort1', 'name')),
                                fb_dirs_first0=bool(getattr(state, 'file_browser_dirs_first0', False)),
                                fb_dirs_first1=bool(getattr(state, 'file_browser_dirs_first1', False)),
                                fb_reverse0=bool(getattr(state, 'file_browser_reverse0', False)),
                                fb_reverse1=bool(getattr(state, 'file_browser_reverse1', False)),
                                fb_view0=getattr(state, 'file_browser_view0', None),
                                fb_view1=getattr(state, 'file_browser_view1', None),
                                fb_path0=str(getattr(state, 'file_browser_path0', '') or ''),
                                fb_path1=str(getattr(state, 'file_browser_path1', '') or ''),
                                fb_side=int(getattr(state, 'file_browser_side', 0)),
                            )
                            net.send({
                                "type": getattr(T,'PREFS_SET','prefs_set'),
                                "values": {
                                    "fb_show_hidden0": bool(getattr(state,'file_browser_show_hidden0',True)),
                                    "fb_show_hidden1": bool(getattr(state,'file_browser_show_hidden1',True)),
                                    "fb_sort0": str(getattr(state,'file_browser_sort0','name')),
                                    "fb_sort1": str(getattr(state,'file_browser_sort1','name')),
                                    "fb_dirs_first0": bool(getattr(state,'file_browser_dirs_first0',False)),
                                    "fb_dirs_first1": bool(getattr(state,'file_browser_dirs_first1',False)),
                                    "fb_reverse0": bool(getattr(state,'file_browser_reverse0',False)),
                                    "fb_reverse1": bool(getattr(state,'file_browser_reverse1',False)),
                                    "fb_view0": getattr(state,'file_browser_view0',None),
                                    "fb_view1": getattr(state,'file_browser_view1',None),
                                }
                            })
                    except Exception:
                        pass
                except Exception:
                    pass
                continue
            # Settings modal (F1)
            if ch == getattr(curses, 'KEY_F1', 0):
                try:
                    state.file_browser_settings_mode = True
                    state.file_browser_settings_side = int(getattr(state, 'file_browser_side', 0))
                except Exception:
                    state.file_browser_settings_mode = True
                    state.file_browser_settings_side = 0
                continue
            # Top menu open/close via F9
            if ch == getattr(curses, 'KEY_F9', 0):
                state.file_browser_menu_mode = not bool(getattr(state, 'file_browser_menu_mode', False))
                if state.file_browser_menu_mode:
                    state.file_browser_menu_top = 0
                    state.file_browser_menu_index = 0
                continue
            # View menu interactions
            if getattr(state, 'file_browser_view_mode', False):
                if ch in ('\x1b',) or ch == 27:
                    state.file_browser_view_mode = False
                    continue
                if ch in (curses.KEY_UP,):
                    n = 5
                    state.file_browser_view_index = (state.file_browser_view_index - 1) % n
                    continue
                if ch in (curses.KEY_DOWN,):
                    n = 5
                    state.file_browser_view_index = (state.file_browser_view_index + 1) % n
                    continue
                if ch in ('\n','\r') or ch in (curses.KEY_ENTER, 13):
                    try:
                        if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                            st = state.file_browser_state
                            side = int(getattr(st, 'side', 0))
                            sel = int(getattr(state, 'file_browser_view_index', 0))
                            # 0: Имя А→Я, 1: Имя Я→А, 2: Дата новые, 3: Дата старые, 4: Папки впереди toggle
                            if sel == 0:
                                if side == 0:
                                    st.sort0, st.reverse0 = 'name', False
                                else:
                                    st.sort1, st.reverse1 = 'name', False
                            elif sel == 1:
                                if side == 0:
                                    st.sort0, st.reverse0 = 'name', True
                                else:
                                    st.sort1, st.reverse1 = 'name', True
                            elif sel == 2:
                                if side == 0:
                                    st.sort0, st.reverse0 = 'mtime', True
                                else:
                                    st.sort1, st.reverse1 = 'mtime', True
                            elif sel == 3:
                                if side == 0:
                                    st.sort0, st.reverse0 = 'mtime', False
                                else:
                                    st.sort1, st.reverse1 = 'mtime', False
                            elif sel == 4:
                                if side == 0:
                                    st.dirs_first0 = not bool(st.dirs_first0)
                                else:
                                    st.dirs_first1 = not bool(st.dirs_first1)
                            # Relist active pane with new prefs
                            if side == 0:
                                st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0)
                                st.index0 = min(st.index0, len(st.items0)-1) if st.items0 else 0
                            else:
                                st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1)
                                st.index1 = min(st.index1, len(st.items1)-1) if st.items1 else 0
                        state.file_browser_view_mode = False
                    except Exception:
                        state.file_browser_view_mode = False
                    continue
            # Top menu interactions
            if getattr(state, 'file_browser_menu_mode', False):
                # Left/right switch top menu category
                if ch == curses.KEY_LEFT:
                    state.file_browser_menu_top = (int(getattr(state, 'file_browser_menu_top', 0)) - 1) % 3
                    state.file_browser_menu_index = 0
                    continue
                if ch == curses.KEY_RIGHT:
                    state.file_browser_menu_top = (int(getattr(state, 'file_browser_menu_top', 0)) + 1) % 3
                    state.file_browser_menu_index = 0
                    continue
                if ch == curses.KEY_UP:
                    state.file_browser_menu_index = max(0, int(getattr(state, 'file_browser_menu_index', 0)) - 1)
                    continue
                if ch == curses.KEY_DOWN:
                    # recompute items count for bounds
                    top = int(getattr(state, 'file_browser_menu_top', 0))
                    cnt = 2 if top in (0,2) else 6
                    state.file_browser_menu_index = min(cnt - 1, int(getattr(state, 'file_browser_menu_index', 0)) + 1)
                    continue
                if ch in ('\x1b',) or ch == 27:
                    state.file_browser_menu_mode = False
                    continue
                if ch in ('\n','\r') or ch in (curses.KEY_ENTER, 13):
                    # Execute selection
                    try:
                        top = int(getattr(state, 'file_browser_menu_top', 0))
                        idx = int(getattr(state, 'file_browser_menu_index', 0))
                        # Build current items like in draw
                        if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                            st = state.file_browser_state
                            side = 0 if top == 0 else (1 if top == 2 else int(getattr(st, 'side', 0)))
                            sh = getattr(st, 'show_hidden0' if side == 0 else 'show_hidden1', True)
                        else:
                            # Fallback state (no module): decide target pane from the chosen top tab or current side
                            st = None
                            side = 0 if top == 0 else (1 if top == 2 else int(getattr(state, 'file_browser_side', 0)))
                            try:
                                sh = bool(getattr(state, 'file_browser_show_hidden0', True)) if side == 0 else bool(getattr(state, 'file_browser_show_hidden1', True))
                            except Exception:
                                sh = True
                        if top in (0,2):
                            items = [
                                f"Показать скрытые файлы: {'Выкл' if sh else 'Вкл'}",
                                "Обновить",
                            ]
                        else:
                            items = [
                                f"Сортировать: Дата создания",
                                f"Сортировать: Дата редактирования",
                                f"Сортировать: Дата добавления",
                                "Вид: по папкам",
                                "Вид: по файлам",
                                "Вид: все",
                            ]
                        lab = items[max(0, min(len(items)-1, idx))]
                        if top in (0,2):
                            if lab.startswith('Показать скрытые файлы'):
                                if st is not None:
                                    if side == 0:
                                        st.show_hidden0 = not bool(st.show_hidden0)
                                        st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0, view=st.view0)
                                        st.side = 0
                                    else:
                                        st.show_hidden1 = not bool(st.show_hidden1)
                                        st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1, view=st.view1)
                                        st.side = 1
                                else:
                                    # Fallback state — update the correct pane based on resolved side
                                    if side == 0:
                                        state.file_browser_show_hidden0 = not bool(getattr(state, 'file_browser_show_hidden0', True))
                                        base = state.file_browser_path0 or str(_Path('.').resolve())
                                        items = _list_dir(base, 0)
                                        if not state.file_browser_show_hidden0:
                                            items = [(n, d) for (n, d) in items if (n == '..' or not n.startswith('.'))]
                                        state.file_browser_items0 = items
                                    else:
                                        state.file_browser_show_hidden1 = not bool(getattr(state, 'file_browser_show_hidden1', True))
                                        base = state.file_browser_path1 or str(_Path('.').resolve())
                                        items = _list_dir(base, 1)
                                        if not state.file_browser_show_hidden1:
                                            items = [(n, d) for (n, d) in items if (n == '..' or not n.startswith('.'))]
                                        state.file_browser_items1 = items
                            elif lab == 'Обновить':
                                if st is not None:
                                    if side == 0:
                                        st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0, view=st.view0)
                                        st.side = 0
                                    else:
                                        st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1, view=st.view1)
                                        st.side = 1
                                else:
                                    # Nothing to do, items already rebuilt
                                    pass
                        else:
                            # View menu actions — применяем к обеим панелям
                            if st is not None:
                                if lab.endswith('Дата создания'):
                                    st.sort0, st.reverse0 = 'created', True
                                    st.sort1, st.reverse1 = 'created', True
                                elif lab.endswith('Дата редактирования'):
                                    st.sort0, st.reverse0 = 'modified', True
                                    st.sort1, st.reverse1 = 'modified', True
                                elif lab.endswith('Дата добавления'):
                                    st.sort0, st.reverse0 = 'added', True
                                    st.sort1, st.reverse1 = 'added', True
                                elif lab.endswith('по папкам'):
                                    st.view0 = 'dirs'; st.view1 = 'dirs'
                                elif lab.endswith('по файлам'):
                                    st.view0 = 'files'; st.view1 = 'files'
                                elif lab.endswith('все'):
                                    st.view0 = None; st.view1 = None
                                # Relist both panes
                                st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0, view=st.view0)
                                st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1, view=st.view1)
                                st.index0 = min(st.index0, len(st.items0)-1) if st.items0 else 0
                                st.index1 = min(st.index1, len(st.items1)-1) if st.items1 else 0
                            else:
                                # Fallback view actions: update fallback fields only (rendering uses file list as-is)
                                if lab.endswith('Дата создания'):
                                    state.file_browser_sort0, state.file_browser_reverse0 = 'created', True
                                    state.file_browser_sort1, state.file_browser_reverse1 = 'created', True
                                elif lab.endswith('Дата редактирования'):
                                    state.file_browser_sort0, state.file_browser_reverse0 = 'modified', True
                                    state.file_browser_sort1, state.file_browser_reverse1 = 'modified', True
                                elif lab.endswith('Дата добавления'):
                                    state.file_browser_sort0, state.file_browser_reverse0 = 'added', True
                                    state.file_browser_sort1, state.file_browser_reverse1 = 'added', True
                                elif lab.endswith('по папкам'):
                                    state.file_browser_view0 = 'dirs'; state.file_browser_view1 = 'dirs'
                                elif lab.endswith('по файлам'):
                                    state.file_browser_view0 = 'files'; state.file_browser_view1 = 'files'
                                elif lab.endswith('все'):
                                    state.file_browser_view0 = None; state.file_browser_view1 = None
                        # Persist after change
                        if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                            _save_fb_prefs(state.file_browser_state)
                            try:
                                st = state.file_browser_state
                                net.send({
                                    "type": getattr(T,'PREFS_SET','prefs_set'),
                                    "values": {
                                        "fb_show_hidden0": bool(getattr(st,'show_hidden0',True)),
                                        "fb_show_hidden1": bool(getattr(st,'show_hidden1',True)),
                                        "fb_sort0": str(getattr(st,'sort0','name')),
                                        "fb_sort1": str(getattr(st,'sort1','name')),
                                        "fb_dirs_first0": bool(getattr(st,'dirs_first0',False)),
                                        "fb_dirs_first1": bool(getattr(st,'dirs_first1',False)),
                                        "fb_reverse0": bool(getattr(st,'reverse0',False)),
                                        "fb_reverse1": bool(getattr(st,'reverse1',False)),
                                        "fb_view0": getattr(st,'view0',None),
                                        "fb_view1": getattr(st,'view1',None),
                                    }
                                })
                            except Exception:
                                pass
                        else:
                            _save_fb_prefs_values(
                                fb_show_hidden0=bool(getattr(state, 'file_browser_show_hidden0', True)),
                                fb_show_hidden1=bool(getattr(state, 'file_browser_show_hidden1', True)),
                                fb_sort0=str(getattr(state, 'file_browser_sort0', 'name')),
                                fb_sort1=str(getattr(state, 'file_browser_sort1', 'name')),
                                fb_dirs_first0=bool(getattr(state, 'file_browser_dirs_first0', False)),
                                fb_dirs_first1=bool(getattr(state, 'file_browser_dirs_first1', False)),
                                fb_reverse0=bool(getattr(state, 'file_browser_reverse0', False)),
                                fb_reverse1=bool(getattr(state, 'file_browser_reverse1', False)),
                                fb_view0=getattr(state, 'file_browser_view0', None),
                                fb_view1=getattr(state, 'file_browser_view1', None),
                                fb_path0=str(getattr(state, 'file_browser_path0', '') or ''),
                                fb_path1=str(getattr(state, 'file_browser_path1', '') or ''),
                                fb_side=int(getattr(state, 'file_browser_side', 0)),
                            )
                            try:
                                net.send({
                                    "type": getattr(T,'PREFS_SET','prefs_set'),
                                    "values": {
                                        "fb_show_hidden0": bool(getattr(state,'file_browser_show_hidden0',True)),
                                        "fb_show_hidden1": bool(getattr(state,'file_browser_show_hidden1',True)),
                                        "fb_sort0": str(getattr(state,'file_browser_sort0','name')),
                                        "fb_sort1": str(getattr(state,'file_browser_sort1','name')),
                                        "fb_dirs_first0": bool(getattr(state,'file_browser_dirs_first0',False)),
                                        "fb_dirs_first1": bool(getattr(state,'file_browser_dirs_first1',False)),
                                        "fb_reverse0": bool(getattr(state,'file_browser_reverse0',False)),
                                        "fb_reverse1": bool(getattr(state,'file_browser_reverse1',False)),
                                        "fb_view0": getattr(state,'file_browser_view0',None),
                                        "fb_view1": getattr(state,'file_browser_view1',None),
                                    }
                                })
                            except Exception:
                                pass
                        # Status hint
                        try:
                            state.status = f"Меню: применено — {lab}"
                        except Exception:
                            pass
                    except Exception:
                        pass
                    state.file_browser_menu_mode = False
                    continue
            # Navigate
            if ch in ('k',) or ch == curses.KEY_UP:
                try:
                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                        st, chosen = fb_handle(state.file_browser_state, 'UP')
                        state.file_browser_state = st
                    else:
                        side = int(getattr(state, 'file_browser_side', 0))
                        if side == 0:
                            n = len(state.file_browser_items0)
                            if n > 0:
                                state.file_browser_index0 = (state.file_browser_index0 - 1) % n
                        else:
                            n = len(state.file_browser_items1)
                            if n > 0:
                                state.file_browser_index1 = (state.file_browser_index1 - 1) % n
                except Exception:
                    pass
                continue
            if ch in ('j',) or ch == curses.KEY_DOWN:
                try:
                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                        st, chosen = fb_handle(state.file_browser_state, 'DOWN')
                        state.file_browser_state = st
                    else:
                        side = int(getattr(state, 'file_browser_side', 0))
                        if side == 0:
                            n = len(state.file_browser_items0)
                            if n > 0:
                                state.file_browser_index0 = (state.file_browser_index0 + 1) % n
                        else:
                            n = len(state.file_browser_items1)
                            if n > 0:
                                state.file_browser_index1 = (state.file_browser_index1 + 1) % n
                except Exception:
                    pass
                continue
            # Switch pane with Tab (also support KEY_BTAB)
            if (not getattr(state, 'file_browser_menu_mode', False)) and (not getattr(state, 'file_browser_view_mode', False)) and (
                ch in ('\t', getattr(curses, 'KEY_BTAB', -9999), getattr(curses, 'KEY_TAB', -9999), 9)
                or ch == getattr(curses, 'KEY_STAB', -9999)
            ):
                try:
                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                        st, _ = fb_handle(state.file_browser_state, 'TAB')
                        state.file_browser_state = st
                        state.file_browser_side = int(getattr(st, 'side', 0))
                    else:
                        state.file_browser_side = 1 - int(getattr(state, 'file_browser_side', 0))
                    try:
                        _persist_file_browser_state(state)  # type: ignore[name-defined]
                    except Exception:
                        pass
                except Exception:
                    state.file_browser_side = 0
                continue
            # Open dir / choose file
            if (not getattr(state, 'file_browser_menu_mode', False)) and (not getattr(state, 'file_browser_view_mode', False)) and ((isinstance(ch, str) and ch in ('\n', '\r')) or ch in (curses.KEY_ENTER, 13, curses.KEY_RIGHT)):
                try:
                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                        st, chosen = fb_handle(state.file_browser_state, 'ENTER')
                        state.file_browser_state = st
                        if chosen:
                            try:
                                sp = str(_Path(chosen).expanduser().resolve())
                            except Exception:
                                try:
                                    sp = str(chosen)
                                except Exception:
                                    sp = ''
                            if sp:
                                txt = f"/file {sp}"
                                try:
                                    state.input_buffer = txt
                                    state.input_caret = len(txt)
                                except Exception:
                                    state.input_buffer = txt
                                state.status = f"Выбран файл: {sp}"
                            # In любом случае закрываем модалку после выбора
                            close_file_browser(state)  # type: ignore[name-defined]
                        else:
                            try:
                                _persist_file_browser_state(state)  # type: ignore[name-defined]
                            except Exception:
                                pass
                        continue
                    else:
                        side = int(getattr(state, 'file_browser_side', 0))
                        if side == 0:
                            idx = max(0, min(len(state.file_browser_items0) - 1, int(getattr(state, 'file_browser_index0', 0))))
                            name, is_dir = state.file_browser_items0[idx]
                            cur = state.file_browser_path0
                        else:
                            idx = max(0, min(len(state.file_browser_items1) - 1, int(getattr(state, 'file_browser_index1', 0))))
                            name, is_dir = state.file_browser_items1[idx]
                            cur = state.file_browser_path1
                        if is_dir:
                            nxt = _os.path.join(cur, name) if name != '..' else (_os.path.dirname(cur.rstrip(_os.sep)) or cur)
                            if _os.path.isdir(nxt):
                                if side == 0:
                                    state.file_browser_path0 = nxt
                                    state.file_browser_items0 = _list_dir(nxt, 0)
                                    state.file_browser_index0 = 0
                                else:
                                    state.file_browser_path1 = nxt
                                    state.file_browser_items1 = _list_dir(nxt, 1)
                                    state.file_browser_index1 = 0
                                state.status = f"Файлы: {nxt}"
                                try:
                                    _persist_file_browser_state(state)  # type: ignore[name-defined]
                                except Exception:
                                    pass
                            continue
                        else:
                            try:
                                sp = str(_Path(_os.path.join(cur, name)).expanduser().resolve())
                            except Exception:
                                sp = _os.path.join(cur, name)
                            txt = f"/file {sp}"
                            try:
                                state.input_buffer = txt
                                state.input_caret = len(txt)
                            except Exception:
                                state.input_buffer = txt
                            close_file_browser(state)  # type: ignore[name-defined]
                            state.status = f"Выбран файл: {sp}"
                            continue
                except Exception:
                    # В случае любой ошибки — мягко закрываем модалку, не теряя ввод
                    try:
                        state.status = "Ошибка выбора файла"
                    except Exception:
                        pass
                    close_file_browser(state)  # type: ignore[name-defined]
            # Parent dir via Backspace/Left
            if (not getattr(state, 'file_browser_menu_mode', False)) and (not getattr(state, 'file_browser_view_mode', False)) and (ch in (curses.KEY_BACKSPACE, curses.KEY_LEFT)):
                try:
                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                        st, _ = fb_handle(state.file_browser_state, 'BACKSPACE')
                        state.file_browser_state = st
                        try:
                            _persist_file_browser_state(state)  # type: ignore[name-defined]
                        except Exception:
                            pass
                    else:
                        side = int(getattr(state, 'file_browser_side', 0))
                        cur = (state.file_browser_path0 if side == 0 else state.file_browser_path1) or str(_Path('.').resolve())
                        nxt = _os.path.dirname(cur.rstrip(_os.sep)) or cur
                        if nxt and _os.path.isdir(nxt) and nxt != cur:
                            if side == 0:
                                state.file_browser_path0 = nxt
                                state.file_browser_items0 = _list_dir(nxt, 0)
                                state.file_browser_index0 = 0
                            else:
                                state.file_browser_path1 = nxt
                                state.file_browser_items1 = _list_dir(nxt, 1)
                                state.file_browser_index1 = 0
                            state.status = f"Файлы: {nxt}"
                    try:
                        _persist_file_browser_state(state)  # type: ignore[name-defined]
                    except Exception:
                        pass
                except Exception:
                    pass
                continue
            # Mouse inside file browser modal (top menu, selection, double-click, wheel)
            # Global SGR mouse fallback (macOS Terminal/iTerm2) when curses doesn't map to KEY_MOUSE
            if isinstance(ch, str) and ch.startswith('\x1b[<'):
                try:
                    import re as _re
                    m = _re.match(r"\x1b\[<(?P<b>\d+);(?P<x>\d+);(?P<y>\d+)(?P<t>[Mm])", ch)
                    if not m:
                        m = _re.match(r"\x1b\[(?P<b>\d+);(?P<x>\d+);(?P<y>\d+)(?P<t>[Mm])", ch)
                    if m:
                        Cb = int(m.group('b'))
                        mx = int(m.group('x')) - 1
                        my = int(m.group('y')) - 1
                        is_press = (m.group('t') == 'M')
                        # Map to wheel/click masks similar to curses
                        wheel_up, wheel_down = _compute_wheel_masks()
                        bstate = 0
                        if (Cb & 0x40) and (Cb & 1) == 0:
                            bstate |= wheel_up
                        elif (Cb & 0x40) and (Cb & 1) == 1:
                            bstate |= wheel_down
                        else:
                            if is_press:
                                bstate |= getattr(curses, 'BUTTON1_PRESSED', 0) or getattr(curses, 'BUTTON1_CLICKED', 0)
                            else:
                                bstate |= getattr(curses, 'BUTTON1_RELEASED', 0) or getattr(curses, 'BUTTON1_CLICKED', 0)
                        # Apply the same logic as KEY_MOUSE for the main UI
                        try:
                            h, w = stdscr.getmaxyx()
                            left_w = max(20, min(30, w // 4))
                        except Exception:
                            left_w = 20
                        in_left = (mx < left_w)
                        if bstate & wheel_up:
                            if in_left:
                                rows_cnt = len(build_contact_rows(state))
                                vis = int(getattr(state, 'last_left_h', 10))
                                max_start = max(0, rows_cnt - max(0, vis))
                                cs = max(0, int(getattr(state, 'contacts_scroll', 0)) - 3)
                                state.contacts_scroll = max(0, min(cs, max_start))
                                need_redraw = True
                                try:
                                    if state.selected_index < cs:
                                        state.selected_index = cs
                                    elif state.selected_index >= cs + max(1, vis):
                                        state.selected_index = cs + max(1, vis) - 1
                                        clamp_selection(state, prefer='up')
                                except Exception:
                                    pass
                            else:
                                state.history_scroll += 3
                                _clamp_history_scroll(state)
                                need_redraw = True
                            continue
                        if bstate & wheel_down:
                            if in_left:
                                rows_cnt = len(build_contact_rows(state))
                                vis = int(getattr(state, 'last_left_h', 10))
                                max_start = max(0, rows_cnt - max(0, vis))
                                cs = min(max_start, int(getattr(state, 'contacts_scroll', 0)) + 3)
                                state.contacts_scroll = max(0, min(cs, max_start))
                                need_redraw = True
                                try:
                                    if state.selected_index < cs:
                                        state.selected_index = cs
                                    elif state.selected_index >= cs + max(1, vis):
                                        state.selected_index = cs + max(1, vis) - 1
                                        clamp_selection(state, prefer='down')
                                except Exception:
                                    pass
                            else:
                                state.history_scroll = max(0, state.history_scroll - 3)
                                _clamp_history_scroll(state)
                                need_redraw = True
                            continue
                        # Left pane click selection (similar to KEY_MOUSE contact selection)
                        if in_left:
                            # Map to contacts region (align with draw: start_y=2)
                            start_y = 2
                            try:
                                h, w = stdscr.getmaxyx()
                                vis_h = max(0, h - start_y - 2)
                            except Exception:
                                vis_h = 10
                            rows = build_contact_rows(state)
                            cs = max(0, int(getattr(state, 'contacts_scroll', 0)))
                            max_rows = min(max(0, len(rows) - cs), vis_h)
                            if start_y <= my < start_y + max_rows:
                                local_idx = my - start_y
                                idx = cs + local_idx
                                if 0 <= idx < len(rows):
                                    state.selected_index = idx
                                    clamp_selection(state, prefer='down')
                                    sel = current_selected_id(state)
                                    # Auto mark as read
                                    _maybe_send_message_read(state, net, sel)
                                    state.history_scroll = 0
                            continue
                        # history area selection/copy is intentionally left to terminal or existing logic
                except Exception:
                    pass
            if ch == curses.KEY_MOUSE:
                try:
                    _mid, mx, my, _mz, bstate = curses.getmouse()
                    if KEYLOG_ENABLED or getattr(state, 'debug_mode', False):
                        _dbg(f"[mouse] KEY_MOUSE mx={mx} my={my} bstate={bstate}")
                except Exception:
                    continue
                # Header hotkeys hit-testing (y == 0)
                if my == 0:
                    if handle_hotkey_click(mx, my):
                        continue
                    continue
                try:
                    if KEYLOG_ENABLED or getattr(state, 'debug_mode', False):
                        state.debug_last_mouse = f"KEY_MOUSE x={mx} y={my} bstate={bstate}"
                        try:
                            state.debug_lines.append(state.debug_last_mouse)
                            if len(state.debug_lines) > 300:
                                del state.debug_lines[:len(state.debug_lines) - 300]
                        except Exception:
                            pass
                except Exception:
                    pass
                # Geometry same as renderer
                try:
                    h, w = stdscr.getmaxyx()
                    # Top menu bar click (use exact label geometry if available)
                    if my == 0:
                        top = None
                        try:
                            pos = list(getattr(state, 'file_browser_menu_pos', []) or [])
                            for i, (xs, xe) in enumerate(pos[:3]):
                                if xs <= mx < xe:
                                    top = i
                                    break
                        except Exception:
                            top = None
                        if top is None:
                            # Fallback to rough thirds
                            seg = max(1, w // 3)
                            top = min(2, max(0, mx // seg))
                        state.file_browser_menu_mode = True
                        state.file_browser_menu_top = top
                        state.file_browser_menu_index = 0
                        continue
                    split_x = max(1, w // 2)
                    pane_w0 = max(8, split_x - 1)
                    pane_w1 = max(8, w - split_x - 1)
                    list_top = 3
                    # Determine pane by x coordinate
                    side = 0 if mx < split_x else 1
                    sep_y = max(0, h - 2)
                    list_rows = max(1, sep_y - list_top - 2)
                except Exception:
                    continue
                # Prepare data and compute window starts (mirror SGR branch)
                try:
                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                        st = state.file_browser_state
                        idx0 = int(getattr(st, 'index0', 0)); idx1 = int(getattr(st, 'index1', 0))
                        items0 = list(getattr(st, 'items0', []) or [])
                        items1 = list(getattr(st, 'items1', []) or [])
                        start0 = fb_window(idx0, len(items0), list_rows)
                        start1 = fb_window(idx1, len(items1), list_rows)
                    else:
                        idx0 = int(getattr(state, 'file_browser_index0', 0)); idx1 = int(getattr(state, 'file_browser_index1', 0))
                        items0 = list(getattr(state, 'file_browser_items0', []) or [])
                        items1 = list(getattr(state, 'file_browser_items1', []) or [])
                        start0 = max(0, min(idx0 - list_rows // 2, max(0, len(items0) - list_rows)))
                        start1 = max(0, min(idx1 - list_rows // 2, max(0, len(items1) - list_rows)))
                except Exception:
                    continue
                # Wheel support
                try:
                    wheel_up, wheel_down = _compute_wheel_masks()
                except Exception:
                    wheel_up = wheel_down = 0
                b4_mask = (
                    getattr(curses, 'BUTTON4_PRESSED', 0)
                    | getattr(curses, 'BUTTON4_CLICKED', 0)
                    | getattr(curses, 'BUTTON4_RELEASED', 0)
                    | getattr(curses, 'BUTTON4_DOUBLE_CLICKED', 0)
                    | getattr(curses, 'BUTTON4_TRIPLE_CLICKED', 0)
                )
                b5_mask = (
                    getattr(curses, 'BUTTON5_PRESSED', 0)
                    | getattr(curses, 'BUTTON5_CLICKED', 0)
                    | getattr(curses, 'BUTTON5_RELEASED', 0)
                    | getattr(curses, 'BUTTON5_DOUBLE_CLICKED', 0)
                    | getattr(curses, 'BUTTON5_TRIPLE_CLICKED', 0)
                )
                up_event = bool(bstate & wheel_up) or (wheel_up == 0 and bool(bstate & b4_mask))
                down_event = bool(bstate & wheel_down) or (wheel_down == 0 and bool(bstate & b5_mask))
                if up_event:
                    try:
                        delta = 3
                        if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                            st.index0 = max(0, st.index0 - delta) if side == 0 else st.index0
                            st.index1 = max(0, st.index1 - delta) if side == 1 else st.index1
                        else:
                            if side == 0 and items0:
                                state.file_browser_index0 = max(0, state.file_browser_index0 - delta)
                            if side == 1 and items1:
                                state.file_browser_index1 = max(0, state.file_browser_index1 - delta)
                    except Exception:
                        pass
                    continue
                if down_event:
                    try:
                        delta = 3
                        if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                            st.index0 = min(len(items0)-1, st.index0 + delta) if side == 0 and items0 else st.index0
                            st.index1 = min(len(items1)-1, st.index1 + delta) if side == 1 and items1 else st.index1
                        else:
                            if side == 0 and items0:
                                state.file_browser_index0 = min(len(items0)-1, state.file_browser_index0 + delta)
                            if side == 1 and items1:
                                state.file_browser_index1 = min(len(items1)-1, state.file_browser_index1 + delta)
                    except Exception:
                        pass
                    continue
                # Click inside input box to reposition caret
                try:
                    sep_y = max(0, h - 2)
                    hist_y = 1
                    hist_h = max(0, sep_y - hist_y - 2)
                    input_y = hist_y + hist_h
                    if side == 1 and (input_y + 1) <= my <= (sep_y - 1):
                        # Geometry matches renderer
                        hist_x = split_x + 1
                        hist_w = max(1, w - hist_x - 1)
                        width = max(1, hist_w - 2)
                        txt = getattr(state, 'input_buffer', '') or ''
                        # Map click to wrapped row/col
                        target_row = max(0, my - (input_y + 1))
                        target_col = max(0, mx - hist_x - 1)
                        # Precompute caret positions for each index (0..len)
                        positions = []
                        row = 0
                        col = 0
                        positions.append((row, col))  # caret at position 0
                        for ch in txt:
                            if ch == '\n':
                                row += 1
                                col = 0
                                positions.append((row, col))
                                continue
                            col += 1
                            if col >= width:
                                row += 1
                                col = 0
                            positions.append((row, col))
                        # Pick caret with minimal distance to click (row/col)
                        best_idx = 0
                        best_score = 1e9
                        for idx, (r, c) in enumerate(positions):
                            score = abs(r - target_row) * width + abs(c - target_col)
                            if score < best_score:
                                best_score = score
                                best_idx = idx
                        caret = best_idx
                        try:
                            state.input_caret = caret
                            state.input_sel_start = caret
                            state.input_sel_end = caret
                            state.input_cursor_visible = True
                            state.input_cursor_last_toggle = time.time()
                        except Exception:
                            pass
                        continue
                except Exception:
                    pass
                # Click inside dropdown menu to select/activate (mirror SGR)
                if getattr(state, 'file_browser_menu_mode', False):
                    try:
                        top = int(getattr(state, 'file_browser_menu_top', 0))
                        if top == 0:
                            box_w = max(26, min(40, pane_w0 - 2)); x0 = 0; items_cnt = 2
                        elif top == 2:
                            box_w = max(26, min(40, pane_w1 - 2)); x0 = split_x + 1; items_cnt = 2
                        else:
                            box_w = max(30, min(48, w - 4)); x0 = max(0, (w - box_w)//2); items_cnt = 6
                        y0 = 1
                        inside = (y0 <= my <= y0 + 1 + items_cnt) and (x0 <= mx < x0 + box_w)
                        if inside:
                            row = my - (y0 + 1)
                            if 0 <= row < items_cnt:
                                state.file_browser_menu_index = row
                                try:
                                    idx = row
                                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                        st = state.file_browser_state
                                        # Bind panel to actual mouse X position to avoid mis-detected top on macOS
                                        side2 = 0 if mx < split_x else 1
                                        sh = getattr(st, 'show_hidden0' if side2 == 0 else 'show_hidden1', True)
                                    else:
                                            st = None; side2 = 0; sh = bool(getattr(state, 'file_browser_show_hidden0', True))
                                    if top in (0,2):
                                        items_menu = [
                                            f"Показать скрытые файлы: {'Выкл' if sh else 'Вкл'}",
                                            "Обновить",
                                        ]
                                    else:
                                        items_menu = [
                                            f"Сортировать: Дата создания",
                                            f"Сортировать: Дата редактирования",
                                            f"Сортировать: Дата добавления",
                                            "Вид: по папкам",
                                            "Вид: по файлам",
                                            "Вид: все",
                                        ]
                                    lab = items_menu[max(0, min(len(items_menu)-1, idx))]
                                    if top in (0,2):
                                        if lab.startswith('Показать скрытые файлы'):
                                            if st is not None:
                                                if side2 == 0:
                                                    st.show_hidden0 = not bool(st.show_hidden0)
                                                    st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0, view=st.view0)
                                                    st.side = 0
                                                else:
                                                    st.show_hidden1 = not bool(st.show_hidden1)
                                                    st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1, view=st.view1)
                                                    st.side = 1
                                            else:
                                                if side2 == 0:
                                                    state.file_browser_show_hidden0 = not bool(getattr(state, 'file_browser_show_hidden0', True))
                                                    base = state.file_browser_path0 or str(_Path('.').resolve())
                                                    its = _list_dir(base, 0)
                                                    if not state.file_browser_show_hidden0:
                                                        its = [(n,d) for (n,d) in its if (n=='..' or not n.startswith('.'))]
                                                    state.file_browser_items0 = its
                                                else:
                                                    state.file_browser_show_hidden1 = not bool(getattr(state, 'file_browser_show_hidden1', True))
                                                    base = state.file_browser_path1 or str(_Path('.').resolve())
                                                    its = _list_dir(base, 1)
                                                    if not state.file_browser_show_hidden1:
                                                        its = [(n,d) for (n,d) in its if (n=='..' or not n.startswith('.'))]
                                                    state.file_browser_items1 = its
                                    elif lab == 'Обновить':
                                        if st is not None:
                                            if side2 == 0:
                                                st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0, view=st.view0)
                                                st.side = 0
                                            else:
                                                st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1, view=st.view1)
                                                st.side = 1
                                    else:
                                        if st is not None:
                                            if lab.endswith('Дата создания'):
                                                st.sort0, st.reverse0 = 'created', True
                                                st.sort1, st.reverse1 = 'created', True
                                            elif lab.endswith('Дата редактирования'):
                                                st.sort0, st.reverse0 = 'modified', True
                                                st.sort1, st.reverse1 = 'modified', True
                                            elif lab.endswith('Дата добавления'):
                                                st.sort0, st.reverse0 = 'added', True
                                                st.sort1, st.reverse1 = 'added', True
                                            elif lab.endswith('по папкам'):
                                                st.view0 = 'dirs'; st.view1 = 'dirs'
                                            elif lab.endswith('по файлам'):
                                                st.view0 = 'files'; st.view1 = 'files'
                                            elif lab.endswith('все'):
                                                st.view0 = None; st.view1 = None
                                        else:
                                            if lab.endswith('Дата создания'):
                                                state.file_browser_sort0=state.file_browser_sort1='created'; state.file_browser_reverse0=state.file_browser_reverse1=True
                                            elif lab.endswith('Дата редактирования'):
                                                state.file_browser_sort0=state.file_browser_sort1='modified'; state.file_browser_reverse0=state.file_browser_reverse1=True
                                            elif lab.endswith('Дата добавления'):
                                                state.file_browser_sort0=state.file_browser_sort1='added'; state.file_browser_reverse0=state.file_browser_reverse1=True
                                            elif lab.endswith('по папкам'):
                                                state.file_browser_view0=state.file_browser_view1='dirs'
                                            elif lab.endswith('по файлам'):
                                                state.file_browser_view0=state.file_browser_view1='files'
                                            elif lab.endswith('все'):
                                                state.file_browser_view0=state.file_browser_view1=None
                                except Exception:
                                    pass
                                state.file_browser_menu_mode = False
                                continue
                    except Exception:
                        pass
                # Click inside lists (select row; open on double-click or fast second click)
                if list_top <= my < list_top + list_rows:
                    try:
                        row = my - list_top
                        if side == 0 and items0:
                            new_idx = max(0, min(len(items0)-1, start0 + row))
                            if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                st.index0 = new_idx; st.side = 0
                            else:
                                state.file_browser_index0 = new_idx; state.file_browser_side = 0
                        elif side == 1 and items1:
                            new_idx = max(0, min(len(items1)-1, start1 + row))
                            if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                st.index1 = new_idx; st.side = 1
                            else:
                                state.file_browser_index1 = new_idx; state.file_browser_side = 1
                        # Determine open by double-click or fast second click on same row/side
                        open_now = bool(bstate & getattr(curses, 'BUTTON1_DOUBLE_CLICKED', 0))
                        try:
                            import time as _t
                            now = float(_t.time())
                            last_ts = float(getattr(state, 'fb_last_click_ts', 0.0))
                            last_side = int(getattr(state, 'fb_last_click_side', -1))
                            last_row = int(getattr(state, 'fb_last_click_row', -1))
                            cur_row = (start0 + row) if side == 0 else (start1 + row)
                            # Treat as double if same row/side within 0.4s
                            if (side == last_side) and (cur_row == last_row) and ((now - last_ts) <= 0.4):
                                open_now = True
                            # Update last click info
                            state.fb_last_click_ts = now
                            state.fb_last_click_side = side
                            state.fb_last_click_row = cur_row
                        except Exception:
                            pass
                        if open_now:
                            if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                st2, chosen = fb_handle(state.file_browser_state, 'ENTER')
                                state.file_browser_state = st2
                                if chosen:
                                    try:
                                        sp = str(_Path(chosen).expanduser().resolve())
                                    except Exception:
                                        sp = str(chosen)
                                    if sp:
                                        txt = f"/file {sp}"
                                        state.input_buffer = txt
                                        state.input_caret = len(txt)
                                        state.status = f"Выбран файл: {sp}"
                                    close_file_browser(state)  # type: ignore[name-defined]
                        else:
                            # Fallback: compute current selection and open/choose
                            if side == 0:
                                idx = max(0, min(len(items0)-1, new_idx)); name, is_dir = items0[idx]; cur = state.file_browser_path0
                            else:
                                idx = max(0, min(len(items1)-1, new_idx)); name, is_dir = items1[idx]; cur = state.file_browser_path1
                            if is_dir:
                                nxt = _os.path.join(cur, name) if name != '..' else (_os.path.dirname(cur.rstrip(_os.sep)) or cur)
                                if _os.path.isdir(nxt):
                                    if side == 0:
                                        state.file_browser_path0 = nxt; state.file_browser_items0 = _list_dir(nxt, 0); state.file_browser_index0 = 0
                                    else:
                                        state.file_browser_path1 = nxt; state.file_browser_items1 = _list_dir(nxt, 1); state.file_browser_index1 = 0
                                    state.status = f"Файлы: {nxt}"
                            else:
                                try:
                                    sp = str(_Path(_os.path.join(cur, name)).expanduser().resolve())
                                except Exception:
                                    sp = _os.path.join(cur, name)
                                txt = f"/file {sp}"
                                state.input_buffer = txt; state.input_caret = len(txt)
                                state.status = f"Выбран файл: {sp}"
                                close_file_browser(state)  # type: ignore[name-defined]
                        continue
                    except Exception:
                        pass
                # Click on header rows: set active pane even if not over list
                else:
                    try:
                        if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                            st = state.file_browser_state
                            st.side = 0 if mx < split_x else 1
                        else:
                            state.file_browser_side = 0 if mx < split_x else 1
                    except Exception:
                        pass
            # SGR mouse fallback for terminals that don't map to KEY_MOUSE on macOS
            if isinstance(ch, str) and ch.startswith('\x1b[<') and bool(getattr(state, 'mouse_enabled', False)):
                try:
                    # Format: ESC [ < Cb ; Cx ; Cy (M|m)
                    s = ch
                    # Extract numbers
                    import re as _re
                    m = _re.match(r"\x1b\[<(?P<b>\d+);(?P<x>\d+);(?P<y>\d+)(?P<t>[Mm])", s)
                    if not m:
                        # Some terminals send without '<'
                        m = _re.match(r"\x1b\[(?P<b>\d+);(?P<x>\d+);(?P<y>\d+)(?P<t>[Mm])", s)
                    if not m:
                        pass
                    else:
                        Cb = int(m.group('b'))
                        mx = int(m.group('x')) - 1
                        my = int(m.group('y')) - 1
                        is_press = (m.group('t') == 'M')
                        # Map to curses bstate
                        bstate = 0
                        try:
                            wheel_up, wheel_down = _compute_wheel_masks()
                        except Exception:
                            wheel_up = wheel_down = 0
                        if (Cb & 0x40) and (Cb & 1) == 0:
                            # Wheel up (typically 64)
                            bstate |= wheel_up
                        elif (Cb & 0x40) and (Cb & 1) == 1:
                            # Wheel down (65)
                            bstate |= wheel_down
                        else:
                            if is_press:
                                bstate |= getattr(curses, 'BUTTON1_PRESSED', 0) or getattr(curses, 'BUTTON1_CLICKED', 0)
                            else:
                                bstate |= getattr(curses, 'BUTTON1_RELEASED', 0) or getattr(curses, 'BUTTON1_CLICKED', 0)
                        # Reuse the same geometry/menu handling as KEY_MOUSE
                        try:
                            h, w = stdscr.getmaxyx()
                            # Top menu bar click
                            if my == 0:
                                # Use remembered label positions if available
                                top = None
                                try:
                                    pos = list(getattr(state, 'file_browser_menu_pos', []) or [])
                                    for i, (xs, xe) in enumerate(pos[:3]):
                                        if xs <= mx < xe:
                                            top = i
                                            break
                                except Exception:
                                    top = None
                                if top is None:
                                    seg = max(1, w // 3)
                                    top = min(2, max(0, mx // seg))
                                state.file_browser_menu_mode = True
                                state.file_browser_menu_top = top
                                state.file_browser_menu_index = 0
                                continue
                            split_x = max(1, w // 2)
                            pane_w0 = max(8, split_x - 1)
                            pane_w1 = max(8, w - split_x - 1)
                            list_top = 3
                            sep_y = max(0, h - 2)
                            list_rows = max(1, sep_y - list_top - 2)
                            # Determine pane
                            side = 0 if mx < split_x else 1
                            # Compute window starts
                            if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                st = state.file_browser_state
                                idx0 = int(getattr(st, 'index0', 0)); idx1 = int(getattr(st, 'index1', 0))
                                items0 = list(getattr(st, 'items0', []) or [])
                                items1 = list(getattr(st, 'items1', []) or [])
                                start0 = fb_window(idx0, len(items0), list_rows)
                                start1 = fb_window(idx1, len(items1), list_rows)
                            else:
                                idx0 = int(getattr(state, 'file_browser_index0', 0)); idx1 = int(getattr(state, 'file_browser_index1', 0))
                                items0 = list(getattr(state, 'file_browser_items0', []) or [])
                                items1 = list(getattr(state, 'file_browser_items1', []) or [])
                                start0 = max(0, min(idx0 - list_rows // 2, max(0, len(items0) - list_rows)))
                                start1 = max(0, min(idx1 - list_rows // 2, max(0, len(items1) - list_rows)))
                            # If dropdown menu open: handle as in KEY_MOUSE
                            if getattr(state, 'file_browser_menu_mode', False):
                                top = int(getattr(state, 'file_browser_menu_top', 0))
                                if top == 0:
                                    box_w = max(26, min(40, pane_w0 - 2)); x0 = 0; items_cnt = 2
                                elif top == 2:
                                    box_w = max(26, min(40, pane_w1 - 2)); x0 = split_x + 1; items_cnt = 2
                                else:
                                    box_w = max(30, min(48, w - 4)); x0 = max(0, (w - box_w)//2); items_cnt = 6
                                y0 = 1
                                inside = (y0 <= my <= y0 + 1 + items_cnt) and (x0 <= mx < x0 + box_w)
                                if inside:
                                    row = my - (y0 + 1)
                                    if 0 <= row < items_cnt:
                                        state.file_browser_menu_index = row
                                        # Применим пункт немедленно (как в KEY_MOUSE пути)
                                        try:
                                            idx = row
                                            # Сформируем список пунктов как в отрисовке и применим
                                            if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                                st = state.file_browser_state
                                                # Bind panel to actual mouse X position to avoid mis-detected top on macOS
                                                side2 = 0 if mx < split_x else 1
                                                sh = getattr(st, 'show_hidden0' if side2 == 0 else 'show_hidden1', True)
                                            else:
                                                st = None; side2 = 0; sh = True
                                            if top in (0,2):
                                                items_menu = [
                                                    f"Показать скрытые файлы: {'Выкл' if sh else 'Вкл'}",
                                                    "Обновить",
                                                ]
                                            else:
                                                items_menu = [
                                                    f"Сортировать: Дата создания",
                                                    f"Сортировать: Дата редактирования",
                                                    f"Сортировать: Дата добавления",
                                                    "Вид: по папкам",
                                                    "Вид: по файлам",
                                                    "Вид: все",
                                                ]
                                            lab = items_menu[max(0, min(len(items_menu)-1, idx))]
                                            if top in (0,2):
                                                if lab.startswith('Показать скрытые файлы'):
                                                    if st is not None:
                                                        if side2 == 0:
                                                            st.show_hidden0 = not bool(st.show_hidden0)
                                                            st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0, view=st.view0)
                                                            st.side = 0
                                                        else:
                                                            st.show_hidden1 = not bool(st.show_hidden1)
                                                            st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1, view=st.view1)
                                                            st.side = 1
                                                    else:
                                                        if side2 == 0:
                                                            state.file_browser_show_hidden0 = not bool(getattr(state, 'file_browser_show_hidden0', True))
                                                            base = state.file_browser_path0 or str(_Path('.').resolve())
                                                            its = _list_dir(base, 0)
                                                            if not state.file_browser_show_hidden0:
                                                                its = [(n,d) for (n,d) in its if (n=='..' or not n.startswith('.'))]
                                                            state.file_browser_items0 = its
                                                        else:
                                                            state.file_browser_show_hidden1 = not bool(getattr(state, 'file_browser_show_hidden1', True))
                                                            base = state.file_browser_path1 or str(_Path('.').resolve())
                                                            its = _list_dir(base, 1)
                                                            if not state.file_browser_show_hidden1:
                                                                its = [(n,d) for (n,d) in its if (n=='..' or not n.startswith('.'))]
                                                            state.file_browser_items1 = its
                                                elif lab == 'Обновить':
                                                    if st is not None:
                                                        if side2 == 0:
                                                            st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0, view=st.view0)
                                                            st.side = 0
                                                        else:
                                                            st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1, view=st.view1)
                                                            st.side = 1
                                            else:
                                                if st is not None:
                                                    if lab.endswith('Дата создания'):
                                                        st.sort0, st.reverse0 = 'created', True
                                                        st.sort1, st.reverse1 = 'created', True
                                                    elif lab.endswith('Дата редактирования'):
                                                        st.sort0, st.reverse0 = 'modified', True
                                                        st.sort1, st.reverse1 = 'modified', True
                                                    elif lab.endswith('Дата добавления'):
                                                        st.sort0, st.reverse0 = 'added', True
                                                        st.sort1, st.reverse1 = 'added', True
                                                    elif lab.endswith('по папкам'):
                                                        st.view0 = 'dirs'; st.view1 = 'dirs'
                                                    elif lab.endswith('по файлам'):
                                                        st.view0 = 'files'; st.view1 = 'files'
                                                    elif lab.endswith('все'):
                                                        st.view0 = None; st.view1 = None
                                                    st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0, view=st.view0)
                                                    st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1, view=st.view1)
                                                    st.index0 = min(st.index0, len(st.items0)-1) if st.items0 else 0
                                                    st.index1 = min(st.index1, len(st.items1)-1) if st.items1 else 0
                                                else:
                                                    # Fallback: обновим только флаги
                                                    if lab.endswith('Дата создания'):
                                                        state.file_browser_sort0=state.file_browser_sort1='created'; state.file_browser_reverse0=state.file_browser_reverse1=True
                                                    elif lab.endswith('Дата редактирования'):
                                                        state.file_browser_sort0=state.file_browser_sort1='modified'; state.file_browser_reverse0=state.file_browser_reverse1=True
                                                    elif lab.endswith('Дата добавления'):
                                                        state.file_browser_sort0=state.file_browser_sort1='added'; state.file_browser_reverse0=state.file_browser_reverse1=True
                                                    elif lab.endswith('по папкам'):
                                                        state.file_browser_view0=state.file_browser_view1='dirs'
                                                    elif lab.endswith('по файлам'):
                                                        state.file_browser_view0=state.file_browser_view1='files'
                                                    elif lab.endswith('все'):
                                                        state.file_browser_view0=state.file_browser_view1=None
                                            # Persist and sync prefs to server
                                            try:
                                                if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                                    st2 = state.file_browser_state
                                                    _save_fb_prefs(st2)
                                                    try:
                                                        net.send({
                                                            "type": getattr(T,'PREFS_SET','prefs_set'),
                                                            "values": {
                                                                "fb_show_hidden0": bool(getattr(st2,'show_hidden0',True)),
                                                                "fb_show_hidden1": bool(getattr(st2,'show_hidden1',True)),
                                                                "fb_sort0": str(getattr(st2,'sort0','name')),
                                                                "fb_sort1": str(getattr(st2,'sort1','name')),
                                                                "fb_dirs_first0": bool(getattr(st2,'dirs_first0',False)),
                                                                "fb_dirs_first1": bool(getattr(st2,'dirs_first1',False)),
                                                                "fb_reverse0": bool(getattr(st2,'reverse0',False)),
                                                                "fb_reverse1": bool(getattr(st2,'reverse1',False)),
                                                                "fb_view0": getattr(st2,'view0',None),
                                                                "fb_view1": getattr(st2,'view1',None),
                                                            }
                                                        })
                                                    except Exception:
                                                        pass
                                                else:
                                                    _save_fb_prefs_values(
                                                        fb_show_hidden0=bool(getattr(state, 'file_browser_show_hidden0', True)),
                                                        fb_show_hidden1=bool(getattr(state, 'file_browser_show_hidden1', True)),
                                                        fb_sort0=str(getattr(state, 'file_browser_sort0', 'name')),
                                                        fb_sort1=str(getattr(state, 'file_browser_sort1', 'name')),
                                                        fb_dirs_first0=bool(getattr(state, 'file_browser_dirs_first0', False)),
                                                        fb_dirs_first1=bool(getattr(state, 'file_browser_dirs_first1', False)),
                                                        fb_reverse0=bool(getattr(state, 'file_browser_reverse0', False)),
                                                        fb_reverse1=bool(getattr(state, 'file_browser_reverse1', False)),
                                                        fb_view0=getattr(state, 'file_browser_view0', None),
                                                        fb_view1=getattr(state, 'file_browser_view1', None),
                                                        fb_path0=str(getattr(state, 'file_browser_path0', '') or ''),
                                                        fb_path1=str(getattr(state, 'file_browser_path1', '') or ''),
                                                        fb_side=int(getattr(state, 'file_browser_side', 0)),
                                                    )
                                                    try:
                                                        net.send({
                                                            "type": getattr(T,'PREFS_SET','prefs_set'),
                                                            "values": {
                                                                "fb_show_hidden0": bool(getattr(state,'file_browser_show_hidden0',True)),
                                                                "fb_show_hidden1": bool(getattr(state,'file_browser_show_hidden1',True)),
                                                                "fb_sort0": str(getattr(state,'file_browser_sort0','name')),
                                                                "fb_sort1": str(getattr(state,'file_browser_sort1','name')),
                                                                "fb_dirs_first0": bool(getattr(state,'file_browser_dirs_first0',False)),
                                                                "fb_dirs_first1": bool(getattr(state,'file_browser_dirs_first1',False)),
                                                                "fb_reverse0": bool(getattr(state,'file_browser_reverse0',False)),
                                                                "fb_reverse1": bool(getattr(state,'file_browser_reverse1',False)),
                                                                "fb_view0": getattr(state,'file_browser_view0',None),
                                                                "fb_view1": getattr(state,'file_browser_view1',None),
                                                            }
                                                        })
                                                    except Exception:
                                                        pass
                                            except Exception:
                                                pass
                                            # close menu and set status
                                            state.file_browser_menu_mode = False
                                            try:
                                                state.status = f"Меню: применено — {lab}"
                                            except Exception:
                                                pass
                                        except Exception:
                                            state.file_browser_menu_mode = False
                                    continue
                            # Otherwise click inside lists (select; open on double-click or fast second click)
                            if list_top <= my < list_top + list_rows:
                                row = my - list_top
                                if side == 0 and items0:
                                    new_idx = max(0, min(len(items0)-1, start0 + row))
                                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                        st.index0 = new_idx; st.side = 0
                                    else:
                                        state.file_browser_index0 = new_idx; state.file_browser_side = 0
                                elif side == 1 and items1:
                                    new_idx = max(0, min(len(items1)-1, start1 + row))
                                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                        st.index1 = new_idx; st.side = 1
                                    else:
                                        state.file_browser_index1 = new_idx; state.file_browser_side = 1
                                # Determine open by double-click or fast second click
                                open_now = bool(bstate & getattr(curses, 'BUTTON1_DOUBLE_CLICKED', 0))
                                try:
                                    import time as _t
                                    now = float(_t.time())
                                    last_ts = float(getattr(state, 'fb_last_click_ts', 0.0))
                                    last_side = int(getattr(state, 'fb_last_click_side', -1))
                                    last_row = int(getattr(state, 'fb_last_click_row', -1))
                                    cur_row = (start0 + row) if side == 0 else (start1 + row)
                                    if (side == last_side) and (cur_row == last_row) and ((now - last_ts) <= 0.4):
                                        open_now = True
                                    state.fb_last_click_ts = now
                                    state.fb_last_click_side = side
                                    state.fb_last_click_row = cur_row
                                except Exception:
                                    pass
                                if open_now:
                                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                        st2, chosen = fb_handle(state.file_browser_state, 'ENTER')
                                        state.file_browser_state = st2
                                        if chosen:
                                            try:
                                                sp = str(_Path(chosen).expanduser().resolve())
                                            except Exception:
                                                sp = str(chosen)
                                            if sp:
                                                txt = f"/file {sp}"
                                                state.input_buffer = txt
                                                state.input_caret = len(txt)
                                                state.status = f"Выбран файл: {sp}"
                                            close_file_browser(state)  # type: ignore[name-defined]
                                        else:
                                            try:
                                                _persist_file_browser_state(state)  # type: ignore[name-defined]
                                            except Exception:
                                                pass
                                    else:
                                        # Fallback open/choose
                                        if side == 0:
                                            idx = max(0, min(len(items0)-1, new_idx)); name, is_dir = items0[idx]; cur = state.file_browser_path0
                                        else:
                                            idx = max(0, min(len(items1)-1, new_idx)); name, is_dir = items1[idx]; cur = state.file_browser_path1
                                        if is_dir:
                                            nxt = _os.path.join(cur, name) if name != '..' else (_os.path.dirname(cur.rstrip(_os.sep)) or cur)
                                            if _os.path.isdir(nxt):
                                                if side == 0:
                                                    state.file_browser_path0 = nxt; state.file_browser_items0 = _list_dir(nxt, 0); state.file_browser_index0 = 0
                                                else:
                                                    state.file_browser_path1 = nxt; state.file_browser_items1 = _list_dir(nxt, 1); state.file_browser_index1 = 0
                                                state.status = f"Файлы: {nxt}"
                                        else:
                                            try:
                                                sp = str(_Path(_os.path.join(cur, name)).expanduser().resolve())
                                            except Exception:
                                                sp = _os.path.join(cur, name)
                                            txt = f"/file {sp}"
                                            state.input_buffer = txt; state.input_caret = len(txt)
                                            state.status = f"Выбран файл: {sp}"
                                            close_file_browser(state)  # type: ignore[name-defined]
                                continue
                        except Exception:
                            pass
                except Exception:
                    pass
                # Determine pane by x
                side = 0 if mx < split_x else 1
                # Prepare data and compute window starts
                if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                    st = state.file_browser_state
                    idx0 = int(getattr(st, 'index0', 0)); idx1 = int(getattr(st, 'index1', 0))
                    items0 = list(getattr(st, 'items0', []) or [])
                    items1 = list(getattr(st, 'items1', []) or [])
                    start0 = fb_window(idx0, len(items0), list_rows)
                    start1 = fb_window(idx1, len(items1), list_rows)
                else:
                    idx0 = int(getattr(state, 'file_browser_index0', 0)); idx1 = int(getattr(state, 'file_browser_index1', 0))
                    items0 = list(getattr(state, 'file_browser_items0', []) or [])
                    items1 = list(getattr(state, 'file_browser_items1', []) or [])
                    start0 = max(0, min(idx0 - list_rows // 2, max(0, len(items0) - list_rows)))
                    start1 = max(0, min(idx1 - list_rows // 2, max(0, len(items1) - list_rows)))
                # Wheel support
                wheel_up, wheel_down = _compute_wheel_masks()
                if bstate & wheel_up:
                    delta = 3
                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                        st.index0 = max(0, st.index0 - delta) if side == 0 else st.index0
                        st.index1 = max(0, st.index1 - delta) if side == 1 else st.index1
                    else:
                        if side == 0 and items0:
                            state.file_browser_index0 = max(0, state.file_browser_index0 - delta)
                        if side == 1 and items1:
                            state.file_browser_index1 = max(0, state.file_browser_index1 - delta)
                    continue
                if bstate & wheel_down:
                    delta = 3
                    if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                        st.index0 = min(len(items0)-1, st.index0 + delta) if side == 0 and items0 else st.index0
                        st.index1 = min(len(items1)-1, st.index1 + delta) if side == 1 and items1 else st.index1
                    else:
                        if side == 0 and items0:
                            state.file_browser_index0 = min(len(items0)-1, state.file_browser_index0 + delta)
                        if side == 1 and items1:
                            state.file_browser_index1 = min(len(items1)-1, state.file_browser_index1 + delta)
                    continue
                # Click inside dropdown menu to select/activate
                if getattr(state, 'file_browser_menu_mode', False):
                    # Recompute dropdown rect as in draw
                    top = int(getattr(state, 'file_browser_menu_top', 0))
                    if top == 0:
                        box_w = max(26, min(40, pane_w0 - 2)); x0 = 0
                        items_cnt = 2
                    elif top == 2:
                        box_w = max(26, min(40, pane_w1 - 2)); x0 = split_x + 1
                        items_cnt = 2
                    else:
                        box_w = max(30, min(48, w - 4)); x0 = max(0, (w - box_w)//2)
                        items_cnt = 6
                    y0 = 1
                    inside = (y0 <= my <= y0 + 1 + items_cnt)
                    inside = inside and (x0 <= mx < x0 + box_w)
                    if inside:
                        row = my - (y0 + 1)
                        if 0 <= row < items_cnt:
                            state.file_browser_menu_index = row
                            # Apply immediately on single click
                            try:
                                idx = row
                                # Build items like draw and execute selection (same as Enter logic)
                                if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                    st = state.file_browser_state
                                    side = 0 if top == 0 else (1 if top == 2 else int(getattr(st, 'side', 0)))
                                    sh = getattr(st, 'show_hidden0' if side == 0 else 'show_hidden1', True)
                                else:
                                    st = None; side = 0; sh = True
                                if top in (0,2):
                                    items = [
                                        f"Показать скрытые файлы: {'Выкл' if sh else 'Вкл'}",
                                        "Обновить",
                                    ]
                                else:
                                    items = [
                                        f"Сортировать: Дата создания",
                                        f"Сортировать: Дата редактирования",
                                        f"Сортировать: Дата добавления",
                                        "Вид: по папкам",
                                        "Вид: по файлам",
                                        "Вид: все",
                                    ]
                                lab = items[max(0, min(len(items)-1, idx))]
                                if top in (0,2):
                                    if lab.startswith('Показать скрытые файлы'):
                                        if st is not None:
                                            if side == 0:
                                                st.show_hidden0 = not bool(st.show_hidden0)
                                                st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0, view=st.view0)
                                            else:
                                                st.show_hidden1 = not bool(st.show_hidden1)
                                                st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1, view=st.view1)
                                        else:
                                            # Fallback state
                                            if side == 0:
                                                state.file_browser_show_hidden0 = not bool(getattr(state, 'file_browser_show_hidden0', True))
                                                base = state.file_browser_path0 or str(_Path('.').resolve())
                                                state.file_browser_items0 = _list_dir(base, 0)
                                            else:
                                                state.file_browser_show_hidden1 = not bool(getattr(state, 'file_browser_show_hidden1', True))
                                                base = state.file_browser_path1 or str(_Path('.').resolve())
                                                state.file_browser_items1 = _list_dir(base, 1)
                                    elif lab == 'Обновить':
                                        if st is not None:
                                            if side == 0:
                                                st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0, view=st.view0)
                                            else:
                                                st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1, view=st.view1)
                                        else:
                                            pass
                                else:
                                    if st is not None:
                                        if lab.endswith('Дата создания'):
                                            st.sort0, st.reverse0 = 'created', True
                                            st.sort1, st.reverse1 = 'created', True
                                        elif lab.endswith('Дата редактирования'):
                                            st.sort0, st.reverse0 = 'modified', True
                                            st.sort1, st.reverse1 = 'modified', True
                                        elif lab.endswith('Дата добавления'):
                                            st.sort0, st.reverse0 = 'added', True
                                            st.sort1, st.reverse1 = 'added', True
                                        elif lab.endswith('по папкам'):
                                            st.view0 = 'dirs'; st.view1 = 'dirs'
                                        elif lab.endswith('по файлам'):
                                            st.view0 = 'files'; st.view1 = 'files'
                                        elif lab.endswith('все'):
                                            st.view0 = None; st.view1 = None
                                        # Relist both panes
                                        st.items0 = fb_list(st.path0, show_hidden=st.show_hidden0, sort=st.sort0, dirs_first=st.dirs_first0, reverse=st.reverse0, view=st.view0)
                                        st.items1 = fb_list(st.path1, show_hidden=st.show_hidden1, sort=st.sort1, dirs_first=st.dirs_first1, reverse=st.reverse1, view=st.view1)
                                        st.index0 = min(st.index0, len(st.items0)-1) if st.items0 else 0
                                        st.index1 = min(st.index1, len(st.items1)-1) if st.items1 else 0
                                # Persist prefs after any change
                                if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                                    _save_fb_prefs(state.file_browser_state)
                                else:
                                    _save_fb_prefs_values(
                                        fb_show_hidden0=bool(getattr(state, 'file_browser_show_hidden0', True)),
                                        fb_show_hidden1=bool(getattr(state, 'file_browser_show_hidden1', True)),
                                        fb_sort0=str(getattr(state, 'file_browser_sort0', 'name')),
                                        fb_sort1=str(getattr(state, 'file_browser_sort1', 'name')),
                                        fb_dirs_first0=bool(getattr(state, 'file_browser_dirs_first0', False)),
                                        fb_dirs_first1=bool(getattr(state, 'file_browser_dirs_first1', False)),
                                        fb_reverse0=bool(getattr(state, 'file_browser_reverse0', False)),
                                        fb_reverse1=bool(getattr(state, 'file_browser_reverse1', False)),
                                        fb_view0=getattr(state, 'file_browser_view0', None),
                                        fb_view1=getattr(state, 'file_browser_view1', None),
                                        fb_path0=str(getattr(state, 'file_browser_path0', '') or ''),
                                        fb_path1=str(getattr(state, 'file_browser_path1', '') or ''),
                                        fb_side=int(getattr(state, 'file_browser_side', 0)),
                                    )
                            except Exception:
                                pass
                            state.file_browser_menu_mode = False
                        continue
                # Click inside lists (only when no dropdown menu is open or click outside menu)
                if (not getattr(state, 'file_browser_menu_mode', False)) and (list_top <= my < list_top + list_rows):
                    row = my - list_top
                    if side == 0 and items0:
                        new_idx = max(0, min(len(items0)-1, start0 + row))
                        if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                            st.index0 = new_idx; st.side = 0
                        else:
                            state.file_browser_index0 = new_idx; state.file_browser_side = 0
                    elif side == 1 and items1:
                        new_idx = max(0, min(len(items1)-1, start1 + row))
                        if (not FILE_BROWSER_FALLBACK) and (state.file_browser_state is not None):
                            st.index1 = new_idx; st.side = 1
                        else:
                            state.file_browser_index1 = new_idx; state.file_browser_side = 1
                    # Double click = Enter
                    if bstate & getattr(curses, 'BUTTON1_DOUBLE_CLICKED', 0):
                        # Reuse Enter handler
                        ch2 = '\n'
                        # Fall through to Enter logic next frame
                    continue

        # Search action modal handling
        if state.search_action_mode:
            # Close on ESC
            if ch in ('\x1b',) or ch == 27:
                state.search_action_mode = False
                state.search_action_peer = None
                state.search_action_options = []
                state.search_action_step = 'choose'
                state.search_mode = False
                state.search_query = ''
                state.search_results = []
                state.search_live_id = None
                state.search_live_ok = False
                _touch_contact_rows(state)
                continue
            # Quick hotkey: A — запрос авторизации (не отправляем, если уже друг/авторизован)
            if isinstance(ch, str) and ch.lower() == 'a' and state.search_action_peer:
                rid = state.search_action_peer
                self_id = str(state.self_id or '')
                if rid == self_id or (rid in state.friends) or (rid in state.roster_friends):
                    state.status = f"Контакт уже добавлен и авторизован: {rid}"
                    continue
                if rid in state.blocked:
                    state.status = f"Аккаунт заблокирован вами: {rid}"
                    continue
                if rid in state.pending_requests:
                    state.status = f"Аккаунт ждёт вашей авторизации: {rid}"
                    state.search_action_options = ["Авторизовать", "Отклонить", "Просмотреть профиль", "Отмена"]
                    state.search_action_index = 0
                    state.search_action_step = 'choose'
                    continue
                if rid in state.pending_out:
                    state.status = f"Контакт уже ожидает авторизации (исходящий): {rid}"
                    continue
                try:
                    net.send({"type": "authz_request", "to": rid})
                    state.pending_out.add(rid)
                    state.status = f"Запрос авторизации отправлен: {rid}"
                    try:
                        state.authz_out_pending.add(rid)
                    except Exception:
                        pass
                    # Close search UI
                    state.search_action_mode = False
                    state.search_action_step = 'choose'
                    state.search_mode = False
                    state.search_query = ''
                    state.search_results = []
                    _touch_contact_rows(state)
                    # Move selection to the peer if present in contacts
                    try:
                        rows = build_contact_rows(state)
                        if rid in rows:
                            state.selected_index = rows.index(rid)
                            clamp_selection(state)
                    except Exception:
                        pass
                except Exception:
                    state.status = f"Не удалось отправить запрос: {rid}"
                    state.search_action_mode = False
                continue
            # Enter executes or closes depending on step
            if ch in ('\n', '\r') or ch in (10, 13):
                if state.search_action_step == 'choose':
                    try:
                        choice = state.search_action_options[state.search_action_index]
                    except Exception:
                        choice = 'Отмена'
                    if choice == 'Запросить авторизацию' and state.search_action_peer:
                        rid = state.search_action_peer
                        self_id = str(state.self_id or '')
                        if rid == self_id or (rid in state.friends) or (rid in state.roster_friends):
                            state.status = f"Контакт уже добавлен и авторизован: {rid}"
                            state.search_action_step = 'choose'
                        elif rid in state.blocked:
                            state.status = f"Аккаунт заблокирован вами: {rid}"
                            state.search_action_step = 'choose'
                        elif rid in state.pending_requests:
                            state.status = f"Аккаунт ждёт вашей авторизации: {rid}"
                            state.search_action_options = ["Авторизовать", "Отклонить", "Просмотреть профиль", "Отмена"]
                            state.search_action_index = 0
                            state.search_action_step = 'choose'
                        elif rid in state.pending_out:
                            state.status = f"Контакт уже ожидает авторизации (исходящий): {rid}"
                            state.search_action_step = 'choose'
                        else:
                            try:
                                net.send({"type": "authz_request", "to": rid})
                                state.pending_out.add(rid)
                                state.status = f"Запрос авторизации отправлен: {rid}"
                                try:
                                    state.authz_out_pending.add(rid)
                                except Exception:
                                    pass
                                # Close search UI and clear
                                state.search_action_mode = False
                                state.search_action_step = 'choose'
                                state.search_mode = False
                                state.search_query = ''
                                state.search_results = []
                                _touch_contact_rows(state)
                                try:
                                    rows = build_contact_rows(state)
                                    if rid in rows:
                                        state.selected_index = rows.index(rid)
                                        clamp_selection(state)
                                except Exception:
                                    pass
                            except Exception:
                                state.status = f"Не удалось отправить запрос: {rid}"
                                state.search_action_mode = False
                    elif choice == 'Авторизовать' and state.search_action_peer:
                        try:
                            net.send({"type": "authz_response", "peer": state.search_action_peer, "accept": True})
                            state.status = f"Добавлен в контакты: {state.search_action_peer}"
                            state.lock_selection_peer = state.search_action_peer
                            state.suppress_auto_menu = True
                            # Close modal; server will update roster
                            state.search_action_mode = False
                        except Exception:
                            pass
                    elif choice == 'Отклонить' and state.search_action_peer:
                        try:
                            net.send({"type": "authz_response", "peer": state.search_action_peer, "accept": False})
                            state.status = f"Отклонён запрос от {state.search_action_peer}"
                            state.search_action_mode = False
                        except Exception:
                            pass
                    elif choice == 'Отменить запрос' and state.search_action_peer:
                        try:
                            net.send({"type": "authz_cancel", "peer": state.search_action_peer})
                            state.pending_out.discard(state.search_action_peer)
                            state.status = f"Отменён запрос: {state.search_action_peer}"
                            state.search_action_mode = False
                        except Exception:
                            pass
                    elif choice == 'Просмотреть профиль' and state.search_action_peer:
                        try:
                            net.send({"type": "profile_get", "id": state.search_action_peer})
                        except Exception:
                            pass
                        state.profile_view_mode = True
                        state.profile_view_id = state.search_action_peer
                        state.search_action_mode = False
                    else:
                        state.search_action_mode = False
                else:
                    # For 'waiting' — do NOT close on Enter (only ESC cancels)
                    if state.search_action_step in ('accepted', 'declined'):
                        # Close on Enter after a decision and exit search mode entirely
                        state.search_action_mode = False
                        state.search_action_step = 'choose'
                        state.search_mode = False
                        state.search_query = ''
                        state.search_results = []
                        _touch_contact_rows(state)
                        # Move selection to the peer if present in contacts
                        try:
                            if state.search_action_peer:
                                rows = build_contact_rows(state)
                                if state.search_action_peer in rows:
                                    state.selected_index = rows.index(state.search_action_peer)
                                    clamp_selection(state)
                        except Exception:
                            pass
                        state.search_action_peer = None
                        state.status = ''
                    # If 'waiting', ignore Enter and keep the modal open
                continue
            # Navigation in choose step
            if state.search_action_step == 'choose':
                if ch in ('k',) or ch in (curses.KEY_UP, curses.KEY_LEFT):
                    if state.search_action_index > 0:
                        state.search_action_index -= 1
                    continue
                if ch in ('j',) or ch in (curses.KEY_DOWN, curses.KEY_RIGHT):
                    if state.search_action_index < max(0, len(state.search_action_options) - 1):
                        state.search_action_index += 1
                    continue

        # Live search (F3): handle input immediately and send search requests on-the-fly
        if state.search_mode:
            # Close on ESC
            if ch in ('\x1b',) or ch == 27:
                state.search_mode = False
                state.search_query = ''
                state.search_results = []
                state.search_live_id = None
                state.search_live_ok = False
                _touch_contact_rows(state)
                state.status = "Поиск закрыт"
                continue
            # Submit request when valid (Enter or 'A')
            if ch in ('\n', '\r') or (isinstance(ch, str) and ch == 'A'):
                rid = state.search_live_id if state.search_live_ok else None
                if rid:
                    # Determine if current live hit is a group or a board
                    is_group = False
                    is_board = False
                    try:
                        for item in (state.search_results or []):
                            if str(item.get('id') or '') == str(rid):
                                is_group = bool(item.get('group'))
                                is_board = bool(item.get('board'))
                                break
                    except Exception:
                        is_group = False
                        is_board = False
                    try:
                        if is_group:
                            # If already in groups (owner/member) — просто выбрать чат
                            if rid in state.groups:
                                state.search_mode = False
                                state.search_query = ''
                                state.search_results = []
                                _touch_contact_rows(state)
                                try:
                                    rows = build_contact_rows(state)
                                    if rid in rows:
                                        state.selected_index = rows.index(rid)
                                        clamp_selection(state)
                                except Exception:
                                    pass
                                state.status = f"Чат: {state.groups.get(rid, {}).get('name') or rid}"
                            else:
                                # Отправить заявку на вступление в чат
                                net.send({"type": "group_join_request", "group_id": rid})
                                state.status = f"Заявка на вступление в чат отправлена: {rid}"
                                state.search_mode = False
                                state.search_query = ''
                                state.search_results = []
                                _touch_contact_rows(state)
                        elif is_board:
                            # Boards: if already subscribed — focus; otherwise try to subscribe (join)
                            if rid in getattr(state, 'boards', {}):
                                state.search_mode = False
                                state.search_query = ''
                                state.search_results = []
                                _touch_contact_rows(state)
                                try:
                                    rows = build_contact_rows(state)
                                    if rid in rows:
                                        state.selected_index = rows.index(rid)
                                        clamp_selection(state)
                                except Exception:
                                    pass
                                meta = state.boards.get(rid) or {}
                                state.status = f"Доска: {meta.get('name') or rid}"
                            else:
                                # Subscribe (join) to the board (public)
                                try:
                                    net.send({"type": "board_join", "board_id": rid})
                                    state.status = "Подписка на доску..."
                                except Exception:
                                    state.status = "Ошибка: не удалось подписаться на доску"
                                state.search_mode = False
                                state.search_query = ''
                                state.search_results = []
                                _touch_contact_rows(state)
                        else:
                            # User flow: do NOT request auth if already friend/authorized or blocked
                            is_friend = bool(state.friends.get(rid) or (rid in state.roster_friends))
                            if is_friend:
                                state.status = f"Контакт уже добавлен: {rid}"
                                state.search_mode = False
                                state.search_query = ''
                                state.search_results = []
                                _touch_contact_rows(state)
                                try:
                                    rows = build_contact_rows(state)
                                    if rid in rows:
                                        state.selected_index = rows.index(rid)
                                        clamp_selection(state)
                                except Exception:
                                    pass
                            elif rid in state.blocked or rid in getattr(state, 'blocked_by', set()):
                                # Unhide if it was hidden and focus it for quick unblock
                                try:
                                    state.hidden_blocked.discard(rid)
                                except Exception:
                                    pass
                                state.status = f"Аккаунт заблокирован: {rid} (можно снять блок)"
                                state.search_mode = False
                                state.search_query = ''
                                state.search_results = []
                                _touch_contact_rows(state)
                                try:
                                    rows = build_contact_rows(state)
                                    if rid in rows:
                                        state.selected_index = rows.index(rid)
                                        clamp_selection(state)
                                except Exception:
                                    pass
                            else:
                                # Request authorization for new contact
                                net.send({"type": "authz_request", "to": rid})
                                state.pending_out.add(rid)
                                state.status = f"Запрос авторизации отправлен: {rid}"
                                try:
                                    state.authz_out_pending.add(rid)
                                except Exception:
                                    pass
                                state.search_mode = False
                                state.search_query = ''
                                state.search_results = []
                                _touch_contact_rows(state)
                                try:
                                    rows = build_contact_rows(state)
                                    if rid in rows:
                                        state.selected_index = rows.index(rid)
                                        clamp_selection(state)
                                except Exception:
                                    pass
                    except Exception:
                        state.status = f"Не удалось выполнить действие: {rid}"
                else:
                    state.status = "Не найден — исправьте ID/@логин"
                continue
            # Backspace
            if (isinstance(ch, str) and ch in ('\x7f', '\b')) or ch == curses.KEY_BACKSPACE:
                state.search_query = state.search_query[:-1]
                state.search_query = format_search_id(state.search_query)
                q = (state.search_query or '').strip()
                try:
                    if any(c.isalpha() for c in q) and not q.startswith('@'):
                        q = normalize_handle(q) or q
                except Exception:
                    pass
                try:
                    now = time.time()
                    last = float(getattr(state, 'last_search_sent', 0.0) or 0.0)
                    if now - last >= 0.2:
                        net.send({"type": "search", "query": q})
                        state.last_search_sent = now  # type: ignore[attr-defined]
                except Exception:
                    pass
                continue
            # Printable
            if isinstance(ch, str) and ch.isprintable():
                state.search_query += ch
                state.search_query = format_search_id(state.search_query)
                q = (state.search_query or '').strip()
                try:
                    if any(c.isalpha() for c in q) and not q.startswith('@'):
                        q = normalize_handle(q) or q
                except Exception:
                    pass
                try:
                    now = time.time()
                    last = float(getattr(state, 'last_search_sent', 0.0) or 0.0)
                    if now - last >= 0.2:
                        net.send({"type": "search", "query": q})
                        state.last_search_sent = now  # type: ignore[attr-defined]
                except Exception:
                    pass
                continue

        # Group create modal handling (consistent with other modals)
        if getattr(state, 'group_create_mode', False):
            # Initialize field on first entry
            if not hasattr(state, 'group_create_field'):
                state.group_create_field = 0  # type: ignore[attr-defined]
            if ch in ('\x1b',) or ch == 27:
                state.group_create_mode = False  # type: ignore[attr-defined]
                state.status = "Создание чата отменено"
                continue
            # Field navigation
            if ch in ('\t',) or ch == getattr(curses, 'KEY_BTAB', -9999) or ch in (curses.KEY_UP, curses.KEY_DOWN):
                try:
                    cur = int(getattr(state, 'group_create_field', 0))
                    if ch == curses.KEY_UP:
                        cur = 0
                    elif ch == curses.KEY_DOWN:
                        cur = 1
                    else:
                        cur = 1 - cur
                    state.group_create_field = cur  # type: ignore[attr-defined]
                except Exception:
                    pass
                continue
            # Submit
            if ch in ('\n', '\r') or ch in (10, 13):
                name = (state.group_name_input or '').strip()
                members = (state.group_members_input or '').strip()
                field = int(getattr(state, 'group_create_field', 0))
                if field == 0:
                    if not name:
                        state.status = "Введите название чата"
                    else:
                        state.group_create_field = 1
                    continue
                # field == 1: validate members then create
                if not name:
                    state.status = "Введите название чата"
                    continue
                if not members:
                    state.status = "Укажите участников (ID/@логины)"
                    continue
                # tokens: allow both numeric IDs and @handles, comma/space-separated
                import re as _re
                tokens = [t.strip() for t in _re.split(r"[\s,]+", members) if t.strip()]
                if not tokens:
                    state.modal_message = "Добавьте хотя бы одного участника"
                    state.status = "Введите участников"
                    continue
                # Fast-path: all tokens are ID-like (no handles) → do not block on verify, let server validate
                def _id_like(tok: str) -> bool:
                    try:
                        import re as _re2
                        return bool(_re2.match(r"^(?:\d{3}-\d{2}|\d{3}(?:-\d{3})+)$", tok))
                    except Exception:
                        return False
                if all((_id_like(t) and not t.startswith('@')) for t in tokens):
                    if not ensure_group_members_authorized(state, tokens, net):
                        continue
                    try:
                        net.send({"type": "group_create", "name": name, "members": ",".join(tokens)})
                        state.group_create_mode = False
                        state.group_verify_mode = False
                        state.status = f"Создаём чат: {name}"
                    except Exception:
                        state.status = "Ошибка: не удалось отправить запрос на создание чата"
                    continue
                # Kick off verification or reuse live map
                try:
                    state.group_verify_tokens = tokens
                    # Rebuild map only for current tokens
                    cur_map = {}
                    for t in tokens:
                        cur_map[t] = (getattr(state, 'group_verify_map', {}) or {}).get(t)
                    state.group_verify_map = cur_map
                    state.group_verify_pending = set()
                except Exception:
                    state.group_verify_map = {}
                    state.group_verify_pending = set()
                # Send searches for any unknown tokens (both handles and IDs; normalize handles)
                for t in tokens:
                    known = state.group_verify_map.get(t)
                    if known:
                        continue
                    try:
                        q = normalize_search_token(t)
                        net.send({"type": "search", "query": q})
                        state.group_verify_pending.add(t)
                        state.group_verify_map[t] = None
                    except Exception:
                        state.group_verify_map[t] = None
                if state.group_verify_pending:
                    state.group_verify_mode = True
                    state.status = "Проверяем участников…"
                    continue
                # All resolved — only verified IDs allowed
                ids = [state.group_verify_map.get(t) for t in tokens]
                ids = [str(i) for i in ids if i]
                if not ids:
                    state.modal_message = "Добавьте хотя бы одного участника"
                    state.status = "Введите участников"
                    continue
                try:
                    state.last_group_create_name = name
                    state.last_group_create_intended = set(ids)
                except Exception:
                    state.last_group_create_intended = set(ids)
                if not ensure_group_members_authorized(state, ids, net):
                    continue
                try:
                    net.send({"type": "group_create", "name": name, "members": ",".join(ids)})
                    state.group_create_mode = False
                    state.group_verify_mode = False
                    state.status = f"Создаём чат: {name}"
                except Exception:
                    state.status = "Ошибка: не удалось отправить запрос на создание чата"
                continue
            # Text input within modal
            if isinstance(ch, str):
                if ch in ('\x7f', '\b'):
                    if int(getattr(state, 'group_create_field', 0)) == 0:
                        state.group_name_input = getattr(state, 'group_name_input', '')[:-1]  # type: ignore[attr-defined]
                    else:
                        state.group_members_input = getattr(state, 'group_members_input', '')[:-1]  # type: ignore[attr-defined]
                elif ch.isprintable():
                    if int(getattr(state, 'group_create_field', 0)) == 0:
                        state.group_name_input = getattr(state, 'group_name_input', '') + ch  # type: ignore[attr-defined]
                    else:
                        state.group_members_input = getattr(state, 'group_members_input', '') + ch  # type: ignore[attr-defined]
                        state.group_members_input = format_member_tokens(getattr(state, 'group_members_input', ''))
                        # Live validate tokens on each char
                        try:
                            import re as _re
                            members = getattr(state, 'group_members_input', '')
                            tokens = [t for t in _re.split(r"[\s,]+", members) if t]
                            state.group_verify_tokens = tokens
                            # reset pending for current tokens
                            new_map: Dict[str, Optional[str]] = {}
                            for t in tokens:
                                new_map[t] = (getattr(state, 'group_verify_map', {}) or {}).get(t)
                            state.group_verify_map = new_map
                            state.group_verify_pending = set()
                            for t in tokens:
                                if not state.group_verify_map.get(t):
                                    try:
                                        q = normalize_search_token(t)
                                    except Exception:
                                        q = t
                                    try:
                                        net.send({"type": "search", "query": q})
                                        state.group_verify_pending.add(t)
                                        state.group_verify_map[t] = None
                                    except Exception:
                                        state.group_verify_map[t] = None
                        except Exception:
                            pass
                continue
            else:
                # Handle curses KEY_BACKSPACE when ch is int
                if ch == curses.KEY_BACKSPACE:
                    if int(getattr(state, 'group_create_field', 0)) == 0:
                        state.group_name_input = getattr(state, 'group_name_input', '')[:-1]  # type: ignore[attr-defined]
                    else:
                        state.group_members_input = getattr(state, 'group_members_input', '')[:-1]  # type: ignore[attr-defined]
                        state.group_members_input = format_member_tokens(getattr(state, 'group_members_input', ''))
                        # Live validate on backspace
                        try:
                            import re as _re
                            members = getattr(state, 'group_members_input', '')
                            tokens = [t for t in _re.split(r"[\s,]+", members) if t]
                            state.group_verify_tokens = tokens
                            new_map: Dict[str, Optional[str]] = {}
                            for t in tokens:
                                new_map[t] = (getattr(state, 'group_verify_map', {}) or {}).get(t)
                            state.group_verify_map = new_map
                            state.group_verify_pending = set()
                            for t in tokens:
                                if not state.group_verify_map.get(t):
                                    try:
                                        q = normalize_search_token(t)
                                    except Exception:
                                        q = t
                                    try:
                                        net.send({"type": "search", "query": q})
                                        state.group_verify_pending.add(t)
                                        state.group_verify_map[t] = None
                                    except Exception:
                                        state.group_verify_map[t] = None
                        except Exception:
                            pass
                    continue

        # Board create modal handling
        if getattr(state, 'board_create_mode', False):
            if ch in ('\x1b',) or ch == 27:
                state.board_create_mode = False
                state.status = "Создание доски отменено"
                continue
            # Switch fields
            if ch in ('\t',) or ch in (curses.KEY_UP, curses.KEY_DOWN):
                cur = int(getattr(state, 'board_create_field', 0))
                if ch == curses.KEY_UP:
                    cur = 0
                elif ch == curses.KEY_DOWN:
                    cur = 1
                else:
                    cur = 1 - cur
                state.board_create_field = cur
                continue
            # Submit
            if ch in ('\n', '\r') or ch in (10, 13):
                name = (state.board_name_input or '').strip()
                handle = (state.board_handle_input or '').strip()
                if not name:
                    state.status = "Введите название доски"
                    continue
                try:
                    from modules.profile import normalize_handle  # type: ignore
                except Exception:
                    def normalize_handle(v: str) -> str:
                        import re as _re2
                        v = (v or '').strip().lower()
                        if not v:
                            return ''
                        if not v.startswith('@'):
                            v = '@' + v
                        base = _re2.sub(r"[^a-z0-9_]", "", v[1:])
                        return '@' + base
                nh = normalize_handle(handle) if handle else None
                payload = {"type": "board_create", "name": name}
                if nh:
                    payload['handle'] = nh
                try:
                    net.send(payload)
                    state.board_create_mode = False
                    state.status = f"Создаём доску: {name}"
                except Exception:
                    state.status = "Ошибка: не удалось отправить запрос на создание доски"
                continue
            # Text input
            if isinstance(ch, str):
                if ch in ('\x7f', '\b'):
                    if int(getattr(state, 'board_create_field', 0)) == 0:
                        state.board_name_input = (state.board_name_input or '')[:-1]
                    else:
                        state.board_handle_input = (state.board_handle_input or '')[:-1]
                elif ch.isprintable():
                    if int(getattr(state, 'board_create_field', 0)) == 0:
                        state.board_name_input = (state.board_name_input or '') + ch
                    else:
                        state.board_handle_input = (state.board_handle_input or '') + ch
                continue
            else:
                if ch == curses.KEY_BACKSPACE:
                    if int(getattr(state, 'board_create_field', 0)) == 0:
                        state.board_name_input = (state.board_name_input or '')[:-1]
                    else:
                        state.board_handle_input = (state.board_handle_input or '')[:-1]
                continue

        # Global hotkeys (apply before mode-specific handling)
        if isinstance(ch, str) and ch == '\x15':  # Ctrl+U: manual update check
            latest = interactive_update_check(stdscr, confirm=False)
            if latest and latest == CLIENT_VERSION:
                state.status = f"Клиент актуален (v{CLIENT_VERSION})"
            continue
        if isinstance(ch, str) and ch.lower() == 'i':
            # Quick members view for selected group/board if no conflicting modal
            if state.authed and (not state.search_action_mode) and (not state.action_menu_mode) and (not state.profile_mode) and (not state.profile_view_mode) and (not state.modal_message) and (not state.help_mode) and (not getattr(state, 'group_create_mode', False)) and (not state.group_manage_mode) and (not getattr(state, 'board_create_mode', False)) and (not getattr(state, 'board_manage_mode', False)) and (not state.members_view_mode):
                peer = current_selected_id(state)
                if peer and ((peer in getattr(state, 'groups', {})) or (peer in getattr(state, 'boards', {}))):
                    open_members_view(state, net, peer)
                    continue
        # ===== Обработка модалки подтверждения отправки файла =====
        if getattr(state, 'file_confirm_mode', False):
            # Если исходный ввод содержал маркер '/file', отправляем без подтверждения
            try:
                txt_full = str(getattr(state, 'file_confirm_text_full', '') or '')
            except Exception:
                txt_full = ''
            if '/file' in txt_full.split():
                pth = state.file_confirm_path or ''
                to = state.file_confirm_target
                # Сброс модалки
                state.file_confirm_mode = False
                if to and (not is_separator(to)) and pth:
                    # Проверки для ЛС: дружба и отсутствие блокировок
                    try:
                        blocked_union = set(state.blocked) | set(getattr(state, 'blocked_by', set()))
                    except Exception:
                        blocked_union = set()
                    is_room = bool(to in state.groups or (isinstance(to, str) and to.startswith('b-')))
                    if (not is_room):
                        is_friend = bool(state.friends.get(to) or (to in state.roster_friends))
                        if (to in blocked_union) or (state.self_id in blocked_union):
                            state.status = "Нельзя отправить файл: контакт в блокировке"
                            # Очистить поля и выйти
                            state.file_confirm_path = None
                            state.file_confirm_target = None
                            state.file_confirm_index = 0
                            state.file_confirm_text_full = ""
                            state.file_confirm_prev_text = ""
                            state.file_confirm_prev_caret = 0
                            continue
                        if not is_friend:
                            state.status = "Требуется авторизация: добавьте друг друга в контакты"
                            state.file_confirm_path = None
                            state.file_confirm_target = None
                            state.file_confirm_index = 0
                            state.file_confirm_text_full = ""
                            state.file_confirm_prev_text = ""
                            state.file_confirm_prev_caret = 0
                            continue
                    meta = file_meta_for(pth)
                    if not meta:
                        state.modal_message = "Файл не найден или нет доступа"
                    else:
                        state.file_send_path = str(meta.path)
                        state.file_send_name = meta.name
                        state.file_send_size = meta.size
                        if to in state.groups or (isinstance(to, str) and to.startswith('b-')):
                            state.file_send_room = to
                            state.file_send_to = None
                            net.send({"type": getattr(T, 'FILE_OFFER', 'file_offer'), "room": to, "name": meta.name, "size": meta.size})
                        else:
                            state.file_send_room = None
                            state.file_send_to = to
                            net.send({"type": getattr(T, 'FILE_OFFER', 'file_offer'), "to": to, "name": meta.name, "size": meta.size})
                        try:
                            chan = to
                            nm = meta.name or 'файл'
                            state.conversations.setdefault(chan, []).append(ChatMessage('out', f"Отправка файла [{nm}]…", time.time()))
                        except Exception:
                            pass
                        state.status = f"Подготовка отправки: {meta.name} ({meta.size} байт)"
                # Очистим вспомогательные поля пути
                state.file_confirm_path = None
                state.file_confirm_target = None
                state.file_confirm_index = 0
                state.file_confirm_text_full = ""
                state.file_confirm_prev_text = ""
                state.file_confirm_prev_caret = 0
                continue
            # Стрелки меняют выбор (Да/Нет/Отмена)
            if ch in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT):
                try:
                    from modules.modal_std import nav_index  # type: ignore
                except Exception:
                    def nav_index(idx, count, ch):
                        import curses
                        if ch in (curses.KEY_RIGHT, curses.KEY_DOWN):
                            return (idx + 1) % count
                        if ch in (curses.KEY_LEFT, curses.KEY_UP):
                            return (idx - 1) % count
                        return idx
                try:
                    idx = int(getattr(state, 'file_confirm_index', 0))
                    idx = nav_index(idx, 3, ch)
                    state.file_confirm_index = idx
                except Exception:
                    state.file_confirm_index = 0
                continue
            # Enter — подтвердить/отменить отправку файла
            if (isinstance(ch, str) and ch in ('\n', '\r')) or ch in (curses.KEY_ENTER, 13):
                idx = int(getattr(state, 'file_confirm_index', 0))
                yes = (idx == 0)
                no_text = (idx == 1)
                cancel = (idx == 2)
                pth = state.file_confirm_path or ''
                to = state.file_confirm_target
                txt = state.file_confirm_text_full or ''
                # Сброс модалки
                state.file_confirm_mode = False
                if cancel:
                    state.input_buffer = state.file_confirm_prev_text
                    try:
                        state.input_caret = int(getattr(state, 'file_confirm_prev_caret', 0))
                    except Exception:
                        pass
                    state.status = ""
                elif yes and to and (not is_separator(to)) and pth:
                    meta = file_meta_for(pth)
                    if not meta:
                        state.modal_message = "Файл не найден или нет доступа"
                    else:
                        state.file_send_path = str(meta.path)
                        state.file_send_name = meta.name
                        state.file_send_size = meta.size
                        if to in state.groups or (isinstance(to, str) and to.startswith('b-')):
                            state.file_send_room = to
                            state.file_send_to = None
                            net.send({"type": getattr(T, 'FILE_OFFER', 'file_offer'), "room": to, "name": meta.name, "size": meta.size})
                        else:
                            state.file_send_room = None
                            state.file_send_to = to
                            net.send({"type": getattr(T, 'FILE_OFFER', 'file_offer'), "to": to, "name": meta.name, "size": meta.size})
                        try:
                            chan = to
                            nm = meta.name or 'файл'
                            state.conversations.setdefault(chan, []).append(ChatMessage('out', f"Отправка файла [{nm}]…", time.time()))
                        except Exception:
                            pass
                        state.status = f"Подготовка отправки: {meta.name} ({meta.size} байт)"
                elif no_text:
                    if txt and to and (not is_separator(to)):
                        if to in state.groups or (isinstance(to, str) and to.startswith('b-')):
                            try:
                                net.send({"type": "send", "room": to, "text": txt})
                            except Exception:
                                pass
                        else:
                            try:
                                net.send({"type": "send", "to": to, "text": txt})
                            except Exception:
                                pass
                # Очистим вспомогательные поля пути
                state.file_confirm_path = None
                state.file_confirm_target = None
                state.file_confirm_index = 0
                state.file_confirm_text_full = ""
                state.file_confirm_prev_text = ""
                state.file_confirm_prev_caret = 0
                continue
            # Закрытие по Esc — отмена (вернуть ввод)
            if ch in ('\x1b',) or ch == 27:
                state.input_buffer = state.file_confirm_prev_text
                try:
                    state.input_caret = int(getattr(state, 'file_confirm_prev_caret', 0))
                except Exception:
                    pass
                state.file_confirm_mode = False
                state.file_confirm_path = None
                state.file_confirm_target = None
                state.file_confirm_text_full = ""
                state.file_confirm_index = 0
                state.file_confirm_prev_text = ""
                state.file_confirm_prev_caret = 0
                state.status = ""
                continue
        # ===== Обработка модалки "файл уже существует" =====
        if getattr(state, 'file_exists_mode', False):
            # Esc — оставить как есть
            if ch in ('\x1b',) or ch == 27:
                # Ничего не делаем, оставим выбор по умолчанию "Оставить"
                state.file_exists_mode = False
                continue
            # Стрелки меняют выбор
            if ch in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT):
                try:
                    state.file_exists_index = 1 - int(getattr(state, 'file_exists_index', 0))
                except Exception:
                    state.file_exists_index = 0
                continue
            # Enter — применить выбор (заменить/оставить), реальное действие завершается на DOWNLOAD_COMPLETE
            if (isinstance(ch, str) and ch in ('\n', '\r')) or ch in (curses.KEY_ENTER, 13):
                # Просто закроем модалку; результат будет применён по file_exists_index в момент завершения
                state.file_exists_mode = False
                continue
            # (больше действий не требуется)
        # Quick toggle mute for selected contact (M)
        # Do not trigger inside any input/modal modes (e.g., add/remove members, actions menu, manage modals)
        if (
            isinstance(ch, str)
            and ch.lower() == 'm'
            and (state.input_buffer == '')
            and (not state.profile_mode)
            and (not getattr(state, 'board_member_add_mode', False))
            and (not getattr(state, 'board_member_remove_mode', False))
            and (not getattr(state, 'group_member_add_mode', False))
            and (not getattr(state, 'group_member_remove_mode', False))
            and (not getattr(state, 'group_manage_mode', False))
            and (not getattr(state, 'board_manage_mode', False))
            and (not getattr(state, 'board_create_mode', False))
            and (not getattr(state, 'group_create_mode', False))
            and (not state.search_action_mode)
            and (not state.action_menu_mode)
        ):
            sel = current_selected_id(state)
            if sel and (sel not in state.groups):
                try:
                    val = (sel not in state.muted)
                    net.send({"type": "mute_set", "peer": sel, "value": val})
                    state.status = ("Заглушаем: " if val else "Снимаем заглушку: ") + sel
                except Exception:
                    pass
            continue

        # Attempt to detect xterm 'CSI u' sequences and Meta-keys
        # (e.g., Shift+Enter => ESC [ 13 ; 2 u, Alt+b => ESC b, Alt+f => ESC f,
        #  Ctrl+Left/Right => ESC [ 1 ; 5 D/C)
        if isinstance(ch, str) and ch == '\x1b':
            seq = ''
            for _ in range(8):
                try:
                    nxt = stdscr.get_wch()
                except curses.error:
                    break
                if isinstance(nxt, str):
                    seq += nxt
                else:
                    break
            # Handle CSI-u and CSI~ sequences
            if seq.startswith('['):
                # minimal parse
                try:
                    body = seq[1:]
                    # Plain arrows (ESC [ D/C/A/B) — map to caret moves when input is active
                    if body in ('D','C','A','B'):
                        # Respect overlays: only edit when no blocking modal
                        _over = bool(
                            state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or state.help_mode or getattr(state, 'group_create_mode', False) or state.group_manage_mode
                            or getattr(state, 'file_confirm_mode', False) or getattr(state, 'file_progress_mode', False)
                            or getattr(state, 'board_invite_mode', False) or getattr(state, 'board_added_consent_mode', False)
                            or getattr(state, 'board_member_add_mode', False) or getattr(state, 'board_member_remove_mode', False)
                            or getattr(state, 'group_member_add_mode', False) or getattr(state, 'group_member_remove_mode', False)
                        )
                        if (not _over) and state.input_buffer is not None:
                            try:
                                if body == 'D':
                                    chat_input.move_left(state)
                                elif body == 'C':
                                    chat_input.move_right(state)
                                elif body == 'A':
                                    chat_input.move_up(state)
                                elif body == 'B':
                                    chat_input.move_down(state)
                            except Exception:
                                chat_input.clear_selection(state)
                            continue
                    if body.startswith('13;') and body.endswith('u'):
                        mod = body[3:-1]
                        if mod in ('2', '5'):
                            # Treat as newline insertion at caret
                            try:
                                chat_input.insert_newline(state)
                            except Exception:
                                pass
                            continue
                    # Plain Backspace via CSI-u without modifiers: ESC [ 8 u or ESC [ 127 u
                    if body.endswith('u'):
                        try:
                            keycode_s = body.split(';', 1)[0]
                            keycode = int(''.join([c for c in keycode_s if c.isdigit()]) or '-1')
                        except Exception:
                            keycode = -1
                        # If word-ops disabled (mac default) OR no Ctrl modifier, treat 8/127 as single backspace
                        if keycode in (8, 127) and (not _word_ops_enabled() or (';5' not in body)):
                            try:
                                chat_input.backspace(state)
                            except Exception:
                                pass
                            continue
                    # Ctrl+Left / Ctrl+Right (many terminals): ESC [ 1 ; 5 D / C
                    if _word_ops_enabled() and (body.endswith('D') and (';5' in body or body == '5D')):
                        try:
                            chat_input.move_word_left(state)
                        except Exception:
                            pass
                        continue
                    if _word_ops_enabled() and (body.endswith('C') and (';5' in body or body == '5C')):
                        try:
                            chat_input.move_word_right(state)
                        except Exception:
                            pass
                        continue
                    # CSI-u variants for Ctrl+Backspace/Delete: ESC [ 8 ; 5 u or ESC [ 127 ; 5 u
                    if _word_ops_enabled() and (body.endswith('u') and (';5' in body)):
                        try:
                            keycode = int(body.split(';', 1)[0])
                        except Exception:
                            keycode = -1
                        if keycode in (8, 127):
                            try:
                                chat_input.delete_word_left(state)
                            except Exception:
                                pass
                            continue
                    # CSI Delete (no modifiers): ESC [ 3 ~ → delete one char
                    if body == '3~' or (not _word_ops_enabled() and (body.startswith('3;') and body.endswith('~'))):
                        try:
                            chat_input.delete_forward(state)
                        except Exception:
                            pass
                        continue
                    # CSI "Delete" with modifiers: ESC [ 3 ; 5 ~  (Ctrl+Delete → delete word right)
                    if _word_ops_enabled() and (body.endswith('~') and body.startswith('3;') and (';5' in body)):
                        try:
                            chat_input.delete_word_right(state)
                        except Exception:
                            pass
                        continue
                except Exception:
                    pass
            else:
                # Meta-b / Meta-f (Alt+b / Alt+f)
                try:
                    if _word_ops_enabled() and seq and seq[0].lower() == 'b':
                        chat_input.move_word_left(state)
                        continue
                    if _word_ops_enabled() and seq and seq[0].lower() == 'f':
                        chat_input.move_word_right(state)
                        continue
                    # Meta-d (Alt+d) — delete word right
                    if _word_ops_enabled() and seq and seq[0].lower() == 'd':
                        chat_input.delete_word_right(state)
                        continue
                except Exception:
                    pass
            # If no CSI-u sequence followed, treat as plain ESC and allow overlay handling below

        # Profile modal input handling
        if state.profile_mode and state.authed:
            # Close modal
            if ch == '\x1b' or ch == curses.KEY_EXIT or ch == curses.KEY_CANCEL:
                state.profile_mode = False
                state.status = "Профиль закрыт"
                continue
            # Switch fields
            if ch in ('\t',) or ch == curses.KEY_BTAB or ch in (curses.KEY_UP, curses.KEY_DOWN):
                if ch == curses.KEY_UP:
                    state.profile_field = 0
                elif ch == curses.KEY_DOWN:
                    state.profile_field = 1
                else:
                    state.profile_field = 1 - state.profile_field
                continue
            # Submit (Enter)
            if ch in ('\n', '\r') or ch in (10, 13):
                try:
                    payload = make_profile_set_payload(
                        state.profile_name_input if state.profile_name_input != '' else None,
                        state.profile_handle_input if state.profile_handle_input != '' else None,
                    )
                    net.send(payload)
                    state.status = "Сохраняем профиль..."
                except Exception as e:
                    state.status = f"Ошибка профиля: {e}"
                continue
            # Edit fields
            if isinstance(ch, str):
                if ch in ('\x7f', '\b'):
                    if state.profile_field == 0:
                        state.profile_name_input = state.profile_name_input[:-1]
                    else:
                        state.profile_handle_input = state.profile_handle_input[:-1]
                elif ch == '\x03':
                    state.profile_mode = False
                elif ch.isprintable():
                    if state.profile_field == 0:
                        state.profile_name_input += ch
                    else:
                        state.profile_handle_input += ch
                continue
            else:
                if ch in (curses.KEY_BACKSPACE,):
                    if state.profile_field == 0:
                        state.profile_name_input = state.profile_name_input[:-1]
                    else:
                        state.profile_handle_input = state.profile_handle_input[:-1]
                elif ch == curses.KEY_F2:
                    state.profile_mode = False
                continue

        # Profile view card handling (read-only)
        if state.profile_view_mode:
            # Close on Esc or arrows
            if ch in ('\x1b',) or ch in (curses.KEY_LEFT, curses.KEY_RIGHT):
                state.profile_view_mode = False
                state.profile_view_id = None
                state.status = ""
                continue
            # Send authz request via 'A'
            if isinstance(ch, str) and ch.lower() == 'a':
                if state.profile_view_id:
                    net.send({"type": "authz_request", "to": state.profile_view_id})
                    state.status = f"Запрос авторизации отправлен: {state.profile_view_id}"
                    try:
                        state.authz_out_pending.add(str(state.profile_view_id))
                    except Exception:
                        pass
                continue

        # Group manage input handling
        if state.group_manage_mode:
            if ch in ('\x1b',) or ch == 27:
                state.group_manage_mode = False
                state.group_manage_gid = None
                state.status = "Чат: окно закрыто"
                continue
            if ch in ('\t',) or ch == curses.KEY_BTAB or ch in (curses.KEY_UP, curses.KEY_DOWN):
                try:
                    cur = int(getattr(state, 'group_manage_field', 0))
                    if ch == curses.KEY_UP:
                        cur = 0
                    elif ch == curses.KEY_DOWN:
                        cur = 1
                    else:
                        cur = 1 - cur
                    state.group_manage_field = cur
                except Exception:
                    pass
                continue
            if ch in ('\n', '\r') or ch in (10, 13):
                # Save name if field 0
                curf = int(getattr(state, 'group_manage_field', 0))
                if state.group_manage_gid and curf == 0:
                    new_name = (state.group_manage_name_input or '').strip()
                    if not new_name:
                        state.status = "Введите название чата"
                    else:
                        try:
                            net.send({"type": "group_rename", "group_id": state.group_manage_gid, "name": new_name})
                            state.status = "Сохраняем название чата..."
                        except Exception:
                            pass
                elif state.group_manage_gid and curf == 1:
                    # Save group handle
                    new_h = (state.group_manage_handle_input or '').strip()
                    if not new_h:
                        state.status = "Введите логин чата"
                    else:
                        try:
                            # Normalize with modules..normalize_handle if available
                            try:
                                from modules.profile import normalize_handle  # type: ignore
                            except Exception:
                                def normalize_handle(h: str) -> str:
                                    import re as _re
                                    h = (h or '').strip().lower()
                                    if not h.startswith('@'):
                                        h = '@' + h
                                    base = _re.sub(r"[^a-z0-9_]", "", h[1:])
                                    return '@' + base
                            nh = normalize_handle(new_h)
                            net.send({"type": "group_set_handle", "group_id": state.group_manage_gid, "handle": nh})
                            state.status = "Сохраняем логин чата..."
                        except Exception:
                            pass
                continue
            if isinstance(ch, str):
                if ch in ('\x7f', '\b'):
                    if int(getattr(state, 'group_manage_field', 0)) == 0:
                        state.group_manage_name_input = state.group_manage_name_input[:-1]
                    elif int(getattr(state, 'group_manage_field', 0)) == 1:
                        state.group_manage_handle_input = state.group_manage_handle_input[:-1]
                elif ch.isprintable():
                    if int(getattr(state, 'group_manage_field', 0)) == 0:
                        state.group_manage_name_input += ch
                    elif int(getattr(state, 'group_manage_field', 0)) == 1:
                        state.group_manage_handle_input += ch
                continue
            else:
                if ch == curses.KEY_BACKSPACE:
                    if int(getattr(state, 'group_manage_field', 0)) == 0:
                        state.group_manage_name_input = state.group_manage_name_input[:-1]
                        continue
                    if int(getattr(state, 'group_manage_field', 0)) == 1:
                        state.group_manage_handle_input = state.group_manage_handle_input[:-1]
                        continue

        # Board manage input handling
        if getattr(state, 'board_manage_mode', False):
            if ch in ('\x1b',) or ch == 27:
                state.board_manage_mode = False
                state.board_manage_bid = None
                state.status = "Доска: окно закрыто"
                continue
            if ch in ('\t',) or ch == curses.KEY_BTAB or ch in (curses.KEY_UP, curses.KEY_DOWN):
                try:
                    cur = int(getattr(state, 'board_manage_field', 0))
                    if ch == curses.KEY_UP:
                        cur = 0
                    elif ch == curses.KEY_DOWN:
                        cur = 1
                    else:
                        cur = 1 - cur
                    state.board_manage_field = cur
                except Exception:
                    pass
                continue
            if ch in ('\n', '\r') or ch in (10, 13):
                curf = int(getattr(state, 'board_manage_field', 0))
                if state.board_manage_bid and curf == 0:
                    new_name = (state.board_manage_name_input or '').strip()
                    if not new_name:
                        state.status = "Введите название доски"
                    else:
                        try:
                            net.send({"type": "board_rename", "board_id": state.board_manage_bid, "name": new_name})
                            state.status = "Сохраняем название доски..."
                        except Exception:
                            pass
                elif state.board_manage_bid and curf == 1:
                    new_h = (state.board_manage_handle_input or '').strip()
                    if not new_h:
                        state.status = "Введите логин доски"
                    else:
                        try:
                            try:
                                from modules.profile import normalize_handle  # type: ignore
                            except Exception:
                                def normalize_handle(h: str) -> str:
                                    import re as _re
                                    h = (h or '').strip().lower()
                                    if not h.startswith('@'):
                                        h = '@' + h
                                    base = _re.sub(r"[^a-z0-9_]", "", h[1:])
                                    return '@' + base
                            nh = normalize_handle(new_h)
                            net.send({"type": "board_set_handle", "board_id": state.board_manage_bid, "handle": nh})
                            state.status = "Сохраняем логин доски..."
                        except Exception:
                            pass
                continue
            if isinstance(ch, str):
                if ch in ('\x7f', '\b'):
                    if int(getattr(state, 'board_manage_field', 0)) == 0:
                        state.board_manage_name_input = state.board_manage_name_input[:-1]
                    elif int(getattr(state, 'board_manage_field', 0)) == 1:
                        state.board_manage_handle_input = state.board_manage_handle_input[:-1]
                elif ch.isprintable():
                    if int(getattr(state, 'board_manage_field', 0)) == 0:
                        state.board_manage_name_input += ch
                    elif int(getattr(state, 'board_manage_field', 0)) == 1:
                        state.board_manage_handle_input += ch
                continue
            else:
                if ch == curses.KEY_BACKSPACE:
                    if int(getattr(state, 'board_manage_field', 0)) == 0:
                        state.board_manage_name_input = state.board_manage_name_input[:-1]
                        continue
                    if int(getattr(state, 'board_manage_field', 0)) == 1:
                        state.board_manage_handle_input = state.board_manage_handle_input[:-1]
                        continue

        if getattr(state, 'group_member_add_mode', False):
            if ch in ('\x1b',) or ch == 27:
                state.group_member_add_mode = False
                state.group_member_add_gid = None
                state.group_member_add_input = ""
                state.status = "Добавление участников отменено"
                continue
            if ch in ('\n', '\r') or ch in (10, 13):
                gid = state.group_member_add_gid
                raw = format_member_tokens(state.group_member_add_input or '').strip()
                if not gid:
                    state.group_member_add_mode = False
                    continue
                if not raw:
                    state.status = "Укажите участников"
                    continue
                import re as _re
                tokens = [t.strip() for t in _re.split(r"[\s,]+", raw) if t.strip()]
                if not tokens:
                    state.status = "Укажите участников"
                    continue
                resolved: List[str] = []
                missing: List[str] = []
                for tok in tokens:
                    if tok.startswith('@'):
                        match_id = None
                        for pid, prof in state.profiles.items():
                            if (prof or {}).get('handle') == tok:
                                match_id = pid
                                break
                        if match_id:
                            resolved.append(str(match_id))
                        else:
                            missing.append(tok)
                    else:
                        resolved.append(tok)
                if missing:
                    state.status = "Не найдено: " + ", ".join(missing)
                    continue
                resolved = [str(x) for x in resolved if x]
                if not resolved:
                    state.status = "Укажите участников"
                    continue
                if not ensure_group_members_authorized(state, resolved, net):
                    continue
                try:
                    net.send({"type": "group_add", "group_id": gid, "members": resolved})
                    state.status = "Добавляем участников: " + ", ".join(resolved)
                except Exception:
                    state.status = "Не удалось отправить запрос"
                state.group_member_add_mode = False
                state.group_member_add_gid = None
                state.group_member_add_input = ""
                continue
            if ch in ('\x7f', '\b') or ch == curses.KEY_BACKSPACE:
                state.group_member_add_input = format_member_tokens(state.group_member_add_input[:-1])
                continue
            if isinstance(ch, str) and ch.isprintable():
                state.group_member_add_input = format_member_tokens(state.group_member_add_input + ch)
                continue

        if getattr(state, 'board_member_add_mode', False):
            if ch in ('\x1b',) or ch == 27:
                state.board_member_add_mode = False
                state.board_member_add_bid = None
                state.board_member_add_input = ""
                state.status = "Приглашение отменено"
                continue
            if ch in ('\n', '\r') or ch in (10, 13):
                bid = state.board_member_add_bid
                raw = format_member_tokens(state.board_member_add_input or '').strip()
                if not bid:
                    state.board_member_add_mode = False
                    continue
                if not raw:
                    state.status = "Укажите участников"
                    continue
                import re as _re
                tokens = [t.strip() for t in _re.split(r"[\s,]+", raw) if t.strip()]
                if not tokens:
                    state.status = "Укажите участников"
                    continue
                resolved: List[str] = []
                missing: List[str] = []
                for tok in tokens:
                    if tok.startswith('@'):
                        match_id = None
                        for pid, prof in state.profiles.items():
                            if (prof or {}).get('handle') == tok:
                                match_id = pid
                                break
                        if match_id:
                            resolved.append(str(match_id))
                        else:
                            missing.append(tok)
                    else:
                        resolved.append(tok)
                if missing:
                    state.status = "Не найдено: " + ", ".join(missing)
                    continue
                resolved = [str(x) for x in resolved if x]
                if not resolved:
                    state.status = "Укажите участников"
                    continue
                try:
                    # Сначала запросить разрешение у пользователей (приглашение)
                    net.send({"type": "board_invite", "board_id": bid, "members": resolved})
                    state.status = "Приглашены: " + ", ".join(resolved)
                except Exception:
                    state.status = "Не удалось отправить приглашение"
                state.board_member_add_mode = False
                state.board_member_add_bid = None
                state.board_member_add_input = ""
                continue
            if ch in ('\x7f', '\b') or ch == curses.KEY_BACKSPACE:
                state.board_member_add_input = format_member_tokens(state.board_member_add_input[:-1])
                continue
            if isinstance(ch, str) and ch.isprintable():
                state.board_member_add_input = format_member_tokens(state.board_member_add_input + ch)
                continue

        if getattr(state, 'board_added_consent_mode', False):
            if ch in (curses.KEY_UP, curses.KEY_LEFT):
                try:
                    state.board_added_index = max(0, int(getattr(state, 'board_added_index', 0)) - 1)
                except Exception:
                    state.board_added_index = 0
                continue
            if ch in (curses.KEY_DOWN, curses.KEY_RIGHT):
                try:
                    state.board_added_index = min(1, int(getattr(state, 'board_added_index', 0)) + 1)
                except Exception:
                    state.board_added_index = 0
                continue
            if ch in ('\n', '\r') or ch in (10, 13):
                sel = int(getattr(state, 'board_added_index', 0))
                bid = state.board_added_bid or ''
                if sel == 1 and bid:
                    try:
                        net.send({"type": "board_leave", "board_id": bid})
                        state.status = f"Отклонено участие в доске: {bid}"
                    except Exception:
                        state.status = "Не удалось покинуть доску"
                state.board_added_consent_mode = False
                state.board_added_bid = None
                continue
            if ch in ('\x1b',) or ch == 27:
                state.board_added_consent_mode = False
                state.board_added_bid = None
                state.status = "Приглашение подтверждено"
                continue

        if getattr(state, 'board_member_remove_mode', False):
            if ch in ('\x1b',) or ch == 27:
                state.board_member_remove_mode = False
                state.board_member_remove_bid = None
                state.board_member_remove_options = []
                state.status = "Удаление отменено"
                continue
            if ch in (curses.KEY_UP, curses.KEY_LEFT):
                if state.board_member_remove_options:
                    state.board_member_remove_index = max(0, state.board_member_remove_index - 1)
                continue
            if ch in (curses.KEY_DOWN, curses.KEY_RIGHT):
                if state.board_member_remove_options:
                    state.board_member_remove_index = min(len(state.board_member_remove_options) - 1, state.board_member_remove_index + 1)
                continue
            if ch in ('\n', '\r') or ch in (10, 13):
                bid = state.board_member_remove_bid
                if not bid or not state.board_member_remove_options:
                    state.board_member_remove_mode = False
                    state.board_member_remove_bid = None
                    state.board_member_remove_options = []
                    continue
                idx = max(0, min(state.board_member_remove_index, len(state.board_member_remove_options) - 1))
                target = state.board_member_remove_options[idx]
                try:
                    net.send({"type": "board_remove", "board_id": bid, "members": [target]})
                    state.status = f"Удаляем из доски: {target}"
                except Exception:
                    state.status = "Не удалось отправить запрос"
                state.board_member_remove_mode = False
                state.board_member_remove_bid = None
                state.board_member_remove_options = []
                continue

        if getattr(state, 'board_invite_mode', False):
            # Unified selection: ↑/↓ to choose, Enter to confirm, Esc to cancel
            if ch in (curses.KEY_UP, curses.KEY_LEFT):
                try:
                    state.board_invite_index = max(0, int(getattr(state, 'board_invite_index', 0)) - 1)
                except Exception:
                    state.board_invite_index = 0
                continue
            if ch in (curses.KEY_DOWN, curses.KEY_RIGHT):
                try:
                    state.board_invite_index = min(1, int(getattr(state, 'board_invite_index', 0)) + 1)
                except Exception:
                    state.board_invite_index = 0
                continue
            if ch in ('\n', '\r') or ch in (10, 13):
                bid = state.board_invite_bid or ''
                sel = int(getattr(state, 'board_invite_index', 0))
                if sel == 0 and bid:
                    try:
                        net.send({"type": "board_invite_response", "board_id": bid, "accept": True})
                        state.status = f"Принято приглашение в доску: {state.board_invite_name or bid}"
                    except Exception:
                        state.status = "Не удалось принять приглашение"
                    try:
                        state.board_pending_invites.pop(bid, None)
                    except Exception:
                        pass
                else:
                    state.status = "Приглашение отклонено"
                    try:
                        if bid:
                            net.send({"type": "board_invite_response", "board_id": bid, "accept": False})
                    except Exception:
                        pass
                    try:
                        if bid:
                            state.board_pending_invites.pop(bid, None)
                    except Exception:
                        pass
                state.board_invite_mode = False
                state.board_invite_bid = None
                continue
            if ch in ('\x1b',) or ch == 27:
                bid = state.board_invite_bid or ''
                state.board_invite_mode = False
                state.board_invite_bid = None
                state.status = "Приглашение отклонено"
                try:
                    if bid:
                        net.send({"type": "board_invite_response", "board_id": bid, "accept": False})
                        state.board_pending_invites.pop(bid, None)
                except Exception:
                    pass
                continue
        # When selecting a pending group invite token, open modal only on Enter
        # If selection is on a pending group invite token, open modal on Enter
        if (not getattr(state, 'group_invite_mode', False)) and (not modal_now) and (ch in ('\n', '\r', 10, 13)):
            sel = current_selected_id(state)
            if isinstance(sel, str) and sel.startswith('GINV:'):
                try:
                    gid = sel.split(':', 1)[1]
                except Exception:
                    gid = sel
                meta = (getattr(state, 'group_pending_invites', {}) or {}).get(gid, {})
                state.group_invite_mode = True
                state.group_invite_gid = gid
                state.group_invite_name = str(meta.get('name') or gid)
                state.group_invite_from = str(meta.get('from') or '')
                state.group_invite_index = 0
                continue
        if getattr(state, 'group_invite_mode', False):
            if ch in ('\t',):
                try:
                    state.group_invite_index = (int(getattr(state, 'group_invite_index', 0)) + 1) % 3
                except Exception:
                    state.group_invite_index = 0
                continue
            if ch == curses.KEY_BTAB:
                try:
                    state.group_invite_index = (int(getattr(state, 'group_invite_index', 0)) - 1) % 3
                except Exception:
                    state.group_invite_index = 0
                continue
            if ch in ('\n', '\r') or ch in (10, 13):
                gid = state.group_invite_gid or ''
                sel = int(getattr(state, 'group_invite_index', 0))
                inviter = state.group_invite_from
                if sel == 0 and gid:
                    try:
                        net.send({"type": "group_invite_response", "group_id": gid, "accept": True})
                        state.status = f"Принято приглашение в группу: {state.group_invite_name or gid}"
                    except Exception:
                        state.status = "Не удалось принять приглашение"
                elif sel == 2 and inviter:
                    try:
                        net.send({"type": "block_set", "peer": inviter, "value": True})
                        state.status = f"Пользователь {inviter} заблокирован"
                    except Exception:
                        state.status = "Не удалось заблокировать"
                    try:
                        if gid:
                            net.send({"type": "group_invite_response", "group_id": gid, "accept": False})
                    except Exception:
                        pass
                else:
                    try:
                        if gid:
                            net.send({"type": "group_invite_response", "group_id": gid, "accept": False})
                    except Exception:
                        pass
                    state.status = "Приглашение отклонено"
                try:
                    if gid:
                        state.group_pending_invites.pop(gid, None)
                except Exception:
                    pass
                state.group_invite_mode = False
                state.group_invite_gid = None
                continue
            if ch in ('\x1b',) or ch == 27:
                state.group_invite_mode = False
                state.group_invite_gid = None
                state.status = "Приглашение отклонено"
                continue
            # ignore arrows to avoid hijacking list navigation while modal open
            if ch in (curses.KEY_UP, curses.KEY_LEFT, curses.KEY_DOWN, curses.KEY_RIGHT):
                continue
        if getattr(state, 'group_member_remove_mode', False):
            if ch in ('\x1b',) or ch == 27:
                state.group_member_remove_mode = False
                state.group_member_remove_gid = None
                state.group_member_remove_options = []
                state.status = "Удаление участника отменено"
                continue
            if ch in (curses.KEY_UP, curses.KEY_LEFT):
                if state.group_member_remove_options:
                    state.group_member_remove_index = max(0, state.group_member_remove_index - 1)
                continue
            if ch in (curses.KEY_DOWN, curses.KEY_RIGHT):
                if state.group_member_remove_options:
                    state.group_member_remove_index = min(len(state.group_member_remove_options) - 1, state.group_member_remove_index + 1)
                continue
            if ch in ('\n', '\r') or ch in (10, 13):
                gid = state.group_member_remove_gid
                if not gid or not state.group_member_remove_options:
                    state.group_member_remove_mode = False
                    state.group_member_remove_gid = None
                    state.group_member_remove_options = []
                    continue
                idx = max(0, min(state.group_member_remove_index, len(state.group_member_remove_options) - 1))
                target = state.group_member_remove_options[idx]
                try:
                    net.send({"type": "group_remove", "group_id": gid, "members": [target]})
                    state.status = f"Удаляем участника: {target}"
                except Exception:
                    state.status = "Не удалось отправить запрос"
                state.group_member_remove_mode = False
                state.group_member_remove_gid = None
                state.group_member_remove_options = []
                continue

        if not state.authed:
            # auth mode switching
            if ch == '\t' or ch == curses.KEY_BTAB or ch == curses.KEY_LEFT or ch == curses.KEY_RIGHT:
                state.auth_mode = 'register' if state.auth_mode == 'login' else 'login'
                state.login_field = 0
                state.login_msg = ""
                continue
            if isinstance(ch, str):
                if ch == '\x1b':  # ESC — exit from auth screen
                    running = False
                    break
                if ch == '\x03':  # Ctrl+C — явный выход
                    running = False
                    break
                elif ch in ('\n', '\r'):
                    if state.auth_mode == 'login':
                        if state.login_field == 0:
                            state.login_field = 1
                        else:
                            if not state.id_input:
                                state.login_msg = "Введите ID/@логин"
                            elif not state.pw1:
                                state.login_msg = "Введите пароль"
                            else:
                                state.login_msg = "Входим..."
                                state.last_submit_pw = state.pw1
                                try:
                                    state.last_auth_id = state.id_input.strip()
                                    import time as _t
                                    state.login_pending_since = float(_t.time())
                                except Exception:
                                    state.login_pending_since = 0.0
                                net.send({"type": "auth", "id": state.last_auth_id or state.id_input.strip(), "password": state.pw1})
                    else:  # register
                        if state.login_field == 0:
                            state.login_field = 1
                        else:
                            if not state.pw1:
                                state.login_msg = "Пароль не может быть пустым"
                            elif state.pw1 != state.pw2:
                                state.login_msg = "Пароли не совпадают"
                                state.pw1 = state.pw2 = ""
                                state.login_field = 0
                            else:
                                state.login_msg = "Регистрируем..."
                                state.last_submit_pw = state.pw1
                                net.send({"type": "register", "password": state.pw1})
                    continue
                elif ch in ('\x7f', '\b'):
                    if state.auth_mode == 'login':
                        if state.login_field == 0:
                            state.id_input = state.id_input[:-1]
                        else:
                            state.pw1 = state.pw1[:-1]
                    else:
                        if state.login_field == 0:
                            state.pw1 = state.pw1[:-1]
                        else:
                            state.pw2 = state.pw2[:-1]
                    continue
                elif ch == 'k':
                    state.login_field = 0
                    continue
                elif ch == 'j':
                    state.login_field = 1
                    continue
                else:
                    if ch.isprintable():
                        if state.auth_mode == 'login':
                            if state.login_field == 0:
                                state.id_input += ch
                            else:
                                state.pw1 += ch
                        else:
                            if state.login_field == 0:
                                state.pw1 += ch
                            else:
                                state.pw2 += ch
                    continue
            else:
                if ch == curses.KEY_UP:
                    state.login_field = 0
                    continue
                elif ch == curses.KEY_DOWN:
                    state.login_field = 1
                    continue
                elif ch == 27:  # ESC — exit from auth screen (int)
                    running = False
                    break
                elif ch == curses.KEY_MOUSE:
                    try:
                        _mid, mx, my, _mz, bstate = curses.getmouse()
                    except Exception:
                        continue
                    # React only to left button clicks on tab line (y==0)
                    btnmask = 0
                    for name in ("BUTTON1_CLICKED", "BUTTON1_PRESSED", "BUTTON1_RELEASED", "BUTTON1_DOUBLE_CLICKED"):
                        btnmask |= getattr(curses, name, 0)
                    if not (bstate & btnmask):
                        continue
                    if my == 0:
                        # Recreate tab layout to compute spans
                        tabs = [" Вход ", " Регистрация "]
                        active = 0 if state.auth_mode == 'login' else 1
                        x_cursor = 1  # leading space used in title
                        for i, t in enumerate(tabs):
                            chunk = f"[{t}]  " if i == active else f" {t}   "
                            start = x_cursor
                            end = x_cursor + len(chunk)
                            if start <= mx < end:
                                state.auth_mode = 'login' if i == 0 else 'register'
                                state.login_field = 0
                                state.login_msg = ""
                                break
                            x_cursor = end
                    continue
                elif ch in (curses.KEY_BACKSPACE,):
                    if state.auth_mode == 'login':
                        if state.login_field == 0:
                            state.id_input = state.id_input[:-1]
                        else:
                            state.pw1 = state.pw1[:-1]
                    else:
                        if state.login_field == 0:
                            state.pw1 = state.pw1[:-1]
                        else:
                            state.pw2 = state.pw2[:-1]
                    continue

        # Handle wide-char and key events for chat / search / prompts
        if isinstance(ch, str):
            if ch == '\x03':  # Ctrl+C
                running = False
                break
            if ch == '\x1b':  # ESC (string variant)
                # Close overlays if any, otherwise exit
                sel = None
                try:
                    sel = current_selected_id(state)
                except Exception:
                    pass
                overlays_active = bool(state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or state.help_mode)
                # Cancel pending outgoing auth request for current chat if no overlays
                if sel and (sel in getattr(state, 'authz_out_pending', set())) and not overlays_active:
                    try:
                        net.send({"type": "authz_cancel", "peer": sel})
                    except Exception:
                        pass
                    try:
                        state.pending_out.discard(sel)
                        state.authz_out_pending.discard(sel)
                    except Exception:
                        pass
                    state.status = f"Отменён запрос: {sel}"
                    continue
                if state.action_menu_mode:
                    # If user dismissed incoming auth actions, don't auto-pop it back on navigation.
                    if _is_auth_actions_menu(state):
                        try:
                            peer = str(state.action_menu_peer or "").strip()
                            if peer:
                                state.authz_menu_snoozed.add(peer)
                        except Exception:
                            pass
                    state.action_menu_mode = False
                    state.action_menu_peer = None
                    state.action_menu_options = []
                    state.status = ""
                    continue
                if state.search_mode:
                    state.search_mode = False
                    state.search_query = ""
                    state.search_results = []
                    state.status = "Поиск закрыт"
                    continue
                if state.help_mode:
                    state.help_mode = False
                    state.status = ""
                    continue
                running = False
                break
            # Actions menu navigation via Tab ONLY for authorization menu; other menus use arrows
            if state.action_menu_mode and state.action_menu_options and ch == '\t' and _is_auth_actions_menu(state):
                try:
                    n = len(state.action_menu_options)
                    if n > 0:
                        state.action_menu_index = (state.action_menu_index + 1) % n
                except Exception:
                    pass
                continue
            if ch == '\x1b':  # ESC
                # If outgoing auth overlay is active for the current chat, cancel it
                try:
                    sel = current_selected_id(state)
                except Exception:
                    sel = None
                overlays_active = bool(state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or state.help_mode)
                if sel and (sel in getattr(state, 'authz_out_pending', set())) and not overlays_active:
                    try:
                        net.send({"type": "authz_cancel", "peer": sel})
                    except Exception:
                        pass
                    try:
                        state.pending_out.discard(sel)
                    except Exception:
                        pass
                    try:
                        state.authz_out_pending.discard(sel)
                    except Exception:
                        pass
                    state.status = f"Отменён запрос: {sel}"
                    continue
                if state.action_menu_mode:
                    # If user dismissed incoming auth actions, don't auto-pop it back on navigation.
                    if _is_auth_actions_menu(state):
                        try:
                            peer = str(state.action_menu_peer or "").strip()
                            if peer:
                                state.authz_menu_snoozed.add(peer)
                        except Exception:
                            pass
                    state.action_menu_mode = False
                    state.action_menu_peer = None
                    state.action_menu_options = []
                    state.status = ""
                    continue
                if state.search_mode:
                    state.search_mode = False
                    state.search_query = ""
                    state.search_results = []
                    state.status = "Поиск закрыт"
                else:
                    if state.help_mode:
                        state.help_mode = False
                        state.status = ""
                        continue
                    # Single ESC exits app when no overlays are active
                    running = False
                    break
                continue
            # Быстрые действия в поиске: Y/N — принять/отклонить, если это входящий
            # Кнопка A в режиме поиска обрабатывается ОТДЕЛЬНО в блоке поиска (наверх по файлу),
            # чтобы не мешать вводу символа 'a' в строке.
            if state.search_mode and (not getattr(state, 'board_invite_mode', False)) and ch.lower() in ('y', 'n'):
                sel = current_selected_id(state)
                if sel and (sel not in state.groups):
                    if ch.lower() == 'y' and sel in state.pending_requests:
                        net.send({"type": "authz_response", "peer": sel, "accept": True})
                        state.lock_selection_peer = sel
                        state.suppress_auto_menu = True
                        state.status = f"Добавлен в контакты: {sel}"
                        try:
                            state.pending_requests = [p for p in state.pending_requests if p != sel]
                        except Exception:
                            pass
                        continue
                    if ch.lower() == 'n' and sel in state.pending_requests:
                        net.send({"type": "authz_response", "peer": sel, "accept": False})
                        state.status = f"Отклонён запрос от {sel}"
                        try:
                            state.pending_requests = [p for p in state.pending_requests if p != sel]
                        except Exception:
                            pass
                        continue
            # Глобальные горячие клавиши (когда поле ввода пустое)
            if (state.input_buffer == '') and (not getattr(state, 'board_invite_mode', False)) and ch.lower() in ('a', 'y', 'n') and not state.profile_mode:
                sel = current_selected_id(state)
                # Only allow A/Y/N for user peers (not groups/boards/tokens)
                if sel and (sel not in state.groups) and (sel not in getattr(state, 'boards', {})) and not (isinstance(sel, str) and (sel.startswith('BINV:') or sel.startswith('JOIN:') or sel.startswith('b-'))):
                    if ch.lower() == 'a':
                        if not state.friends.get(sel) and sel not in state.roster_friends:
                            try:
                                net.send({"type": "authz_request", "to": sel})
                                state.pending_out.add(sel)
                                state.status = f"Запрос авторизации отправлен: {sel}"
                                try:
                                    state.authz_out_pending.add(sel)
                                except Exception:
                                    pass
                            except Exception:
                                state.status = f"Не удалось отправить запрос: {sel}"
                        else:
                            state.status = f"Уже в контактах: {sel}"
                        continue
                    if ch.lower() == 'y' and sel in state.pending_requests:
                        net.send({"type": "authz_response", "peer": sel, "accept": True})
                        state.lock_selection_peer = sel
                        state.suppress_auto_menu = True
                        state.status = f"Добавлен в контакты: {sel}"
                        try:
                            state.pending_requests = [p for p in state.pending_requests if p != sel]
                        except Exception:
                            pass
                        continue
                    if ch.lower() == 'n' and sel in state.pending_requests:
                        net.send({"type": "authz_response", "peer": sel, "accept": False})
                        state.status = f"Отклонён запрос от {sel}"
                        try:
                            state.pending_requests = [p for p in state.pending_requests if p != sel]
                        except Exception:
                            pass
                        continue
            # Автоприглашение (Y/N) отключено: действия доступны через меню/выбор
            if state.search_mode:
                if ch in ('\n', '\r'):
                    # Perform search; сервер вернёт search_result. Окно поиска не закрываем,
                    # пока не получим подтверждение (или не найдём пользователя).
                    q = (state.search_query or '').strip()
                    # Нормализация: если это похоже на логин (с буквами) и нет '@', добавим его.
                    try:
                        if any(c.isalpha() for c in q) and not q.startswith('@'):
                            # используем нормализатор логинов, если доступен
                            h = normalize_handle(q)
                            if h:
                                q = h
                    except Exception:
                        pass
                    # Локальная первичная проверка контакт-листа
                    try:
                        def _local_rel_for(qs: str):
                            rid = None
                            # По точному ID
                            if qs and not qs.startswith('@'):
                                known = set(state.friends.keys()) | set(state.roster_friends.keys()) | set(state.pending_requests) | set(state.pending_out) | set(state.blocked)
                                if qs in known:
                                    rid = qs
                            # По @логину из локального кэша профилей
                            if (rid is None) and qs.startswith('@'):
                                for pid, prof in (state.profiles or {}).items():
                                    try:
                                        if (prof or {}).get('handle') == qs:
                                            rid = pid
                                            break
                                    except Exception:
                                        pass
                            if not rid:
                                return None
                            if rid in state.blocked:
                                return rid, 'blocked'
                            if (rid in state.friends) or (rid in state.roster_friends):
                                return rid, 'friend'
                            if rid in state.pending_requests:
                                return rid, 'incoming'
                            if rid in state.pending_out:
                                return rid, 'outgoing'
                            return rid, 'known'

                        local = _local_rel_for(q)
                        if local:
                            rid, rel = local
                            mapping = {
                                'friend': f"Контакт уже добавлен и авторизован: {rid}",
                                'incoming': f"Аккаунт ждёт вашей авторизации: {rid}",
                                'outgoing': f"Контакт добавлен, ждёт авторизации (исходящий): {rid}",
                                'blocked': f"Аккаунт добавлен и заблокирован вами: {rid}",
                                'known': f"Контакт известен: {rid}",
                            }
                            state.status = mapping.get(rel, f"Контакт: {rid}")
                    except Exception:
                        pass
                    # Close conflicting overlays
                    state.action_menu_mode = False
                    state.profile_view_mode = False
                    state.profile_mode = False
                    state.modal_message = None
                    net.send({"type": "search", "query": q})
                    state.status = "Поиск..."
                    # Остаёмся в режиме поиска; Overlay останется видимым до результата
                    continue
                elif ch in ('\x7f', '\b'):
                    state.search_query = state.search_query[:-1]
                    continue
                else:
                    if ch.isprintable():
                        state.search_query += ch
                        continue
            if ch == '\n':
                # Ctrl+J (LF) — новая строка в поле ввода
                state.input_buffer += '\n'
                continue
            elif ch == '\r':
                # Enter (CR) — отправка
                if state.action_menu_mode and state.action_menu_options and state.action_menu_peer:
                    opt = state.action_menu_options[state.action_menu_index]
                    peer = state.action_menu_peer
                    # Execute action
                    try:
                        # Унифицированный пункт: Отправить файл (открыть менеджер файлов)
                        if opt == 'Отправить файл':
                            # Закрыть меню действий и открыть модалку выбора файла
                            state.action_menu_mode = False
                            state.action_menu_peer = None
                            state.action_menu_options = []
                            try:
                                start_file_browser(state)  # type: ignore[name-defined]
                                state.status = "Выберите файл для отправки"
                            except Exception:
                                state.file_browser_mode = True
                            continue
                        # Group join request token handling
                        if isinstance(peer, str) and peer.startswith('JOIN:'):
                            try:
                                _, gid, rid = peer.split(':', 2)
                            except Exception:
                                gid, rid = '', ''
                            if opt == 'Принять в чат' and gid and rid:
                                net.send({"type": "group_join_response", "group_id": gid, "peer": rid, "accept": True})
                                state.status = f"Принят в чат: {rid} → {gid}"
                            elif opt == 'Отклонить' and gid and rid:
                                net.send({"type": "group_join_response", "group_id": gid, "peer": rid, "accept": False})
                                state.status = f"Отклонён запрос в чат: {rid}"
                            elif opt == 'Профиль пользователя' and rid:
                                try:
                                    net.send({"type": "profile_get", "id": rid})
                                except Exception:
                                    pass
                                state.profile_view_mode = True
                                state.profile_view_id = rid
                            elif opt == 'Профиль чата' and gid:
                                try:
                                    net.send({"type": "group_info", "group_id": gid})
                                except Exception:
                                    pass
                            # keep rest of choices intact
                        # Board invite token handling via actions menu
                        elif isinstance(peer, str) and peer.startswith('BINV:'):
                            bid = ''
                            try:
                                bid = peer.split(':', 1)[1]
                            except Exception:
                                bid = ''
                            if opt == 'Принять приглашение' and bid:
                                net.send({"type": "board_join", "board_id": bid})
                                state.status = f"Принято приглашение в доску: {bid}"
                                try:
                                    state.board_pending_invites.pop(bid, None)
                                except Exception:
                                    pass
                                state.action_menu_mode = False
                                state.action_menu_peer = None
                                state.action_menu_options = []
                                continue
                            if opt == 'Отклонить' and bid:
                                try:
                                    state.board_pending_invites.pop(bid, None)
                                except Exception:
                                    pass
                                state.status = "Приглашение отклонено"
                                state.action_menu_mode = False
                                state.action_menu_peer = None
                                state.action_menu_options = []
                                continue
                        elif (peer in state.groups) and (opt == 'Добавить участника'):
                            state.group_member_add_mode = True
                            state.group_member_add_gid = peer
                            state.group_member_add_input = ""
                            state.action_menu_mode = False
                            state.status = f"Добавление участников: {peer}"
                            try:
                                net.send({"type": "group_info", "group_id": peer})
                            except Exception:
                                pass
                        elif (peer in state.groups) and (opt == 'Удалить участника'):
                            gmeta = state.groups.get(peer) or {}
                            owner_id = str(gmeta.get('owner_id') or '')
                            members = [m for m in list(gmeta.get('members') or []) if m and m != owner_id]
                            state.group_member_remove_mode = True
                            state.group_member_remove_gid = peer
                            state.group_member_remove_options = members
                            state.group_member_remove_index = 0
                            state.action_menu_mode = False
                            state.status = f"Удаление участника из {peer}"
                            try:
                                net.send({"type": "group_info", "group_id": peer})
                            except Exception:
                                pass
                        elif (peer in state.groups) and (opt == 'Рассформировать чат'):
                            net.send({"type": "group_disband", "group_id": peer})
                            state.status = f"Рассформирование чата: {peer}"
                        elif (peer in state.groups) and (opt == 'Участники'):
                            open_members_view(state, net, peer)
                            state.action_menu_mode = False
                            state.action_menu_peer = None
                            state.action_menu_options = []
                            continue
                        elif (peer in state.groups) and (opt == 'Покинуть чат'):
                            try:
                                net.send({"type": "group_leave", "group_id": peer})
                                state.status = f"Покидаем чат: {peer}"
                            except Exception:
                                pass
                            state.action_menu_mode = False
                            state.action_menu_peer = None
                            state.action_menu_options = []
                            continue
                        elif (peer in state.groups) and (opt in ('Заглушить чат', 'Включить уведомления')):
                            try:
                                muted_set = getattr(state, 'group_muted')
                            except Exception:
                                muted_set = set()
                            if opt == 'Заглушить чат':
                                try:
                                    muted_set.add(peer)
                                except Exception:
                                    pass
                                state.status = f"🔕 Уведомления отключены: {peer}"
                            else:
                                try:
                                    if peer in muted_set:
                                        muted_set.discard(peer)
                                except Exception:
                                    pass
                                state.status = f"🔔 Уведомления включены: {peer}"
                            try:
                                state.group_muted = muted_set
                            except Exception:
                                pass
                        elif (peer in state.groups) and (opt == 'Профиль чата'):
                            try:
                                net.send({"type": "group_info", "group_id": peer})
                            except Exception:
                                pass
                            state.group_manage_mode = True
                            state.group_manage_gid = peer
                            state.group_manage_field = 0
                            ginfo = state.groups.get(peer) or {}
                            state.group_manage_name_input = str(ginfo.get('name') or peer)
                            try:
                                state.group_manage_handle_input = str(ginfo.get('handle') or '')
                            except Exception:
                                state.group_manage_handle_input = ''
                            state.group_manage_member_count = len(list(ginfo.get('members') or []))
                        elif (peer in getattr(state, 'boards', {})) and (opt == 'Добавить участника'):
                            state.board_member_add_mode = True
                            state.board_member_add_bid = peer
                            state.board_member_add_input = ""
                            state.action_menu_mode = False
                            state.status = f"Приглашение участников в доску: {peer}"
                            try:
                                net.send({"type": "board_info", "board_id": peer})
                            except Exception:
                                pass
                        elif (peer in getattr(state, 'boards', {})) and (opt == 'Участники'):
                            open_members_view(state, net, peer)
                            state.action_menu_mode = False
                            state.action_menu_peer = None
                            state.action_menu_options = []
                            continue
                        elif (peer in getattr(state, 'boards', {})) and (opt == 'Удалить участника'):
                            bmeta = (getattr(state, 'boards', {}) or {}).get(peer) or {}
                            owner_id = str(bmeta.get('owner_id') or '')
                            members = [m for m in list(bmeta.get('members') or []) if m and m != owner_id]
                            state.board_member_remove_mode = True
                            state.board_member_remove_bid = peer
                            state.board_member_remove_options = members
                            state.board_member_remove_index = 0
                            state.action_menu_mode = False
                            state.status = f"Удаление участника из доски {peer}"
                            try:
                                net.send({"type": "board_info", "board_id": peer})
                            except Exception:
                                pass
                        elif (peer in getattr(state, 'boards', {})) and (opt == 'Рассформировать доску'):
                            try:
                                net.send({"type": "board_disband", "board_id": peer})
                                state.status = f"Рассформирование доски: {peer}"
                            except Exception:
                                pass
                        elif (peer in getattr(state, 'boards', {})) and (opt in ('Заглушить доску', 'Включить уведомления', 'Покинуть доску')):
                            if opt == 'Покинуть доску':
                                try:
                                    net.send({"type": "board_leave", "board_id": peer})
                                    state.status = f"Покидаем доску: {peer}"
                                except Exception:
                                    pass
                                # keep processing next event
                                continue
                            try:
                                muted_set = getattr(state, 'board_muted')
                            except Exception:
                                muted_set = set()
                            if opt == 'Заглушить доску':
                                try:
                                    muted_set.add(peer)
                                except Exception:
                                    pass
                                state.status = f"🔕 Уведомления отключены: {peer}"
                            else:
                                try:
                                    muted_set.discard(peer)
                                except Exception:
                                    pass
                                state.status = f"🔔 Уведомления включены: {peer}"
                            try:
                                state.board_muted = muted_set
                            except Exception:
                                pass
                        elif opt == 'Запросить авторизацию':
                            net.send({"type": "authz_request", "to": peer})
                            state.pending_out.add(peer)
                            state.status = f"Запрос авторизации отправлен: {peer}"
                            try:
                                state.authz_out_pending.add(peer)
                            except Exception:
                                pass
                        elif opt == 'Авторизовать':
                            net.send({"type": "authz_response", "peer": peer, "accept": True})
                            state.status = f"Добавлен в контакты: {peer}"
                            state.lock_selection_peer = peer
                            state.suppress_auto_menu = True
                            try:
                                state.pending_requests = [p for p in state.pending_requests if p != peer]
                            except Exception:
                                pass
                        elif opt == 'Отклонить':
                            net.send({"type": "authz_response", "peer": peer, "accept": False})
                            state.status = f"Отклонён запрос от {peer}"
                            try:
                                state.pending_requests = [p for p in state.pending_requests if p != peer]
                            except Exception:
                                pass
                        elif opt == 'Отменить запрос':
                            net.send({"type": "authz_cancel", "peer": peer})
                            state.status = f"Отменён запрос: {peer}"
                        elif opt == 'Удалить из контактов':
                            # Если контакт заблокирован (или вы у него в blocked_by) — не отправляем friend_remove,
                            # чтобы сохранить авторизацию и избежать спама: просто скрываем его из списка друзей.
                            try:
                                is_block_ctx = (peer in state.blocked) or (peer in getattr(state, 'blocked_by', set()))
                            except Exception:
                                is_block_ctx = False
                            if is_block_ctx:
                                try:
                                    state.roster_friends.pop(peer, None)
                                except Exception:
                                    pass
                                # Дополнительно уберём из локальной карты друзей (на случай fallback-пути)
                                try:
                                    state.friends.pop(peer, None)
                                except Exception:
                                    pass
                                # Скрыть заблокированный контакт из левого списка (останется доступен через поиск)
                                try:
                                    state.hidden_blocked.add(peer)
                                except Exception:
                                    pass
                                state.status = f"Скрыт из контактов (заблокирован): {peer}"
                            else:
                                net.send({"type": "friend_remove", "peer": peer})
                                # Сразу локально уберём из friends/roster_friends для мгновенной реакции UI
                                try:
                                    state.friends.pop(peer, None)
                                except Exception:
                                    pass
                                try:
                                    state.roster_friends.pop(peer, None)
                                except Exception:
                                    pass
                                state.status = f"Удаление из контактов: {peer}"
                        elif opt == 'Очистить чат':
                            try:
                                net.send({"type": "chat_clear", "peer": peer})
                                state.status = f"Очищаем чат с {peer}..."
                            except Exception:
                                state.status = f"Не удалось очистить чат: {peer}"
                        elif opt in ('Заглушить', 'Снять заглушку'):
                            val = (opt == 'Заглушить')
                            net.send({"type": "mute_set", "peer": peer, "value": val})
                        elif opt in ('Заблокировать', 'Разблокировать'):
                            val = (opt == 'Заблокировать')
                            net.send({"type": "block_set", "peer": peer, "value": val})
                        elif opt in ('Профиль пользователя',):
                            try:
                                net.send({"type": "profile_get", "id": peer})
                            except Exception:
                                pass
                            state.profile_view_mode = True
                            state.profile_view_id = peer
                        elif opt in ('Просмотреть профиль', 'Профиль пользователя'):
                            try:
                                net.send({"type": "profile_get", "id": peer})
                            except Exception:
                                pass
                            state.profile_view_mode = True
                            state.profile_view_id = peer
                    except Exception:
                        pass
                    # Close menu
                    state.action_menu_mode = False
                    state.action_menu_peer = None
                    state.action_menu_options = []
                    state.history_scroll = 0
                    continue
                to = current_selected_id(state)
                text = state.input_buffer.strip()
                if text:
                    def _debug_line(msg: str):
                        try:
                            state.debug_lines.append(msg)
                            if len(state.debug_lines) > 300:
                                del state.debug_lines[:len(state.debug_lines) - 300]
                        except Exception:
                            pass
                    def _run_selftest(mode: str):
                        if mode not in ('update', 'file', 'reg'):
                            state.status = "Используйте: /selftest update|file|reg"
                            return
                        state.status = f"Selftest {mode}..."
                        _debug_line(f"[selftest] start {mode}")
                        try:
                            if mode == 'update':
                                base = _get_update_base_url()
                                if not base:
                                    _debug_line("[selftest:update] UPDATE_URL не задан")
                                else:
                                    mani = _load_manifest(base, timeout=4.0)
                                    if mani:
                                        entries = _safe_manifest_entries(mani)
                                        _debug_line(f"[selftest:update] manifest ok version={mani.get('version')} files={len(entries)}")
                                    else:
                                        _debug_line("[selftest:update] не удалось загрузить manifest")
                            elif mode == 'file':
                                try:
                                    tmp_dir = FILES_DIR / "selftest"
                                    tmp_dir.mkdir(parents=True, exist_ok=True)
                                    tmp_path = tmp_dir / "probe.txt"
                                    tmp_path.write_text("selftest-file-payload", encoding='utf-8')
                                    _debug_line(f"[selftest:file] подготовлен файл {tmp_path} size={tmp_path.stat().st_size}")
                                except Exception as e:
                                    _debug_line(f"[selftest:file] ошибка подготовки файла: {e}")
                            elif mode == 'reg':
                                try:
                                    rid = "selftest-" + str(int(time.time()))
                                    payload = {"type": "register", "id": rid, "password": "testpw"}
                                    state.selftest_reg = True
                                    state.selftest_reg_pw = "testpw"
                                    net.send(payload)
                                    _debug_line(f"[selftest:reg] отправлен запрос: {payload}")
                                except Exception as e:
                                    _debug_line(f"[selftest:reg] ошибка отправки: {e}")
                        except Exception as e:
                            _debug_line(f"[selftest] ошибка: {e}")
                    # Slash-commands
                    if text.startswith('/'):
                        cmd = text.split()[0]
                        arg = text[len(cmd):].strip()
                        if cmd == '/profile':
                            net.send({"type": "profile_get"})
                            state.status = "Запрос профиля..."
                        elif cmd in ('/search', '/seach'):
                            # Open search mode (alias: legacy typo /seach)
                            try:
                                state.profile_mode = False
                                state.profile_view_mode = False
                                state.action_menu_mode = False
                                state.search_action_mode = False
                                state.modal_message = None
                            except Exception:
                                pass
                            state.search_mode = True
                            state.search_query = arg or ""
                            state.search_results = []
                            state.search_live_id = None
                            state.search_live_ok = False
                            state.selected_index = 0
                            state.status = "Режим поиска: введите ID/@логин и Enter"
                            # If query is provided, send search request immediately.
                            if arg:
                                try:
                                    net.send({"type": "search", "query": arg})
                                    state.last_search_sent = time.time()
                                except Exception:
                                    pass
                        elif cmd == '/chat':
                            # Open group chat creation modal (same as F5)
                            try:
                                state.search_action_mode = False
                                state.search_action_peer = None
                                state.search_action_options = []
                                state.search_mode = False
                                state.profile_view_mode = False
                                state.profile_mode = False
                                state.modal_message = None
                                state.group_create_mode = True  # type: ignore[attr-defined]
                                state.group_create_field = 0    # type: ignore[attr-defined]
                                state.group_name_input = getattr(state, 'group_name_input', '') or ""  # type: ignore[attr-defined]
                                state.group_members_input = getattr(state, 'group_members_input', '') or ""  # type: ignore[attr-defined]
                                state.status = "Создание чата: введите имя и участников"
                            except Exception:
                                pass
                        elif cmd == '/board':
                            # Open board creation modal (same as F6)
                            try:
                                state.search_action_mode = False
                                state.search_action_peer = None
                                state.search_action_options = []
                                state.search_mode = False
                                state.profile_view_mode = False
                                state.profile_mode = False
                                state.modal_message = None
                                state.board_create_mode = True
                                state.board_create_field = 0
                                state.board_name_input = getattr(state, 'board_name_input', '') or ""
                                state.board_handle_input = getattr(state, 'board_handle_input', '') or ""
                                state.status = "Создание доски: введите имя и логин"
                            except Exception:
                                pass
                        elif cmd == '/help':
                            # Toggle help overlay (same as F1)
                            try:
                                state.help_mode = not bool(state.help_mode)
                                state.status = "Справка открыта" if state.help_mode else ""
                            except Exception:
                                pass
                        elif cmd == '/exit':
                            running = False
                            break
                        elif cmd in ('/update', '/filetest', '/regtest'):
                            # Алиасы для selftest
                            mode = 'update' if cmd == '/update' else ('file' if cmd == '/filetest' else 'reg')
                            _run_selftest(mode)
                            state.input_buffer = ""
                            state.input_caret = 0
                            state.history_scroll = 0
                            need_redraw = True
                            continue
                        elif cmd == '/selftest':
                            mode = arg.split()[0] if arg else ''
                            if mode not in ('update', 'file', 'reg'):
                                state.status = "Используйте: /selftest update|file|reg"
                            else:
                                _run_selftest(mode)
                            state.input_buffer = ""
                            state.input_caret = 0
                            state.history_scroll = 0
                            need_redraw = True
                            continue
                        elif cmd == '/setname':
                            try:
                                # Trim optional quotes
                                if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
                                    arg1 = arg[1:-1]
                                else:
                                    arg1 = arg
                                payload = make_profile_set_payload(arg1, None)
                                net.send(payload)
                                state.status = "Обновляем имя..."
                            except Exception as e:
                                state.status = f"Ошибка имени: {e}"
                        elif cmd == '/sethandle':
                            try:
                                h = normalize_handle(arg)
                                payload = make_profile_set_payload(None, h)
                                net.send(payload)
                                state.status = "Обновляем логин..."
                            except Exception as e:
                                state.status = f"Ошибка логина: {e}"
                        elif cmd == '/whois':
                            if arg:
                                if arg.startswith('@'):
                                    net.send({"type": "profile_get", "handle": arg})
                                else:
                                    net.send({"type": "profile_get", "id": arg})
                                state.status = f"Запрос профиля {arg}..."
                        elif cmd == '/file':
                            # Send a file to the selected contact/group/board
                            if not to or is_separator(to):
                                state.status = "Выберите контакт или чат для отправки файла"
                            else:
                                path = arg or ''
                                if not path:
                                    state.status = "Укажите путь к файлу"
                                else:
                                    meta = file_meta_for(path)
                                    if not meta:
                                        state.status = "Файл не найден или нет доступа"
                                    else:
                                        # Проверки перед отправкой (директ-сообщения): авторизация и блокировки
                                        try:
                                            blocked_union = set(state.blocked) | set(getattr(state, 'blocked_by', set()))
                                        except Exception:
                                            blocked_union = set()
                                        # Если выбран не group/board — это ЛС; требуется дружба и отсутствие блокировок
                                        is_room = bool(to in state.groups or (isinstance(to, str) and to.startswith('b-')))
                                        if not state.authed:
                                            state.status = "Вы не авторизованы"
                                            state.input_buffer = ""; state.input_caret = 0
                                            state.history_scroll = 0
                                            continue
                                        if (not is_room):
                                            is_friend = bool(state.friends.get(to) or (to in state.roster_friends))
                                            if (to in blocked_union) or (state.self_id in blocked_union):
                                                state.status = "Нельзя отправить файл: контакт в блокировке"
                                                state.input_buffer = ""; state.input_caret = 0
                                                state.history_scroll = 0
                                                continue
                                            if not is_friend:
                                                state.status = "Требуется авторизация: добавьте друг друга в контакты"
                                                state.input_buffer = ""; state.input_caret = 0
                                                state.history_scroll = 0
                                                continue
                                        # Гарантированно отключить любые модалки подтверждения/прогресса перед прямой отправкой
                                        try:
                                            state.file_confirm_mode = False
                                            state.file_confirm_path = None
                                            state.file_confirm_target = None
                                            state.file_confirm_text_full = ""
                                            state.file_confirm_index = 0
                                            state.file_confirm_prev_text = ""
                                            state.file_confirm_prev_caret = 0
                                            state.file_exists_mode = False
                                        except Exception:
                                            pass
                                        state.file_send_path = str(meta.path)
                                        state.file_send_name = meta.name
                                        state.file_send_size = meta.size
                                        # Determine target: room (group/board) vs direct
                                        if to in state.groups or (isinstance(to, str) and to.startswith('b-')):
                                            state.file_send_room = to
                                            state.file_send_to = None
                                            net.send({"type": getattr(T, 'FILE_OFFER', 'file_offer'), "room": to, "name": meta.name, "size": meta.size})
                                        else:
                                            state.file_send_room = None
                                            state.file_send_to = to
                                            net.send({"type": getattr(T, 'FILE_OFFER', 'file_offer'), "to": to, "name": meta.name, "size": meta.size})
                                        # Show in channel that we started sending
                                        try:
                                            chan = to
                                            nm = meta.name or 'файл'
                                            state.conversations.setdefault(chan, []).append(ChatMessage('out', f"Отправка файла [{nm}]…", time.time()))
                                        except Exception:
                                            pass
                                        state.status = f"Подготовка отправки: {meta.name} ({meta.size} байт)"
                            state.input_buffer = ""
                            try:
                                state.input_caret = 0
                            except Exception:
                                pass
                            state.history_scroll = 0
                            continue
                        elif cmd == '/ok' or cmd.startswith('/ok'):
                            # Accept latest pending file offer for selected channel
                            key = to or ''
                            if not key:
                                state.status = "Нет выбранного адресата"
                            else:
                                # Поддержка /ok<ID>
                                num = None
                                if cmd != '/ok':
                                    try:
                                        num = int(cmd[3:])
                                    except Exception:
                                        num = None
                                fid = None
                                if num is not None and num > 0:
                                    try:
                                        fid = (state.file_offer_map.get(key, {}) or {}).get(num)
                                    except Exception:
                                        fid = None
                                if not fid:
                                    # fallback — первый ожидающий оффер
                                    fids = list(state.incoming_by_peer.get(key, []))
                                    if not fids:
                                        state.status = "Нет входящих файлов"
                                    else:
                                        fid = fids.pop(0)
                                        state.incoming_by_peer[key] = fids
                                if fid:
                                    net.send({"type": getattr(T, 'FILE_ACCEPT', 'file_accept'), "file_id": fid})
                                    state.status = "Запрос на прием файла отправлен"
                            state.input_buffer = ""
                            try:
                                state.input_caret = 0
                            except Exception:
                                pass
                            state.history_scroll = 0
                            continue
                        else:
                            state.status = "Неизвестная команда"
                        state.input_buffer = ""
                        try:
                            state.input_caret = 0
                        except Exception:
                            pass
                        try:
                            state.input_caret = 0
                        except Exception:
                            pass
                        state.history_scroll = 0
                        continue
                    if not to or is_separator(to):
                        state.status = "Выберите контакт или чат"
                    elif to in state.groups or to in getattr(state, 'boards', {}):
                        net.send({"type": "send", "room": to, "text": text})
                        conv = state.conversations.setdefault(to, [])
                        ts = time.time()
                        conv.append(ChatMessage('out', text, ts, status='sent'))
                        _dbg(f"[send] room={to} text={text!r}")
                        try:
                            append_history_record({"id": None, "from": state.self_id, "text": text, "ts": ts, "room": to})
                        except Exception:
                            pass
                        logging.getLogger('client').debug("Msg to ROOM %s: %s", to, text)
                    else:
                        # Blocked peer: do not allow sending; show modal
                        if to in state.blocked:
                            state.modal_message = "Вы заблокировали этот аккаунт"
                            # keep input buffer (user can copy or decide later)
                            continue
                        if to in getattr(state, 'blocked_by', set()):
                            state.modal_message = "Аккаунт заблокировал вас"
                            # keep input buffer (user can copy or decide later)
                            continue
                        net.send({"type": "send", "to": to, "text": text})
                        _dbg(f"[send] to={to} text={text!r}")
                        conv = state.conversations.setdefault(to, [])
                        ts = time.time()
                        conv.append(ChatMessage('out', text, ts))
                        try:
                            append_history_record({"id": None, "from": state.self_id, "to": to, "text": text, "ts": ts})
                        except Exception:
                            pass
                        try:
                            _write_user_history_line(str(to), 'out', str(text), float(ts))
                        except Exception:
                            pass
                        logging.getLogger('client').debug("Msg to %s: %s", to, text)
                    # After any send, exit search-related overlays to avoid blanking the chat
                    try:
                        state.search_mode = False
                        state.search_action_mode = False
                        state.search_action_peer = None
                        state.search_action_options = []
                        state.search_action_step = 'choose'
                        state.search_query = ""
                        state.search_results = []
                        state.search_live_id = None
                        state.search_live_ok = False
                    except Exception:
                        pass
                state.input_buffer = ""
                try:
                    state.input_caret = 0
                except Exception:
                    pass
                try:
                    state.input_caret = 0
                except Exception:
                    pass
                state.history_scroll = 0
            elif ch in ('\x7f', '\b'):
                if state.action_menu_mode and state.action_menu_options:
                    # Treat backspace as ESC in menu
                    state.action_menu_mode = False
                    state.action_menu_peer = None
                    state.action_menu_options = []
                else:
                    try:
                        chat_input.backspace(state)
                    except Exception:
                        # Fallback: trim one char at end
                        state.input_buffer = state.input_buffer[:-1]
                        try:
                            state.input_caret = min(len(state.input_buffer), int(getattr(state, 'input_caret', len(state.input_buffer))))
                            state.input_sel_start = state.input_caret
                            state.input_sel_end = state.input_caret
                        except Exception:
                            pass
            elif ch in ('k', 'j') and state.vi_keys and (state.input_buffer == '') and (not state.profile_mode) and (not state.search_mode):
                # Special case: allow fast profile browsing when profile view is open
                if state.profile_view_mode:
                    if ch == 'k':
                        if state.selected_index > 0:
                            state.selected_index -= 1
                            clamp_selection(state, prefer='up')
                    else:  # 'j'
                        rows = build_contact_rows(state)
                        if state.selected_index < max(0, len(rows) - 1):
                            state.selected_index += 1
                            clamp_selection(state, prefer='down')
                    sel = current_selected_id(state)
                    if sel and (sel not in state.groups):
                        try:
                            net.send({"type": "profile_get", "id": sel})
                        except Exception:
                            pass
                        state.profile_view_id = sel
                    # Do not mark messages read in profile browsing
                    state.history_scroll = 0
                    continue
                # Do not move selection with vi-keys when other overlays/menus are active
                if state.action_menu_mode or state.search_action_mode or getattr(state, 'group_create_mode', False) or state.group_manage_mode or state.group_verify_mode:
                    continue
                if ch == 'k':
                    if state.selected_index > 0:
                        state.selected_index -= 1
                        clamp_selection(state)
                        # Inform server messages are read for selected peer
                        sel = current_selected_id(state)
                        _maybe_send_message_read(state, net, sel)
                        state.history_scroll = 0
                else:  # 'j'
                    rows = build_contact_rows(state)
                    if state.selected_index < max(0, len(rows) - 1):
                        state.selected_index += 1
                        clamp_selection(state)
                        sel = current_selected_id(state)
                        _maybe_send_message_read(state, net, sel)
                        state.history_scroll = 0
            else:
                # Append/insert unicode into input (supports Russian)
                if ch == '\n':
                    try:
                        chat_input.insert_newline(state)
                    except Exception:
                        pass
                elif ch in ('\x7f', '\b'):
                    # Backspace (string variant) — учитывать каретку
                    try:
                        chat_input.backspace(state)
                    except Exception:
                        pass
                elif ch == '\x01':  # Ctrl+A — move caret to start of line
                    try:
                        chat_input.move_line_start(state)
                    except Exception:
                        pass
                elif ch == '\x05':  # Ctrl+E — move caret to end of line
                    try:
                        chat_input.move_line_end(state)
                    except Exception:
                        pass
                elif ch == '\x0b':  # Ctrl+K — delete to end of line
                    try:
                        chat_input.delete_to_line_end(state)
                    except Exception:
                        pass
                elif ch.isprintable():
                    try:
                        chat_input.insert_text(state, ch)
                    except Exception:
                        pass
                    # Do not auto-open slash suggestions on typing '/'
        else:
            # ch is int (special keys)
            # Treat raw 27 as ESC (some terminals deliver ESC as int)
            if ch == 27:
                # Mirror the string ESC handling
                sel = None
                try:
                    sel = current_selected_id(state)
                except Exception:
                    pass
                overlays_active = bool(state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or state.help_mode)
                if sel and (sel in getattr(state, 'authz_out_pending', set())) and not overlays_active:
                    try:
                        net.send({"type": "authz_cancel", "peer": sel})
                    except Exception:
                        pass
                    try:
                        state.pending_out.discard(sel)
                    except Exception:
                        pass
                    try:
                        state.authz_out_pending.discard(sel)
                    except Exception:
                        pass
                    state.status = f"Отменён запрос: {sel}"
                    continue
                if state.action_menu_mode:
                    # If user dismissed incoming auth actions, don't auto-pop it back on navigation.
                    if _is_auth_actions_menu(state):
                        try:
                            peer = str(state.action_menu_peer or "").strip()
                            if peer:
                                state.authz_menu_snoozed.add(peer)
                        except Exception:
                            pass
                    state.action_menu_mode = False
                    state.action_menu_peer = None
                    state.action_menu_options = []
                    state.status = ""
                    continue
                if state.search_mode:
                    state.search_mode = False
                    state.search_query = ""
                    state.search_results = []
                    state.status = "Поиск закрыт"
                    continue
                if state.help_mode:
                    state.help_mode = False
                    state.status = ""
                    continue
                running = False
                break
            # If there is text in input and no modal overlays, arrow keys operate within input field
            _over = bool(
                state.search_action_mode or state.action_menu_mode or state.profile_mode or state.profile_view_mode or state.modal_message or state.help_mode or getattr(state, 'group_create_mode', False) or state.group_manage_mode
                or getattr(state, 'file_confirm_mode', False) or getattr(state, 'file_progress_mode', False)
                or getattr(state, 'board_invite_mode', False) or getattr(state, 'board_added_consent_mode', False)
                or getattr(state, 'board_member_add_mode', False) or getattr(state, 'board_member_remove_mode', False)
                or getattr(state, 'group_member_add_mode', False) or getattr(state, 'group_member_remove_mode', False)
            )
            if (not _over) and state.input_buffer:
                # Treat raw 127/8 ints as Backspace for terminals that don't map to KEY_BACKSPACE
                if ch in (127, 8):
                    try:
                        chat_input.backspace(state)
                    except Exception:
                        pass
                    continue
                if ch == curses.KEY_LEFT:
                    try:
                        chat_input.move_left(state)
                    except Exception:
                        pass
                    continue
                if ch == curses.KEY_RIGHT:
                    try:
                        chat_input.move_right(state)
                    except Exception:
                        pass
                    continue
                if ch == curses.KEY_UP:
                    try:
                        prev = int(getattr(state, 'input_caret', 0))
                        chat_input.move_up(state)
                        if int(getattr(state, 'input_caret', 0)) == prev:
                            # Browse history if cannot move up further
                            hist = list(getattr(state, 'input_history', []))
                            idx = int(getattr(state, 'input_history_index', -1))
                            if hist:
                                idx = len(hist) - 1 if idx == -1 else max(0, idx - 1)
                                state.input_history_index = idx
                                chat_input.set_text(state, hist[idx], caret_at_end=True)
                    except Exception:
                        pass
                    continue
                if ch == curses.KEY_DOWN:
                    try:
                        prev = int(getattr(state, 'input_caret', 0))
                        chat_input.move_down(state)
                        if int(getattr(state, 'input_caret', 0)) == prev:
                            # Browse history forward
                            hist = list(getattr(state, 'input_history', []))
                            idx = int(getattr(state, 'input_history_index', -1))
                            if hist and idx != -1:
                                if idx < len(hist) - 1:
                                    idx += 1
                                    chat_input.set_text(state, hist[idx], caret_at_end=True)
                                    state.input_history_index = idx
                                else:
                                    # Exit history mode
                                    chat_input.set_text(state, '', caret_at_end=False)
                                    state.input_history_index = -1
                    except Exception:
                        pass
                    continue
                if ch == curses.KEY_DC:  # Delete key
                    try:
                        chat_input.delete_forward(state)
                    except Exception:
                        pass
                    continue
                if ch == getattr(curses, 'KEY_HOME', -9999):
                    try:
                        chat_input.move_line_start(state)
                    except Exception:
                        pass
                    continue
                if ch == getattr(curses, 'KEY_END', -9999):
                    try:
                        chat_input.move_line_end(state)
                    except Exception:
                        pass
                    continue
            if ch == curses.KEY_UP:
                # In actions menu: arrow up selects previous option (except for authorization menu — Tab-only)
                if state.action_menu_mode and state.action_menu_options and not _is_auth_actions_menu(state):
                    try:
                        _dbg(f"[key UP] action_menu idx={state.action_menu_index}")
                        if state.action_menu_index > 0:
                            state.action_menu_index -= 1
                    except Exception:
                        pass
                    continue
            elif ch == curses.KEY_DOWN:
                # In actions menu: arrow down selects next option (except for authorization menu — Tab-only)
                if state.action_menu_mode and state.action_menu_options and not _is_auth_actions_menu(state):
                    try:
                        _dbg(f"[key DOWN] action_menu idx={state.action_menu_index}")
                        if state.action_menu_index < max(0, len(state.action_menu_options) - 1):
                            state.action_menu_index += 1
                    except Exception:
                        pass
                    continue
            elif ch == curses.KEY_F1:
                # Toggle help overlay
                _dbg("[key F1] toggle help")
                state.help_mode = not state.help_mode
                state.status = "Справка открыта" if state.help_mode else ""
                continue
            # Shift-Tab cycles backwards ONLY for authorization menu
            if state.action_menu_mode and state.action_menu_options and ch == curses.KEY_BTAB and _is_auth_actions_menu(state):
                try:
                    n = len(state.action_menu_options)
                    if n > 0:
                        state.action_menu_index = (state.action_menu_index - 1) % n
                except Exception:
                    pass
                continue
            # Shift+Enter (if terminal exposes it as KEY_SENTER)
            if ch == getattr(curses, 'KEY_SENTER', -9999):
                try:
                    chat_input.insert_newline(state)
                except Exception:
                    pass
                continue
            if ch == curses.KEY_F2:
                _dbg("[key F2] profile toggle")
                if state.profile_mode:
                    state.profile_mode = False
                    state.status = "Профиль закрыт"
                else:
                    state.profile_mode = True
                    state.profile_field = 0
                    state.status = "Профиль: редактирование"
                    state.profile_name_input = ""
                    state.profile_handle_input = ""
                    try:
                        net.send({"type": "profile_get"})
                    except Exception:
                        pass
                continue
            if ch == curses.KEY_F3:
                # Enter search mode
                _dbg("[key F3] search mode")
                try:
                    state.profile_mode = False
                    state.profile_view_mode = False
                    state.action_menu_mode = False
                    state.search_action_mode = False
                    state.modal_message = None
                except Exception:
                    pass
                state.search_mode = True
                state.search_query = ""
                state.search_results = []
                state.search_live_id = None
                state.search_live_ok = False
                state.selected_index = 0
                state.status = "Режим поиска: введите ID/@логин и Enter"
                continue
            if ch == curses.KEY_F5:
                # Open group create modal
                _dbg("[key F5] group create modal")
                try:
                    # Close conflicting overlays
                    state.search_action_mode = False
                    state.search_action_peer = None
                    state.search_action_options = []
                    state.search_mode = False
                    state.profile_view_mode = False
                    state.profile_mode = False
                    state.modal_message = None
                    state.group_create_mode = True  # type: ignore[attr-defined]
                    state.group_create_field = 0    # type: ignore[attr-defined]
                    state.group_name_input = getattr(state, 'group_name_input', '') or ""  # type: ignore[attr-defined]
                    state.group_members_input = getattr(state, 'group_members_input', '') or ""  # type: ignore[attr-defined]
                    state.status = "Создание чата: введите имя и участников"
                except Exception:
                    pass
                continue
            if ch == curses.KEY_F6:
                # Open board create modal
                _dbg("[key F6] board create modal")
                try:
                    state.search_action_mode = False
                    state.search_action_peer = None
                    state.search_action_options = []
                    state.search_mode = False
                    state.profile_view_mode = False
                    state.profile_mode = False
                    state.modal_message = None
                    state.board_create_mode = True
                    state.board_create_field = 0
                    state.board_name_input = getattr(state, 'board_name_input', '') or ""
                    state.board_handle_input = getattr(state, 'board_handle_input', '') or ""
                    state.status = "Создание доски: введите имя и логин"
                except Exception:
                    pass
                continue
            # Normalize function key escape sequences for terminals that don't map to curses.KEY_Fx
            try:
                if isinstance(ch, str):
                    # Common xterm sequences for F-keys
                    if ch in ('\x1b[18~',):  # F7
                        _dbg("[key F7 seq] normalize to KEY_F7"); ch = curses.KEY_F7
                    elif ch in ('\x1b[15~',):  # F5
                        _dbg("[key F5 seq] normalize to KEY_F5"); ch = curses.KEY_F5
                    elif ch in ('\x1b[17~',):  # F6
                        _dbg("[key F6 seq] normalize to KEY_F6"); ch = curses.KEY_F6
                    elif ch in ('\x1b[24~',):  # F12
                        try:
                            ch = curses.KEY_F12
                        except Exception:
                            pass
            except Exception:
                pass
            if ch == curses.KEY_F7:
                # Open two-pane file browser modal (isolated initializer)
                _dbg("[key F7] file browser")
                try:
                    start_file_browser(state)  # type: ignore[name-defined]
                except Exception:
                    close_file_browser(state)  # type: ignore[name-defined]
                continue
            # F12 — toggle debug overlay
            try:
                if ch == getattr(curses, 'KEY_F12', 277) or (isinstance(ch, str) and ch == '\x1b[24~'):
                    state.debug_mode = not getattr(state, 'debug_mode', False)
                    try:
                        state.history_dirty = True
                        state._hist_draw_state = None
                    except Exception:
                        pass
                    if state.debug_mode:
                        # Добавим строку, чтобы сразу было видно в логе
                        try:
                            state.debug_lines.append("DEBUG включён")
                            if len(state.debug_lines) > 300:
                                del state.debug_lines[:len(state.debug_lines) - 300]
                        except Exception:
                            pass
                    state.status = 'DEBUG: ON' if state.debug_mode else ''
                    # Re-arm mouse tracking after overlay toggles (some terminals reset modes on CSI sequences).
                    try:
                        _apply_mouse(bool(getattr(state, 'mouse_enabled', True)))
                    except Exception:
                        pass
                    try:
                        if bool(getattr(state, 'mouse_enabled', False)):
                            _maybe_enable_tmux_mouse(state)
                    except Exception:
                        pass
                    # Drop any partial ESC/mouse sequences that could corrupt subsequent input handling.
                    try:
                        curses.flushinp()
                    except Exception:
                        pass
                    continue
            except Exception:
                pass
            # Handle raw mouse clicks (SGR 1006 / URXVT 1015 / X10) for terminals where curses.KEY_MOUSE is flaky.
            if isinstance(ch, str) and bool(getattr(state, 'mouse_enabled', False)):
                parsed = _parse_sgr_mouse(ch)
                if parsed:
                    mx, my, bstate = parsed
                    # Header hotkeys
                    if my == 0:
                        if handle_hotkey_click(mx, my):
                            continue
                        continue
                    # Safety: wheel can still arrive here on some terminals; handle it.
                    try:
                        wheel_up, wheel_down = _compute_wheel_masks()
                        direction = _wheel_direction_from_bstate(bstate, wheel_up, wheel_down)
                        if direction:
                            _handle_wheel_scroll(state, stdscr, mx, my, direction)
                            need_redraw = True
                            continue
                    except Exception:
                        pass
                    try:
                        try:
                            h, w = stdscr.getmaxyx()
                            left_w = int(getattr(state, 'last_left_w', 0) or 0) or max(20, min(30, w // 4))
                        except Exception:
                            left_w = 20
                        if mx < left_w:
                            try:
                                dbg = bool(getattr(state, 'debug_mode', False))
                            except Exception:
                                dbg = False
                            btn1_any = (
                                getattr(curses, 'BUTTON1_PRESSED', 0)
                                | getattr(curses, 'BUTTON1_CLICKED', 0)
                                | getattr(curses, 'BUTTON1_RELEASED', 0)
                                | getattr(curses, 'BUTTON1_DOUBLE_CLICKED', 0)
                            )
                            btn3_any = (
                                getattr(curses, 'BUTTON3_PRESSED', 0)
                                | getattr(curses, 'BUTTON3_CLICKED', 0)
                                | getattr(curses, 'BUTTON3_RELEASED', 0)
                                | getattr(curses, 'BUTTON3_DOUBLE_CLICKED', 0)
                            )
                            if bstate & btn3_any:
                                open_actions_menu_for_selection(state, net)
                                try:
                                    if dbg:
                                        state.debug_lines.append(f"[mouse] right_click x={mx} y={my} bstate={bstate}")
                                        if len(state.debug_lines) > 300:
                                            del state.debug_lines[:len(state.debug_lines) - 300]
                                except Exception:
                                    pass
                                continue
                            if bstate & btn1_any:
                                rows = build_contact_rows(state)
                                idx = None
                                tok = None
                                # Prefer draw_ui's hit-map (screen Y -> row index) for correctness.
                                try:
                                    ymap = getattr(state, 'contacts_y_map', None)
                                    if isinstance(ymap, dict):
                                        idx = ymap.get(int(my))
                                except Exception:
                                    idx = None
                                # Fallback: derive from scroll window if hit-map absent.
                                cs = 0
                                vis_h = 0
                                max_rows2 = 0
                                if idx is None:
                                    start_y = 2  # contacts_start_y in draw_ui
                                    try:
                                        h, w = stdscr.getmaxyx()
                                        vis_h = max(0, h - start_y - 2)
                                    except Exception:
                                        vis_h = int(getattr(state, 'last_left_h', 10)) or 10
                                    cs = max(0, int(getattr(state, 'contacts_scroll', 0)))
                                    max_rows2 = min(max(0, len(rows) - cs), vis_h)
                                    if start_y <= my < start_y + max_rows2:
                                        idx = cs + (my - start_y)
                                try:
                                    if dbg:
                                        state.debug_lines.append(
                                            f"[mouse] btn1 x={mx} y={my} bstate={bstate} idx={idx} cs={cs} rows={len(rows)} vis_h={vis_h} max_rows={max_rows2}"
                                        )
                                        if len(state.debug_lines) > 300:
                                            del state.debug_lines[:len(state.debug_lines) - 300]
                                except Exception:
                                    pass
                                if isinstance(idx, int) and 0 <= idx < len(rows):
                                    try:
                                        tok = rows[idx]
                                    except Exception:
                                        tok = None
                                    state.selected_index = int(idx)
                                    clamp_selection(state, prefer='down', rows=rows)
                                    sel = current_selected_id(state, rows=rows)
                                    _maybe_send_message_read(state, net, sel)
                                    state.history_scroll = 0
                                    try:
                                        if dbg:
                                            state.debug_lines.append(f"[mouse] hit idx={idx} tok={repr(tok)}")
                                            state.debug_lines.append(f"[mouse] select {sel} idx={idx} x={mx} y={my}")
                                            if len(state.debug_lines) > 300:
                                                del state.debug_lines[:len(state.debug_lines) - 300]
                                    except Exception:
                                        pass
                                else:
                                    try:
                                        if dbg:
                                            state.debug_lines.append(f"[mouse] btn1 miss y={my} idx={idx}")
                                            if len(state.debug_lines) > 300:
                                                del state.debug_lines[:len(state.debug_lines) - 300]
                                    except Exception:
                                        pass
                                need_redraw = True
                                continue
                    except Exception as e:
                        try:
                            if bool(getattr(state, 'debug_mode', False)):
                                state.debug_lines.append(f"[mouse] ERROR {type(e).__name__}: {e}")
                                if len(state.debug_lines) > 300:
                                    del state.debug_lines[:len(state.debug_lines) - 300]
                        except Exception:
                            pass
                    continue

            if ch == curses.KEY_MOUSE:
                try:
                    _mid, mx, my, _mz, bstate = curses.getmouse()
                except Exception:
                    continue
                try:
                    if getattr(state, 'debug_mode', False):
                        state.debug_last_mouse = f"KEY_MOUSE x={mx} y={my} bstate={bstate}"
                        try:
                            state.debug_lines.append(state.debug_last_mouse)
                            if len(state.debug_lines) > 300:
                                del state.debug_lines[:len(state.debug_lines) - 300]
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    state.mouse_events_total = int(getattr(state, 'mouse_events_total', 0)) + 1
                    state.mouse_last_seen_ts = time.time()
                except Exception:
                    pass
                # Mouse wheel scroll
                wheel_up, wheel_down = _compute_wheel_masks()
                direction = _wheel_direction_from_bstate(bstate, wheel_up, wheel_down)
                if direction:
                    _handle_wheel_scroll(state, stdscr, mx, my, direction)
                    need_redraw = True
                    continue
                # Left-pane (contacts): right-click opens menu; left-click selects
                try:
                    try:
                        h, w = stdscr.getmaxyx()
                        left_w = int(getattr(state, 'last_left_w', 0) or 0) or max(20, min(30, w // 4))
                    except Exception:
                        left_w = 20
                    in_left = (mx < left_w)
                    if in_left:
                        # Treat any left-button event as a click for selection purposes
                        btn1_any = (
                            getattr(curses, 'BUTTON1_PRESSED', 0)
                            | getattr(curses, 'BUTTON1_CLICKED', 0)
                            | getattr(curses, 'BUTTON1_RELEASED', 0)
                            | getattr(curses, 'BUTTON1_DOUBLE_CLICKED', 0)
                        )
                        btn3_any = (
                            getattr(curses, 'BUTTON3_PRESSED', 0)
                            | getattr(curses, 'BUTTON3_CLICKED', 0)
                            | getattr(curses, 'BUTTON3_RELEASED', 0)
                            | getattr(curses, 'BUTTON3_DOUBLE_CLICKED', 0)
                        )
                        if bstate & btn3_any:
                            open_actions_menu_for_selection(state, net)
                            continue
                        if bstate & btn1_any:
                            rows = build_contact_rows(state)
                            start_y = 2  # contacts_start_y in draw_ui
                            try:
                                h, w = stdscr.getmaxyx()
                                vis_h = max(0, h - start_y - 2)
                            except Exception:
                                vis_h = int(getattr(state, 'last_left_h', 10)) or 10
                            cs = max(0, int(getattr(state, 'contacts_scroll', 0)))
                            max_rows = min(max(0, len(rows) - cs), vis_h)
                            if start_y <= my < start_y + max_rows:
                                local_idx = my - start_y
                                idx = cs + local_idx
                                if 0 <= idx < len(rows):
                                    state.selected_index = idx
                                    clamp_selection(state, prefer='down')
                                    sel = current_selected_id(state)
                                    _maybe_send_message_read(state, net, sel)
                                    state.history_scroll = 0
                            # Do not fall through to history selection for left-pane clicks
                            continue
                except Exception:
                    pass
                # Compute geometry for panels
                h, w = stdscr.getmaxyx()
                left_w = int(getattr(state, 'last_left_w', 0) or 0) or max(20, min(30, w // 4))
                input_h = 3
                history_h = h - input_h - 2
                hist_y = 2
                hist_x = left_w + 2
                hist_w = w - hist_x - 2
                hist_h = max(1, history_h)
                # Selection with left button in history area
                btn1_press = (
                    getattr(curses, 'BUTTON1_PRESSED', 0)
                    | getattr(curses, 'BUTTON1_CLICKED', 0)
                    | getattr(curses, 'BUTTON1_DOUBLE_CLICKED', 0)
                )
                # Some terminals (notably macOS Terminal) may send CLICKED/DOUBLE without RELEASED
                btn1_release = (
                    getattr(curses, 'BUTTON1_RELEASED', 0)
                    | getattr(curses, 'BUTTON1_CLICKED', 0)
                    | getattr(curses, 'BUTTON1_DOUBLE_CLICKED', 0)
                    | getattr(curses, 'BUTTON1_TRIPLE_CLICKED', 0)
                )
                if hist_y <= my < hist_y + hist_h and hist_x <= mx < hist_x + hist_w:
                    # Activate chat window on any left-click event inside history area
                    try:
                        sel = current_selected_id(state)
                        _maybe_send_message_read(state, net, sel)
                    except Exception:
                        pass
                    # Handle word/line quick-copy via double/triple click first
                    is_double = bool(bstate & getattr(curses, 'BUTTON1_DOUBLE_CLICKED', 0))
                    is_triple = bool(bstate & getattr(curses, 'BUTTON1_TRIPLE_CLICKED', 0))
                    if is_triple:
                        # Copy full visible line
                        idx = my - hist_y
                        if 0 <= idx < len(state.last_lines):
                            _line_obj = state.last_lines[idx]
                            line = ("".join(_line_obj) if isinstance(_line_obj, list) else str(_line_obj)).rstrip()
                            ok = copy_to_clipboard(line)
                            state.status = "Строка скопирована" if (ok and line) else "Копирование не удалось"
                        state.select_active = False
                        continue
                    if is_double:
                        # Copy word under cursor (letters/digits/_/@/#/.-)
                        idx = my - hist_y
                        if 0 <= idx < len(state.last_lines):
                            _line_obj = state.last_lines[idx]
                            line = ("".join(_line_obj) if isinstance(_line_obj, list) else str(_line_obj)).rstrip('\n')
                            col = max(0, min(hist_w - 1, mx - hist_x))
                            # Clamp to real line length
                            col = min(col, len(line) - 1) if line else 0
                            def is_word_char(ch: str) -> bool:
                                return ch.isalnum() or ch in "_@#.-"
                            if line:
                                l = r = col
                                while l > 0 and is_word_char(line[l - 1]):
                                    l -= 1
                                while r < len(line) and is_word_char(line[r]):
                                    r += 1
                                word = line[l:r].strip()
                            else:
                                word = ""
                            ok = copy_to_clipboard(word)
                            state.status = "Слово скопировано" if (ok and word) else "Копирование не удалось"
                        state.select_active = False
                        continue
                    # Single click on history line: copy full line (user-friendly default)
                    is_single = bool(bstate & getattr(curses, 'BUTTON1_CLICKED', 0)) and not state.select_active
                    if is_single:
                        idx = my - hist_y
                        if 0 <= idx < len(state.last_lines):
                            _line_obj = state.last_lines[idx]
                            line = ("".join(_line_obj) if isinstance(_line_obj, list) else str(_line_obj)).rstrip()
                            ok = copy_to_clipboard(line)
                            state.status = "Строка скопирована" if (ok and line) else "Копирование не удалось"
                        state.select_active = False
                        continue
                # Only react to left button click-like events
                btnmask = 0
                for name in ("BUTTON1_CLICKED", "BUTTON1_PRESSED", "BUTTON1_RELEASED", "BUTTON1_DOUBLE_CLICKED"):
                    btnmask |= getattr(curses, name, 0)
                if not (bstate & btnmask):
                    continue
                # Map to contacts region (align with draw: contacts_start_y=2)
                h, w = stdscr.getmaxyx()
                left_w = int(getattr(state, 'last_left_w', 0) or 0) or max(20, min(30, w // 4))
                start_y = 2
                rows = build_contact_rows(state)
                # Visible height of contacts pane: reserve 2 last rows (separator + footer)
                vis_h = max(0, h - start_y - 2)
                cs = max(0, int(getattr(state, 'contacts_scroll', 0)))
                max_rows = min(max(0, len(rows) - cs), vis_h)
                if 0 <= mx < left_w and start_y <= my < start_y + max_rows:
                    local_idx = my - start_y
                    idx = cs + local_idx
                    if 0 <= idx < len(rows):
                        state.selected_index = idx
                        clamp_selection(state, prefer='down')
                        # Keep selection fully visible and avoid auto-jump to top
                        try:
                            if state.selected_index < cs:
                                cs = state.selected_index
                            elif state.selected_index >= cs + max(1, vis_h):
                                cs = state.selected_index - max(1, vis_h) + 1
                            # Avoid ending on a separator if there are items below
                            end_idx = min(len(rows) - 1, cs + max(1, vis_h) - 1)
                            if end_idx < (len(rows) - 1) and is_separator(rows[end_idx]):
                                cs = min(cs + 1, max(0, len(rows) - max(1, vis_h)))
                            state.contacts_scroll = max(0, min(cs, max(0, len(rows) - max(1, vis_h))))
                        except Exception:
                            pass
                        sel = current_selected_id(state)
                        # Авто‑меню для входящих запросов при клике по элементу
                        state.action_menu_mode = False
                        state.action_menu_peer = None
                        state.action_menu_options = []
                        _maybe_open_authz_actions_menu(state, sel)
                        _maybe_send_message_read(state, net, sel)
                        state.history_scroll = 0
                continue
            if ch == curses.KEY_RIGHT and not state.action_menu_mode:
                # Open read-only profile card for selected contact
                # Close other overlays
                state.search_action_mode = False
                state.modal_message = None
                state.help_mode = False
                rows = build_contact_rows(state)
                if rows:
                    sel = rows[state.selected_index]
                    if is_separator(sel):
                        pass
                    elif sel in state.groups:
                        # Open group manage modal
                        state.group_manage_mode = True
                        state.group_manage_gid = sel
                        state.group_manage_field = 0
                        g = state.groups.get(sel) or {}
                        state.group_manage_name_input = str(g.get('name') or '')
                        try:
                            state.group_manage_handle_input = str(g.get('handle') or '')
                        except Exception:
                            state.group_manage_handle_input = ''
                        state.group_manage_member_count = 0
                        state.status = f"Управление чатом: {state.group_manage_name_input or sel}"
                        try:
                            net.send({"type": "group_info", "group_id": sel})
                        except Exception:
                            pass
                    elif sel in getattr(state, 'boards', {}):
                        # Open board manage modal
                        try:
                            state.board_manage_mode = True
                            state.board_manage_bid = sel
                            state.board_manage_field = 0
                            b = (getattr(state, 'boards', {}) or {}).get(sel) or {}
                            state.board_manage_name_input = str(b.get('name') or '')
                            try:
                                state.board_manage_handle_input = str(b.get('handle') or '')
                            except Exception:
                                state.board_manage_handle_input = ''
                            state.board_manage_member_count = 0
                            state.status = f"Управление доской: {state.board_manage_name_input or sel}"
                            try:
                                net.send({"type": "board_info", "board_id": sel})
                            except Exception:
                                pass
                        except Exception:
                            pass
                    else:
                        state.profile_view_mode = True
                        state.profile_view_id = sel
                        state.status = f"Профиль пользователя {sel}"
                        try:
                            net.send({"type": "profile_get", "id": sel})
                        except Exception:
                            pass
                continue
            if ch == curses.KEY_LEFT:
                # Open actions menu for selected contact
                # Close other overlays
                state.search_action_mode = False
                state.profile_view_mode = False
                state.profile_mode = False
                state.modal_message = None
                state.help_mode = False
                sel = current_selected_id(state)
                if sel and isinstance(sel, str) and sel.startswith('GINV:'):
                    try:
                        gid = sel.split(':', 1)[1]
                    except Exception:
                        gid = sel
                    meta = (getattr(state, 'group_pending_invites', {}) or {}).get(gid, {})
                    state.group_invite_mode = True
                    state.group_invite_gid = gid
                    state.group_invite_name = str(meta.get('name') or gid)
                    state.group_invite_from = str(meta.get('from') or '')
                    state.group_invite_index = 0
                    continue
                if sel and isinstance(sel, str) and sel.startswith('BINV:'):
                    try:
                        bid = sel.split(':', 1)[1]
                    except Exception:
                        bid = sel
                    meta = (getattr(state, 'board_pending_invites', {}) or {}).get(bid, {})
                    state.board_invite_mode = True
                    state.board_invite_bid = bid
                    state.board_invite_name = str(meta.get('name') or bid)
                    state.board_invite_from = str(meta.get('from') or '')
                    state.board_invite_index = 0
                    continue
                # If selection is on a separator (None) and there are pending requests only, pick first pending for menu
                if (sel is None) and state.pending_requests:
                    try:
                        rows = build_contact_rows(state)
                        # Find first non-separator pending
                        for tok in rows:
                            if (not is_separator(tok)) and (tok in state.pending_requests):
                                sel = tok
                                break
                    except Exception:
                        pass
                if sel and isinstance(sel, str) and sel.startswith('JOIN:'):
                    # Actions for group join request token JOIN:<gid>:<uid>
                    try:
                        _, gid, rid = sel.split(':', 2)
                    except Exception:
                        gid, rid = '', ''
                    options = ["Принять в чат", "Отклонить", "Профиль пользователя", "Профиль чата"]
                    state.action_menu_mode = True
                    state.action_menu_peer = sel
                    state.action_menu_options = options
                    state.action_menu_index = 0
                elif sel and (sel in state.groups):
                    g = state.groups.get(sel) or {}
                    is_owner = (str(g.get('owner_id') or '') == str(state.self_id or ''))
                    options: List[str] = []
                    # Разрешить «Отправить файл» для участников чата (сервер всё равно авторитетен)
                    try:
                        members = set(g.get('members') or [])
                        is_member = (str(state.self_id or '') in members) if members else True
                    except Exception:
                        is_member = True
                    if is_member:
                        options.append("Отправить файл")
                    if is_owner:
                        options.append("Добавить участника")
                        # Показывать "Удалить участника" сразу, не ожидая group_info,
                        # чтобы меню было предсказуемым после запуска клиента.
                        options.append("Удалить участника")
                        options.append("Рассформировать чат")
                        try:
                            net.send({"type": "group_info", "group_id": sel})
                        except Exception:
                            pass
                    else:
                        options.append("Профиль чата")
                    # Toggle notifications for this chat
                    try:
                        muted = sel in getattr(state, 'group_muted', set())
                        options.append("Включить уведомления" if muted else "Заглушить чат")
                    except Exception:
                        options.append("Заглушить чат")
                    # Allow non-владелец to leave the chat
                    options.append("Покинуть чат")
                    if options:
                        state.action_menu_mode = True
                        state.action_menu_peer = sel
                        state.action_menu_options = options
                        state.action_menu_index = 0
                elif sel and (sel in getattr(state, 'boards', {})):
                    # Unified Board actions menu
                    b = (getattr(state, 'boards', {}) or {}).get(sel) or {}
                    is_owner = (str(b.get('owner_id') or '') == str(state.self_id or ''))
                    options: List[str] = []
                    # На доске файлы может отправлять только владелец
                    if is_owner:
                        options.append("Отправить файл")
                    # Toggle notifications for this board (local mute) — not shown for owner
                    if not is_owner:
                        try:
                            muted = sel in getattr(state, 'board_muted', set())
                            options.append("Включить уведомления" if muted else "Заглушить доску")
                        except Exception:
                            options.append("Заглушить доску")
                    if is_owner:
                        options.append("Добавить участника")
                        options.append("Удалить участника")
                        options.append("Рассформировать доску")
                    else:
                        options.append("Покинуть доску")
                    if options:
                        state.action_menu_mode = True
                        state.action_menu_peer = sel
                        state.action_menu_options = options
                        state.action_menu_index = 0
                elif sel and (sel not in state.groups) and (sel not in getattr(state, 'boards', {})):
                    options: List[str] = []
                    try:
                        is_friend = (sel in state.friends) or (sel in state.roster_friends)
                    except Exception:
                        is_friend = (sel in state.friends)
                    try:
                        is_block_ctx = (sel in state.blocked) or (sel in getattr(state, 'blocked_by', set()))
                    except Exception:
                        is_block_ctx = False
                    # Для ЛС: если дружба есть и нет блокировок — показываем «Отправить файл»
                    try:
                        if (state.authed) and is_friend and (not is_block_ctx):
                            options.append("Отправить файл")
                    except Exception:
                        pass
                    if is_friend or is_block_ctx:
                        options.append("Удалить из контактов")
                        options.append("Очистить чат")
                        # Мьют дополнительно для ЛС
                        options.append("Снять заглушку" if sel in state.muted else "Заглушить")
                        options.append("Разблокировать" if sel in state.blocked else "Заблокировать")
                    elif sel in state.pending_requests:
                        options.append("Авторизовать")
                        options.append("Отклонить")
                        options.append("Заблокировать")
                        options.append("Профиль пользователя")
                    elif sel in state.pending_out:
                        options.append("Отменить запрос")
                        options.append("Заблокировать")
                    else:
                        options.append("Запросить авторизацию")
                    state.action_menu_mode = True
                    state.action_menu_peer = sel
                    state.action_menu_options = options
                    state.action_menu_index = 0
                # (removed redundant legacy group/board branches)
                continue
            if state.profile_view_mode and ch in (curses.KEY_LEFT, curses.KEY_RIGHT):
                state.profile_view_mode = False
                state.profile_view_id = None
                continue
            if ch == curses.KEY_UP:
                if state.action_menu_mode and state.action_menu_options:
                    if _is_auth_actions_menu(state):
                        continue
                    try:
                        n = len(state.action_menu_options)
                        if n > 0:
                            state.action_menu_index = (state.action_menu_index - 1) % n
                    except Exception:
                        pass
                    continue
                if state.profile_view_mode:
                    if state.selected_index > 0:
                        state.selected_index -= 1
                        clamp_selection(state, prefer='up')
                        sel = current_selected_id(state)
                        if sel and (sel not in state.groups):
                            try:
                                net.send({"type": "profile_get", "id": sel})
                            except Exception:
                                pass
                            state.profile_view_id = sel
                    continue
                if (not (state.action_menu_mode and state.action_menu_options)) and (state.search_action_mode or state.profile_mode or getattr(state, 'group_create_mode', False) or state.group_manage_mode or state.group_verify_mode):
                    continue
                if state.selected_index > 0:
                    state.selected_index -= 1
                    clamp_selection(state, prefer='up')
                    state.select_active = False
                    state.sel_anchor_y = state.sel_anchor_x = state.sel_cur_y = state.sel_cur_x = -1
                    sel = current_selected_id(state)
                    state.action_menu_mode = False
                    state.action_menu_peer = None
                    state.action_menu_options = []
                    if sel and isinstance(sel, str) and sel.startswith('JOIN:'):
                        state.action_menu_mode = True
                        state.action_menu_peer = sel
                        state.action_menu_options = ["Принять в чат", "Отклонить", "Профиль пользователя", "Профиль чата"]
                        state.action_menu_index = 0
                    _maybe_open_authz_actions_menu(state, sel)
                    _maybe_send_message_read(state, net, sel)
                    try:
                        request_history_if_needed(state, net, sel)
                    except Exception:
                        pass
                    state.history_scroll = 0
            elif ch == curses.KEY_DOWN:
                if state.action_menu_mode and state.action_menu_options:
                    if _is_auth_actions_menu(state):
                        continue
                    try:
                        n = len(state.action_menu_options)
                        if n > 0:
                            state.action_menu_index = (state.action_menu_index + 1) % n
                    except Exception:
                        pass
                    continue
                if state.profile_view_mode:
                    rows = build_contact_rows(state)
                    if state.selected_index < max(0, len(rows) - 1):
                        state.selected_index += 1
                        clamp_selection(state, prefer='down')
                        sel = current_selected_id(state)
                        if sel and (sel not in state.groups):
                            try:
                                net.send({"type": "profile_get", "id": sel})
                            except Exception:
                                pass
                            state.profile_view_id = sel
                    continue
                if (not (state.action_menu_mode and state.action_menu_options)) and (state.search_action_mode or state.profile_mode or getattr(state, 'group_create_mode', False) or state.group_manage_mode or state.group_verify_mode):
                    continue
                rows = build_contact_rows(state)
                if state.selected_index < max(0, len(rows) - 1):
                    state.selected_index += 1
                    clamp_selection(state, prefer='down')
                    state.select_active = False
                    state.sel_anchor_y = state.sel_anchor_x = state.sel_cur_y = state.sel_cur_x = -1
                    sel = current_selected_id(state)
                    state.action_menu_mode = False
                    state.action_menu_peer = None
                    state.action_menu_options = []
                    if sel and isinstance(sel, str) and sel.startswith('JOIN:'):
                        state.action_menu_mode = True
                        state.action_menu_peer = sel
                        state.action_menu_options = ["Принять в чат", "Отклонить", "Профиль пользователя", "Профиль чата"]
                        state.action_menu_index = 0
                    _maybe_open_authz_actions_menu(state, sel)
                    _maybe_send_message_read(state, net, sel)
                    try:
                        request_history_if_needed(state, net, sel)
                    except Exception:
                        pass
                    state.history_scroll = 0
            elif ch == curses.KEY_PPAGE:  # PageUp
                page = max(1, int(getattr(state, 'last_hist_h', 3)) - 1)
                state.history_scroll += page
                _clamp_history_scroll(state)
            elif ch == curses.KEY_NPAGE:  # PageDown
                page = max(1, int(getattr(state, 'last_hist_h', 3)) - 1)
                state.history_scroll = max(0, state.history_scroll - page)
                _clamp_history_scroll(state)
            elif ch == getattr(curses, 'KEY_SR', 259):  # Scroll wheel up (ncurses alias)
                try:
                    step = max(1, int(getattr(state, 'last_hist_h', 3)) // 2)
                except Exception:
                    step = 3
                state.history_scroll += step
                _clamp_history_scroll(state)
            elif ch == getattr(curses, 'KEY_SF', 262):  # Scroll wheel down (ncurses alias)
                try:
                    step = max(1, int(getattr(state, 'last_hist_h', 3)) // 2)
                except Exception:
                    step = 3
                state.history_scroll = max(0, state.history_scroll - step)
                _clamp_history_scroll(state)
            elif ch in (curses.KEY_BACKSPACE,):
                try:
                    chat_input.backspace(state)
                except Exception:
                    pass
            elif ch == 10:
                # Ctrl+J (LF) — вставить перевод строки
                try:
                    chat_input.insert_newline(state)
                except Exception:
                    pass
            elif ch in (curses.KEY_ENTER, 13):
                to = current_selected_id(state)
                text = state.input_buffer.strip()
                # Быстрый путь: если выбрана строка приглашения доски и нет текста — открыть модалку
                if (not text) and to and isinstance(to, str) and to.startswith('BINV:') and not getattr(state, 'board_invite_mode', False):
                    try:
                        bid = to.split(':', 1)[1]
                        meta = (getattr(state, 'board_pending_invites', {}) or {}).get(bid, {})
                        state.board_invite_mode = True
                        state.board_invite_bid = bid
                        state.board_invite_name = str(meta.get('name') or bid)
                        state.board_invite_from = str(meta.get('from') or '')
                        state.board_invite_index = 0
                        continue
                    except Exception:
                        pass
                if text:
                    # If the whole input looks like a path — предложить прикрепление как файла
                    try:
                        cand = extract_path_candidate(text)
                    except Exception:
                        cand = None
                    # Доп. грубый детектор: если модуль не распознал, попробуем найти последний '/path.ext'
                    if not cand and ('/' in text or '\\' in text):
                        try:
                            import re as _re
                            # Сначала пробуем общий путь без привязки к расширению
                            m = _re.findall(r"(/[^\s]+)", text)
                            if m:
                                # Возьмём последний видимый путь и обрежем хвостовую пунктуацию/галочки
                                t = m[-1].strip()
                                for ch in ['\"', "'", '»', '”', '’', '✓', '…', ',', '.', ';', ':', '!', '?', ')', ']', '}']:
                                    if t.endswith(ch):
                                        t = t[:-1].strip()
                                cand = t
                        except Exception:
                            cand = None
                    if cand and (text.strip() == cand.strip()):
                        if not to or is_separator(to):
                            state.status = "Выберите контакт или чат для отправки"
                        else:
                            # Откроем модалку с подтверждением прикрепления файла
                            state.file_confirm_mode = True
                            state.file_confirm_path = cand
                            state.file_confirm_target = to
                            state.file_confirm_text_full = text
                            state.file_confirm_index = 0
                            # Сохраним исходный ввод для "Отмена"
                            state.file_confirm_prev_text = state.input_buffer
                            try:
                                state.file_confirm_prev_caret = int(getattr(state, 'input_caret', 0))
                            except Exception:
                                state.file_confirm_prev_caret = 0
                            try:
                                import os as _os
                                from pathlib import Path as _Path
                                p = _Path(cand).expanduser()
                                if p.exists():
                                    state.status = f"Обнаружен путь: {p.name}. Enter — подтвердить прикрепление"
                                else:
                                    state.status = f"Путь обнаружен, но файл не найден"
                            except Exception:
                                state.status = "Обнаружен путь. Enter — подтвердить прикрепление"
                        state.history_scroll = 0
                        continue
                    # Если путь присутствует в составе текста
                    if cand and text.strip() != cand.strip():
                        # Правило: если в тексте явно встречается маркер /file — трактуем как команду без подтверждения
                        try:
                            tokens = [t.strip() for t in text.replace('\n', ' ').split(' ') if t.strip()]
                        except Exception:
                            tokens = []
                        if '/file' in tokens:
                            if not to or is_separator(to):
                                state.status = "Выберите контакт или чат для отправки файла"
                            else:
                                # Проверки для ЛС: дружба и отсутствие блокировок
                                try:
                                    blocked_union = set(state.blocked) | set(getattr(state, 'blocked_by', set()))
                                except Exception:
                                    blocked_union = set()
                                is_room = bool(to in state.groups or (isinstance(to, str) and to.startswith('b-')))
                                if (not is_room):
                                    is_friend = bool(state.friends.get(to) or (to in state.roster_friends))
                                    if (to in blocked_union) or (state.self_id in blocked_union):
                                        state.status = "Нельзя отправить файл: контакт в блокировке"
                                        state.input_buffer = ""; state.input_caret = 0
                                        state.history_scroll = 0
                                        continue
                                    if not is_friend:
                                        state.status = "Требуется авторизация: добавьте друг друга в контакты"
                                        state.input_buffer = ""; state.input_caret = 0
                                        state.history_scroll = 0
                                        continue
                                meta = file_meta_for(cand)
                                if not meta:
                                    state.modal_message = "Файл не найден или нет доступа"
                                else:
                                    state.file_send_path = str(meta.path)
                                    state.file_send_name = meta.name
                                    state.file_send_size = meta.size
                                    if to in state.groups or (isinstance(to, str) and to.startswith('b-')):
                                        state.file_send_room = to
                                        state.file_send_to = None
                                        net.send({"type": getattr(T, 'FILE_OFFER', 'file_offer'), "room": to, "name": meta.name, "size": meta.size})
                                    else:
                                        state.file_send_room = None
                                        state.file_send_to = to
                                        net.send({"type": getattr(T, 'FILE_OFFER', 'file_offer'), "to": to, "name": meta.name, "size": meta.size})
                                    try:
                                        chan = to
                                        nm = meta.name or 'файл'
                                        state.conversations.setdefault(chan, []).append(ChatMessage('out', f"Отправка файла [{nm}]…", time.time()))
                                    except Exception:
                                        pass
                                    state.status = f"Подготовка отправки: {meta.name} ({meta.size} байт)"
                            state.input_buffer = ""
                            try:
                                state.input_caret = 0
                            except Exception:
                                pass
                            state.history_scroll = 0
                            continue
                        # Иначе попросим подтверждение отправки файла
                        if not to or is_separator(to):
                            state.status = "Выберите контакт или чат"
                        else:
                            state.file_confirm_mode = True
                            state.file_confirm_path = cand
                            state.file_confirm_target = to
                            state.file_confirm_text_full = text
                            state.file_confirm_index = 0
                            state.file_confirm_prev_text = state.input_buffer
                            try:
                                state.file_confirm_prev_caret = int(getattr(state, 'input_caret', 0))
                            except Exception:
                                state.file_confirm_prev_caret = 0
                        state.history_scroll = 0
                        continue
                    # Slash-commands (duplicate for special-key enter)
                    if text.startswith('/'):
                        cmd = text.split()[0]
                        arg = text[len(cmd):].strip()
                        if cmd == '/profile':
                            net.send({"type": "profile_get"})
                            state.status = "Запрос профиля..."
                        elif cmd == '/setname':
                            try:
                                if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
                                    arg1 = arg[1:-1]
                                else:
                                    arg1 = arg
                                payload = make_profile_set_payload(arg1, None)
                                net.send(payload)
                                state.status = "Обновляем имя..."
                            except Exception as e:
                                state.status = f"Ошибка имени: {e}"
                        elif cmd == '/sethandle':
                            try:
                                h = normalize_handle(arg)
                                payload = make_profile_set_payload(None, h)
                                net.send(payload)
                                state.status = "Обновляем логин..."
                            except Exception as e:
                                state.status = f"Ошибка логина: {e}"
                        elif cmd == '/whois':
                            if arg:
                                if arg.startswith('@'):
                                    net.send({"type": "profile_get", "handle": arg})
                                else:
                                    net.send({"type": "profile_get", "id": arg})
                                state.status = f"Запрос профиля {arg}..."
                        else:
                            state.status = "Неизвестная команда"
                        state.input_buffer = ""
                        try:
                            state.input_caret = 0
                        except Exception:
                            pass
                        state.history_scroll = 0
                        continue
                    if not to or is_separator(to):
                        state.status = "Выберите контакт или чат"
                    elif to in state.groups:
                        net.send({"type": "send", "room": to, "text": text})
                        conv = state.conversations.setdefault(to, [])
                        ts = time.time()
                        conv.append(ChatMessage('out', text, ts))
                        try:
                            append_history_record({"id": None, "from": state.self_id, "text": text, "ts": ts, "room": to})
                        except Exception:
                            pass
                        logging.getLogger('client').debug("Msg to GROUP %s: %s", to, text)
                    else:
                        # Enforce authorization client-side: auto-send auth request instead of DM
                        is_friend = bool(state.friends.get(to) or (to in state.roster_friends))
                        if not is_friend:
                            if to not in state.pending_out:
                                try:
                                    net.send({"type": "authz_request", "to": to})
                                    state.pending_out.add(to)
                                    state.status = f"Требуется авторизация. Запрос отправлен: {to}"
                                    state.modal_message = f"Запрос авторизации отправлен: {to}"
                                except Exception:
                                    state.status = f"Требуется авторизация для {to}"
                            else:
                                state.status = f"Запрос авторизации к {to} уже отправлен"
                        else:
                            net.send({"type": "send", "to": to, "text": text})
                            conv = state.conversations.setdefault(to, [])
                            ts = time.time()
                            conv.append(ChatMessage('out', text, ts))
                            try:
                                append_history_record({"id": None, "from": state.self_id, "to": to, "text": text, "ts": ts})
                            except Exception:
                                pass
                            logging.getLogger('client').debug("Msg to %s: %s", to, text)
                        # Add to input history (store original typed content)
                        try:
                            if state.input_buffer:
                                hist = list(getattr(state, 'input_history', []))
                                if not hist or hist[-1] != state.input_buffer:
                                    hist.append(state.input_buffer)
                                    state.input_history = hist
                                state.input_history_index = -1
                        except Exception:
                            pass
                state.input_buffer = ""
                try:
                    state.input_caret = 0
                except Exception:
                    pass
                state.history_scroll = 0

    net.stop()


if __name__ == '__main__':
    # Lightweight CLI flags to control logging/behavior without env juggling.
    # Examples:
    #   ./client.py DEBUG            -> LOG_LEVEL=DEBUG (logs to file)
    #   ./client.py INFO --stderr    -> LOG_LEVEL=INFO + also log to stderr
    #   ./client.py - DEBUG --log-file=/tmp/client-debug.log --json
    #   ./client.py --no-mouse --vi
    def _apply_cli_args(argv: List[str]) -> None:
        try:
            args = list(argv[1:])
            # Simple levels: DEBUG/INFO/WARNING/ERROR
            for a in args:
                av = (a or '').strip()
                if av in ('-', '--'):
                    continue
                u = av.upper()
                if u in ('DEBUG', 'INFO', 'WARNING', 'ERROR'):
                    os.environ['LOG_LEVEL'] = u
            # --log-file=<path> or -l <path>
            for i, a in enumerate(args):
                if a.startswith('--log-file='):
                    os.environ['CLIENT_LOG_FILE'] = a.split('=', 1)[1]
                elif a in ('-l', '--log-file') and i + 1 < len(args):
                    os.environ['CLIENT_LOG_FILE'] = args[i + 1]
            # --stderr -> also log to stderr (startup only); suppressed inside TUI to avoid breaking UI
            # Use --stderr-tui to force logs to stderr during TUI (not recommended)
            if '--stderr' in args:
                os.environ['CLIENT_LOG_STDERR'] = '1'
            if '--stderr-tui' in args:
                os.environ['CLIENT_STDERR_TUI'] = '1'
            # --json -> JSON logs instead of plain text
            if '--json' in args:
                os.environ['LOG_JSON'] = '1'
            # --no-mouse -> disable in-app mouse (use terminal selection)
            if '--no-mouse' in args:
                os.environ['CLIENT_MOUSE'] = '0'
            # --vi -> enable vi-like keys (k/j in list)
            if '--vi' in args:
                os.environ['CLIENT_VI_KEYS'] = '1'
            # Help
            if '--help' in args or '-h' in args:
                print('Usage: ./client.py [LEVEL] [options]\n' \
                      '  LEVEL: DEBUG|INFO|WARNING|ERROR (sets LOG_LEVEL)\n' \
                      'Options:\n' \
                      '  --log-file=PATH | -l PATH   Write logs to PATH (rotating)\n' \
                      '  --stderr                    Also log to stderr before TUI (suppressed in TUI)\n' \
                      '  --stderr-tui                Force log to stderr during TUI (may break UI)\n' \
                      '  --json                      JSON-formatted logs\n' \
                      '  --no-mouse                  Disable in-app mouse handling\n' \
                      '  --vi                        Enable vi-like keys in list (k/j)\n' \
                      'Examples:\n' \
                      '  ./client.py DEBUG --stderr\n' \
                      '  ./client.py - DEBUG --log-file=/tmp/client.log --json\n')
                sys.exit(0)
        except SystemExit:
            raise
        except Exception:
            # Do not break startup on bad flags
            pass

    # Isolated initializer for F7 file manager modal
    def _persist_file_browser_state(state: ClientState) -> None:
        """Persist current file manager prefs (path + view options) to config."""
        try:
            st = getattr(state, 'file_browser_state', None)
            if isinstance(st, FileManagerState):
                base0 = _fm_norm_path(str(st.path or '.'))
                _save_fb_prefs_values(
                    fb_show_hidden0=bool(st.show_hidden),
                    fb_sort0=str(st.sort or 'name'),
                    fb_dirs_first0=bool(st.dirs_first),
                    fb_reverse0=bool(st.reverse),
                    fb_view0=st.view,
                    fb_path0=base0,
                    fb_side=0,
                )
                return
            # Legacy fallback (older two-pane state)
            base0 = _fm_norm_path(str(getattr(state, 'file_browser_path0', '.') or '.'))
            base1 = _fm_norm_path(str(getattr(state, 'file_browser_path1', '.') or '.'))
            _save_fb_prefs_values(
                fb_show_hidden0=bool(getattr(state, 'file_browser_show_hidden0', False)),
                fb_show_hidden1=bool(getattr(state, 'file_browser_show_hidden1', False)),
                fb_sort0=str(getattr(state, 'file_browser_sort0', 'name')),
                fb_sort1=str(getattr(state, 'file_browser_sort1', 'name')),
                fb_dirs_first0=bool(getattr(state, 'file_browser_dirs_first0', True)),
                fb_dirs_first1=bool(getattr(state, 'file_browser_dirs_first1', True)),
                fb_reverse0=bool(getattr(state, 'file_browser_reverse0', False)),
                fb_reverse1=bool(getattr(state, 'file_browser_reverse1', False)),
                fb_view0=getattr(state, 'file_browser_view0', None),
                fb_view1=getattr(state, 'file_browser_view1', None),
                fb_path0=base0,
                fb_path1=base1,
                fb_side=int(getattr(state, 'file_browser_side', 0)),
            )
        except Exception:
            logging.getLogger('client').exception("Failed to persist file browser prefs")

    def close_file_browser(state: ClientState) -> None:
        """Close file browser, persist preferences and reset modes."""
        try:
            _persist_file_browser_state(state)
        except Exception:
            pass
        state.file_browser_view_mode = False
        state.file_browser_settings_mode = False
        state.file_browser_menu_mode = False
        state.file_browser_mode = False
        state.status = ""

    def start_file_browser(state: ClientState) -> None:
        try:
            import os as _os
            from pathlib import Path as _Path

            # Close conflicting overlays
            state.search_action_mode = False
            state.action_menu_mode = False
            state.profile_mode = False
            state.profile_view_mode = False
            state.modal_message = None

            state.file_browser_view_mode = False
            state.file_browser_settings_mode = False
            state.file_browser_menu_mode = False

            # Proactively fetch server-side prefs when authed
            try:
                if getattr(state, 'authed', False):
                    net.send({"type": getattr(T, 'PREFS_GET', 'prefs_get')})
            except Exception:
                pass

            prefs = _get_fb_prefs()
            start = _fm_norm_path('.')
            try:
                p0 = prefs.get('fb_path0')
                if p0 and _os.path.isdir(str(p0)):
                    start = _fm_norm_path(str(p0))
                else:
                    home = _fm_norm_path('~')
                    if _os.path.isdir(home):
                        start = home
            except Exception:
                pass

            fm = FileManagerState(
                path=start,
                show_hidden=bool(prefs.get('fb_show_hidden0', False)),
                sort=str(prefs.get('fb_sort0', 'name')),
                dirs_first=bool(prefs.get('fb_dirs_first0', True)),
                reverse=bool(prefs.get('fb_reverse0', False)),
                view=prefs.get('fb_view0', None),
            )
            _fm_relist(fm)

            state.file_browser_mode = True
            state.file_browser_side = 0
            state.file_browser_state = fm

            # Mirror into legacy fields for compatibility with older helpers
            try:
                state.file_browser_path0 = str(fm.path or '')
                state.file_browser_show_hidden0 = bool(fm.show_hidden)
                state.file_browser_sort0 = str(fm.sort or 'name')
                state.file_browser_dirs_first0 = bool(fm.dirs_first)
                state.file_browser_reverse0 = bool(fm.reverse)
                state.file_browser_view0 = fm.view
            except Exception:
                pass
            state.status = ""
        except Exception:
            close_file_browser(state)  # type: ignore[name-defined]
    def _preflight_diagnostics():
        try:
            pyver = sys.version.replace('\n', ' ')
            if sys.version_info < (3, 7):
                print(f"[client] Unsupported Python version: {pyver}. Need >= 3.7.", file=sys.stderr)
            if 'TERM' not in os.environ:
                print("[client] Warning: TERM env not set; curses may misbehave", file=sys.stderr)
            lvl = os.environ.get('LOG_LEVEL', 'INFO').upper()
            if PROFILE_MODULE_FALLBACK and lvl in ('DEBUG', 'TRACE'):
                print("[client] Note: modules. not found; using embedded validators", file=sys.stderr)
        except Exception:
            pass

    def _enter_alt_screen():
        try:
            sys.stdout.write("\x1b[?1049h")
            sys.stdout.flush()
        except Exception:
            pass

    def _exit_alt_screen():
        try:
            # Best-effort terminal restore: show cursor, reset attrs, leave alt screen, move to new line
            try:
                curses.echo()
                curses.nocbreak()
            except Exception:
                pass
            try:
                # Disable mouse/focus reporting so shell won't receive SGR sequences after exit
                _term_write("\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1015l\x1b[?1006l\x1b[?1007l\x1b[?1004l", tmux_passthrough=True)
            except Exception:
                pass
            try:
                # If we auto-enabled tmux mouse for this session, restore it when we know it was previously off.
                if os.environ.get('TMUX') and bool(globals().get('__TMUX_MOUSE_AUTO_ENABLED__')):
                    prev = globals().get('__TMUX_MOUSE_PREV__')
                    if prev is not None:
                        prev_norm = str(prev).strip().lower()
                        if prev_norm in ('off', '0', 'false', 'no', ''):
                            subprocess.run(
                                ['tmux', 'set', '-g', 'mouse', 'off'],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=0.25,
                                check=False,
                            )
                            globals()['__TMUX_MOUSE_AUTO_ENABLED__'] = False
            except Exception:
                pass
            sys.stdout.write("\x1b[?25h\x1b[0m\x1b[?1049l\r\n")
            sys.stdout.flush()
        except Exception:
            pass

    try:
        _apply_cli_args(sys.argv)
        _preflight_diagnostics()
        # Enable system locale for unicode input/output
        try:
            locale.setlocale(locale.LC_ALL, '')
        except Exception:
            pass
        # Inform user when --stderr is provided but suppressed during TUI
        try:
            if os.environ.get('CLIENT_LOG_STDERR') and not os.environ.get('CLIENT_STDERR_TUI'):
                sys.stderr.write("[client] --stderr: logging to stderr is suppressed during TUI; see log/client*.log or use --stderr-tui to force (may break UI)\n")
                sys.stderr.flush()
        except Exception:
            pass
        _enter_alt_screen()
        # Install Unix signal handlers to play well with macOS terminals (suspend/resume)
        try:
            import signal as _sig
            # Global flag to re-apply mouse modes after resume
            globals()['__RESUME_MOUSE__'] = False

            def _on_tstp(signum, frame):
                # Best-effort: disable mouse/paste/focus before suspending
                try:
                    _term_write("\x1b[?1000l\x1b[?1002l\x1b[?1015l\x1b[?1006l\x1b[?1007l\x1b[?1004l\x1b[?25h", tmux_passthrough=True)
                except Exception:
                    pass
                try:
                    _set_bracketed_paste(False)
                except Exception:
                    pass
                try:
                    curses.endwin()
                except Exception:
                    pass
                _sig.signal(_sig.SIGTSTP, _sig.SIG_DFL)
                os.kill(os.getpid(), _sig.SIGTSTP)

            def _on_cont(signum, frame):
                try:
                    globals()['__RESUME_MOUSE__'] = True
                except Exception:
                    pass
            try:
                _sig.signal(_sig.SIGTSTP, _on_tstp)
                _sig.signal(_sig.SIGCONT, _on_cont)
            except Exception:
                pass
        except Exception:
            pass
        try:
            curses.wrapper(main)
        except curses.error as e:
            # Частая причина: запуск не в TTY или поломанный TERM
            msg1 = f"[client] Ошибка инициализации TUI: {e}\n"
            msg2 = "[client] Проверьте TERM, что stdin/stdout не перенаправлены, и запустите в обычном терминале.\n"
            msg3 = f"[client] isatty(stdin)={sys.stdin.isatty()} isatty(stdout)={sys.stdout.isatty()}\n"
            try:
                sys.stderr.write(msg1)
                sys.stderr.write(msg2)
                sys.stderr.write(msg3)
                sys.stderr.flush()
            except Exception:
                pass
            try:
                log_dir = Path(__file__).resolve().parents[1] / "var" / "log"
                log_dir.mkdir(parents=True, exist_ok=True)
                with open(log_dir / "client-startup.log", "a", encoding="utf-8") as f:
                    f.write(msg1)
                    f.write(msg2)
                    f.write(msg3)
            except Exception:
                pass
            raise SystemExit(2)
        finally:
            _exit_alt_screen()
    except KeyboardInterrupt:
        logging.getLogger('client').info("KeyboardInterrupt; exiting")
    except Exception as e:
        # Print a concise error summary to help diagnose startup issues
        try:
            print(f"[client] Fatal error: {e}", file=sys.stderr)
            print(f"[client] Python: {sys.version.split()[0]} ({sys.executable})", file=sys.stderr)
            print(f"[client] PROFILE_FALLBACK={PROFILE_MODULE_FALLBACK}", file=sys.stderr)
        except Exception:
            pass
        raise
MODALS_MODULE_FALLBACK = MODALS_MODULE_FALLBACK
