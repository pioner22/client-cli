from __future__ import annotations

"""
Pure helpers for input behaviors to make them testable without curses.

- handle_history_navigation: Up/Down behavior with multiline editor and history
- compute_suggestions: decide slash/file suggestions span and items
- apply_suggestion: replace span with selected suggestion
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from .text_editor import TextEditor
from .multiline_editor import EditorView
from .slash_commands import suggest as slash_suggest
from .file_suggest import get_file_system_suggestions


def handle_history_navigation(
    text: str,
    caret: int,
    width: int,
    direction: str,
    history: List[str],
    hist_index: int,
) -> Tuple[str, int, int]:
    """Return (new_text, new_caret, new_hist_index) for Up/Down.

    - If movement inside multiline is possible, prefer moving caret.
    - If at top (Up) or bottom (Down) and cannot move further, browse history.
    - hist_index: -1 means not browsing yet; otherwise index into history.
    """
    direction = (direction or '').lower()
    text = text or ''
    caret = max(0, min(len(text), int(caret)))
    width = max(1, int(width))

    ed = TextEditor(text, caret)
    before = ed.caret
    if direction == 'up':
        ed.move_up()
        moved = ed.caret != before
        if moved:
            return ed.text, ed.caret, hist_index
        # browse history
        if not history:
            return text, caret, hist_index
        idx = len(history) - 1 if hist_index == -1 else max(0, hist_index - 1)
        s = history[idx]
        return s, len(s), idx
    elif direction == 'down':
        ed.move_down()
        moved = ed.caret != before
        if moved:
            return ed.text, ed.caret, hist_index
        if not history or hist_index == -1:
            return text, caret, hist_index
        if hist_index < len(history) - 1:
            idx = hist_index + 1
            s = history[idx]
            return s, len(s), idx
        # Exit history mode at end
        return '', 0, -1
    else:
        return text, caret, hist_index


@dataclass
class SuggestionResult:
    kind: str  # 'slash' | 'file'
    items: List[object]
    start: int
    end: int


def _slash_suggestions(text: str, caret: int, limit: int) -> Optional[SuggestionResult]:
    if not text.startswith('/'):
        return None
    end = text.find(' ')
    if end == -1 or caret <= end:
        end = caret if end == -1 else max(caret, end)
        items = slash_suggest(text[:end] or '/', limit=limit)
        if items:
            return SuggestionResult(kind='slash', items=items, start=0, end=end)
    return None


def _token_span(text: str, caret: int) -> tuple[int, int]:
    a = text.rfind(' ', 0, caret)
    a2 = text.rfind('\n', 0, caret)
    a = max(a, a2) + 1
    b = caret
    while b < len(text) and not text[b].isspace():
        b += 1
    return a, b


def _looks_like_file_token(token: str) -> bool:
    return bool(
        token
        and (
            token.startswith('/')
            or token.startswith('~')
            or token.startswith('./')
            or token.startswith('../')
            or ('/' in token)
            or ('\\' in token)
        )
    )


def compute_suggestions(text: str, caret: int, cwd: Optional[Union[str, Path]] = None, limit: int = 10) -> Optional[SuggestionResult]:
    text = text or ''
    caret = max(0, min(len(text), int(caret)))
    slash = _slash_suggestions(text, caret, limit)
    if slash is not None:
        return slash
    a, b = _token_span(text, caret)
    token = text[a:b]
    if _looks_like_file_token(token):
        items = get_file_system_suggestions(token, cwd=cwd or Path('.').resolve(), limit=limit)
        if items:
            return SuggestionResult(kind='file', items=items, start=a, end=b)
    return None


def apply_suggestion(text: str, start: int, end: int, replacement: str) -> Tuple[str, int]:
    text = text or ''
    start = max(0, int(start))
    end = max(start, int(end))
    replacement = replacement or ''
    s = text[:start] + replacement + text[end:]
    return s, start + len(replacement)


__all__ = [
    'handle_history_navigation',
    'compute_suggestions',
    'apply_suggestion',
    'SuggestionResult',
]


# ===== Additional helpers for client hotkeys/routing =====

def classify_peer(sel: object, groups: Optional[set[str]], boards: Optional[dict[str, object]]) -> str:
    """Classify selected token into: 'user' | 'group' | 'board' | 'token' | 'unknown'.

    - group if sel in groups
    - board if sel is a string starting with 'b-' or present in boards dict
    - token if sel is a special UI token like 'BINV:' or 'JOIN:'
    - user if a non-empty string otherwise
    """
    try:
        gset = set(groups or set())
        bmap = dict(boards or {})
        if sel in gset:
            return 'group'
        if isinstance(sel, str):
            if sel.startswith('BINV:') or sel.startswith('JOIN:'):
                return 'token'
            if sel.startswith('b-') or (sel in bmap):
                return 'board'
            if sel:
                return 'user'
        return 'unknown'
    except Exception:
        return 'unknown'


def should_trigger_auth_hotkey(input_buffer: str, board_invite_mode: bool, profile_mode: bool, sel: object, groups: Optional[set[str]], boards: Optional[dict[str, object]]) -> bool:
    """Return True if pressing 'A' should send an authz_request.

    Mirrors client logic: only when input is empty, no board invite/profile modes,
    and selection is a user (not group/board/special token).
    """
    if (input_buffer or '').strip():
        return False
    if board_invite_mode or profile_mode:
        return False
    cls = classify_peer(sel, groups, boards)
    return cls == 'user'


def file_offer_target(sel: object, groups: Optional[set[str]], boards: Optional[dict[str, object]]) -> Optional[dict[str, str]]:
    """Map current selection to target payload for FILE_OFFER: {'room': id} or {'to': id}.

    Returns None if selection invalid/absent.
    """
    try:
        if not sel or not isinstance(sel, str):
            return None
        if sel in set(groups or set()) or sel.startswith('b-') or (sel in dict(boards or {})):
            return {'room': sel}
        return {'to': sel}
    except Exception:
        return None


__all__ += ['classify_peer', 'should_trigger_auth_hotkey', 'file_offer_target']


# ===== File send policy (client-side pre-checks) =====
def _combined_blocks(blocked: Optional[set[str]], blocked_by: Optional[set[str]]) -> set[str]:
    return set(blocked or set()) | set(blocked_by or set())


def _combined_friends(friends: Optional[dict[str, bool]], roster_friends: Optional[dict[str, dict]]) -> set[str]:
    return set((friends or {}).keys()) | set((roster_friends or {}).keys())


def _can_send_direct(sel: object, *, blocked: Optional[set[str]], blocked_by: Optional[set[str]], friends: Optional[dict[str, bool]], roster_friends: Optional[dict[str, dict]]) -> tuple[bool, str]:
    bset = _combined_blocks(blocked, blocked_by)
    if isinstance(sel, str) and (sel in bset):
        return False, 'blocked'
    fset = _combined_friends(friends, roster_friends)
    if isinstance(sel, str) and (sel not in fset):
        return False, 'not_friends'
    return True, 'ok'


def can_send_file(
    authed: bool,
    sel: object,
    groups: Optional[set[str]],
    boards: Optional[dict[str, object]],
    friends: Optional[dict[str, bool]],
    roster_friends: Optional[dict[str, dict]],
    blocked: Optional[set[str]],
    blocked_by: Optional[set[str]],
    self_id: Optional[str],
) -> tuple[bool, str]:
    """Return (ok, reason) for sending a file according to client policy.

    - Requires authentication
    - For direct messages (user): requires confirmed friendship and no blocks
    - For rooms (groups/boards): allowed client-side; server remains authoritative
    Reasons: 'ok' | 'not_authed' | 'blocked' | 'not_friends' | 'invalid_target'
    """
    if not authed:
        return False, 'not_authed'
    cls = classify_peer(sel, groups or set(), boards or {})
    if cls in ('group', 'board'):
        return True, 'ok'
    if cls != 'user':
        return False, 'invalid_target'
    return _can_send_direct(sel, blocked=blocked, blocked_by=blocked_by, friends=friends, roster_friends=roster_friends)


__all__ += ['can_send_file']
