from __future__ import annotations

"""
Unified modal rules for the TUI.

- modal_active(state): return True if any modal/overlay is active.
- nav_index(idx, count, ch): handle arrow key navigation with wrap-around.

Import this from client code to keep behavior consistent.
"""

import curses  # type: ignore
from typing import Any


def modal_active(state: Any) -> bool:
    try:
        return bool(
            getattr(state, 'action_menu_mode', False)
            or getattr(state, 'search_action_mode', False)
            or getattr(state, 'profile_mode', False)
            or getattr(state, 'profile_view_mode', False)
            or getattr(state, 'debug_mode', False)
            or getattr(state, 'file_browser_mode', False)
            or getattr(state, 'group_create_mode', False)
            or getattr(state, 'group_manage_mode', False)
            or getattr(state, 'group_verify_mode', False)
            or getattr(state, 'group_member_add_mode', False)
            or getattr(state, 'group_member_remove_mode', False)
            or getattr(state, 'board_create_mode', False)
            or getattr(state, 'board_manage_mode', False)
            or getattr(state, 'board_member_add_mode', False)
            or getattr(state, 'board_member_remove_mode', False)
            or getattr(state, 'board_invite_mode', False)
            or getattr(state, 'board_added_consent_mode', False)
            or getattr(state, 'file_confirm_mode', False)
            or getattr(state, 'file_progress_mode', False)
            or getattr(state, 'suggest_mode', False)
            or getattr(state, 'update_prompt_mode', False)
            or getattr(state, 'modal_message', None)
            or getattr(state, 'help_mode', False)
        )
    except Exception:
        # Never blank the UI just because a getter raised; treat as no modal.
        return False


def nav_index(idx: int, count: int, ch: Any) -> int:
    """Return new index for arrow keys (wrap-around)."""
    if count <= 0:
        return 0
    try:
        if ch in (curses.KEY_RIGHT, curses.KEY_DOWN):
            return (idx + 1) % count
        if ch in (curses.KEY_LEFT, curses.KEY_UP):
            return (idx - 1) % count
        return idx
    except Exception:
        return idx


__all__ = ['modal_active', 'nav_index']
