from __future__ import annotations

"""Utilities for user presence/status display.

Defines simple helpers to represent online/offline presence in TUI.
"""

ONLINE = "online"
OFFLINE = "offline"


def status_icon(is_online: bool) -> str:
    """Return a unicode icon for presence: filled dot for online, hollow for offline."""
    return "●" if is_online else "○"


__all__ = ["ONLINE", "OFFLINE", "status_icon"]
