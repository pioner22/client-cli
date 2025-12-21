from __future__ import annotations

"""
Shared helpers for the "Boards" (Доски) feature.

This module is intentionally lightweight and UI/DB-agnostic so it can be
imported by both server and client code. It provides:
  - Board dataclass with a safe to_payload() mapper
  - Handle normalization/validation wrappers (reuse profile rules)
  - Board id utilities (b-<hex>)
  - Small payload builders for common client→server actions

Server-side storage and handlers will live in server/ (separately), while
this module focuses on shared types and helpers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re

try:
    # Reuse existing profile handle rules for consistency
    from .profile import normalize_handle as _normalize_handle
    from .profile import validate_handle as _validate_handle
except Exception:  # pragma: no cover - fallback if module not available
    _HANDLE_RE = re.compile(r"^@[a-z0-9_]{3,16}$")

    def _normalize_handle(handle: Optional[str]) -> Optional[str]:
        if handle is None:
            return None
        h = (handle or "").strip().lower()
        if not h:
            return None
        if not h.startswith("@"):
            h = "@" + h
        base = re.sub(r"[^a-z0-9_]", "", h[1:])
        return "@" + base

    def _validate_handle(handle: Optional[str]) -> Tuple[bool, Optional[str]]:
        if handle is None:
            return True, None
        return (True, None) if _HANDLE_RE.match(handle) else (False, "handle_invalid")

# Board ids follow prefix b- + hex (length >= 6)
BOARD_ID_RE = re.compile(r"^b-[0-9a-f]{6,32}$", re.IGNORECASE)


def is_board_id(token: Optional[str]) -> bool:
    try:
        return bool(token) and bool(BOARD_ID_RE.match(str(token)))
    except Exception:
        return False


def normalize_board_handle(handle: Optional[str]) -> Optional[str]:
    """Normalize a board handle using the same rules as user handles."""
    return _normalize_handle(handle)


def validate_board_handle(handle: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Validate a board handle; returns (ok, reason)."""
    return _validate_handle(handle)


@dataclass
class Board:
    id: str
    name: str
    owner_id: str
    members: List[str]
    handle: Optional[str] = None
    visibility: str = "public"  # MVP: public boards are readable/joinable; only owner can post

    def to_payload(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "owner_id": self.owner_id,
            "handle": self.handle,
            "members": list(self.members),
        }


# CSV helper for member lists (reuse groups semantics)
try:  # pragma: no cover - mirror groups utility when available
    from .groups import parse_members_csv as parse_members_csv  # type: ignore
except Exception:
    def parse_members_csv(text: str) -> List[str]:  # fallback
        out: List[str] = []
        for raw in (text or "").split(','):
            v = raw.strip()
            if v:
                out.append(v)
        return out


# Client→Server payload helpers (convenience only)
def make_board_create_payload(name: str, handle: Optional[str] = None, members: Optional[List[str]] = None) -> Dict[str, object]:
    nh = normalize_board_handle(handle)
    ok, reason = validate_board_handle(nh)
    if not ok:
        raise ValueError(f"invalid handle: {reason}")
    payload: Dict[str, object] = {"type": "board_create", "name": (name or "").strip()}
    if nh is not None:
        payload["handle"] = nh
    if members:
        payload["members"] = list(members)
    return payload


def make_board_send_payload(board_id: str, text: str) -> Dict[str, object]:
    if not is_board_id(board_id):
        raise ValueError("invalid board_id format; expected b-<hex>")
    return {"type": "send", "room": board_id, "text": text}


__all__ = [
    "Board",
    "BOARD_ID_RE",
    "is_board_id",
    "normalize_board_handle",
    "validate_board_handle",
    "parse_members_csv",
    "make_board_create_payload",
    "make_board_send_payload",
]
