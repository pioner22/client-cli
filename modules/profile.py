from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


MAX_DISPLAY_NAME = 64
MAX_STATUS_TEXT = 96
MAX_BIO = 280
HANDLE_RE = re.compile(r"^@[a-z0-9_]{3,16}$")


def _collapse_spaces(text: str) -> str:
    return " ".join(text.split())


def normalize_display_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    name = _collapse_spaces(name.strip())
    if not name:
        return None
    # Ограничение по длине
    return name[:MAX_DISPLAY_NAME]


def validate_display_name(name: Optional[str]) -> Tuple[bool, Optional[str]]:
    if name is None:
        return True, None
    if not name.strip():
        return False, "empty"
    if len(name.strip()) > MAX_DISPLAY_NAME:
        return False, "too_long"
    return True, None


def normalize_handle(handle: Optional[str]) -> Optional[str]:
    if handle is None:
        return None
    h = handle.strip().lower()
    if not h:
        return None
    if not h.startswith("@"):
        h = "@" + h
    # Разрешаем только a-z, 0-9 и _ после @
    base = h[1:]
    base = re.sub(r"[^a-z0-9_]", "", base)
    h = "@" + base
    return h


def validate_handle(handle: Optional[str]) -> Tuple[bool, Optional[str]]:
    if handle is None:
        return True, None
    if not HANDLE_RE.match(handle):
        return False, "handle_invalid"
    return True, None


def normalize_status_text(status: Optional[str]) -> Optional[str]:
    if status is None:
        return None
    status = _collapse_spaces(status.strip())
    if not status:
        return None
    return status[:MAX_STATUS_TEXT]


def validate_status_text(status: Optional[str]) -> Tuple[bool, Optional[str]]:
    if status is None:
        return True, None
    if not str(status).strip():
        return False, "empty"
    if len(str(status)) > MAX_STATUS_TEXT:
        return False, "too_long"
    return True, None


def normalize_bio(bio: Optional[str]) -> Optional[str]:
    if bio is None:
        return None
    b = str(bio).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not b:
        return None
    # Collapse excessive whitespace, but keep newlines.
    lines = [" ".join(line.split()) for line in b.split("\n")]
    b2 = "\n".join(lines).strip()
    if not b2:
        return None
    return b2[:MAX_BIO]


def validate_bio(bio: Optional[str]) -> Tuple[bool, Optional[str]]:
    if bio is None:
        return True, None
    if not str(bio).strip():
        return False, "empty"
    if len(str(bio)) > MAX_BIO:
        return False, "too_long"
    return True, None


@dataclass
class Profile:
    """Профиль пользователя поверх его уникального ID.

    id: неизменяемый внутренний идентификатор (например, "930-77").
    display_name: удобочитаемое имя, до 64 символов.
    handle: логин в формате "@user" (латиница/цифры/нижнее подчёркивание, 3–16).
    """

    id: str
    display_name: Optional[str] = None
    handle: Optional[str] = None
    bio: Optional[str] = None
    status: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def make_profile_set_payload(display_name: Optional[str], handle: Optional[str], bio: Optional[str] = None, status: Optional[str] = None) -> dict:
    """Сформировать NDJSON-пакет для обновления профиля (клиент → сервер)."""
    nd = normalize_display_name(display_name)
    nh = normalize_handle(handle)
    nb = normalize_bio(bio)
    ns = normalize_status_text(status)
    ok_name, err_name = validate_display_name(nd)
    if not ok_name:
        raise ValueError(f"invalid display_name: {err_name}")
    ok_handle, err_handle = validate_handle(nh)
    if not ok_handle:
        raise ValueError(f"invalid handle: {err_handle}")
    ok_bio, err_bio = validate_bio(nb)
    if not ok_bio:
        raise ValueError(f"invalid bio: {err_bio}")
    ok_status, err_status = validate_status_text(ns)
    if not ok_status:
        raise ValueError(f"invalid status: {err_status}")
    payload: dict = {"type": "profile_set"}
    if nd is not None:
        payload["display_name"] = nd
    if nh is not None:
        payload["handle"] = nh
    if nb is not None:
        payload["bio"] = nb
    if ns is not None:
        payload["status"] = ns
    return payload


__all__ = [
    "Profile",
    "normalize_display_name",
    "validate_display_name",
    "normalize_handle",
    "validate_handle",
    "normalize_bio",
    "validate_bio",
    "normalize_status_text",
    "validate_status_text",
    "make_profile_set_payload",
]
