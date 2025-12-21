from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


MAX_DISPLAY_NAME = 64
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

    def to_dict(self) -> dict:
        return asdict(self)


def make_profile_set_payload(display_name: Optional[str], handle: Optional[str]) -> dict:
    """Сформировать NDJSON-пакет для обновления профиля (клиент → сервер)."""
    nd = normalize_display_name(display_name)
    nh = normalize_handle(handle)
    ok_name, err_name = validate_display_name(nd)
    if not ok_name:
        raise ValueError(f"invalid display_name: {err_name}")
    ok_handle, err_handle = validate_handle(nh)
    if not ok_handle:
        raise ValueError(f"invalid handle: {err_handle}")
    payload: dict = {"type": "profile_set"}
    if nd is not None:
        payload["display_name"] = nd
    if nh is not None:
        payload["handle"] = nh
    return payload


__all__ = [
    "Profile",
    "normalize_display_name",
    "validate_display_name",
    "normalize_handle",
    "validate_handle",
    "make_profile_set_payload",
]
