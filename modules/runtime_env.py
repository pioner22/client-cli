from __future__ import annotations

import os
from typing import Optional

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def bool_from_raw(raw: object | None, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    try:
        return str(raw).strip().lower() in _TRUE_VALUES
    except Exception:
        return bool(default)


def flag_from_raw(raw: object | None, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    try:
        txt = str(raw).strip().lower()
    except Exception:
        return bool(default)
    if txt in _FALSE_VALUES:
        return False
    if txt in _TRUE_VALUES:
        return True
    return bool(default)


def env_bool(name: str, default: bool = False) -> bool:
    try:
        raw = os.environ.get(name)
    except Exception:
        return bool(default)
    return bool_from_raw(raw, default)


def env_flag(name: str, default: bool = False) -> bool:
    try:
        raw = os.environ.get(name)
    except Exception:
        return bool(default)
    return flag_from_raw(raw, default)


def env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
    except Exception:
        return int(default)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
    except Exception:
        return float(default)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def env_optional_str(name: str) -> Optional[str]:
    try:
        raw = os.environ.get(name)
    except Exception:
        return None
    try:
        value = str(raw or "").strip()
    except Exception:
        return None
    return value or None


def safe_int(value: object, default: int = 0, *, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        out = int(value)  # type: ignore[arg-type]
    except Exception:
        out = int(default)
    if min_value is not None:
        out = max(int(min_value), int(out))
    if max_value is not None:
        out = min(int(max_value), int(out))
    return int(out)


def safe_float(value: object, default: float = 0.0, *, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        out = float(default)
    if min_value is not None:
        out = max(float(min_value), float(out))
    if max_value is not None:
        out = min(float(max_value), float(out))
    return float(out)


def env_int_clamped(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    return safe_int(env_int(name, default), default, min_value=min_value, max_value=max_value)


def env_float_clamped(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    return safe_float(env_float(name, default), default, min_value=min_value, max_value=max_value)


def env_int_nonneg(name: str, default: int) -> int:
    return env_int_clamped(name, default, min_value=0)


def env_float_nonneg(name: str, default: float) -> float:
    return env_float_clamped(name, default, min_value=0.0)


def env_float_positive(name: str, default: float) -> float:
    value = env_float(name, default)
    if value <= 0:
        return float(default)
    return float(value)


__all__ = [
    "bool_from_raw",
    "env_bool",
    "env_flag",
    "env_float",
    "env_float_clamped",
    "env_float_nonneg",
    "env_float_positive",
    "env_int",
    "env_int_clamped",
    "env_int_nonneg",
    "env_optional_str",
    "flag_from_raw",
    "safe_float",
    "safe_int",
]
