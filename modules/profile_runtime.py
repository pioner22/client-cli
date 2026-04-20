from __future__ import annotations

import re
from typing import Optional, Tuple

try:
    from module.profile import (  # type: ignore
        normalize_bio,
        normalize_display_name,
        normalize_handle,
        normalize_status_text,
        validate_bio,
        validate_display_name,
        validate_handle,
        validate_status_text,
    )
except Exception:
    try:
        from server.module.profile import (  # type: ignore
            normalize_bio,
            normalize_display_name,
            normalize_handle,
            normalize_status_text,
            validate_bio,
            validate_display_name,
            validate_handle,
            validate_status_text,
        )
    except Exception:
        _HANDLE_RE = re.compile(r"^@[a-z0-9_]{3,16}$")

        def normalize_display_name(name: Optional[str]) -> Optional[str]:
            if name is None:
                return None
            name = " ".join(name.strip().split())
            if not name:
                return None
            return name[:64]

        def validate_display_name(name: Optional[str]) -> Tuple[bool, Optional[str]]:
            if name is None:
                return True, None
            if not name.strip():
                return False, "empty"
            if len(name.strip()) > 64:
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
            base = re.sub(r"[^a-z0-9_]", "", h[1:])
            return "@" + base if base else None

        def validate_handle(handle: Optional[str]) -> Tuple[bool, Optional[str]]:
            if handle is None:
                return True, None
            return (True, None) if _HANDLE_RE.match(handle) else (False, "handle_invalid")

        def normalize_bio(bio: Optional[str]) -> Optional[str]:
            if bio is None:
                return None
            b = str(bio).replace("\r\n", "\n").replace("\r", "\n").strip()
            if not b:
                return None
            lines = [" ".join(line.split()) for line in b.split("\n")]
            out = "\n".join(lines).strip()
            return out[:280] if out else None

        def validate_bio(bio: Optional[str]) -> Tuple[bool, Optional[str]]:
            if bio is None:
                return True, None
            if not str(bio).strip():
                return False, "empty"
            if len(str(bio)) > 280:
                return False, "too_long"
            return True, None

        def normalize_status_text(status: Optional[str]) -> Optional[str]:
            if status is None:
                return None
            s = " ".join(str(status).strip().split())
            return s[:96] if s else None

        def validate_status_text(status: Optional[str]) -> Tuple[bool, Optional[str]]:
            if status is None:
                return True, None
            if not str(status).strip():
                return False, "empty"
            if len(str(status)) > 96:
                return False, "too_long"
            return True, None


__all__ = [
    "normalize_bio",
    "normalize_display_name",
    "normalize_handle",
    "normalize_status_text",
    "validate_bio",
    "validate_display_name",
    "validate_handle",
    "validate_status_text",
]
