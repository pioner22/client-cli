from __future__ import annotations

"""
File-system suggestions for typeahead. Returns path candidates for Tab-complete.
"""

import os
from pathlib import Path
from typing import Iterable, List, Optional, Union


def _visible(p: Path) -> bool:
    try:
        return not p.name.startswith('.')
    except Exception:
        return True


def get_file_system_suggestions(token: str, cwd: Optional[Union[str, Path]] = None, limit: int = 20) -> List[str]:
    token = (token or '').strip()
    base = Path(cwd or os.getcwd())
    # Expand ~ and relative forms
    p = Path(token).expanduser()
    # If token ends with a separator, suggest inside that directory
    if token.endswith(os.sep) or token.endswith('/'):
        dir_path = (base / p).resolve() if not p.is_absolute() else p
        parent = dir_path
        stem = ''
    else:
        parent = ((base / p).resolve().parent if not p.is_absolute() else p.parent)
        stem = p.name
    out: List[str] = []
    try:
        for entry in sorted(parent.iterdir(), key=lambda x: x.name.lower()):
            if stem and not entry.name.startswith(stem):
                continue
            if not _visible(entry):
                continue
            label = str(entry)
            if entry.is_dir():
                label += os.sep
            out.append(label)
            if len(out) >= max(1, int(limit)):
                break
    except Exception:
        return []
    return out


__all__ = ["get_file_system_suggestions"]
