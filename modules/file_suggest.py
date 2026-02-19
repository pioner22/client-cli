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


def _resolve_parent_and_stem(token: str, base: Path) -> tuple[Path, str]:
    p = Path(token).expanduser()
    if token.endswith(os.sep) or token.endswith('/'):
        parent = (base / p).resolve() if not p.is_absolute() else p
        return parent, ''
    parent = (base / p).resolve().parent if not p.is_absolute() else p.parent
    return parent, p.name


def _iter_suggestions(parent: Path, stem: str, limit: int) -> List[str]:
    out: List[str] = []
    max_items = max(1, int(limit))
    for entry in sorted(parent.iterdir(), key=lambda x: x.name.lower()):
        if stem and not entry.name.startswith(stem):
            continue
        if not _visible(entry):
            continue
        label = str(entry)
        if entry.is_dir():
            label += os.sep
        out.append(label)
        if len(out) >= max_items:
            break
    return out


def get_file_system_suggestions(token: str, cwd: Optional[Union[str, Path]] = None, limit: int = 20) -> List[str]:
    token = (token or '').strip()
    base = Path(cwd or os.getcwd())
    parent, stem = _resolve_parent_and_stem(token, base)
    try:
        return _iter_suggestions(parent, stem, limit)
    except Exception:
        return []


__all__ = ["get_file_system_suggestions"]
