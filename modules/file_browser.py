from __future__ import annotations

"""
Pure helpers for a two‑pane file browser (MC‑like) used by the client F7 modal.

This module contains no curses calls and can be fully unit‑tested. The client
UI can consume these helpers to keep rendering predictable and robust.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union
import os
import time


Entry = Tuple[str, bool]  # (name, is_dir)


def list_dir(path: Union[str, Path], *, return_err: bool = False) -> Union[Tuple[List[Entry], Optional[str]], List[Entry]]:
    """Return directory listing including '..' as the first entry when possible, plus error string.

    - Returns entries sorted by name (case‑insensitive)
    - Hidden files (starting with '.') are included
    - On any error returns a minimal listing with only '..' when applicable and error message
    """
    try:
        p = Path(path).expanduser().resolve()
    except Exception:
        p = Path(".").resolve()
    items: List[Entry] = []
    err: Optional[str] = None
    try:
        names = sorted(os.listdir(p), key=lambda s: s.lower())
        for name in names:
            full = p / name
            is_dir = False
            try:
                is_dir = full.is_dir()
            except Exception:
                is_dir = False
            items.append((name, is_dir))
    except PermissionError:
        err = "Permission denied"
        items = []
    except FileNotFoundError:
        err = "Not found"
        items = []
    except Exception as e:
        err = str(e)
        items = []
    # Prepend '..' if parent exists and is different
    try:
        parent = p.parent
        if parent and parent != p:
            items.insert(0, ('..', True))
    except Exception:
        pass
    return (items, err) if return_err else items


def list_dir_opts(
    path: Union[str, Path],
    *,
    show_hidden: bool = True,
    sort: str = 'name',  # 'name' | 'mtime' | 'ctime' | 'created' | 'changed' | 'added'
    dirs_first: bool = False,
    reverse: bool = False,
    view: Optional[str] = None,  # None|'dirs'|'files'
    return_err: bool = False,
) -> Union[List[Entry], Tuple[List[Entry], Optional[str]]]:
    """Return directory listing with filtering/sorting options.

    - show_hidden: include dotfiles when True
    - sort: 'name' (case-insensitive) or 'mtime' (stat mtime)
    - dirs_first: place directories before files
    - reverse: reverse ordering
    Always includes '..' entry if parent exists.
    Returns list by default; set return_err=True to also get error text.
    """
    base_items, err = list_dir(path, return_err=True)
    items: List[Entry] = []
    parent_row: Optional[Entry] = None
    for name, is_dir in base_items:
        if name == '..':
            parent_row = ('..', True)
            continue
        if (not show_hidden) and name.startswith('.'):
            continue
        if view == 'dirs' and (not is_dir):
            continue
        if view == 'files' and is_dir:
            continue
        items.append((name, is_dir))
    # Sorting
    if sort in ('mtime', 'modified', 'ctime', 'created', 'changed', 'added'):
        p = Path(path).expanduser()
        def mtime_key(entry: Entry) -> float:
            try:
                full = (p / entry[0])
                st = full.stat()
                if sort in ('mtime', 'modified'):
                    return st.st_mtime
                if sort in ('created',):
                    # Prefer birthtime when available (macOS); fallback to ctime
                    ts = getattr(st, 'st_birthtime', None)
                    return float(ts if ts is not None else st.st_ctime)
                # changed/added/ctime – best effort via st_ctime
                return st.st_ctime
            except Exception:
                return 0.0
        items.sort(key=mtime_key, reverse=reverse)
    else:
        items.sort(key=lambda e: e[0].lower(), reverse=reverse)
    if dirs_first:
        items = [e for e in items if e[1]] + [e for e in items if not e[1]]
    # Prepend parent row
    if parent_row is not None:
        items.insert(0, parent_row)
    return (items, err) if return_err else items


def open_entry(cur_path: Union[str, Path], name: str) -> Tuple[str, Optional[str]]:
    """Return (new_path, chosen_file_path) for Enter on an entry.

    - If name == '..' → go up (new_path = parent), no chosen file
    - If entry is a directory → enter it (new_path = cur/name), no chosen file
    - Else (file) → chosen_file_path = absolute path (new_path unchanged)
    """
    try:
        base = Path(cur_path).expanduser().resolve()
    except Exception:
        base = Path(".").resolve()
    if name == '..':
        try:
            parent = base.parent
            return (str(parent if parent else base), None)
        except Exception:
            return (str(base), None)
    full = base / name
    try:
        if full.is_dir():
            return (str(full.resolve()), None)
        if full.is_file():
            return (str(base), str(full.resolve()))
    except Exception:
        return (str(base), None)
    return (str(base), None)


def compute_window(idx: int, total: int, rows: int) -> int:
    """Compute start index for a centered window of size rows around idx."""
    rows = max(1, int(rows))
    total = max(0, int(total))
    if total <= rows:
        return 0
    idx = max(0, min(int(idx), total - 1))
    start = idx - rows // 2
    start = max(0, min(start, total - rows))
    return start


def fmt_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return time.strftime('%b %e %H:%M', time.localtime(ts))
    except Exception:
        return ''


def _human_size(bytes_val: int) -> str:
    try:
        units = ['B', 'K', 'M', 'G', 'T']
        b = float(bytes_val)
        for u in units:
            if b < 1024.0:
                return f"{b:.0f}{u}"
            b /= 1024.0
        return f"{b:.0f}P"
    except Exception:
        return str(bytes_val)


def row_for(name: str, is_dir: bool, base: Path, name_w: int, selected: bool) -> str:
    """Build a single list row similar to the client's MC‑style columns.

    Columns: prefix/name | size | mtime. Size shown only for files; '..' shows -ВВЕРХ-
    """
    name_w = max(4, int(name_w))
    prefix = '▸' if selected else ' '
    label = ('/' if is_dir else ' ') + name
    size_s = ''
    mtime_s = ''
    full = (base / name) if name != '..' else base
    try:
        if name == '..':
            size_s = '-ВВЕРХ-'
        elif full.is_file():
            try:
                size_s = _human_size(full.stat().st_size)
            except Exception:
                size_s = str(full.stat().st_size)
        if full.exists():
            mtime_s = fmt_mtime(full)
    except Exception:
        pass
    return f"║{prefix}{label.ljust(name_w)}│{size_s.rjust(6)} │ {mtime_s.ljust(12)}║"


@dataclass
class FileBrowserState:
    side: int = 0  # 0=left, 1=right
    path0: str = str(Path('.').resolve())
    path1: str = str(Path('.').resolve())
    items0: List[Entry] = None  # type: ignore[assignment]
    items1: List[Entry] = None  # type: ignore[assignment]
    err0: Optional[str] = None
    err1: Optional[str] = None
    index0: int = 0
    index1: int = 0
    last_dir0: Optional[str] = None  # track last entered dir to reselect on parent
    last_dir1: Optional[str] = None
    # View prefs per pane
    show_hidden0: bool = True
    show_hidden1: bool = True
    sort0: str = 'name'      # 'name' | 'mtime'
    sort1: str = 'name'
    dirs_first0: bool = False
    dirs_first1: bool = False
    reverse0: bool = False
    reverse1: bool = False
    # Filter view per pane: None|'dirs'|'files'
    view0: Optional[str] = None
    view1: Optional[str] = None

    def __post_init__(self) -> None:
        self.items0, self.err0 = list_dir_opts(self.path0, show_hidden=self.show_hidden0, sort=self.sort0, dirs_first=self.dirs_first0, reverse=self.reverse0, view=self.view0, return_err=True)
        self.items1, self.err1 = list_dir_opts(self.path1, show_hidden=self.show_hidden1, sort=self.sort1, dirs_first=self.dirs_first1, reverse=self.reverse1, view=self.view1, return_err=True)


def init_browser(start: Optional[Union[str, Path]] = None) -> FileBrowserState:
    base = str(Path(start or '.').expanduser().resolve())
    return FileBrowserState(0, base, base)


def handle_key(state: FileBrowserState, key: str) -> Tuple[FileBrowserState, Optional[str]]:
    """Apply key: 'UP','DOWN','LEFT','RIGHT','TAB','ENTER','BACKSPACE','ESC'.
    Returns (new_state, chosen_file_path).
    """
    key = key.upper()
    s = FileBrowserState(
        side=state.side,
        path0=state.path0,
        path1=state.path1,
        items0=list(state.items0),
        items1=list(state.items1),
        index0=state.index0,
        index1=state.index1,
    )
    if key == 'TAB':
        s.side = 1 - s.side
        return s, None
    # Choose active pane
    act_items = s.items0 if s.side == 0 else s.items1
    act_idx = s.index0 if s.side == 0 else s.index1
    act_path = s.path0 if s.side == 0 else s.path1
    act_err = s.err0 if s.side == 0 else s.err1
    last_dir = s.last_dir0 if s.side == 0 else s.last_dir1
    if key in ('UP',):
        if act_items:
            act_idx = (act_idx - 1) % len(act_items)
    elif key in ('DOWN',):
        if act_items:
            act_idx = (act_idx + 1) % len(act_items)
    elif key in ('BACKSPACE', 'LEFT'):
        # go up
        newp, _ = open_entry(act_path, '..')
        act_path = newp
        # apply pane-specific prefs
        if s.side == 0:
            act_items, act_err = list_dir_opts(act_path, show_hidden=s.show_hidden0, sort=s.sort0, dirs_first=s.dirs_first0, reverse=s.reverse0, view=s.view0, return_err=True)
        else:
            act_items, act_err = list_dir_opts(act_path, show_hidden=s.show_hidden1, sort=s.sort1, dirs_first=s.dirs_first1, reverse=s.reverse1, view=s.view1, return_err=True)
        # try reselect last visited dir in parent
        if last_dir:
            try:
                names = [n for n, _ in act_items]
                if last_dir in names:
                    act_idx = names.index(last_dir)
                else:
                    act_idx = 0
            except Exception:
                act_idx = 0
        else:
            act_idx = 0
    elif key in ('ENTER', 'RIGHT'):
        if not act_items:
            return s, None
        name, is_dir = act_items[max(0, min(act_idx, len(act_items) - 1))]
        if is_dir or name == '..':
            newp, _ = open_entry(act_path, name)
            act_path = newp
            # remember last visited dir to restore selection on parent
            last_dir = name if name not in ('..',) else last_dir
            if s.side == 0:
                act_items, act_err = list_dir_opts(act_path, show_hidden=s.show_hidden0, sort=s.sort0, dirs_first=s.dirs_first0, reverse=s.reverse0, view=s.view0, return_err=True)
            else:
                act_items, act_err = list_dir_opts(act_path, show_hidden=s.show_hidden1, sort=s.sort1, dirs_first=s.dirs_first1, reverse=s.reverse1, view=s.view1, return_err=True)
            act_idx = 0
        else:
            # choose file
            _, chosen = open_entry(act_path, name)
            return s, chosen
    elif key in ('ESC',):
        return s, None
    # Write back
    if s.side == 0:
        s.path0, s.items0, s.index0 = act_path, act_items, act_idx
        s.err0 = act_err
        s.last_dir0 = last_dir
    else:
        s.path1, s.items1, s.index1 = act_path, act_items, act_idx
        s.err1 = act_err
        s.last_dir1 = last_dir
    return s, None


__all__ = [
    'Entry', 'list_dir', 'list_dir_opts', 'open_entry', 'compute_window', 'fmt_mtime', 'row_for',
    'FileBrowserState', 'init_browser', 'handle_key',
]
