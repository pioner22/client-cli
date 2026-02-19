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


def _resolve_list_path(path: Union[str, Path]) -> Path:
    try:
        return Path(path).expanduser().resolve()
    except Exception:
        return Path(".").resolve()


def _scan_entries(path: Path) -> Tuple[List[Entry], Optional[str]]:
    items: List[Entry] = []
    err: Optional[str] = None
    try:
        names = sorted(os.listdir(path), key=lambda s: s.lower())
        for name in names:
            full = path / name
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
    return items, err


def _prepend_parent(items: List[Entry], path: Path) -> List[Entry]:
    out = list(items)
    try:
        parent = path.parent
        if parent and parent != path:
            out.insert(0, ("..", True))
    except Exception:
        pass
    return out


def list_dir(path: Union[str, Path], *, return_err: bool = False) -> Union[Tuple[List[Entry], Optional[str]], List[Entry]]:
    """Return directory listing including '..' as the first entry when possible, plus error string.

    - Returns entries sorted by name (case‑insensitive)
    - Hidden files (starting with '.') are included
    - On any error returns a minimal listing with only '..' when applicable and error message
    """
    p = _resolve_list_path(path)
    items, err = _scan_entries(p)
    items = _prepend_parent(items, p)
    return (items, err) if return_err else items


def _filter_base_items(base_items: List[Entry], *, show_hidden: bool, view: Optional[str]) -> tuple[Optional[Entry], List[Entry]]:
    parent_row: Optional[Entry] = None
    items: List[Entry] = []
    for name, is_dir in base_items:
        if name == "..":
            parent_row = ("..", True)
            continue
        if (not show_hidden) and name.startswith("."):
            continue
        if view == "dirs" and (not is_dir):
            continue
        if view == "files" and is_dir:
            continue
        items.append((name, is_dir))
    return parent_row, items


def _mtime_key_factory(path: Union[str, Path], sort: str):
    base = Path(path).expanduser()

    def _mtime_key(entry: Entry) -> float:
        try:
            full = base / entry[0]
            st = full.stat()
            if sort in ("mtime", "modified"):
                return st.st_mtime
            if sort in ("created",):
                ts = getattr(st, "st_birthtime", None)
                return float(ts if ts is not None else st.st_ctime)
            return st.st_ctime
        except Exception:
            return 0.0

    return _mtime_key


def _sort_items(items: List[Entry], *, path: Union[str, Path], sort: str, reverse: bool) -> List[Entry]:
    out = list(items)
    if sort in ("mtime", "modified", "ctime", "created", "changed", "added"):
        out.sort(key=_mtime_key_factory(path, sort), reverse=reverse)
    else:
        out.sort(key=lambda e: e[0].lower(), reverse=reverse)
    return out


def _apply_dirs_first(items: List[Entry], dirs_first: bool) -> List[Entry]:
    if not dirs_first:
        return items
    return [e for e in items if e[1]] + [e for e in items if not e[1]]


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
    parent_row, items = _filter_base_items(base_items, show_hidden=show_hidden, view=view)
    items = _sort_items(items, path=path, sort=sort, reverse=reverse)
    items = _apply_dirs_first(items, dirs_first)
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


def _clone_for_key(state: FileBrowserState) -> FileBrowserState:
    # Keep constructor shape to preserve existing behavior.
    return FileBrowserState(
        side=state.side,
        path0=state.path0,
        path1=state.path1,
        items0=list(state.items0),
        items1=list(state.items1),
        index0=state.index0,
        index1=state.index1,
    )


def _active_pane_values(s: FileBrowserState) -> tuple[List[Entry], int, str, Optional[str], Optional[str]]:
    items = s.items0 if s.side == 0 else s.items1
    idx = s.index0 if s.side == 0 else s.index1
    path = s.path0 if s.side == 0 else s.path1
    err = s.err0 if s.side == 0 else s.err1
    last_dir = s.last_dir0 if s.side == 0 else s.last_dir1
    return items, idx, path, err, last_dir


def _reload_active_items(s: FileBrowserState, path: str) -> tuple[List[Entry], Optional[str]]:
    if s.side == 0:
        return list_dir_opts(
            path,
            show_hidden=s.show_hidden0,
            sort=s.sort0,
            dirs_first=s.dirs_first0,
            reverse=s.reverse0,
            view=s.view0,
            return_err=True,
        )
    return list_dir_opts(
        path,
        show_hidden=s.show_hidden1,
        sort=s.sort1,
        dirs_first=s.dirs_first1,
        reverse=s.reverse1,
        view=s.view1,
        return_err=True,
    )


def _cycle_index(idx: int, items: List[Entry], delta: int) -> int:
    if not items:
        return idx
    return (idx + delta) % len(items)


def _index_for_last_dir(items: List[Entry], last_dir: Optional[str]) -> int:
    if not last_dir:
        return 0
    try:
        names = [n for n, _ in items]
        return names.index(last_dir) if last_dir in names else 0
    except Exception:
        return 0


def _handle_parent_nav(s: FileBrowserState, act_path: str, last_dir: Optional[str]) -> tuple[str, List[Entry], Optional[str], int]:
    new_path, _ = open_entry(act_path, "..")
    items, err = _reload_active_items(s, new_path)
    idx = _index_for_last_dir(items, last_dir)
    return new_path, items, err, idx


def _handle_enter_nav(
    s: FileBrowserState,
    act_items: List[Entry],
    act_idx: int,
    act_path: str,
    last_dir: Optional[str],
) -> tuple[str, List[Entry], Optional[str], int, Optional[str], Optional[str]]:
    name, is_dir = act_items[max(0, min(act_idx, len(act_items) - 1))]
    if not (is_dir or name == ".."):
        _, chosen = open_entry(act_path, name)
        return act_path, act_items, None, act_idx, last_dir, chosen
    new_path, _ = open_entry(act_path, name)
    new_last = name if name not in ("..",) else last_dir
    items, err = _reload_active_items(s, new_path)
    return new_path, items, err, 0, new_last, None


def _write_back_active(
    s: FileBrowserState,
    *,
    path: str,
    items: List[Entry],
    idx: int,
    err: Optional[str],
    last_dir: Optional[str],
) -> None:
    if s.side == 0:
        s.path0, s.items0, s.index0 = path, items, idx
        s.err0 = err
        s.last_dir0 = last_dir
        return
    s.path1, s.items1, s.index1 = path, items, idx
    s.err1 = err
    s.last_dir1 = last_dir


def handle_key(state: FileBrowserState, key: str) -> Tuple[FileBrowserState, Optional[str]]:
    """Apply key: 'UP','DOWN','LEFT','RIGHT','TAB','ENTER','BACKSPACE','ESC'.
    Returns (new_state, chosen_file_path).
    """
    key = key.upper()
    s = _clone_for_key(state)
    if key == 'TAB':
        s.side = 1 - s.side
        return s, None
    act_items, act_idx, act_path, act_err, last_dir = _active_pane_values(s)
    if key in ('UP',):
        act_idx = _cycle_index(act_idx, act_items, -1)
    elif key in ('DOWN',):
        act_idx = _cycle_index(act_idx, act_items, 1)
    elif key in ('BACKSPACE', 'LEFT'):
        act_path, act_items, act_err, act_idx = _handle_parent_nav(s, act_path, last_dir)
    elif key in ('ENTER', 'RIGHT'):
        if not act_items:
            return s, None
        act_path, act_items, act_err, act_idx, last_dir, chosen = _handle_enter_nav(s, act_items, act_idx, act_path, last_dir)
        if chosen:
            return s, chosen
    elif key in ('ESC',):
        return s, None
    _write_back_active(s, path=act_path, items=act_items, idx=act_idx, err=act_err, last_dir=last_dir)
    return s, None


__all__ = [
    'Entry', 'list_dir', 'list_dir_opts', 'open_entry', 'compute_window', 'fmt_mtime', 'row_for',
    'FileBrowserState', 'init_browser', 'handle_key',
]
