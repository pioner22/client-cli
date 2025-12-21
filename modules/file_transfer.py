from __future__ import annotations

"""
Lightweight helpers for file transfer (client-side + shared protocol constants).

Responsibilities:
- Detect local filesystem path intent in input text (absolute or relative)
- Prepare metadata for an outgoing file offer
- Chunking helpers (base64) with a conservative default chunk size

Server logic lives in server/server.py; this module avoids any direct
dependencies on curses/UI and can be imported by both client and server.
"""

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple, Union


# Default payload-safe chunk size (bytes before base64). Keep below MSG_MAX_BYTES.
DEFAULT_CHUNK_SIZE = 32 * 1024  # 32 KiB


@dataclass
class FileMeta:
    path: Path
    name: str
    size: int


def is_path_like(token: str) -> bool:
    """Return True if token looks like a filesystem path.

    Rules:
    - Starts with '/' or './' or '../' or a drive-letter (Windows), or contains os.sep
    - No spaces at both ends; must contain at least one path separator or be absolute
    """
    if not token:
        return False
    t = token.strip()
    if not t:
        return False
    # Если в токене есть пробелы — это, скорее всего, не отдельный путь,
    # а часть обычной фразы. В таком случае не считаем весь токен путём.
    if any(ch.isspace() for ch in t):
        return False
    # Windows drive (C:\...) or UNC (\\server\share)
    if len(t) >= 2 and t[1] == ':' and (t[0].isalpha()):
        return True
    if t.startswith('\\\\'):
        return True
    # Unix-like
    if t.startswith('/') or t.startswith('./') or t.startswith('../'):
        return True
    # Contains separators
    return (os.sep in t) or ('/' in t) or ('\\' in t)


def extract_path_candidate(text: str) -> Optional[str]:
    """Extract a plausible filesystem path from a text message.

    - If the entire text is a path, return it.
    - Otherwise, scan for the last whitespace-delimited token that looks like a path.
    """
    if not text:
        return None
    s = (text or '').strip()
    if not s:
        return None
    def _sanitize(p: str) -> str:
        if not isinstance(p, str):
            return p
        t = p.strip()
        # Снимем обрамляющие кавычки
        if len(t) >= 2 and t[0] in "\"'«“„””" and t[-1] in "\"'»“„””":
            t = t[1:-1].strip()
        # Удалим хвостовые не-символьные маркеры (пунктуация/галочки/многоточия)
        trail = set(list('.,;:!?✓…') + [')', ']', '}', '»', '”', '’'])
        while len(t) > 1 and t[-1] in trail:
            t = t[:-1]
            t = t.strip()
        return t

    # 1) Прямая передача: вся строка — это путь
    if is_path_like(s):
        return _sanitize(s)
    # 2) Поиск абсолютных/относительных путей внутри текста (последний)
    #    - Unix/Posix: /..., ./..., ../...
    #    - Windows: C:\..., \\server\share\...
    # Учитываем кавычки и скобки в качестве разделителей вокруг пути
    # Абсолютные пути (Unix/Windows)
    abs_pat = re.compile(r'''(?:^|\s|[\[\(\{"'«»“”„])((?:[A-Za-z]:)?[/\\][^\s\]\)"'«»“”„]+?\.(?:[A-Za-z0-9]{1,6}))''')
    hits = abs_pat.findall(s)
    if hits:
        return _sanitize(hits[-1])
    # Относительные пути с каталогами
    rel_pat = re.compile(r'''(?:^|\s|[\[\(\{"'«»“”„])((?:\./|\.\./)?(?:[A-Za-z0-9._-]+[/\\])+[A-Za-z0-9._-]+\.(?:[A-Za-z0-9]{1,6}))''')
    hits = rel_pat.findall(s)
    if hits:
        return _sanitize(hits[-1])
    # 3) Имя файла в скобках/тексте (без каталогов), если похоже на файл по расширению
    name_pat = re.compile(r"([A-Za-z0-9._-]+\.(?:png|jpe?g|gif|webp|bmp|pdf|txt|zip|tar|gz|7z|mp4|mov|mkv))", re.IGNORECASE)
    hits = name_pat.findall(s)
    if hits:
        return _sanitize(hits[-1])
    # 4) Общий абсолютный/UNC путь без требования расширения (последний)
    #    Пример: /Users/user/file (с любым допустимым именем), C:\\path\\to\\file
    general_pat = re.compile(r'''(?:^|\s|[\[\(\{"'«»“”„])((?:[A-Za-z]:)?[/\\][^\s\]\)"'«»“”„]+)''')
    hits = general_pat.findall(s)
    if hits:
        return _sanitize(hits[-1])
    # 5) Fallback: последний токен как путь, если выглядит как путь
    parts = [p for p in s.replace('\n', ' ').split(' ') if p]
    for tok in reversed(parts):
        if is_path_like(tok):
            return _sanitize(tok)
    return None


def file_meta_for(path: Union[str, os.PathLike[str]]) -> Optional[FileMeta]:
    try:
        p = Path(path).expanduser().resolve()
    except Exception:
        return None
    try:
        st = p.stat()
    except Exception:
        return None
    if not p.is_file():
        return None
    try:
        return FileMeta(path=p, name=p.name, size=int(st.st_size))
    except Exception:
        return None


def sanitize_remote_filename(name: str, *, default: str = "file") -> str:
    """Return a safe basename for a remote-provided filename.

    Defense-in-depth for incoming file transfers: prevents path traversal and
    strips control/special characters that may break the terminal UI.
    """
    try:
        s = str(name or "")
    except Exception:
        s = ""
    s = s.strip()
    # Normalize path separators and keep only basename.
    s = s.replace("\\", "/").split("/")[-1].strip()
    if not s or s in (".", ".."):
        s = default
    # Replace unsafe characters; keep a conservative charset.
    try:
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    except Exception:
        # If regex fails, fall back to a constant safe name.
        s = default
    if not s or s in (".", ".."):
        s = default
    # Keep filenames reasonably short to avoid filesystem/path issues.
    try:
        max_len = 128
        if len(s) > max_len:
            root, ext = os.path.splitext(s)
            ext = ext[:16]
            s = (root[: max(1, max_len - len(ext))] + ext)[:max_len]
    except Exception:
        pass
    return s


def iter_base64_chunks(p: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Generator[Tuple[int, str], None, None]:
    """Yield (seq, base64_data) chunks for a file on disk.

    seq is zero-based chunk number. Data is base64-encoded without newlines.
    """
    seq = 0
    with open(p, 'rb') as f:
        while True:
            buf = f.read(max(1, int(chunk_size)))
            if not buf:
                break
            yield seq, base64.b64encode(buf).decode('ascii')
            seq += 1


def progress_percent(done: int, total: int) -> int:
    try:
        if total <= 0:
            return 0
        pct = int((max(0, done) / float(total)) * 100)
        return max(0, min(100, pct))
    except Exception:
        return 0


__all__ = [
    'DEFAULT_CHUNK_SIZE',
    'FileMeta',
    'is_path_like',
    'extract_path_candidate',
    'file_meta_for',
    'sanitize_remote_filename',
    'iter_base64_chunks',
    'progress_percent',
]
