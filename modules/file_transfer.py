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
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple, Union


# Default payload-safe chunk size (bytes before base64). Keep below MSG_MAX_BYTES.
DEFAULT_CHUNK_SIZE = 32 * 1024  # 32 KiB

_QUOTE_LEFT = "\"'«“„””"
_QUOTE_RIGHT = "\"'»“„””"
_TRAILING_MARKS = set(list(".,;:!?✓…") + [")", "]", "}", "»", "”", "’"])
_ABS_PATH_RE = re.compile(r"""(?:^|\s|[\[\(\{"'«»“”„])((?:[A-Za-z]:)?[/\\][^\s\]\)"'«»“”„]+?\.(?:[A-Za-z0-9]{1,6}))""")
_REL_PATH_RE = re.compile(r"""(?:^|\s|[\[\(\{"'«»“”„])((?:\./|\.\./)?(?:[A-Za-z0-9._-]+[/\\])+[A-Za-z0-9._-]+\.(?:[A-Za-z0-9]{1,6}))""")
_NAME_PATH_RE = re.compile(
    r"([A-Za-z0-9._-]+\.(?:png|jpe?g|gif|webp|bmp|pdf|txt|zip|tar|gz|7z|mp4|mov|mkv))",
    re.IGNORECASE,
)
_GENERAL_PATH_RE = re.compile(r"""(?:^|\s|[\[\(\{"'«»“”„])((?:[A-Za-z]:)?[/\\][^\s\]\)"'«»“”„]+)""")


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


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] in _QUOTE_LEFT and value[-1] in _QUOTE_RIGHT:
        return value[1:-1].strip()
    return value


def _strip_trailing_marks(value: str) -> str:
    out = value
    while len(out) > 1 and out[-1] in _TRAILING_MARKS:
        out = out[:-1].strip()
    return out


def _sanitize_path_candidate(value: str) -> str:
    if not isinstance(value, str):
        return value
    return _strip_trailing_marks(_strip_wrapping_quotes(value.strip()))


def _find_last_regex_hit(pattern: re.Pattern[str], text: str) -> Optional[str]:
    hits = pattern.findall(text)
    if not hits:
        return None
    return _sanitize_path_candidate(hits[-1])


def _extract_file_command_payload(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    head, sep, rest = s.partition(" ")
    if head != "/file" or not sep:
        return None
    payload = rest.strip()
    return payload or None


def _decode_exact_json_string(payload: str) -> Optional[str]:
    if not payload.startswith('"'):
        return None
    try:
        parsed, end = json.JSONDecoder().raw_decode(payload)
    except Exception:
        return None
    if not isinstance(parsed, str) or payload[end:].strip():
        return None
    return _sanitize_path_candidate(parsed)


def _decode_exact_single_quoted_string(payload: str) -> Optional[str]:
    if len(payload) < 2 or payload[0] != "'" or payload[-1] != "'":
        return None
    return _sanitize_path_candidate(payload[1:-1])


def _find_embedded_path(text: str) -> Optional[str]:
    for pattern in (_ABS_PATH_RE, _REL_PATH_RE, _NAME_PATH_RE, _GENERAL_PATH_RE):
        hit = _find_last_regex_hit(pattern, text)
        if hit:
            return hit
    return None


def _find_path_like_token(text: str) -> Optional[str]:
    parts = [p for p in text.replace('\n', ' ').split(' ') if p]
    for tok in reversed(parts):
        if is_path_like(tok):
            return _sanitize_path_candidate(tok)
    return None


def extract_file_command_path(text: str) -> Optional[str]:
    """Extract the path payload from a dedicated ``/file`` command."""
    payload = _extract_file_command_payload(text)
    if payload is None:
        return None
    for parser in (_decode_exact_json_string, _decode_exact_single_quoted_string):
        parsed = parser(payload)
        if parsed is not None:
            return parsed
    return _sanitize_path_candidate(payload)


def format_file_command(path: Union[str, os.PathLike[str]]) -> str:
    """Return a ``/file`` command that safely round-trips paths with spaces."""
    raw = str(path or "")
    if not raw:
        return "/file"
    if any(ch.isspace() for ch in raw) or '"' in raw:
        return f"/file {json.dumps(raw, ensure_ascii=False)}"
    return f"/file {raw}"


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

    cmd_path = extract_file_command_path(s)
    if cmd_path:
        return cmd_path

    # 1) Прямая передача: вся строка — это путь
    if is_path_like(s):
        return _sanitize_path_candidate(s)
    # 2) Поиск абсолютных/относительных путей внутри текста (последний)
    #    - Unix/Posix: /..., ./..., ../...
    #    - Windows: C:\..., \\server\share\...
    #    Учитываем кавычки и скобки в качестве разделителей вокруг пути
    hit = _find_embedded_path(s)
    if hit:
        return hit
    # 3) Fallback: последний токен как путь, если выглядит как путь
    return _find_path_like_token(s)


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


def _basename_only(name: str, default: str) -> str:
    s = str(name or "").strip()
    s = s.replace("\\", "/").split("/")[-1].strip()
    if not s or s in (".", ".."):
        return default
    return s


def _sanitize_filename_charset(name: str, default: str) -> str:
    try:
        out = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    except Exception:
        out = default
    if not out or out in (".", ".."):
        return default
    return out


def _truncate_filename(name: str, *, max_len: int = 128, max_ext: int = 16) -> str:
    if len(name) <= max_len:
        return name
    root, ext = os.path.splitext(name)
    ext = ext[:max_ext]
    return (root[: max(1, max_len - len(ext))] + ext)[:max_len]


def sanitize_remote_filename(name: str, *, default: str = "file") -> str:
    """Return a safe basename for a remote-provided filename.

    Defense-in-depth for incoming file transfers: prevents path traversal and
    strips control/special characters that may break the terminal UI.
    """
    try:
        safe = _basename_only(str(name or ""), default)
    except Exception:
        safe = default
    safe = _sanitize_filename_charset(safe, default)
    try:
        safe = _truncate_filename(safe, max_len=128, max_ext=16)
    except Exception:
        pass
    return safe


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
    'extract_file_command_path',
    'extract_path_candidate',
    'format_file_command',
    'file_meta_for',
    'sanitize_remote_filename',
    'iter_base64_chunks',
    'progress_percent',
]
