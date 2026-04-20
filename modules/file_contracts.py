from __future__ import annotations

from pathlib import Path

SAFE_DOWNLOAD_MIME_BY_EXT: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".ico": "image/x-icon",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".avif": "image/avif",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".ogv": "video/ogg",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp",
    ".3g2": "video/3gpp2",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
    ".pdf": "application/pdf",
}

ATTACHMENT_MIME_BY_EXT: dict[str, str] = {
    **SAFE_DOWNLOAD_MIME_BY_EXT,
    ".svg": "image/svg+xml",
    ".txt": "text/plain",
    ".json": "application/json",
}

IMAGE_EXTS = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".png",
    ".tiff",
    ".webp",
}

VIDEO_EXTS = {
    ".3gp",
    ".3gpp",
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogv",
    ".webm",
    ".wmv",
}

AUDIO_EXTS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}


def extract_name_ext(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    leaf = raw.split("?", 1)[0].split("#", 1)[0].split("/")[-1].split("\\")[-1].strip().lower()
    if not leaf:
        return ""
    return Path(leaf).suffix.lower()


def resolve_ext(*, name: str = "", path: str | Path | None = None) -> str:
    ext = extract_name_ext(name)
    if ext:
        return ext
    if path is None:
        return ""
    try:
        return Path(path).suffix.lower()
    except Exception:
        return ""


def guess_attachment_mime_by_name(name: str) -> str | None:
    ext = extract_name_ext(name)
    if not ext:
        return None
    return ATTACHMENT_MIME_BY_EXT.get(ext)


def guess_safe_download_mime_by_name(name: str) -> str | None:
    ext = extract_name_ext(name)
    if not ext:
        return None
    return SAFE_DOWNLOAD_MIME_BY_EXT.get(ext)


def is_image_like(mime: str | None, *, name: str = "", path: str | Path | None = None) -> bool:
    mime_norm = str(mime or "").strip().lower()
    if mime_norm.startswith("image/"):
        return True
    return resolve_ext(name=name, path=path) in IMAGE_EXTS


def is_video_like(mime: str | None, *, name: str = "", path: str | Path | None = None) -> bool:
    mime_norm = str(mime or "").strip().lower()
    if mime_norm.startswith("video/"):
        return True
    return resolve_ext(name=name, path=path) in VIDEO_EXTS


def is_audio_like(mime: str | None, *, name: str = "", path: str | Path | None = None) -> bool:
    mime_norm = str(mime or "").strip().lower()
    if mime_norm.startswith("audio/"):
        return True
    return resolve_ext(name=name, path=path) in AUDIO_EXTS


__all__ = [
    "ATTACHMENT_MIME_BY_EXT",
    "AUDIO_EXTS",
    "IMAGE_EXTS",
    "SAFE_DOWNLOAD_MIME_BY_EXT",
    "VIDEO_EXTS",
    "extract_name_ext",
    "guess_attachment_mime_by_name",
    "guess_safe_download_mime_by_name",
    "is_audio_like",
    "is_image_like",
    "is_video_like",
    "resolve_ext",
]
