#!/usr/bin/env python3
"""
Эфемерный загрузчик клиента: тянет актуальный dist/client.py с сервера,
проверяет sha256 из dist/version.json и запускает клиент в памяти (без записи на диск).

Запуск:
  PYTHONDONTWRITEBYTECODE=1 UPDATE_URL=https://host/chat SERVER_ADDR=host:7777 \
  python3 -B scripts/bootstrap.py

Либо в один пайп (если bootstrap.py опубликован по HTTPS):
  PYTHONDONTWRITEBYTECODE=1 UPDATE_URL=https://host/chat SERVER_ADDR=host:7777 \
  curl -fsSL https://host/chat/bootstrap.py | python3 -B -

Требуемые переменные окружения:
  UPDATE_URL   — базовый URL директории с dist/version.json и dist/client.py
  SERVER_ADDR  — адрес сервера чата (host:port)

Дополнительно:
  EPHEMERAL=1 — включить «без следов»: клиент не пишет логи/историю и не автообновляется на диск.
"""
import os
import sys
import json
import hashlib
import urllib.request
import base64
import tempfile
import runpy
from pathlib import Path
from typing import Any, Dict, List, Optional


class BootstrapError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _parse_pubkey_env():
    """Read UPDATE_PUBKEY from env and decode into raw 32-byte public key.

    Accepts hex (64 chars) or base64.
    """
    try:
        pk = os.environ.get('UPDATE_PUBKEY')
        if not pk:
            return None
        s = pk.strip()
        # Try hex
        try:
            b = bytes.fromhex(s)
            if len(b) == 32:
                return b
        except Exception:
            pass
        # Try base64
        try:
            b = base64.b64decode(s, validate=True)
            if len(b) == 32:
                return b
        except Exception:
            pass
        return None
    except Exception:
        return None


def _verify_ed25519_signature(message: bytes, signature: bytes, pubkey: bytes) -> bool:
    """Verify Ed25519 signature using available backends.

    Tries cryptography, then ed25519 (pure Python). Returns False if all fail.
    """
    # cryptography backend
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # type: ignore
        from cryptography.exceptions import InvalidSignature  # type: ignore
        try:
            pk = Ed25519PublicKey.from_public_bytes(pubkey)
            pk.verify(signature, message)
            return True
        except InvalidSignature:
            return False
        except Exception:
            pass
    except Exception:
        pass
    # ed25519 pure-python backend
    try:
        import ed25519  # type: ignore
        try:
            vk = ed25519.VerifyingKey(pubkey)
            vk.verify(signature, message)
            return True
        except ed25519.BadSignatureError:
            return False
        except Exception:
            pass
    except Exception:
        pass
    return False


def _is_truthy_env(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default)).strip().lower()
    return v in ("1", "true", "yes", "on")


def _require_update_url() -> str:
    base = os.environ.get("UPDATE_URL")
    if not base:
        raise BootstrapError(2, "UPDATE_URL is not set")
    return base


def _resolve_manifest_verification_pubkey() -> Optional[bytes]:
    # Security: require manifest signature verification unless explicitly in insecure dev mode.
    insecure_dev = _is_truthy_env("ALLOW_INSECURE_DEV", default="0")
    pub = _parse_pubkey_env()
    if pub is None and not insecure_dev:
        raise BootstrapError(
            7,
            "UPDATE_PUBKEY is required for manifest signature verification. Set UPDATE_PUBKEY (hex/base64) or ALLOW_INSECURE_DEV=1 for local testing.",
        )
    if pub is None and insecure_dev:
        print(
            "[bootstrap] WARNING: ALLOW_INSECURE_DEV=1: running without manifest signature verification (development/testing only).",
            file=sys.stderr,
        )
    return pub


def _fetch_bytes(url: str, *, timeout: float, label: str) -> bytes:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "yagodka-bootstrap"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        raise BootstrapError(3, f"Failed to fetch {label}: {e}") from e


def _parse_manifest_signature(sig_b: bytes) -> Optional[bytes]:
    try:
        if len(sig_b) == 64:
            return sig_b
        sig_txt = sig_b.strip()
        try:
            return base64.b64decode(sig_txt, validate=True)
        except Exception:
            try:
                return bytes.fromhex(sig_txt.decode("ascii"))
            except Exception:
                return None
    except Exception:
        return None


def _load_manifest(mani_b: bytes) -> Dict[str, Any]:
    try:
        return json.loads(mani_b.decode("utf-8"))
    except Exception as e:
        raise BootstrapError(4, f"Failed to parse manifest.json: {e}") from e


def _extract_manifest_entries(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for item in manifest.get("files") or []:
        path_txt = str(item.get("path") or "").strip()
        sha = str(item.get("sha256") or "").strip()
        try:
            size = int(item.get("size") or 0)
        except Exception:
            size = 0
        p = Path(path_txt)
        if not path_txt or p.is_absolute() or ".." in p.parts or len(sha) != 64 or size <= 0:
            continue
        entries.append({"path": path_txt, "sha256": sha, "size": size})
    if not entries:
        raise BootstrapError(4, "Manifest has no valid entries")
    return entries


def _download_one_entry(base: str, entry: Dict[str, Any], tmp_root: Path) -> None:
    rel = Path(entry["path"])
    dest = tmp_root / rel
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise BootstrapError(5, f"Failed to prepare dest dir for {rel}: {e}") from e
    url = base.rstrip("/") + "/" + rel.as_posix()
    try:
        blob = _fetch_bytes(url, timeout=10.0, label=str(rel))
    except BootstrapError as e:
        raise BootstrapError(5, e.message) from e
    if len(blob) != entry["size"]:
        raise BootstrapError(6, f"Size mismatch for {rel}")
    if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise BootstrapError(6, f"Hash mismatch for {rel}")
    try:
        dest.write_bytes(blob)
    except Exception as e:
        raise BootstrapError(5, f"Failed to write {rel}: {e}") from e
    try:
        dest.chmod(0o755 if dest.suffix == ".py" else 0o644)
    except Exception:
        pass


def _download_entries(base: str, entries: List[Dict[str, Any]], tmp_root: Path) -> None:
    for entry in entries:
        _download_one_entry(base, entry, tmp_root)


def _verify_manifest_signature_if_enabled(base: str, mani_b: bytes, pub: Optional[bytes]) -> None:
    if pub is None:
        return
    sig_b = _fetch_bytes(base.rstrip("/") + "/manifest.sig", timeout=6.0, label="manifest.sig")
    sig = _parse_manifest_signature(sig_b)
    if not sig or len(sig) != 64:
        raise BootstrapError(7, "Invalid manifest signature format")
    if not _verify_ed25519_signature(mani_b, sig, pub):
        raise BootstrapError(7, "Manifest signature verification failed")


def _find_client_path(tmp_root: Path, entries: List[Dict[str, Any]]) -> Path:
    client_entry = next((e for e in entries if Path(e["path"]).name == "client.py"), None)
    if not client_entry:
        raise BootstrapError(4, "No client.py in manifest")
    client_path = tmp_root / Path(client_entry["path"])
    if not client_path.exists():
        raise BootstrapError(5, "client.py missing after download")
    return client_path


def _exec_client(client_path: Path, tmp_root: Path) -> None:
    os.environ.setdefault("EPHEMERAL", "1")
    os.environ.setdefault("CLIENT_AUTO_UPDATE", "0")
    sys.path.insert(0, str(tmp_root))
    runpy.run_path(str(client_path), run_name="__main__")


def main() -> int:
    try:
        base = _require_update_url()
        pub = _resolve_manifest_verification_pubkey()

        mani_b = _fetch_bytes(base.rstrip("/") + "/manifest.json", timeout=6.0, label="manifest.json")
        _verify_manifest_signature_if_enabled(base, mani_b, pub)

        manifest = _load_manifest(mani_b)
        entries = _extract_manifest_entries(manifest)

        tmp_root = Path(tempfile.mkdtemp(prefix="yagodka-bootstrap-"))
        try:
            _download_entries(base, entries, tmp_root)
            client_path = _find_client_path(tmp_root, entries)
            _exec_client(client_path, tmp_root)
            return 0
        finally:
            pass
    except BootstrapError as e:
        print(f"[bootstrap] {e.message}", file=sys.stderr)
        return e.code


if __name__ == '__main__':
    raise SystemExit(main())
