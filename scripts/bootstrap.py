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
from pathlib import Path


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


def main() -> int:
    base = os.environ.get('UPDATE_URL')
    if not base:
        print('[bootstrap] UPDATE_URL is not set', file=sys.stderr)
        return 2
    # Security: require manifest signature verification unless explicitly in insecure dev mode.
    insecure_dev = str(os.environ.get('ALLOW_INSECURE_DEV', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
    pub = _parse_pubkey_env()
    if pub is None and not insecure_dev:
        print('[bootstrap] UPDATE_PUBKEY is required for manifest signature verification. Set UPDATE_PUBKEY (hex/base64) or ALLOW_INSECURE_DEV=1 for local testing.', file=sys.stderr)
        return 7
    if pub is None and insecure_dev:
        print('[bootstrap] WARNING: ALLOW_INSECURE_DEV=1: running without manifest signature verification (development/testing only).', file=sys.stderr)
    mani_url = base.rstrip('/') + '/manifest.json'
    try:
        mani_req = urllib.request.Request(mani_url, headers={'User-Agent': 'yagodka-bootstrap'})
        with urllib.request.urlopen(mani_req, timeout=6.0) as r:
            mani_b = r.read()
    except Exception as e:
        print(f'[bootstrap] Failed to fetch manifest.json: {e}', file=sys.stderr)
        return 3
    if pub is not None:
        try:
            sig_req = urllib.request.Request(base.rstrip('/') + '/manifest.sig', headers={'User-Agent': 'yagodka-bootstrap'})
            with urllib.request.urlopen(sig_req, timeout=6.0) as rs:
                sig_b = rs.read()
        except Exception as e:
            print(f'[bootstrap] Failed to fetch manifest.sig: {e}', file=sys.stderr)
            return 3
        sig = None
        try:
            if len(sig_b) == 64:
                sig = sig_b
            else:
                sig_txt = sig_b.strip()
                try:
                    sig = base64.b64decode(sig_txt, validate=True)
                except Exception:
                    try:
                        sig = bytes.fromhex(sig_txt.decode('ascii'))
                    except Exception:
                        sig = None
        except Exception:
            sig = None
        if not sig or len(sig) != 64:
            print('[bootstrap] Invalid manifest signature format', file=sys.stderr)
            return 7
        if not _verify_ed25519_signature(mani_b, sig, pub):
            print('[bootstrap] Manifest signature verification failed', file=sys.stderr)
            return 7
    try:
        manifest = json.loads(mani_b.decode('utf-8'))
    except Exception as e:
        print(f'[bootstrap] Failed to parse manifest.json: {e}', file=sys.stderr)
        return 4
    entries = []
    for item in manifest.get('files') or []:
        path_txt = str(item.get('path') or '').strip()
        sha = str(item.get('sha256') or '').strip()
        try:
            size = int(item.get('size') or 0)
        except Exception:
            size = 0
        p = Path(path_txt)
        if not path_txt or p.is_absolute() or '..' in p.parts or len(sha) != 64 or size <= 0:
            continue
        entries.append({'path': path_txt, 'sha256': sha, 'size': size})
    if not entries:
        print('[bootstrap] Manifest has no valid entries', file=sys.stderr)
        return 4
    tmp_root = Path(tempfile.mkdtemp(prefix="yagodka-bootstrap-"))
    try:
        for entry in entries:
            rel = Path(entry['path'])
            dest = tmp_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = base.rstrip('/') + '/' + rel.as_posix()
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'yagodka-bootstrap'}), timeout=10.0) as r:
                    blob = r.read()
            except Exception as e:
                print(f"[bootstrap] Failed to fetch {rel}: {e}", file=sys.stderr)
                return 5
            if len(blob) != entry['size']:
                print(f"[bootstrap] Size mismatch for {rel}", file=sys.stderr)
                return 6
            if hashlib.sha256(blob).hexdigest() != entry['sha256']:
                print(f"[bootstrap] Hash mismatch for {rel}", file=sys.stderr)
                return 6
            dest.write_bytes(blob)
            try:
                dest.chmod(0o755 if dest.suffix == '.py' else 0o644)
            except Exception:
                pass
        client_entry = next((e for e in entries if Path(e['path']).name == 'client.py'), None)
        if not client_entry:
            print('[bootstrap] No client.py in manifest', file=sys.stderr)
            return 4
        client_path = tmp_root / Path(client_entry['path'])
        if not client_path.exists():
            print('[bootstrap] client.py missing after download', file=sys.stderr)
            return 5
        os.environ.setdefault('EPHEMERAL', '1')
        os.environ.setdefault('CLIENT_AUTO_UPDATE', '0')
        sys.path.insert(0, str(tmp_root))
        g = {'__name__': '__main__', '__file__': str(client_path)}
        with open(client_path, 'r', encoding='utf-8') as f:
            code = f.read()
        exec(compile(code, str(client_path), 'exec'), g)
        return 0
    finally:
        pass


if __name__ == '__main__':
    raise SystemExit(main())
