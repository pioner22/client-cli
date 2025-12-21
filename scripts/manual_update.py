#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import json
import os
import os.path as _ospath
import sys
import tarfile
import urllib.error
import urllib.request
import tempfile
import shutil
import time
from pathlib import Path
from typing import Optional

EXPECTED_ROOT_DIRS = {"bin", "modules", "config", "scripts", "var"}
FORBIDDEN_ROOT_FILES = {
    "client.py",
    "client.py.bak",
    "bootstrap.py",
    "pubkey.txt",
    "schema.json",
    "version.json",
    "version.json.bak",
}
REQUIRED_VAR_SUBDIRS = {"files", "history", "log", "update", "users"}

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _is_within_root(root: Path, target: Path) -> bool:
    """Return True if target (after resolving symlinks) stays within root."""
    try:
        root_real = _ospath.realpath(str(root))
        target_real = _ospath.realpath(str(target))
        return _ospath.commonpath([root_real, target_real]) == root_real
    except Exception:
        return False


def fetch(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "yagodka-manual-update"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_with_progress(url: str, expected_size: int, label: str, timeout: float = 20.0, enable: bool = True) -> bytes:
    """Stream download with a simple progress bar (TTY only)."""
    try_progress = enable and expected_size > 0 and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not try_progress:
        return fetch(url, timeout=timeout)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "yagodka-manual-update"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = expected_size
            buf = bytearray()
            chunk_size = 64 * 1024
            bar_w = 30
            last_pct = -1
            start = time.time()
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                buf.extend(chunk)
                pct = int(min(100, (len(buf) / total) * 100)) if total else 0
                if pct != last_pct:
                    filled = int((pct / 100) * bar_w)
                    bar = "#" * filled + "-" * (bar_w - filled)
                    sys.stdout.write(f"\r[client:update] {label} [{bar}] {pct:3d}%")
                    sys.stdout.flush()
                    last_pct = pct
            if last_pct >= 0:
                elapsed = time.time() - start
                sys.stdout.write(f"\r[client:update] {label} [{'#'*bar_w}] 100% ({elapsed:.1f}s)\n")
                sys.stdout.flush()
            return bytes(buf)
    except Exception:
        return fetch(url, timeout=timeout)


def _parse_pubkey(root: Path) -> Optional[bytes]:
    """Resolve update pubkey from env, config/pubkey.txt or client_config.json."""
    candidates: list[str] = []
    pk = os.environ.get("UPDATE_PUBKEY")
    if pk:
        candidates.append(pk.strip())
    try:
        cfg_pk = (root / "config" / "pubkey.txt").read_text(encoding="utf-8").strip()
        if cfg_pk:
            candidates.append(cfg_pk)
    except Exception:
        pass
    try:
        cfg_json = root / "config" / "client_config.json"
        if cfg_json.exists():
            cfg = json.loads(cfg_json.read_text(encoding="utf-8"))
            txt = cfg.get("update_pubkey") or ""
            if isinstance(txt, str) and txt.strip():
                candidates.append(txt.strip())
    except Exception:
        pass
    for s in candidates:
        try:
            b = bytes.fromhex(s)
            if len(b) == 32:
                return b
        except Exception:
            pass
        try:
            b = base64.b64decode(s, validate=True)
            if len(b) == 32:
                return b
        except Exception:
            pass
    return None


def _decode_signature(sig_b: bytes) -> Optional[bytes]:
    if len(sig_b) == 64:
        return sig_b
    s = sig_b.strip()
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        pass
    try:
        return bytes.fromhex(s.decode("ascii"))
    except Exception:
        return None
    return None


def _progress_enabled() -> bool:
    val = str(os.environ.get("CLIENT_UPDATE_PROGRESS", "1")).strip().lower()
    if val in ("0", "no", "false", "off"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _verify_signature(message: bytes, signature: bytes, pubkey: bytes) -> Optional[bool]:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # type: ignore
        pk = Ed25519PublicKey.from_public_bytes(pubkey)
        pk.verify(signature, message)
        return True
    except Exception:
        pass
    try:
        import ed25519  # type: ignore
        vk = ed25519.VerifyingKey(pubkey)
        vk.verify(signature, message)
        return True
    except Exception:
        return None


def _fix_structure(root: Path) -> bool:
    """
    Ensure client tree has expected layout and no forbidden files.
    Returns True if structure is OK (after remediation), False otherwise.
    """
    ok = True
    try:
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        present_dirs = {p.name for p in root.iterdir() if p.is_dir()}
        for d in EXPECTED_ROOT_DIRS:
            if d not in present_dirs:
                try:
                    (root / d).mkdir(parents=True, exist_ok=True)
                except Exception:
                    ok = False
        for fname in FORBIDDEN_ROOT_FILES:
            try:
                (root / fname).unlink()
            except FileNotFoundError:
                pass
            except Exception:
                ok = False
        # Remove stray runtime/modules duplicates
        for stray in (root / "runtime", root / "var" / "runtime", root / "modules 2"):
            try:
                if stray.exists():
                    import shutil
                    shutil.rmtree(stray)
            except Exception:
                ok = False
        var_dir = root / "var"
        for sub in REQUIRED_VAR_SUBDIRS:
            try:
                (var_dir / sub).mkdir(parents=True, exist_ok=True)
            except Exception:
                ok = False
    except Exception:
        ok = False
    return ok


def _safe_extract_tar(tar_path: Path, dest: Path, preserve_var: bool = True, preserve_config: bool = True) -> bool:
    """Extract tarball to dest.

    preserve_var: keep local var/ contents (logs/history/users).
    preserve_config: keep existing config/client_config.json.
    """
    try:
        dest = dest.resolve()
        tmpdir = dest / ".update_tmp_extract"
        if tmpdir.exists():
            # Refuse to operate on a symlink to avoid deleting/copying outside dest.
            if tmpdir.is_symlink() or not tmpdir.is_dir():
                return False
            shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                safe_members: list[tarfile.TarInfo] = []
                for m in tf.getmembers():
                    name = str(m.name or "")
                    # Normalize separators so ".." checks work even with backslashes.
                    norm = name.replace("\\", "/")
                    p = Path(norm)
                    if not norm or p.is_absolute() or ".." in p.parts:
                        continue
                    # Only allow regular files and directories; drop links/devices/etc.
                    if not (m.isfile() or m.isdir()):
                        continue
                    safe_members.append(m)
                if not safe_members:
                    return False
                # Extract into tmpdir without creating symlinks/hardlinks.
                for m in safe_members:
                    norm = str(m.name or "").replace("\\", "/")
                    out_path = tmpdir / norm
                    if not _is_within_root(tmpdir, out_path.parent):
                        return False
                    if m.isdir():
                        out_path.mkdir(parents=True, exist_ok=True)
                        continue
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(m)
                    if src is None:
                        continue
                    with src:
                        with open(out_path, "wb") as dst_f:
                            shutil.copyfileobj(src, dst_f)

            entries = list(tmpdir.iterdir())
            if not entries:
                return False
            # If archive has a single top-level dir, drop it
            roots = {p.relative_to(tmpdir).parts[0] for p in entries}
            base = tmpdir
            if len(roots) == 1:
                candidate = tmpdir / next(iter(roots))
                if candidate.is_dir():
                    base = candidate
            # Ensure extracted release fully replaces managed dirs to avoid stale/extra files.
            preserved_cfg: Optional[bytes] = None
            cfg_rel = Path("config/client_config.json")
            cfg_path = dest / cfg_rel
            if preserve_config:
                try:
                    if cfg_path.exists() and cfg_path.is_file():
                        preserved_cfg = cfg_path.read_bytes()
                except Exception:
                    preserved_cfg = None
            try:
                managed_top = {p.name for p in base.iterdir()}
            except Exception:
                managed_top = set()
            for name in sorted(managed_top):
                if preserve_var and name == "var":
                    continue
                if name == "config":
                    if not (base / "config").exists():
                        continue
                    config_dir = dest / "config"
                    if config_dir.exists():
                        if config_dir.is_symlink() or not config_dir.is_dir():
                            return False
                        try:
                            shutil.rmtree(config_dir, ignore_errors=False)
                        except Exception:
                            return False
                    continue
                target = dest / name
                if not target.exists():
                    continue
                if target.is_symlink():
                    return False
                if target.is_dir():
                    try:
                        shutil.rmtree(target, ignore_errors=False)
                    except Exception:
                        return False
                else:
                    target.unlink(missing_ok=True)

            for f in base.rglob("*"):
                if f.is_dir():
                    continue
                if f.is_symlink():
                    # Should not happen (we don't extract links), but keep defensive.
                    return False
                rel = f.relative_to(base)
                if preserve_var and rel.parts and rel.parts[0] == "var":
                    continue
                if preserve_config and rel == cfg_rel and preserved_cfg is not None:
                    continue
                target = dest / rel
                # Ensure the destination parent doesn't escape via symlinks.
                if not _is_within_root(dest, target.parent):
                    return False
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
                try:
                    with os.fdopen(fd, "wb") as out_f, open(f, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
                    try:
                        os.chmod(tmp_name, 0o755 if target.name.endswith(".py") else 0o644)
                    except Exception:
                        pass
                    os.replace(tmp_name, target)
                finally:
                    try:
                        if _ospath.exists(tmp_name):
                            os.unlink(tmp_name)
                    except Exception:
                        pass
            if preserve_config and preserved_cfg is not None:
                try:
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    cfg_path.write_bytes(preserved_cfg)
                except Exception:
                    return False
            return True
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        return False


def main() -> int:
    override_root = os.environ.get("MANUAL_UPDATE_ROOT", "").strip()
    root = Path(override_root).expanduser().resolve() if override_root else Path(__file__).resolve().parents[1]
    base = os.environ.get("UPDATE_URL", "").strip()
    if not base and len(sys.argv) >= 2:
        base = sys.argv[1].strip()
    if not base:
        try:
            cfg = json.loads((root / "config" / "client_config.json").read_text(encoding="utf-8"))
            base = str(cfg.get("update_url") or "").strip()
        except Exception:
            base = ""
    if not base:
        print("[client:update] UPDATE_URL is required", file=sys.stderr)
        return 2
    base = base.rstrip("/")
    if not _fix_structure(root):
        print("[client:update] Client structure invalid and could not be fixed", file=sys.stderr)
        return 2
    bin_path = root / "bin" / "client.py"
    state_dir = root / "var" / "update"
    state_dir.mkdir(parents=True, exist_ok=True)

    try:
        mani_b = fetch(base + "/manifest.json")
    except Exception as exc:  # pragma: no cover - network
        print(f"[client:update] Failed to download manifest.json: {exc}", file=sys.stderr)
        return 3

    pub = _parse_pubkey(root)
    # TOFU (explicit): если ключ не задан, можем получить pubkey.txt и сохранить
    if pub is None and _env_bool("MANUAL_UPDATE_ALLOW_TOFU", False):
        try:
            fetched_pk = fetch(base + "/pubkey.txt")
            pk_txt = fetched_pk.decode("ascii", errors="ignore").strip()
            try:
                b = bytes.fromhex(pk_txt)
                if len(b) == 32:
                    pub = b
            except Exception:
                pub = None
            if pub:
                dest_pk = root / "config" / "pubkey.txt"
                dest_pk.parent.mkdir(parents=True, exist_ok=True)
                dest_pk.write_text(pk_txt + "\n", encoding="utf-8")
        except Exception:
            pub = None

    if not pub:
        print("[client:update] UPDATE_PUBKEY is required (env or config/pubkey.txt)", file=sys.stderr)
        return 4

    try:
        sig_b = fetch(base + "/manifest.sig")
        sig = _decode_signature(sig_b)
        if not sig or len(sig) != 64:
            print("[client:update] Invalid manifest signature format", file=sys.stderr)
            return 4
        ver_ok = _verify_signature(mani_b, sig, pub)
        if ver_ok is False:
            print("[client:update] Manifest signature verification failed", file=sys.stderr)
            return 4
        if ver_ok is None:
            print("[client:update] No ed25519 backend available; install 'cryptography' or 'ed25519'", file=sys.stderr)
            return 4
    except Exception as exc:  # pragma: no cover - network
        print(f"[client:update] Failed to verify manifest signature: {exc}", file=sys.stderr)
        return 4

    try:
        manifest = json.loads(mani_b.decode("utf-8"))
    except Exception as exc:
        print(f"[client:update] Bad manifest.json: {exc}", file=sys.stderr)
        return 4

    entries = []
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
        print("[client:update] Manifest has no valid entries", file=sys.stderr)
        return 4

    require_archive = str(os.environ.get("ALLOW_NO_ARCHIVE", "")).strip().lower() not in ("1", "true", "yes", "on")
    has_archive = any(e["path"] == "client-release.tar.gz" for e in entries)
    if require_archive and not has_archive:
        print("[client:update] Manifest missing client-release.tar.gz; update server dist and retry", file=sys.stderr)
        return 4

    def _map_destination(rel: Path) -> Path:
        if rel.as_posix() == "client.py":
            return root / "bin" / "client.py"
        if rel.as_posix() == "schema.json":
            return root / "config" / "protocol" / "schema.json"
        if rel.as_posix() == "bootstrap.py":
            return root / "scripts" / "bootstrap.py"
        if rel.as_posix() == "pubkey.txt":
            return root / "config" / "pubkey.txt"
        if rel.as_posix() == "version.json":
            return root / "var" / "update" / "version.json"
        if rel.as_posix() == "scripts/manual_update.py":
            return root / "scripts" / "manual_update.py"
        if rel.as_posix() == "client-release.tar.gz":
            return state_dir / "client-release.tar.gz"
        return root / rel

    # Fast-path: if manifest/root_hash matches cached one and all files already match hashes, skip
    try:
        cached = json.loads((state_dir / "manifest.json").read_text(encoding="utf-8"))
        if cached.get("root_hash") == manifest.get("root_hash") and cached.get("version") == manifest.get("version"):
            all_ok = True
            for entry in entries:
                rel = Path(entry["path"])
                dest = _map_destination(rel).resolve()
                if not dest.exists():
                    all_ok = False
                    break
                try:
                    h = hashlib.sha256()
                    with open(dest, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
                    if h.hexdigest() != entry["sha256"] or dest.stat().st_size != entry["size"]:
                        all_ok = False
                        break
                except Exception:
                    all_ok = False
                    break
            if all_ok:
                print(f"[client:update] Уже установлена версия {manifest.get('version')}, обновление не требуется")
                return 0
    except Exception:
        pass

    changed = 0
    tar_dest: Optional[Path] = None
    tar_hash_ok = False
    progress = _progress_enabled()
    for entry in entries:
        rel = Path(entry["path"])
        dest = _map_destination(rel).resolve()
        if rel.as_posix() == "client-release.tar.gz":
            tar_dest = dest
        if root not in dest.parents and root != dest:
            print(f"[client:update] Unsafe path in manifest: {rel}", file=sys.stderr)
            return 4
        need = True
        if dest.exists():
            try:
                h = hashlib.sha256()
                with open(dest, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                cur_hash = h.hexdigest()
                if cur_hash == entry["sha256"] and dest.stat().st_size == entry["size"]:
                    need = False
                    if rel.as_posix() == "client-release.tar.gz":
                        tar_hash_ok = True
            except Exception:
                need = True
        if not need:
            continue
        try:
            blob = fetch_with_progress(base + "/" + rel.as_posix(), entry["size"], rel.as_posix(), timeout=20.0, enable=progress)
        except Exception as exc:  # pragma: no cover - network
            print(f"[client:update] Failed to download {rel}: {exc}", file=sys.stderr)
            return 5
        if len(blob) != entry["size"]:
            print(f"[client:update] Size mismatch for {rel}", file=sys.stderr)
            return 5
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            print(f"[client:update] Hash mismatch for {rel}", file=sys.stderr)
            return 5
        tmp_path = state_dir / (rel.as_posix().replace("/", "_") + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(blob)
        tmp_path.chmod(0o755 if rel.name.endswith(".py") else 0o644)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.replace(dest)
        changed += 1
        if rel.as_posix() == "client-release.tar.gz":
            tar_dest = dest
            tar_hash_ok = True

    # Final cleanup to remove any forbidden leftovers
    _fix_structure(root)

    # Apply full archive if present in manifest (already verified hash/size)
    if tar_dest and tar_dest.exists() and tar_hash_ok:
        if not _safe_extract_tar(tar_dest, root, preserve_var=True, preserve_config=True):
            print("[client:update] Failed to extract client-release.tar.gz", file=sys.stderr)
            return 5
    elif tar_dest:
        print("[client:update] Warning: client-release.tar.gz not applied (hash/size mismatch)", file=sys.stderr)
    else:
        print("[client:update] Warning: client-release.tar.gz not applied (missing in manifest)", file=sys.stderr)

    (state_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[client:update] Проверено файлов: {len(entries)}, обновлено: {changed}, версия: {manifest.get('version')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
