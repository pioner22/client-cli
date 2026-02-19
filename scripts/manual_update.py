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


def _collect_pubkey_candidates(root: Path) -> list[str]:
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
    return candidates


def _decode_pubkey_candidate(candidate: str) -> Optional[bytes]:
    try:
        b = bytes.fromhex(candidate)
        if len(b) == 32:
            return b
    except Exception:
        pass
    try:
        b = base64.b64decode(candidate, validate=True)
        if len(b) == 32:
            return b
    except Exception:
        pass
    return None


def _parse_pubkey(root: Path) -> Optional[bytes]:
    """Resolve update pubkey from env, config/pubkey.txt or client_config.json."""
    for candidate in _collect_pubkey_candidates(root):
        decoded = _decode_pubkey_candidate(candidate)
        if decoded is not None:
            return decoded
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


def _ensure_expected_root_dirs(root: Path) -> bool:
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
    except Exception:
        return False
    return ok


def _remove_forbidden_root_files(root: Path) -> bool:
    ok = True
    for fname in FORBIDDEN_ROOT_FILES:
        try:
            (root / fname).unlink()
        except FileNotFoundError:
            pass
        except Exception:
            ok = False
    return ok


def _remove_stray_dirs(root: Path) -> bool:
    ok = True
    for stray in (root / "runtime", root / "var" / "runtime", root / "modules 2"):
        try:
            if stray.exists():
                shutil.rmtree(stray)
        except Exception:
            ok = False
    return ok


def _ensure_var_subdirs(root: Path) -> bool:
    ok = True
    var_dir = root / "var"
    for sub in REQUIRED_VAR_SUBDIRS:
        try:
            (var_dir / sub).mkdir(parents=True, exist_ok=True)
        except Exception:
            ok = False
    return ok


def _fix_structure(root: Path) -> bool:
    """
    Ensure client tree has expected layout and no forbidden files.
    Returns True if structure is OK (after remediation), False otherwise.
    """
    return all(
        (
            _ensure_expected_root_dirs(root),
            _remove_forbidden_root_files(root),
            _remove_stray_dirs(root),
            _ensure_var_subdirs(root),
        )
    )


def _prepare_tmp_extract_dir(dest: Path) -> Optional[Path]:
    tmpdir = dest / ".update_tmp_extract"
    if tmpdir.exists():
        if tmpdir.is_symlink() or not tmpdir.is_dir():
            return None
        shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir


def _collect_safe_members(tf: tarfile.TarFile) -> list[tarfile.TarInfo]:
    safe_members: list[tarfile.TarInfo] = []
    for member in tf.getmembers():
        name = str(member.name or "")
        norm = name.replace("\\", "/")
        p = Path(norm)
        if not norm or p.is_absolute() or ".." in p.parts:
            continue
        if not (member.isfile() or member.isdir()):
            continue
        safe_members.append(member)
    return safe_members


def _extract_safe_members(tf: tarfile.TarFile, tmpdir: Path, members: list[tarfile.TarInfo]) -> bool:
    for member in members:
        norm = str(member.name or "").replace("\\", "/")
        out_path = tmpdir / norm
        if not _is_within_root(tmpdir, out_path.parent):
            return False
        if member.isdir():
            out_path.mkdir(parents=True, exist_ok=True)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        src = tf.extractfile(member)
        if src is None:
            continue
        with src:
            with open(out_path, "wb") as dst_f:
                shutil.copyfileobj(src, dst_f)
    return True


def _resolve_extract_base(tmpdir: Path) -> Optional[Path]:
    entries = list(tmpdir.iterdir())
    if not entries:
        return None
    roots = {p.relative_to(tmpdir).parts[0] for p in entries}
    if len(roots) == 1:
        candidate = tmpdir / next(iter(roots))
        if candidate.is_dir():
            return candidate
    return tmpdir


def _read_preserved_cfg(dest: Path, *, preserve_config: bool, cfg_rel: Path) -> Optional[bytes]:
    if not preserve_config:
        return None
    cfg_path = dest / cfg_rel
    try:
        if cfg_path.exists() and cfg_path.is_file():
            return cfg_path.read_bytes()
    except Exception:
        return None
    return None


def _cleanup_top_entry(name: str, *, base: Path, dest: Path, preserve_var: bool) -> bool:
    if preserve_var and name == "var":
        return True
    if name == "config":
        return _cleanup_config_top(base, dest)
    return _cleanup_target_path(dest / name)


def _cleanup_config_top(base: Path, dest: Path) -> bool:
    if not (base / "config").exists():
        return True
    config_dir = dest / "config"
    if not config_dir.exists():
        return True
    if config_dir.is_symlink() or not config_dir.is_dir():
        return False
    try:
        shutil.rmtree(config_dir, ignore_errors=False)
        return True
    except Exception:
        return False


def _cleanup_target_path(target: Path) -> bool:
    if not target.exists():
        return True
    if target.is_symlink():
        return False
    if target.is_dir():
        try:
            shutil.rmtree(target, ignore_errors=False)
            return True
        except Exception:
            return False
    target.unlink(missing_ok=True)
    return True


def _cleanup_managed_top(base: Path, dest: Path, *, preserve_var: bool) -> bool:
    try:
        managed_top = {p.name for p in base.iterdir()}
    except Exception:
        managed_top = set()
    for name in sorted(managed_top):
        if not _cleanup_top_entry(name, base=base, dest=dest, preserve_var=preserve_var):
            return False
    return True


def _should_skip_copy(
    rel: Path,
    *,
    preserve_var: bool,
    preserve_config: bool,
    cfg_rel: Path,
    preserved_cfg: Optional[bytes],
) -> bool:
    if preserve_var and rel.parts and rel.parts[0] == "var":
        return True
    if preserve_config and rel == cfg_rel and preserved_cfg is not None:
        return True
    return False


def _copy_file_atomic(src_file: Path, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as out_f, open(src_file, "rb") as in_f:
            shutil.copyfileobj(in_f, out_f)
        try:
            os.chmod(tmp_name, 0o755 if target.name.endswith(".py") else 0o644)
        except Exception:
            pass
        os.replace(tmp_name, target)
        return True
    finally:
        try:
            if _ospath.exists(tmp_name):
                os.unlink(tmp_name)
        except Exception:
            pass


def _copy_extracted_files(
    base: Path,
    dest: Path,
    *,
    preserve_var: bool,
    preserve_config: bool,
    cfg_rel: Path,
    preserved_cfg: Optional[bytes],
) -> bool:
    for src_file in base.rglob("*"):
        if src_file.is_dir():
            continue
        if src_file.is_symlink():
            return False
        rel = src_file.relative_to(base)
        if _should_skip_copy(
            rel,
            preserve_var=preserve_var,
            preserve_config=preserve_config,
            cfg_rel=cfg_rel,
            preserved_cfg=preserved_cfg,
        ):
            continue
        target = dest / rel
        if not _is_within_root(dest, target.parent):
            return False
        if not _copy_file_atomic(src_file, target):
            return False
    return True


def _restore_preserved_cfg(dest: Path, cfg_rel: Path, preserved_cfg: Optional[bytes], *, preserve_config: bool) -> bool:
    if not (preserve_config and preserved_cfg is not None):
        return True
    cfg_path = dest / cfg_rel
    try:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_bytes(preserved_cfg)
        return True
    except Exception:
        return False


def _safe_extract_tar(tar_path: Path, dest: Path, preserve_var: bool = True, preserve_config: bool = True) -> bool:
    """Extract tarball to dest.

    preserve_var: keep local var/ contents (logs/history/users).
    preserve_config: keep existing config/client_config.json.
    """
    try:
        dest = dest.resolve()
        tmpdir = _prepare_tmp_extract_dir(dest)
        if tmpdir is None:
            return False
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                safe_members = _collect_safe_members(tf)
                if not safe_members:
                    return False
                if not _extract_safe_members(tf, tmpdir, safe_members):
                    return False
            base = _resolve_extract_base(tmpdir)
            if base is None:
                return False
            cfg_rel = Path("config/client_config.json")
            preserved_cfg = _read_preserved_cfg(dest, preserve_config=preserve_config, cfg_rel=cfg_rel)
            if not _cleanup_managed_top(base, dest, preserve_var=preserve_var):
                return False
            if not _copy_extracted_files(
                base,
                dest,
                preserve_var=preserve_var,
                preserve_config=preserve_config,
                cfg_rel=cfg_rel,
                preserved_cfg=preserved_cfg,
            ):
                return False
            if not _restore_preserved_cfg(dest, cfg_rel, preserved_cfg, preserve_config=preserve_config):
                return False
            return True
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        return False


def _resolve_update_root() -> Path:
    override_root = os.environ.get("MANUAL_UPDATE_ROOT", "").strip()
    if override_root:
        return Path(override_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _resolve_update_base(root: Path) -> str:
    base = os.environ.get("UPDATE_URL", "").strip()
    if not base and len(sys.argv) >= 2:
        base = sys.argv[1].strip()
    if not base:
        try:
            cfg = json.loads((root / "config" / "client_config.json").read_text(encoding="utf-8"))
            base = str(cfg.get("update_url") or "").strip()
        except Exception:
            base = ""
    return base.rstrip("/")


def _fetch_manifest_bytes(base: str) -> tuple[Optional[bytes], Optional[int]]:
    try:
        return fetch(base + "/manifest.json"), None
    except Exception as exc:  # pragma: no cover - network
        print(f"[client:update] Failed to download manifest.json: {exc}", file=sys.stderr)
        return None, 3


def _maybe_tofu_pubkey(root: Path, base: str, pub: Optional[bytes]) -> Optional[bytes]:
    if pub is not None or not _env_bool("MANUAL_UPDATE_ALLOW_TOFU", False):
        return pub
    try:
        fetched_pk = fetch(base + "/pubkey.txt")
        pk_txt = fetched_pk.decode("ascii", errors="ignore").strip()
        candidate = _decode_pubkey_candidate(pk_txt)
        if candidate is None:
            return None
        dest_pk = root / "config" / "pubkey.txt"
        dest_pk.parent.mkdir(parents=True, exist_ok=True)
        dest_pk.write_text(pk_txt + "\n", encoding="utf-8")
        return candidate
    except Exception:
        return None


def _verify_manifest_signature(base: str, mani_b: bytes, pub: Optional[bytes]) -> Optional[int]:
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
        return None
    except Exception as exc:  # pragma: no cover - network
        print(f"[client:update] Failed to verify manifest signature: {exc}", file=sys.stderr)
        return 4


def _parse_manifest_entries(manifest: dict) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
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
    return entries


def _validate_archive_presence(entries: list[dict[str, object]]) -> Optional[int]:
    require_archive = str(os.environ.get("ALLOW_NO_ARCHIVE", "")).strip().lower() not in ("1", "true", "yes", "on")
    has_archive = any(e["path"] == "client-release.tar.gz" for e in entries)
    if require_archive and not has_archive:
        print("[client:update] Manifest missing client-release.tar.gz; update server dist and retry", file=sys.stderr)
        return 4
    return None


def _map_destination(root: Path, state_dir: Path, rel: Path) -> Path:
    rel_txt = rel.as_posix()
    if rel_txt == "client.py":
        return root / "bin" / "client.py"
    if rel_txt == "schema.json":
        return root / "config" / "protocol" / "schema.json"
    if rel_txt == "bootstrap.py":
        return root / "scripts" / "bootstrap.py"
    if rel_txt == "pubkey.txt":
        return root / "config" / "pubkey.txt"
    if rel_txt == "version.json":
        return root / "var" / "update" / "version.json"
    if rel_txt == "scripts/manual_update.py":
        return root / "scripts" / "manual_update.py"
    if rel_txt == "client-release.tar.gz":
        return state_dir / "client-release.tar.gz"
    return root / rel


def _entry_matches(dest: Path, entry: dict[str, object]) -> bool:
    if not dest.exists():
        return False
    try:
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest() == entry["sha256"] and dest.stat().st_size == entry["size"]
    except Exception:
        return False


def _is_manifest_already_applied(state_dir: Path, manifest: dict, entries: list[dict[str, object]], root: Path) -> bool:
    try:
        cached = json.loads((state_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    if cached.get("root_hash") != manifest.get("root_hash") or cached.get("version") != manifest.get("version"):
        return False
    for entry in entries:
        rel = Path(str(entry["path"]))
        dest = _map_destination(root, state_dir, rel).resolve()
        if not _entry_matches(dest, entry):
            return False
    return True


def _download_entry_to_dest(base: str, state_dir: Path, rel: Path, entry: dict[str, object], dest: Path, *, progress: bool) -> Optional[int]:
    try:
        blob = fetch_with_progress(base + "/" + rel.as_posix(), int(entry["size"]), rel.as_posix(), timeout=20.0, enable=progress)
    except Exception as exc:  # pragma: no cover - network
        print(f"[client:update] Failed to download {rel}: {exc}", file=sys.stderr)
        return 5
    if len(blob) != int(entry["size"]):
        print(f"[client:update] Size mismatch for {rel}", file=sys.stderr)
        return 5
    if hashlib.sha256(blob).hexdigest() != str(entry["sha256"]):
        print(f"[client:update] Hash mismatch for {rel}", file=sys.stderr)
        return 5
    tmp_path = state_dir / (rel.as_posix().replace("/", "_") + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(blob)
    tmp_path.chmod(0o755 if rel.name.endswith(".py") else 0o644)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(dest)
    return None


def _download_updates(
    *,
    root: Path,
    state_dir: Path,
    base: str,
    entries: list[dict[str, object]],
) -> tuple[int, Optional[Path], bool, Optional[int]]:
    changed = 0
    tar_dest: Optional[Path] = None
    tar_hash_ok = False
    progress = _progress_enabled()
    for entry in entries:
        rel = Path(str(entry["path"]))
        dest = _map_destination(root, state_dir, rel).resolve()
        if rel.as_posix() == "client-release.tar.gz":
            tar_dest = dest
        if root not in dest.parents and root != dest:
            print(f"[client:update] Unsafe path in manifest: {rel}", file=sys.stderr)
            return changed, tar_dest, tar_hash_ok, 4
        if _entry_matches(dest, entry):
            if rel.as_posix() == "client-release.tar.gz":
                tar_hash_ok = True
            continue
        err = _download_entry_to_dest(base, state_dir, rel, entry, dest, progress=progress)
        if err is not None:
            return changed, tar_dest, tar_hash_ok, err
        changed += 1
        if rel.as_posix() == "client-release.tar.gz":
            tar_dest = dest
            tar_hash_ok = True
    return changed, tar_dest, tar_hash_ok, None


def _apply_release_archive(root: Path, tar_dest: Optional[Path], tar_hash_ok: bool) -> Optional[int]:
    if tar_dest and tar_dest.exists() and tar_hash_ok:
        if not _safe_extract_tar(tar_dest, root, preserve_var=True, preserve_config=True):
            print("[client:update] Failed to extract client-release.tar.gz", file=sys.stderr)
            return 5
        return None
    if tar_dest:
        print("[client:update] Warning: client-release.tar.gz not applied (hash/size mismatch)", file=sys.stderr)
    else:
        print("[client:update] Warning: client-release.tar.gz not applied (missing in manifest)", file=sys.stderr)
    return None


def _prepare_update_context() -> tuple[Optional[Path], Optional[str], Optional[Path], Optional[int]]:
    root = _resolve_update_root()
    base = _resolve_update_base(root)
    if not base:
        print("[client:update] UPDATE_URL is required", file=sys.stderr)
        return None, None, None, 2
    if not _fix_structure(root):
        print("[client:update] Client structure invalid and could not be fixed", file=sys.stderr)
        return None, None, None, 2
    state_dir = root / "var" / "update"
    state_dir.mkdir(parents=True, exist_ok=True)
    return root, base, state_dir, None


def _load_verified_manifest(root: Path, base: str) -> tuple[Optional[dict], Optional[int]]:
    mani_b, err = _fetch_manifest_bytes(base)
    if err is not None or mani_b is None:
        return None, err or 3
    pub = _maybe_tofu_pubkey(root, base, _parse_pubkey(root))
    err = _verify_manifest_signature(base, mani_b, pub)
    if err is not None:
        return None, err
    try:
        return json.loads(mani_b.decode("utf-8")), None
    except Exception as exc:
        print(f"[client:update] Bad manifest.json: {exc}", file=sys.stderr)
        return None, 4


def _prepare_manifest_entries(manifest: dict) -> tuple[Optional[list[dict[str, object]]], Optional[int]]:
    entries = _parse_manifest_entries(manifest)
    if not entries:
        print("[client:update] Manifest has no valid entries", file=sys.stderr)
        return None, 4
    err = _validate_archive_presence(entries)
    if err is not None:
        return None, err
    return entries, None


def main() -> int:
    root, base, state_dir, err = _prepare_update_context()
    if err is not None:
        return err
    if root is None or base is None or state_dir is None:
        return 2
    manifest, err = _load_verified_manifest(root, base)
    if err is not None:
        return err
    if manifest is None:
        return 4
    entries, err = _prepare_manifest_entries(manifest)
    if err is not None:
        return err
    if entries is None:
        return 4
    if _is_manifest_already_applied(state_dir, manifest, entries, root):
        print(f"[client:update] Уже установлена версия {manifest.get('version')}, обновление не требуется")
        return 0
    changed, tar_dest, tar_hash_ok, err = _download_updates(root=root, state_dir=state_dir, base=base, entries=entries)
    if err is not None:
        return err

    # Final cleanup to remove any forbidden leftovers
    _fix_structure(root)
    err = _apply_release_archive(root, tar_dest, tar_hash_ok)
    if err is not None:
        return err

    (state_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[client:update] Проверено файлов: {len(entries)}, обновлено: {changed}, версия: {manifest.get('version')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
