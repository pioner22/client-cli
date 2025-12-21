#!/usr/bin/env python3
"""Simple structural sanity check for the client layout.

Validates that sources live in the expected locations and that legacy
paths like modules/module/ or runtime/dll/*.py are absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_MODULES = {
    "authorization.py",
    "boards.py",
    "chat_input.py",
    "contacts.py",
    "cursor.py",
    "formatting.py",
    "error_codes.py",
    "file_browser.py",
    "file_suggest.py",
    "file_transfer.py",
    "groups.py",
    "input_logic.py",
    "modal_std.py",
    "multiline_editor.py",
    "profile.py",
    "protocol.py",
    "selection_editor.py",
    "slash_commands.py",
    "status.py",
    "text_editor.py",
    "ui_modals.py",
    "ui_utils.py",
}


def main() -> int:
    errors: list[str] = []

    def expect(path: Path, kind: str) -> None:
        # Create missing dirs/files on the fly to avoid failing launch for users
        if path.suffix == "" and kind.endswith("/"):
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception:
                    errors.append(f"missing {kind}: {path}")
        elif path.suffix == ".json":
            if not path.exists():
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
                except Exception:
                    errors.append(f"missing {kind}: {path}")
        else:
            if not path.exists():
                errors.append(f"missing {kind}: {path}")

    bin_dir = ROOT / "bin"
    modules_dir = ROOT / "modules"
    config_dir = ROOT / "config"
    var_dir = ROOT / "var"

    expect(bin_dir / "client.py", "bin/client.py")
    expect(bin_dir / "state.py", "bin/state.py")

    expect(config_dir / "client_config.json", "config/client_config.json")
    expect(config_dir / "protocol" / "schema.json", "config/protocol/schema.json")

    for sub in ("files", "history", "log", "users", "update"):
        expect(var_dir / sub, f"var/{sub}/")

    if not modules_dir.is_dir():
        errors.append(f"modules directory is missing: {modules_dir}")
    else:
        # Ensure expected module files exist directly under modules/
        present = {p.name for p in modules_dir.glob("*.py")}
        missing = sorted(EXPECTED_MODULES - present)
        extra = sorted(present - EXPECTED_MODULES - {"__init__.py"})
        if missing:
            errors.append(f"modules missing: {', '.join(missing)}")
        if extra:
            errors.append(f"unexpected modules in modules/: {', '.join(extra)}")

    legacy_pkg = modules_dir / "module"
    if legacy_pkg.exists():
        errors.append(f"legacy package should not exist: {legacy_pkg}")

    runtime_dir = ROOT / "runtime"
    if runtime_dir.exists():
        errors.append(f"runtime directory should be absent: {runtime_dir}")

    stray_modules = modules_dir.parent / "modules 2"
    if stray_modules.exists():
        errors.append(f"unexpected duplicate modules dir: {stray_modules}")

    # Dist artifacts should not live at client root (they belong to server/dist/)
    forbidden_root = [
        ROOT / "client.py",
        ROOT / "bootstrap.py",
        ROOT / "pubkey.txt",
        ROOT / "schema.json",
        ROOT / "version.json",
    ]
    for f in forbidden_root:
        if f.exists():
            errors.append(f"unexpected dist artifact at root: {f}")

    if errors:
        for e in errors:
            print(f"[structure] {e}")
        return 1
    print("[structure] OK: client layout matches expected shape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
