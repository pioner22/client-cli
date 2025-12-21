from __future__ import annotations

"""
Standard builders for modal overlays (menus, prompts) used by the TUI client.

These helpers return string lines ready to be drawn via a center-box renderer
(e.g., client._draw_center_box), ensuring a consistent look across modals.
"""

from typing import Iterable, List, Optional


def build_menu_modal_lines(
    title: str,
    options: Iterable[str],
    selected_index: int = 0,
    subtitle_lines: Optional[Iterable[str]] = None,
    footer: str = "Enter — выбрать | Esc — закрыть | ↑/↓ — выбор",
) -> List[str]:
    lines: List[str] = []
    lines.append(f" {title} ")
    if subtitle_lines:
        for s in subtitle_lines:
            s = str(s or "").strip()
            if s:
                lines.append(s)
    idx = max(0, int(selected_index))
    opts = list(options)
    for i, opt in enumerate(opts):
        prefix = "> " if i == idx else "  "
        lines.append(f"{prefix}{opt}")
    if footer:
        lines.append(footer)
    return lines


__all__ = ["build_menu_modal_lines"]

