from __future__ import annotations

"""
Slash-commands registry and suggestions.

Currently supports only '/file' as requested. Designed to be extended.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str


SLASH_COMMANDS: List[SlashCommand] = [
    SlashCommand(name='/file', description='Передача файлов'),
    SlashCommand(name='/chat', description='Создай свой чат'),
    SlashCommand(name='/board', description='Создать свою новостную ленту'),
    SlashCommand(name='/profile', description='Редактирование профиля'),
    SlashCommand(name='/search', description='Поиск контактов, чатов, досок'),
    SlashCommand(name='/help', description='Справочная информация'),
    SlashCommand(name='/exit', description='Закрыть и выйди из месенжера'),
]


def suggest(prefix: str, limit: int = 10) -> List[SlashCommand]:
    p = (prefix or '').strip()
    if not p.startswith('/'):
        return []
    # If only '/' typed, show full menu in declared order (up to limit)
    if p == '/':
        return SLASH_COMMANDS[: max(1, int(limit))]
    res: List[SlashCommand] = []
    lp = p.lower()
    for cmd in SLASH_COMMANDS:
        if cmd.name.lower().startswith(lp):
            res.append(cmd)
            if len(res) >= max(1, int(limit)):
                break
    return res


__all__ = ["SlashCommand", "SLASH_COMMANDS", "suggest"]
