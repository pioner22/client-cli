from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Roster:
    """Модель списка контактов пользователя.

    friends: подтверждённые контакты (двусторонняя авторизация)
    pending_in: входящие запросы на авторизацию (ожидают решения пользователя)
    pending_out: исходящие запросы, отправленные пользователем
    online: список ID онлайн-контактов (не обязательно только из friends)
    """

    friends: List[str]
    pending_in: List[str]
    pending_out: List[str]
    online: List[str]

    def to_payload(self) -> Dict[str, object]:
        return {
            "type": "roster",
            "friends": self.friends,
            "pending_in": self.pending_in,
            "pending_out": self.pending_out,
            "online": self.online,
        }


def make_roster_payload(friends: List[str], pending_in: List[str], pending_out: List[str], online: List[str]) -> Dict[str, object]:
    return Roster(friends, pending_in, pending_out, online).to_payload()


__all__ = ["Roster", "make_roster_payload"]

