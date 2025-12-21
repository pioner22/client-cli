from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple


@dataclass
class Group:
    id: str
    name: str
    owner_id: str
    members: List[str]

    def to_payload(self) -> Dict[str, object]:
        return {"id": self.id, "name": self.name, "owner_id": self.owner_id, "members": list(self.members)}


def parse_members_csv(text: str) -> List[str]:
    """Parse a delimited string of member tokens into a list of IDs/handles.

    Accept both commas and any whitespace as separators to be resilient to
    different client implementations (e.g., "a,b c\td").
    """
    try:
        import re as _re
        return [t.strip() for t in _re.split(r"[\s,]+", text or "") if t.strip()]
    except Exception:
        items: List[str] = []
        for raw in (text or "").split(','):
            v = raw.strip()
            if v:
                items.append(v)
        return items


__all__ = ["Group", "parse_members_csv"]
