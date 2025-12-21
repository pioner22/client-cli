from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# NDJSON message/type constants for authz-related flows
AUTHZ_REQUEST = "authz_request"
AUTHZ_RESPONSE = "authz_response"
AUTHZ_PENDING = "authz_pending"
AUTHZ_ACCEPTED = "authz_accepted"
AUTHZ_DECLINED = "authz_declined"
AUTHZ_CANCELLED = "authz_cancelled"

# Error reasons
ERR_NOT_AUTHORIZED = "not_authorized"


@dataclass
class AuthzState:
    """Authorization state snapshot for a user.

    friends: confirmed contacts (bidirectional authorization)
    pending_in: incoming requests waiting for user's decision
    pending_out: outgoing requests sent by the user
    """

    friends: List[str]
    pending_in: List[str]
    pending_out: List[str]

    def to_roster_payload(self, online: Optional[List[str]] = None) -> Dict[str, object]:
        # Keep compatibility with existing "roster" payload
        return {
            "type": "roster",
            "friends": list(self.friends),
            "pending_in": list(self.pending_in),
            "pending_out": list(self.pending_out),
            "online": list(online or []),
        }


def make_authz_request_payload(to_id: str, note: Optional[str] = None) -> Dict[str, object]:
    payload: Dict[str, object] = {"type": AUTHZ_REQUEST, "to": to_id}
    if note is not None:
        payload["note"] = note
    return payload


def make_authz_response_payload(peer_id: str, accept: bool) -> Dict[str, object]:
    return {"type": AUTHZ_RESPONSE, "peer": peer_id, "accept": bool(accept)}


def make_dm_blocked_payload(to_id: str, reason: str = ERR_NOT_AUTHORIZED) -> Dict[str, object]:
    return {"type": "message_blocked", "to": to_id, "reason": reason}


def is_authorized(friends_map: Dict[str, bool], peer_id: Optional[str]) -> bool:
    if not peer_id:
        return False
    return bool(friends_map.get(peer_id))


__all__ = [
    "AUTHZ_REQUEST",
    "AUTHZ_RESPONSE",
    "AUTHZ_PENDING",
    "AUTHZ_ACCEPTED",
    "AUTHZ_DECLINED",
    "AUTHZ_CANCELLED",
    "ERR_NOT_AUTHORIZED",
    "AuthzState",
    "make_authz_request_payload",
    "make_authz_response_payload",
    "make_dm_blocked_payload",
    "is_authorized",
]
