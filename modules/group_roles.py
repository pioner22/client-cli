from __future__ import annotations

from typing import Mapping


GROUP_ROLE_OWNER = "owner"
GROUP_ROLE_ADMIN = "admin"
GROUP_ROLE_MODERATOR = "moderator"
GROUP_ROLE_MEMBER = "member"

GROUP_ROLES = frozenset(
    {
        GROUP_ROLE_OWNER,
        GROUP_ROLE_ADMIN,
        GROUP_ROLE_MODERATOR,
        GROUP_ROLE_MEMBER,
    }
)
GROUP_ASSIGNABLE_ROLES = frozenset({GROUP_ROLE_ADMIN, GROUP_ROLE_MODERATOR, GROUP_ROLE_MEMBER})

GROUP_PERMISSION_INVITE = "invite"
GROUP_PERMISSION_APPROVE_JOIN = "approve_join"
GROUP_PERMISSION_SET_INFO = "set_info"
GROUP_PERMISSION_SET_HANDLE = "set_handle"
GROUP_PERMISSION_REMOVE_MEMBER = "remove_member"
GROUP_PERMISSION_MODERATE_POST = "moderate_post"
GROUP_PERMISSION_SET_ROLES = "set_roles"

_ROLE_PERMISSIONS: dict[str, tuple[str, ...]] = {
    GROUP_ROLE_OWNER: (
        GROUP_PERMISSION_INVITE,
        GROUP_PERMISSION_APPROVE_JOIN,
        GROUP_PERMISSION_SET_INFO,
        GROUP_PERMISSION_SET_HANDLE,
        GROUP_PERMISSION_REMOVE_MEMBER,
        GROUP_PERMISSION_MODERATE_POST,
        GROUP_PERMISSION_SET_ROLES,
    ),
    GROUP_ROLE_ADMIN: (
        GROUP_PERMISSION_INVITE,
        GROUP_PERMISSION_APPROVE_JOIN,
        GROUP_PERMISSION_SET_INFO,
        GROUP_PERMISSION_REMOVE_MEMBER,
        GROUP_PERMISSION_MODERATE_POST,
    ),
    GROUP_ROLE_MODERATOR: (
        GROUP_PERMISSION_APPROVE_JOIN,
        GROUP_PERMISSION_MODERATE_POST,
    ),
    GROUP_ROLE_MEMBER: (),
}


def normalize_group_role(raw: object, *, default: str = GROUP_ROLE_MEMBER) -> str:
    role = str(raw or "").strip().lower()
    if role in GROUP_ROLES:
        return role
    return default if default in GROUP_ROLES else GROUP_ROLE_MEMBER


def normalize_assignable_group_role(raw: object) -> str | None:
    role = normalize_group_role(raw)
    return role if role in GROUP_ASSIGNABLE_ROLES else None


def coerce_group_roles(
    *,
    owner_id: object,
    members: list[object] | tuple[object, ...] | set[object],
    raw_roles: Mapping[object, object] | None = None,
) -> dict[str, str]:
    owner = str(owner_id or "").strip()
    raw = raw_roles or {}
    member_ids = {str(uid or "").strip() for uid in members or []}
    member_ids.discard("")
    if owner:
        member_ids.add(owner)

    roles: dict[str, str] = {}
    for uid in sorted(member_ids):
        if uid == owner:
            roles[uid] = GROUP_ROLE_OWNER
            continue
        role = normalize_group_role(raw.get(uid), default=GROUP_ROLE_MEMBER)
        if role == GROUP_ROLE_OWNER:
            role = GROUP_ROLE_MEMBER
        roles[uid] = role
    return roles


def group_role_for_user(*, user_id: object, owner_id: object, roles: Mapping[object, object] | None = None) -> str:
    uid = str(user_id or "").strip()
    owner = str(owner_id or "").strip()
    if uid and uid == owner:
        return GROUP_ROLE_OWNER
    return normalize_group_role((roles or {}).get(uid), default=GROUP_ROLE_MEMBER)


def group_permissions_for_role(role: object) -> list[str]:
    normalized = normalize_group_role(role)
    return list(_ROLE_PERMISSIONS.get(normalized, ()))


def can_group_role(role: object, permission: str) -> bool:
    return permission in _ROLE_PERMISSIONS.get(normalize_group_role(role), ())


def group_permission_payload(*, user_id: object, owner_id: object, roles: Mapping[object, object]) -> dict[str, object]:
    role = group_role_for_user(user_id=user_id, owner_id=owner_id, roles=roles)
    return {"my_role": role, "permissions": group_permissions_for_role(role)}


def group_approver_ids(*, owner_id: object, roles: Mapping[object, object]) -> list[str]:
    owner = str(owner_id or "").strip()
    out: list[str] = []
    seen: set[str] = set()
    for uid, role in roles.items():
        suid = str(uid or "").strip()
        if not suid or suid in seen:
            continue
        if suid == owner or can_group_role(role, GROUP_PERMISSION_APPROVE_JOIN):
            seen.add(suid)
            out.append(suid)
    if owner and owner not in seen:
        out.insert(0, owner)
    return out


def can_manage_group_member(*, actor_role: object, target_role: object) -> bool:
    actor = normalize_group_role(actor_role)
    target = normalize_group_role(target_role)
    if target == GROUP_ROLE_OWNER:
        return False
    if actor == GROUP_ROLE_OWNER:
        return True
    if actor == GROUP_ROLE_ADMIN:
        return target in {GROUP_ROLE_MODERATOR, GROUP_ROLE_MEMBER}
    return False
