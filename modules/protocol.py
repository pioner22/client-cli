from __future__ import annotations

"""
Shared protocol message type constants for server and client.

Keep all NDJSON "type" strings centralized to reduce duplication across
server and client code. This module provides a simple namespace `T` with
string constants and also re-exports authz-related type names from
module.authorization for convenience.

Note: Only literal values are defined here; payload shapes remain
documented in the server/client code and tests.
"""

try:
    # Reuse existing authz constants to avoid duplication
    from .authorization import (
        AUTHZ_REQUEST,
        AUTHZ_RESPONSE,
        AUTHZ_PENDING,
        AUTHZ_ACCEPTED,
        AUTHZ_DECLINED,
        AUTHZ_CANCELLED,
    )
except Exception:  # pragma: no cover - fallback if module not available
    AUTHZ_REQUEST = "authz_request"  # type: ignore
    AUTHZ_RESPONSE = "authz_response"  # type: ignore
    AUTHZ_PENDING = "authz_pending"  # type: ignore
    AUTHZ_ACCEPTED = "authz_accepted"  # type: ignore
    AUTHZ_DECLINED = "authz_declined"  # type: ignore
    AUTHZ_CANCELLED = "authz_cancelled"  # type: ignore


class T:
    # Session / handshake
    WELCOME = "welcome"
    SESSION_REPLACED = "session_replaced"

    # Auth / register
    AUTH = "auth"
    AUTH_OK = "auth_ok"
    AUTH_FAIL = "auth_fail"
    REGISTER = "register"
    REGISTER_OK = "register_ok"
    REGISTER_FAIL = "register_fail"

    # Generic
    ERROR = "error"
    PING = "ping"
    PONG = "pong"

    # Contacts / presence / roster
    CONTACTS = "contacts"
    CONTACT_JOINED = "contact_joined"
    CONTACT_LEFT = "contact_left"
    PRESENCE_UPDATE = "presence_update"
    ROSTER = "roster"
    ROSTER_FULL = "roster_full"
    FRIENDS = "friends"
    USERS = "users"

    # Profiles / prefs
    PROFILE = "profile"
    PROFILE_GET = "profile_get"
    PROFILE_SET = "profile_set"
    PROFILE_SET_RESULT = "profile_set_result"
    PROFILE_UPDATED = "profile_updated"
    AVATAR_SET = "avatar_set"
    AVATAR_SET_RESULT = "avatar_set_result"
    AVATAR_CLEAR = "avatar_clear"
    AVATAR_CLEAR_RESULT = "avatar_clear_result"
    AVATAR_GET = "avatar_get"
    AVATAR = "avatar"
    PREFS = "prefs"
    PREFS_GET = "prefs_get"
    PREFS_SET = "prefs_set"
    MUTE_SET = "mute_set"
    MUTE_SET_RESULT = "mute_set_result"
    BLOCK_SET = "block_set"
    BLOCK_SET_RESULT = "block_set_result"

    # Messaging
    SEND = "send"
    MESSAGE = "message"
    MESSAGE_DELIVERED = "message_delivered"
    MESSAGE_QUEUED = "message_queued"
    MESSAGE_BLOCKED = "message_blocked"
    MESSAGE_DELIVERED_TO_DEVICE = "message_delivered_to_device"
    MESSAGE_READ = "message_read"
    MESSAGE_READ_ACK = "message_read_ack"
    UNREAD_COUNTS = "unread_counts"
    ROOM_READS = "room_reads"

    # Reactions (1 per user)
    REACTION_SET = "reaction_set"
    REACTION_UPDATE = "reaction_update"

    # Update
    UPDATE_REQUIRED = "update_required"

    # PWA Web Push
    PWA_PUSH_SUBSCRIBE = "pwa_push_subscribe"
    PWA_PUSH_SUBSCRIBE_RESULT = "pwa_push_subscribe_result"
    PWA_PUSH_UNSUBSCRIBE = "pwa_push_unsubscribe"
    PWA_PUSH_UNSUBSCRIBE_RESULT = "pwa_push_unsubscribe_result"

    # History / search
    HISTORY = "history"
    HISTORY_RESULT = "history_result"
    SEARCH = "search"
    SEARCH_RESULT = "search_result"
    LIST = "list"

    # History management
    ROOM_CLEAR = "room_clear"
    ROOM_CLEARED = "room_cleared"

    # Groups
    GROUPS = "groups"
    GROUP_CREATE = "group_create"
    GROUP_CREATE_RESULT = "group_create_result"
    GROUP_ADD = "group_add"
    GROUP_ADD_RESULT = "group_add_result"
    GROUP_REMOVE = "group_remove"
    GROUP_REMOVE_RESULT = "group_remove_result"
    GROUP_LEAVE = "group_leave"
    GROUP_LEAVE_RESULT = "group_leave_result"
    GROUP_RENAME = "group_rename"
    GROUP_RENAME_RESULT = "group_rename_result"
    GROUP_DISBAND = "group_disband"
    GROUP_DISBAND_RESULT = "group_disband_result"
    GROUP_INFO = "group_info"
    GROUP_INFO_RESULT = "group_info_result"
    GROUP_SET_INFO = "group_set_info"
    GROUP_SET_INFO_RESULT = "group_set_info_result"
    GROUP_POST_SET = "group_post_set"
    GROUP_POST_SET_RESULT = "group_post_set_result"
    GROUP_POST_UPDATE = "group_post_update"
    GROUP_ADDED = "group_added"
    GROUP_UPDATED = "group_updated"
    GROUP_REMOVED = "group_removed"
    GROUP_INVITE = "group_invite"
    GROUP_INVITE_RESULT = "group_invite_result"
    GROUP_INVITE_RESPONSE = "group_invite_response"

    # Boards (Доски)
    BOARDS = "boards"
    BOARD_CREATE = "board_create"
    BOARD_CREATE_RESULT = "board_create_result"
    BOARD_ADD = "board_add"
    BOARD_ADD_RESULT = "board_add_result"
    BOARD_REMOVE = "board_remove"
    BOARD_REMOVE_RESULT = "board_remove_result"
    BOARD_DISBAND = "board_disband"
    BOARD_DISBAND_RESULT = "board_disband_result"
    BOARD_INFO = "board_info"
    BOARD_INFO_RESULT = "board_info_result"
    BOARD_SET_INFO = "board_set_info"
    BOARD_SET_INFO_RESULT = "board_set_info_result"
    BOARD_RENAME = "board_rename"
    BOARD_RENAME_RESULT = "board_rename_result"
    BOARD_SET_HANDLE = "board_set_handle"
    BOARD_SET_HANDLE_RESULT = "board_set_handle_result"
    BOARD_JOIN = "board_join"
    BOARD_JOIN_RESULT = "board_join_result"
    BOARD_LEAVE = "board_leave"
    BOARD_LEAVE_RESULT = "board_leave_result"
    BOARD_INVITE = "board_invite"
    BOARD_INVITE_RESULT = "board_invite_result"
    BOARD_INVITE_RESPONSE = "board_invite_response"
    BOARD_INVITE_RESPONSE_RESULT = "board_invite_response_result"
    BOARD_ADDED = "board_added"
    BOARD_UPDATED = "board_updated"
    BOARD_REMOVED = "board_removed"

    # Files (offer/upload/download over NDJSON; base64 chunks)
    FILE_OFFER = "file_offer"                # metadata offer: {to|room, name, size}
    FILE_OFFER_RESULT = "file_offer_result"  # ack for sender: {ok, file_id?, reason?}
    FILE_ACCEPT = "file_accept"              # recipient accepts: {file_id}
    FILE_REJECT = "file_reject"              # recipient rejects: {file_id}
    FILE_CHUNK = "file_chunk"                # upload/download chunk: {file_id, seq, data}
    FILE_UPLOAD_COMPLETE = "file_upload_complete"  # sender -> server: {file_id}
    FILE_DOWNLOAD_BEGIN = "file_download_begin"    # server -> recipient: {file_id, name, size, from, room?}
    FILE_DOWNLOAD_COMPLETE = "file_download_complete"  # server -> recipient
    FILE_ERROR = "file_error"                # any-side error
    FILE_ACCEPT_NOTICE = "file_accept_notice"  # server -> sender: recipient accepted
    FILE_RECEIVED = "file_received"            # server -> sender: recipient finished download
    FILE_GET = "file_get"                      # request (re)download: {file_id}
    FILE_PREVIEW_READY = "file_preview_ready"  # server -> users: thumb/meta ready (clients should re-fetch file_url)

    # Calls (signaling for WebRTC/Jitsi)
    CALL_CREATE = "call_create"
    CALL_CREATE_RESULT = "call_create_result"
    CALL_INVITE = "call_invite"
    CALL_INVITE_ACK = "call_invite_ack"
    CALL_ACCEPT = "call_accept"
    CALL_REJECT = "call_reject"
    CALL_END = "call_end"
    CALL_STATE = "call_state"

    # Authorization (friendship) — values from module.authorization
    AUTHZ_REQUEST = AUTHZ_REQUEST
    AUTHZ_RESPONSE = AUTHZ_RESPONSE
    AUTHZ_PENDING = AUTHZ_PENDING
    AUTHZ_ACCEPTED = AUTHZ_ACCEPTED
    AUTHZ_DECLINED = AUTHZ_DECLINED
    AUTHZ_CANCELLED = AUTHZ_CANCELLED


__all__ = [
    # Namespace
    "T",
    # Re-exported authz constants for convenience
    "AUTHZ_REQUEST",
    "AUTHZ_RESPONSE",
    "AUTHZ_PENDING",
    "AUTHZ_ACCEPTED",
    "AUTHZ_DECLINED",
    "AUTHZ_CANCELLED",
]
