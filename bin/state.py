from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ChatMessage:
    direction: str  # 'in' or 'out'
    text: str
    ts: float
    sender: Optional[str] = None
    msg_id: Optional[int] = None
    broadcast: bool = False
    status: Optional[str] = None  # 'sent'|'queued'|'delivered'|'read'


@dataclass
class ClientState:
    self_id: Optional[str] = None
    # Online contacts (legacy server snapshot)
    contacts: List[str] = field(default_factory=list)
    # Rich roster from server: friend id -> {online: bool, last_seen_at: Optional[str], unread: int}
    roster_friends: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # Full directory of user IDs provided by server (excluding self)
    directory: Set[str] = field(default_factory=set)
    selected_index: int = 0
    conversations: Dict[str, List[ChatMessage]] = field(default_factory=dict)
    input_buffer: str = ""
    input_caret: int = 0
    input_sel_start: int = 0
    input_sel_end: int = 0
    input_cursor_visible: bool = True
    input_cursor_last_toggle: float = 0.0
    runtime_modules_ready: bool = False
    history_scroll: int = 0
    # Scroll offset for the left contacts list (topmost visible row index)
    contacts_scroll: int = 0
    # Last computed visible height of the left contacts pane
    last_left_h: int = 0
    # File browser modal state
    file_browser_mode: bool = False
    # Two-pane file browser state (MC-like)
    file_browser_side: int = 0  # 0=left, 1=right
    file_browser_path0: str = ''
    file_browser_path1: str = ''
    file_browser_items0: List[Tuple[str, bool]] = field(default_factory=list)  # (name, is_dir)
    file_browser_items1: List[Tuple[str, bool]] = field(default_factory=list)
    file_browser_index0: int = 0
    file_browser_index1: int = 0
    # Pure state holder (module.file_browser)
    file_browser_state: object = None
    # File browser view options menu
    file_browser_view_mode: bool = False
    file_browser_view_index: int = 0
    # File browser top menu (MC-like)
    file_browser_menu_mode: bool = False
    file_browser_menu_top: int = 0  # 0=Левая панель,1=Файл,2=Команда,3=Настройки,4=Правая панель
    file_browser_menu_index: int = 0
    # Geometry of top menu labels for precise mouse hit-testing: [(x_start,x_end), ...]
    file_browser_menu_pos: List[Tuple[int, int]] = field(default_factory=list)
    # Main top hotkeys hit-testing: list of (name, x_start, x_end) for 'F1'..'F7'
    main_hotkeys_pos: List[Tuple[str, int, int]] = field(default_factory=list)
    # Fallback preferences (when FILE_BROWSER_FALLBACK is True)
    file_browser_show_hidden0: bool = True
    file_browser_show_hidden1: bool = True
    file_browser_sort0: str = 'name'
    file_browser_sort1: str = 'name'
    file_browser_dirs_first0: bool = False
    file_browser_dirs_first1: bool = False
    file_browser_reverse0: bool = False
    file_browser_reverse1: bool = False
    file_browser_view0: Optional[str] = None   # None|'dirs'|'files'
    file_browser_view1: Optional[str] = None
    status: str = "Connecting..."
    authed: bool = False
    auth_pw: str = ""
    last_submit_pw: str = ""
    # Последний запрос авторизации (для повторной отправки при переподключении)
    last_auth_id: str = ""
    login_pending_since: float = 0.0
    login_retry_count: int = 0
    # Версия сервера
    server_version: Optional[str] = None
    # Client integrity vs server
    last_local_sha: Optional[str] = None
    last_server_sha: Optional[str] = None
    last_integrity_size_ok: bool = False
    last_integrity_hash_ok: bool = False
    # Auth UI
    auth_mode: str = "login"  # "login" or "register"
    login_field: int = 0  # login: 0=id,1=pw ; register: 0=pw1,1=pw2
    id_input: str = ""
    pw1: str = ""
    pw2: str = ""
    login_msg: str = ""
    # Notifications / непрочитанные
    unread: Dict[str, int] = field(default_factory=dict)
    # Друзья (авторизованные контакты)
    friends: Dict[str, bool] = field(default_factory=dict)
    # Поиск
    search_mode: bool = False
    search_query: str = ""
    search_results: List[dict] = field(default_factory=list)
    # Запросы авторизации
    pending_requests: List[str] = field(default_factory=list)
    authz_prompt_from: Optional[str] = None
    # Исходящие запросы авторизации (ожидают решения собеседника)
    pending_out: Set[str] = field(default_factory=set)
    # Преференции уведомлений
    muted: Set[str] = field(default_factory=set)
    blocked: Set[str] = field(default_factory=set)
    # Заглушённые групповые чаты (локально; серверных предпочтений для групп нет)
    group_muted: Set[str] = field(default_factory=set)
    # Заглушённые доски (локально)
    board_muted: Set[str] = field(default_factory=set)
    # Peers that have blocked the current user
    blocked_by: Set[str] = field(default_factory=set)
    # Скрытые заблокированные контакты (не показывать в левом списке)
    hidden_blocked: Set[str] = field(default_factory=set)
    # Меню действий
    action_menu_mode: bool = False
    action_menu_peer: Optional[str] = None
    action_menu_options: List[str] = field(default_factory=list)
    action_menu_index: int = 0
    # Простая модалка для уведомлений (центр экрана)
    modal_message: Optional[str] = None
    # Профиль (модальное окно)
    profile_mode: bool = False
    profile_field: int = 0  # 0=name, 1=handle
    profile_name_input: str = ""
    profile_handle_input: str = ""
    # Профили контактов: id -> {display_name, handle}
    profiles: Dict[str, Dict[str, Optional[str]]] = field(default_factory=dict)
    # Просмотр карточки выбранного пользователя
    profile_view_mode: bool = False
    profile_view_id: Optional[str] = None
    # Статусы онлайн/оффлайн
    statuses: Dict[str, bool] = field(default_factory=dict)
    # Помощь
    help_mode: bool = False
    # Selection state for mouse copy
    select_active: bool = False
    sel_anchor_y: int = -1
    sel_anchor_x: int = -1
    sel_cur_y: int = -1
    sel_cur_x: int = -1
    # Last drawn history geometry + lines snapshot for selection mapping
    last_hist_y: int = 0
    last_hist_x: int = 0
    last_hist_h: int = 0
    last_hist_w: int = 0
    last_lines: List[str] = field(default_factory=list)
    last_start: int = 0
    # История: локальный индекс последних id по каналам
    history_last_ids: Dict[str, int] = field(default_factory=dict)
    history_loaded: bool = False
    # Mouse capture mode (enables wheel + in-app selection). Can be toggled with F4.
    mouse_enabled: bool = True
    # ===== Debug overlay (F12): last key/mouse events and raw sequences =====
    debug_mode: bool = False
    debug_lines: List[str] = field(default_factory=list)
    debug_last_key: str = ""
    debug_last_seq: str = ""
    debug_last_mouse: str = ""
    # Incremental ESC sequence buffer (SGR/X10 across frames)
    esc_seq_buf: str = ""
    esc_seq_started_at: float = 0.0
    # Optional VI-style navigation (J/K). Disabled by default.
    vi_keys: bool = False
    # Search flow: action modal for found user
    search_action_mode: bool = False
    search_action_peer: Optional[str] = None
    search_action_options: List[str] = field(default_factory=list)
    search_action_index: int = 0
    search_action_step: str = "choose"  # choose|waiting|accepted|declined
    # Groups: group_id -> {name, owner_id, members}
    groups: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # Boards: board_id -> {name, owner_id, handle, members?}
    boards: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # Pending group join requests for groups owned by the current user: gid -> set(user_id)
    group_join_requests: Dict[str, Set[str]] = field(default_factory=dict)
    # Group create modal state
    group_create_mode: bool = False
    group_create_field: int = 0
    group_name_input: str = ""
    group_members_input: str = ""
    # Board create modal state
    board_create_mode: bool = False
    board_create_field: int = 0  # 0=name, 1=handle
    board_name_input: str = ""
    board_handle_input: str = ""
    # Group pre-validation flow (resolve @handles -> ids via search)
    group_verify_mode: bool = False
    group_verify_tokens: List[str] = field(default_factory=list)
    group_verify_map: Dict[str, Optional[str]] = field(default_factory=dict)
    group_verify_pending: Set[str] = field(default_factory=set)
    # Track last group creation intent to invite non-friends
    last_group_create_name: Optional[str] = None
    last_group_create_intended: Set[str] = field(default_factory=set)
    last_group_create_gid: Optional[str] = None
    # Group manage modal
    group_manage_mode: bool = False
    group_manage_gid: Optional[str] = None
    group_manage_field: int = 0  # 0=name, 1=members (readonly)
    group_manage_name_input: str = ""
    group_manage_member_count: int = 0
    # Board manage modal
    board_manage_mode: bool = False
    board_manage_bid: Optional[str] = None
    board_manage_field: int = 0  # 0=name, 1=handle
    board_manage_name_input: str = ""
    board_manage_handle_input: str = ""
    # Input history for Up/Down browsing
    input_history: List[str] = field(default_factory=list)
    input_history_index: int = -1  # -1: not browsing; otherwise index in input_history
    # Suggestions/typeahead
    suggest_mode: bool = False
    suggest_kind: str = ""  # 'slash' | 'file'
    suggest_items: List[str] = field(default_factory=list)
    suggest_index: int = 0
    suggest_start: int = 0
    suggest_end: int = 0
    # Debounce for /search requests
    last_search_sent: float = 0.0
    board_manage_member_count: int = 0
    # Board participant management
    board_member_add_mode: bool = False
    board_member_add_bid: Optional[str] = None
    board_member_add_input: str = ""
    board_member_remove_mode: bool = False
    board_member_remove_bid: Optional[str] = None
    board_member_remove_options: List[str] = field(default_factory=list)
    board_member_remove_index: int = 0
    # Board invite prompt (incoming)
    board_invite_mode: bool = False
    board_invite_bid: Optional[str] = None
    board_invite_name: str = ""
    board_invite_from: Optional[str] = None
    board_invite_index: int = 0
    # Pending board invites to display under "Ожидают авторизацию": bid -> {name, from}
    board_pending_invites: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Track pending invites to differentiate consensual join vs forced add
    board_recent_invites: Set[str] = field(default_factory=set)
    # Consent modal for unexpected add
    board_added_consent_mode: bool = False
    board_added_bid: Optional[str] = None
    board_added_index: int = 0
    # Known boards set to detect new boards from snapshot events
    known_boards: Set[str] = field(default_factory=set)
    # Mark that initial boards snapshot has been processed to avoid consent modal on first load
    boards_initialized: bool = False
    # Group participant management
    group_member_add_mode: bool = False
    group_member_add_gid: Optional[str] = None
    group_member_add_input: str = ""
    group_member_remove_mode: bool = False
    group_member_remove_gid: Optional[str] = None
    group_member_remove_options: List[str] = field(default_factory=list)
    group_member_remove_index: int = 0
    # Authorization UX helpers
    lock_selection_peer: Optional[str] = None  # keep selection anchored to this peer after auth until ack
    suppress_auto_menu: bool = False          # prevent auto-open of action menu for next pending contact
    # History probes: peers for which we sent a server-side history check after re-auth
    history_probe_peers: Set[str] = field(default_factory=set)
    # Outgoing authorization requests that should show persistent overlay in the chat
    authz_out_pending: Set[str] = field(default_factory=set)
    # Live search (F3) status
    search_live_id: Optional[str] = None
    search_live_ok: bool = False
    # Cursor stability: remember last applied hardware cursor to avoid flicker
    cursor_last_y: int = -1
    cursor_last_x: int = -1
    cursor_last_vis: int = 0
    # ===== Файлы: подтверждение отправки при пути в тексте =====
    file_confirm_mode: bool = False
    file_confirm_path: Optional[str] = None
    file_confirm_target: Optional[str] = None
    file_confirm_text_full: str = ""
    file_confirm_index: int = 0  # 0=Да, 1=Нет, 2=Отмена
    file_confirm_prev_text: str = ""
    file_confirm_prev_caret: int = 0
    # ===== Файлы: прогресс скачивания =====
    file_progress_mode: bool = False
    file_progress_name: str = ""
    file_progress_pct: int = 0
    file_progress_file_id: Optional[str] = None
    # ===== Файлы: индексы офферов по каналам (для /ok<ID>) =====
    file_offer_counters: Dict[str, int] = field(default_factory=dict)  # chan -> next int id
    # ===== UI: ESC exit guard (double-press to exit)
    last_esc_ts: float = 0.0
    # File browser: last click info (for simulating double-click on macOS)
    fb_last_click_ts: float = 0.0
    fb_last_click_side: int = -1
    fb_last_click_row: int = -1
    file_offer_map: Dict[str, Dict[int, str]] = field(default_factory=dict)  # chan -> {num: fid}
    file_offer_rev: Dict[str, Tuple[str, int]] = field(default_factory=dict)  # fid -> (chan, num)
    # ===== Файлы: модалка при конфликте имён (заменить?) =====
    file_exists_mode: bool = False
    file_exists_fid: Optional[str] = None
    file_exists_name: str = ""
    file_exists_target: str = ""
    file_exists_index: int = 0  # 0=Заменить, 1=Оставить
    # File transfer (send)
    file_send_path: Optional[str] = None
    file_send_name: Optional[str] = None
    file_send_size: int = 0
    file_send_to: Optional[str] = None
    file_send_room: Optional[str] = None
    file_send_file_id: Optional[str] = None
    file_send_seq: int = 0
    file_send_fp: Optional[object] = None
    file_send_bytes: int = 0
    # File transfer (receive)
    incoming_file_offers: Dict[str, dict] = field(default_factory=dict)  # file_id -> meta
    incoming_by_peer: Dict[str, List[str]] = field(default_factory=list)  # peer/room -> [file_id]
    file_recv_open: Dict[str, dict] = field(default_factory=dict)  # file_id -> {fp, name, size, from, received, path}


__all__ = ["ChatMessage", "ClientState"]
