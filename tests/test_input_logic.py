import tempfile
import unittest
from pathlib import Path

from modules.input_logic import (
    apply_suggestion,
    can_send_file,
    classify_peer,
    compute_suggestions,
    file_offer_target,
    handle_history_navigation,
    should_trigger_auth_hotkey,
)


class TestInputLogic(unittest.TestCase):
    def test_history_navigation_enters_history_on_up(self):
        text, caret, idx = handle_history_navigation("x", 1, width=10, direction="up", history=["a", "b"], hist_index=-1)
        self.assertEqual((text, idx), ("b", 1))
        self.assertEqual(caret, len(text))

    def test_history_navigation_exits_on_down_at_end(self):
        text, caret, idx = handle_history_navigation("b", 1, width=10, direction="down", history=["a", "b"], hist_index=1)
        self.assertEqual((text, caret, idx), ("", 0, -1))

    def test_apply_suggestion_replaces_span(self):
        s, caret = apply_suggestion("hello wor", 6, 9, "world")
        self.assertEqual(s, "hello world")
        self.assertEqual(caret, 11)

    def test_classify_peer(self):
        self.assertEqual(classify_peer("g1", groups={"g1"}, boards={}), "group")
        self.assertEqual(classify_peer("b-1", groups=set(), boards={}), "board")
        self.assertEqual(classify_peer("JOIN:1", groups=set(), boards={}), "token")
        self.assertEqual(classify_peer("123-456-789", groups=set(), boards={}), "user")

    def test_should_trigger_auth_hotkey(self):
        ok = should_trigger_auth_hotkey("", board_invite_mode=False, profile_mode=False, sel="123-456-789", groups=set(), boards={})
        self.assertEqual(ok, True)
        ok2 = should_trigger_auth_hotkey("x", board_invite_mode=False, profile_mode=False, sel="123-456-789", groups=set(), boards={})
        self.assertEqual(ok2, False)

    def test_file_offer_target(self):
        self.assertEqual(file_offer_target("g1", groups={"g1"}, boards={}), {"room": "g1"})
        self.assertEqual(file_offer_target("b-1", groups=set(), boards={}), {"room": "b-1"})
        self.assertEqual(file_offer_target("123-456-789", groups=set(), boards={}), {"to": "123-456-789"})

    def test_can_send_file_dm_requires_friend(self):
        ok, reason = can_send_file(
            authed=True,
            sel="222-222-222",
            groups=set(),
            boards={},
            friends={},
            roster_friends={},
            blocked=set(),
            blocked_by=set(),
            self_id="111-111-111",
        )
        self.assertEqual((ok, reason), (False, "not_friends"))

    def test_compute_suggestions_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "hello.txt").write_text("x", encoding="utf-8")
            r = compute_suggestions("./he", caret=4, cwd=p, limit=10)
            self.assertTrue(r is None or r.kind in ("file", "slash"))


if __name__ == "__main__":
    unittest.main()

