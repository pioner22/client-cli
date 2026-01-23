import unittest

from modules.formatting import apply_format


class TestFormatting(unittest.TestCase):
    def test_bold_wraps_selection(self):
        r = apply_format("bold", "hello world", caret=0, sel_start=0, sel_end=5)
        self.assertEqual(r.text, "**hello** world")

    def test_italic_word_under_caret(self):
        r = apply_format("italic", "hello world", caret=7, sel_start=0, sel_end=0)
        self.assertEqual(r.text, "hello _world_")

    def test_upper_on_selection(self):
        r = apply_format("upper", "hello world", caret=0, sel_start=6, sel_end=11)
        self.assertEqual(r.text, "hello WORLD")

    def test_link_with_url_and_label(self):
        r = apply_format("link", "hi", caret=0, sel_start=0, sel_end=0, link_text="Y", link_url="https://yagodka.org")
        self.assertEqual(r.text, "[Y](https://yagodka.org)")


if __name__ == "__main__":
    unittest.main()

