import tempfile
import unittest
from pathlib import Path

from modules.file_transfer import extract_file_command_path, extract_path_candidate, file_meta_for, format_file_command


class TestFileTransfer(unittest.TestCase):
    def test_extract_file_command_path_supports_spaces(self):
        path = "/Users/admin/Library/Application Support/Yagodka/test file.txt"
        self.assertEqual(extract_file_command_path(f"/file {path}"), path)

    def test_extract_file_command_path_supports_json_quoted_paths(self):
        path = "/tmp/my file.txt"
        self.assertEqual(extract_file_command_path(f'/file "{path}"'), path)

    def test_format_file_command_round_trips_with_extract_path_candidate(self):
        with tempfile.TemporaryDirectory(prefix="yg file ") as td:
            path = Path(td) / "hello world.txt"
            path.write_text("x", encoding="utf-8")
            command = format_file_command(path)
            self.assertEqual(extract_path_candidate(command), str(path))
            meta = file_meta_for(extract_path_candidate(command) or "")
            self.assertIsNotNone(meta)
            self.assertEqual(meta.name, "hello world.txt")

    def test_extract_path_candidate_preserves_file_command_with_spaces(self):
        cmd = '/file "/Users/admin/Library/Application Support/Yagodka/test file.txt"'
        self.assertEqual(
            extract_path_candidate(cmd),
            "/Users/admin/Library/Application Support/Yagodka/test file.txt",
        )


if __name__ == "__main__":
    unittest.main()
