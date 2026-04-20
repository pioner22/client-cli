import importlib.util
import unittest
from pathlib import Path


CLIENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLIENT_ROOT.parents[0]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStructureContract(unittest.TestCase):
    def test_expected_modules_match_actual_layout(self):
        check_structure = _load_module(
            "client_cli_check_structure",
            CLIENT_ROOT / "scripts" / "check_structure.py",
        )
        verify_client_structure = _load_module(
            "project_verify_client_structure",
            PROJECT_ROOT / "scripts" / "verify_client_structure.py",
        )
        actual_modules = {p.name for p in (CLIENT_ROOT / "modules").glob("*.py") if p.name != "__init__.py"}

        self.assertEqual(check_structure.EXPECTED_MODULES, actual_modules)
        self.assertEqual(verify_client_structure.EXPECTED_MODULES, actual_modules)
        self.assertEqual(check_structure.EXPECTED_MODULES, verify_client_structure.EXPECTED_MODULES)

    def test_both_structure_checkers_accept_the_source_tree(self):
        check_structure = _load_module(
            "client_cli_check_structure_run",
            CLIENT_ROOT / "scripts" / "check_structure.py",
        )
        verify_client_structure = _load_module(
            "project_verify_client_structure_run",
            PROJECT_ROOT / "scripts" / "verify_client_structure.py",
        )

        self.assertEqual(check_structure.main(), 0)
        self.assertEqual(verify_client_structure.check(CLIENT_ROOT), [])


if __name__ == "__main__":
    unittest.main()
