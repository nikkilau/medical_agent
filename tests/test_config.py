import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from micro_mdt.config import load_env_file


class ConfigTests(unittest.TestCase):
    def test_missing_env_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            load_env_file(Path(tmp) / "missing.env")

    def test_loads_key_value_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n"
                "# comment\n"
                "MICRO_MDT_API_KEY=test-key\n"
                "MICRO_MDT_BASE_URL=https://api.deepseek.com/v1\n"
                "MICRO_MDT_MODEL=deepseek-chat\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                load_env_file(env_path)
                self.assertEqual(os.environ["MICRO_MDT_API_KEY"], "test-key")
                self.assertEqual(os.environ["MICRO_MDT_BASE_URL"], "https://api.deepseek.com/v1")
                self.assertEqual(os.environ["MICRO_MDT_MODEL"], "deepseek-chat")

    def test_strips_optional_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "MICRO_MDT_API_KEY='quoted-key'\n"
                'MICRO_MDT_MODEL="deepseek-chat"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                load_env_file(env_path)
                self.assertEqual(os.environ["MICRO_MDT_API_KEY"], "quoted-key")
                self.assertEqual(os.environ["MICRO_MDT_MODEL"], "deepseek-chat")

    def test_existing_environment_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("MICRO_MDT_MODEL=from-file\n", encoding="utf-8")
            with patch.dict(os.environ, {"MICRO_MDT_MODEL": "from-env"}, clear=True):
                load_env_file(env_path)
                self.assertEqual(os.environ["MICRO_MDT_MODEL"], "from-env")


if __name__ == "__main__":
    unittest.main()
