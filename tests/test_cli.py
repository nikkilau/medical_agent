import unittest
from unittest.mock import patch

import micro_mdt.cli as cli


class CliTests(unittest.TestCase):
    def test_web_mode_passes_openai_compatible_options(self):
        with patch("micro_mdt.cli.load_env_file") as load_env_file:
            with patch("webapp.run_server") as run_server:
                result = cli.main([
                    "--web",
                    "--provider",
                    "openai-compatible",
                    "--base-url",
                    "https://api.deepseek.com/v1",
                    "--model",
                    "deepseek-chat",
                    "--port",
                    "9090",
                ])

        self.assertEqual(result, 0)
        load_env_file.assert_called_once_with()
        run_server.assert_called_once_with(
            port=9090,
            provider="openai-compatible",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )


if __name__ == "__main__":
    unittest.main()
