"""CLI smoke tests for ``halfmap-qc``."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from cryoem_mrc.cli import main
from cryoem_mrc.tui import print_help


class TestHalfmapQcCli(unittest.TestCase):
    def test_version(self) -> None:
        rc = main(["--version"])
        self.assertEqual(rc, 0)

    def test_help_command(self) -> None:
        with patch("sys.stdout"):
            rc = main(["help"])
        self.assertEqual(rc, 0)

    def test_help_flag(self) -> None:
        with patch("sys.stdout"):
            rc = main(["--help"])
        self.assertEqual(rc, 0)

    def test_no_argv_non_tty_prints_help(self) -> None:
        with patch("sys.stdin") as stdin, patch("sys.stdout"):
            stdin.isatty.return_value = False
            rc = main([])
        self.assertEqual(rc, 0)

    def test_unknown_command(self) -> None:
        with patch("sys.stdout"), patch("sys.stderr"):
            rc = main(["not-a-command"])
        self.assertEqual(rc, 2)

    def test_help_text_mentions_install(self) -> None:
        from cryoem_mrc.tui import HELP_TEXT

        self.assertIn("pip install cryoem-halfmap-qc", HELP_TEXT)
        self.assertNotIn("git+", HELP_TEXT)

    def test_cohort_removed_from_cli(self) -> None:
        with patch("sys.stdout"), patch("sys.stderr"):
            rc = main(["cohort"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
