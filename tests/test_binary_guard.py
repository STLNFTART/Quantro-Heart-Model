"""Tests for the repository binary guard utilities (v1.0.0)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.check_no_binary import ScanResult, _is_binary_blob, scan_repository


class TestBinaryGuard(unittest.TestCase):
    """Verify binary detection heuristics for Donte's workflow."""

    def test_is_binary_blob_detects_null_bytes(self) -> None:
        """Files containing NULL bytes must be flagged as binary."""

        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"text\0binary")
            candidate = Path(handle.name)

        try:
            self.assertTrue(_is_binary_blob(candidate))
        finally:
            candidate.unlink(missing_ok=True)

    def test_scan_repository_flags_binary_files(self) -> None:
        """Repository scan should isolate binary offenders."""

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            text_file = repo_root / "notes.txt"
            binary_file = repo_root / "image.bin"
            text_file.write_text("Hello Donte!", encoding="utf-8")
            binary_file.write_bytes(b"\xff\x00\xff")

            results = scan_repository(repo_root)
            flagged = {result.path.name: result for result in results if result.is_binary}

            self.assertIn("image.bin", flagged)
            self.assertNotIn("notes.txt", flagged)
            self.assertIsInstance(flagged["image.bin"], ScanResult)


if __name__ == "__main__":  # pragma: no cover - support direct execution
    unittest.main()
