"""Tests for the repository binary guard utilities (v1.1.0)."""
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

    def test_is_binary_blob_detects_late_binary_sequences(self) -> None:
        """Binary markers appearing after the initial chunk must still be caught."""

        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"A" * 8192)  # Write a long ASCII prefix exceeding one chunk.
            handle.write(b"\xff")  # Append a non-UTF-8 byte beyond the first chunk.
            candidate = Path(handle.name)

        try:
            self.assertTrue(_is_binary_blob(candidate))
        finally:
            candidate.unlink(missing_ok=True)

    def test_is_binary_blob_detects_invalid_utf8(self) -> None:
        """High-ASCII sequences that fail UTF-8 decoding are treated as binary."""

        with tempfile.NamedTemporaryFile(delete=False) as handle:
            # Craft a byte pattern lacking NULL bytes but invalid under UTF-8.
            handle.write(b"\xff\xfe\xfd\xfa")
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

    def test_scan_repository_ascii_only_flags_unicode_payloads(self) -> None:
        """Strict ASCII mode should report UTF-8 text containing extended glyphs."""

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            ascii_file = repo_root / "notes.txt"
            unicode_file = repo_root / "unicode.txt"
            ascii_file.write_text("Hello Donte!", encoding="ascii")
            unicode_file.write_text("Voltage alpha \u03b1", encoding="utf-8")

            results = scan_repository(repo_root, ascii_only=True)
            findings = {result.path.name: result for result in results}

            self.assertFalse(findings["notes.txt"].ascii_violation)
            self.assertTrue(findings["unicode.txt"].ascii_violation)
            self.assertFalse(findings["unicode.txt"].is_binary)


if __name__ == "__main__":  # pragma: no cover - support direct execution
    unittest.main()
