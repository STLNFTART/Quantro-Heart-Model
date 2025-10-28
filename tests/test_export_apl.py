"""Tests for the ASCII-to-Unicode APL exporter (v1.0.0)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.export_apl import decode_apl_template, export_templates


class TestExportApl(unittest.TestCase):
    """Validate regeneration of Unicode APL scripts from templates."""

    def test_decode_apl_template_round_trip(self) -> None:
        """The decoder should recover Unicode glyphs from escape sequences."""

        encoded = "\\u2190\\u03b1 + \\u03b2"
        decoded = decode_apl_template(encoded)
        self.assertEqual(decoded, "←α + β")

    def test_export_templates_creates_files(self) -> None:
        """Exporting templates should write Unicode APL files to disk."""

        source_dir = Path("apl_sources")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            exported = export_templates(source_dir, output_dir, overwrite=False)

            self.assertGreater(len(exported), 0)
            for path in exported:
                content = path.read_text(encoding="utf-8")
                self.assertIn("⍝", content)
                self.assertFalse("\\u" in content)

    def test_export_templates_prevent_overwrite(self) -> None:
        """Overwriting without the flag should raise an informative error."""

        source_dir = Path("apl_sources")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            export_templates(source_dir, output_dir, overwrite=False)
            with self.assertRaises(FileExistsError):
                export_templates(source_dir, output_dir, overwrite=False)


if __name__ == "__main__":  # pragma: no cover - direct execution support
    unittest.main()
