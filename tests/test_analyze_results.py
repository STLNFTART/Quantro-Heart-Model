"""Unit tests for analyze_results.py (v1.4.0)."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from analyze_results import (
    AnalysisParameters,
    _prepare_output_path,
    _resolve_output_format,
    compute_rolling_average,
    compute_rolling_averages,
    filter_results,
    run_analysis,
)


class TestAnalyzeResults(unittest.TestCase):
    """Test suite validating filtering and rolling average utilities."""

    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "model": ["SIR", "SIR", "MM"],
                "mode": ["Residual", "Control", "Residual"],
                "val1": [1.0, 3.0, 5.0],
            }
        )

    def test_filter_results_by_model_and_mode(self) -> None:
        """Filtering should isolate a single row when both keys match."""

        filtered = filter_results(self.data, model="SIR", mode="Residual")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["val1"], 1.0)

    def test_compute_rolling_average(self) -> None:
        """Rolling mean should respect ``min_periods`` and ``center`` toggles."""

        subset = self.data.iloc[:3].copy()
        subset["val1"] = [1.0, 3.0, 5.0]
        centered = compute_rolling_average(
            subset,
            "val1",
            window_size=2,
            min_periods=2,
            center=True,
        )
        self.assertTrue(centered.isna().iloc[0])
        self.assertAlmostEqual(centered.iloc[1], 2.0)
        self.assertAlmostEqual(centered.iloc[2], 4.0)

    def test_compute_rolling_averages_multiple_windows(self) -> None:
        """Multiple rolling averages should be computed with unique windows."""

        windows = compute_rolling_averages(
            self.data,
            "val1",
            [1, 2, 2, 3],
            min_periods=1,
            center=False,
        )
        self.assertEqual(sorted(windows.keys()), [1, 2, 3])
        self.assertListEqual(windows[1].tolist(), self.data["val1"].tolist())
        self.assertListEqual(windows[2].round(3).tolist(), [1.0, 2.0, 4.0])

    def test_run_analysis_creates_plot(self) -> None:
        """Full pipeline execution should create the requested figure artifact."""

        with TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            csv_path = temp_dir / "results.csv"
            output_path = temp_dir / "plot.svg"

            synthetic = pd.DataFrame(
                {
                    "model": ["SIR"] * 5,
                    "mode": ["Residual"] * 5,
                    "lambda": [0.5, 1.0, 1.5, 2.0, 2.5],
                    "val1": [10.0, 9.0, 8.0, 7.0, 6.0],
                }
            )
            synthetic.to_csv(csv_path, index=False)

            params = AnalysisParameters(
                input_path=csv_path,
                model="SIR",
                mode="Residual",
                column="val1",
                window=2,
                window_list=[3],
                min_periods=1,
                center=False,
                fig_width=6.0,
                fig_height=4.0,
                x_axis="lambda",
                output=output_path,
                output_format="auto",
                label="unit_test",
            )

            produced_path = run_analysis(params)
            self.assertTrue(produced_path.exists())
            self.assertTrue(produced_path.read_text(encoding="utf-8").startswith("<?xml"))

    def test_resolve_output_format_enforces_svg_only(self) -> None:
        """Auto output format should resolve to SVG and reject raster suffixes."""

        svg_path = Path("artifacts/plot.svg")
        suffixless = Path("artifacts/plot")
        png_path = Path("artifacts/plot.png")

        self.assertEqual(_resolve_output_format(svg_path, "auto"), "svg")
        self.assertEqual(_resolve_output_format(suffixless, "auto"), "svg")

        with self.assertRaises(ValueError):
            _resolve_output_format(svg_path, "gif")
        with self.assertRaises(ValueError):
            _resolve_output_format(png_path, "auto")

    def test_prepare_output_path_appends_or_replaces_suffix(self) -> None:
        """Helper should append or correct suffixes to match the export format."""

        base = Path("artifacts/result")
        appended = _prepare_output_path(base, "svg")
        self.assertEqual(appended.suffix, ".svg")

        mismatched = Path("artifacts/result.png")
        corrected = _prepare_output_path(mismatched, "svg")
        self.assertEqual(corrected.suffix, ".svg")


if __name__ == "__main__":  # pragma: no cover - test runner entry point
    unittest.main()
