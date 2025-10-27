"""Unit tests for analyze_results.py (v1.1.0)."""
from __future__ import annotations

import unittest

import pandas as pd

from analyze_results import (
    compute_rolling_average,
    compute_rolling_averages,
    filter_results,
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


if __name__ == "__main__":  # pragma: no cover - test runner entry point
    unittest.main()
