"""Unit tests for analyze_results.py (v1.0.0)."""
from __future__ import annotations

import unittest

import pandas as pd

from analyze_results import compute_rolling_average, filter_results


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
        """Rolling mean should reduce to cumulative mean with window >= len."""

        series = compute_rolling_average(self.data.iloc[:2], "val1", window_size=2)
        self.assertListEqual(series.round(3).tolist(), [1.0, 2.0])


if __name__ == "__main__":  # pragma: no cover - test runner entry point
    unittest.main()
