"""Unit tests for the batch analysis runner (v1.0.0)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from batch_analysis import execute_batch, load_batch_config


class TestBatchAnalysis(unittest.TestCase):
    """Validate batch configuration loading and execution semantics."""

    def setUp(self) -> None:
        self.synthetic = pd.DataFrame(
            {
                "model": ["SIR", "SIR", "SIR"],
                "mode": ["Residual", "Residual", "Residual"],
                "lambda": [0.5, 1.0, 1.5],
                "val1": [2.0, 3.0, 4.0],
                "val2": [1.0, 1.5, 2.0],
            }
        )

    def test_load_batch_config(self) -> None:
        """Configuration loader should expand relative paths and defaults."""

        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config_path = base / "config.json"
            csv_path = base / "results.csv"
            output_path = base / "artifacts" / "plot.svg"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.synthetic.to_csv(csv_path, index=False)

            config_payload = {
                "tasks": [
                    {
                        "name": "unit_batch",
                        "input": "results.csv",
                        "output": "artifacts/plot.svg",
                        "output_format": "svg",
                        "window": 2,
                        "window_list": [3],
                    }
                ]
            }
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")

            tasks = load_batch_config(config_path)
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertEqual(task.name, "unit_batch")
            self.assertTrue(task.parameters.input_path.is_absolute())
            self.assertTrue(task.parameters.output.is_absolute())
            self.assertEqual(task.parameters.output_format, "svg")

            # Ensure round-trip execution produces the artifact.
            generated_paths = execute_batch(tasks)
            self.assertTrue(task.parameters.output.exists())
            self.assertEqual(generated_paths[0].suffix, ".svg")
            self.assertTrue(task.parameters.output.read_text(encoding="utf-8").startswith("<?xml"))

    def test_invalid_config_raises(self) -> None:
        """Malformed configuration should trigger a ValueError."""

        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "broken.json"
            config_path.write_text(json.dumps({"tasks": "not-a-list"}), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_batch_config(config_path)


if __name__ == "__main__":  # pragma: no cover - test runner entry point
    unittest.main()
