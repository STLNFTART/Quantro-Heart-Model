"""Tests for the vectorized SIR sweep utilities (v1.0.0)."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vectorized_sweep import (
    VectorizedSIRConfig,
    _append_rolling_mean,
    _simulate_sir_vectorized,
    run_vectorized_sweep,
)


class TestVectorizedSweep(unittest.TestCase):
    """Validate vectorized simulation accuracy and rolling averages."""

    def setUp(self) -> None:
        self.grid = pd.DataFrame(
            {
                "beta": [0.24, 0.24],
                "gamma": [0.1, 0.1],
                "contact_multiplier": [1.0, 0.8],
                "simulation_id": [0, 1],
            }
        )

    def test_vectorized_matches_scalar_simulation(self) -> None:
        """Vectorized solver should agree with scalar Euler integration."""

        duration = 1.0
        dt = 0.25
        population = 1000.0
        infected0 = 10.0
        recovered0 = 0.0

        vectorized = _simulate_sir_vectorized(
            self.grid,
            time_step=dt,
            duration=duration,
            population=population,
            initial_infected=infected0,
            initial_recovered=recovered0,
        )

        # Compare against a manual scalar implementation for the first simulation.
        time_steps = int(math.floor(duration / dt)) + 1
        s = np.empty(time_steps)
        i = np.empty(time_steps)
        r = np.empty(time_steps)
        s[0] = population - infected0 - recovered0
        i[0] = infected0
        r[0] = recovered0
        beta = 0.24
        gamma = 0.1
        for idx in range(1, time_steps):
            force = beta * s[idx - 1] * i[idx - 1] / population
            rec = gamma * i[idx - 1]
            s[idx] = s[idx - 1] - dt * force
            i[idx] = i[idx - 1] + dt * (force - rec)
            r[idx] = r[idx - 1] + dt * rec

        first_sim = vectorized[vectorized["simulation_id"] == 0]
        np.testing.assert_allclose(first_sim["susceptible"].to_numpy(), s, rtol=1e-6)
        np.testing.assert_allclose(first_sim["infectious"].to_numpy(), i, rtol=1e-6)
        np.testing.assert_allclose(first_sim["recovered"].to_numpy(), r, rtol=1e-6)

    def test_rolling_mean_appended(self) -> None:
        """Rolling mean column should be added per simulation."""

        df = _simulate_sir_vectorized(
            self.grid,
            time_step=0.5,
            duration=2.0,
            population=500.0,
            initial_infected=5.0,
            initial_recovered=0.0,
        )
        enriched = _append_rolling_mean(df, window=2)
        self.assertIn("rolling_infectious", enriched.columns)
        windowed = (
            enriched[enriched["simulation_id"] == 0]["rolling_infectious"].tolist()
        )
        self.assertGreaterEqual(windowed[1], windowed[0])

    def test_run_vectorized_sweep_complete_pipeline(self) -> None:
        """High-level API should return a complete DataFrame with expected columns."""

        config = VectorizedSIRConfig(
            betas=[0.2, 0.3],
            gammas=[0.1],
            contact_multipliers=[1.0],
            time_step=0.5,
            duration=1.5,
            population=1000.0,
            initial_infected=5.0,
            initial_recovered=0.0,
            rolling_window=2,
            output_csv=Path("artifacts/test.csv"),
            output_plot=Path("artifacts/test.svg"),
            verbose=False,
        )
        result = run_vectorized_sweep(config)
        self.assertGreater(len(result), 0)
        expected_columns = {
            "time",
            "susceptible",
            "infectious",
            "recovered",
            "beta",
            "gamma",
            "contact_multiplier",
            "simulation_id",
            "rolling_infectious",
        }
        self.assertTrue(expected_columns.issubset(result.columns))


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    unittest.main()
