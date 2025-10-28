"""Vectorized SIR parameter sweep utilities for Quantro Heart Model (v1.0.0).

This module provides a CLI and reusable API for executing simultaneous SIR
simulations across a grid of transmission (beta) and recovery (gamma)
parameters. The solver uses explicit Euler integration with vectorized NumPy
operations so multiple parameter combinations share a single time loop. The
results are exported as a CSV (text artifact) and an SVG visualization to stay
compatible with hosts that forbid binary blobs.

Usage example
-------------

```bash
python3 vectorized_sweep.py \
  --betas 0.18 0.22 0.26 \
  --gammas 0.08 0.1 \
  --contact-multipliers 0.9 1.0 1.1 \
  --time-step 0.25 \
  --duration 60 \
  --population 1000 \
  --initial-infected 12 \
  --initial-recovered 1 \
  --rolling-window 5 \
  --output-csv artifacts/vectorized_sir_results.csv \
  --output-plot artifacts/vectorized_sir_infectious.svg
```

Both CLI arguments and JSON configurations (via ``--config``) are supported.
Refer to ``configs/vectorized_sir.json`` for a documented template.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Logger configured via :func:`configure_logging`.
LOGGER = logging.getLogger("quantro.vectorized_sweep")


@dataclass(frozen=True)
class VectorizedSIRConfig:
    """Configuration container for vectorized SIR parameter sweeps."""

    betas: Sequence[float]
    gammas: Sequence[float]
    contact_multipliers: Sequence[float]
    time_step: float
    duration: float
    population: float
    initial_infected: float
    initial_recovered: float
    rolling_window: int
    output_csv: Path
    output_plot: Path
    verbose: bool = False


def configure_logging(verbose: bool) -> None:
    """Configure consistent logging for the module."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LOGGER.debug("Logging configured (verbose=%s)", verbose)


def _validate_positive_sequence(values: Sequence[float], name: str) -> np.ndarray:
    """Ensure ``values`` is a non-empty array of positive floats."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"At least one value must be provided for {name}.")
    if np.any(array <= 0):
        raise ValueError(f"All {name} values must be positive (SI units).")
    return array


def _validate_non_negative(value: float, name: str) -> float:
    """Ensure a scalar value is non-negative."""

    if value < 0:
        raise ValueError(f"{name} must be non-negative (SI units).")
    return value


def _generate_parameter_grid(
    betas: np.ndarray,
    gammas: np.ndarray,
    contact_multipliers: np.ndarray,
) -> pd.DataFrame:
    """Return a DataFrame enumerating every parameter combination."""

    records: List[dict[str, float]] = []
    for beta, gamma, multiplier in product(betas, gammas, contact_multipliers):
        records.append(
            {
                "beta": float(beta),
                "gamma": float(gamma),
                "contact_multiplier": float(multiplier),
            }
        )
    grid = pd.DataFrame.from_records(records)
    grid["simulation_id"] = np.arange(len(grid), dtype=int)
    LOGGER.debug("Generated parameter grid with %d combinations", len(grid))
    return grid


def _simulate_sir_vectorized(
    grid: pd.DataFrame,
    *,
    time_step: float,
    duration: float,
    population: float,
    initial_infected: float,
    initial_recovered: float,
) -> pd.DataFrame:
    """Simulate SIR dynamics for all parameter combinations simultaneously."""

    steps = int(np.floor(duration / time_step)) + 1
    times = np.linspace(0.0, time_step * (steps - 1), steps)

    beta = grid["beta"].to_numpy(dtype=float)
    gamma = grid["gamma"].to_numpy(dtype=float)
    effective_beta = beta * grid["contact_multiplier"].to_numpy(dtype=float)

    infected0 = np.full(beta.shape, initial_infected, dtype=float)
    recovered0 = np.full(beta.shape, initial_recovered, dtype=float)
    susceptible0 = np.full(beta.shape, population - initial_infected - initial_recovered, dtype=float)

    susceptible = np.empty((beta.size, steps), dtype=float)
    infected = np.empty_like(susceptible)
    recovered = np.empty_like(susceptible)

    susceptible[:, 0] = susceptible0
    infected[:, 0] = infected0
    recovered[:, 0] = recovered0

    # Perform vectorized Euler integration.
    for idx in range(1, steps):
        current_s = susceptible[:, idx - 1]
        current_i = infected[:, idx - 1]
        current_r = recovered[:, idx - 1]

        force_of_infection = (effective_beta * current_s * current_i) / population
        recovery = gamma * current_i

        susceptible[:, idx] = current_s - time_step * force_of_infection
        infected[:, idx] = current_i + time_step * (force_of_infection - recovery)
        recovered[:, idx] = current_r + time_step * recovery

        # Numerical guard rails keep state variables within bounds.
        susceptible[:, idx] = np.clip(susceptible[:, idx], 0.0, population)
        infected[:, idx] = np.clip(infected[:, idx], 0.0, population)
        recovered[:, idx] = np.clip(recovered[:, idx], 0.0, population)

    records: List[pd.DataFrame] = []
    for sim_index, params in grid.iterrows():
        df = pd.DataFrame(
            {
                "time": times,
                "susceptible": susceptible[sim_index],
                "infectious": infected[sim_index],
                "recovered": recovered[sim_index],
                "beta": params["beta"],
                "gamma": params["gamma"],
                "contact_multiplier": params["contact_multiplier"],
                "simulation_id": params["simulation_id"],
            }
        )
        records.append(df)

    result = pd.concat(records, ignore_index=True)
    LOGGER.debug(
        "Simulated %d trajectories with %d time steps each (dt=%.3f s)",
        grid.shape[0],
        steps,
        time_step,
    )
    return result


def _append_rolling_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Append rolling averages of the infectious compartment per simulation."""

    if window <= 0:
        raise ValueError("Rolling window must be a positive integer.")

    df = df.sort_values(["simulation_id", "time"]).reset_index(drop=True)
    df["rolling_infectious"] = (
        df.groupby("simulation_id")["infectious"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    return df


def run_vectorized_sweep(config: VectorizedSIRConfig) -> pd.DataFrame:
    """Execute the full vectorized sweep and return the results DataFrame."""

    betas = _validate_positive_sequence(config.betas, "beta")
    gammas = _validate_positive_sequence(config.gammas, "gamma")
    multipliers = _validate_positive_sequence(config.contact_multipliers, "contact multiplier")
    if config.time_step <= 0:
        raise ValueError("time_step must be positive (seconds).")
    if config.duration <= 0:
        raise ValueError("duration must be positive (seconds).")
    _validate_positive_sequence([config.population], "population")
    _validate_non_negative(config.initial_infected, "initial_infected")
    _validate_non_negative(config.initial_recovered, "initial_recovered")
    if config.initial_infected + config.initial_recovered > config.population:
        raise ValueError("Initial infected plus recovered cannot exceed population.")

    grid = _generate_parameter_grid(betas, gammas, multipliers)
    results = _simulate_sir_vectorized(
        grid,
        time_step=config.time_step,
        duration=config.duration,
        population=config.population,
        initial_infected=config.initial_infected,
        initial_recovered=config.initial_recovered,
    )
    results = _append_rolling_mean(results, config.rolling_window)
    return results


def save_results(df: pd.DataFrame, output_path: Path) -> Path:
    """Persist the sweep results as a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved sweep results to %s", output_path)
    return output_path


def plot_infectious_curves(df: pd.DataFrame, output_path: Path) -> Path:
    """Create an SVG plot overlaying infectious trajectories and rolling means."""

    if output_path.suffix.lower() != ".svg":
        raise ValueError("Only SVG plots are supported to maintain text-only artifacts.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for simulation_id, group in df.groupby("simulation_id"):
        label = (
            f"sim {simulation_id} | beta={group['beta'].iat[0]:.3f}, "
            f"gamma={group['gamma'].iat[0]:.3f}, c={group['contact_multiplier'].iat[0]:.2f}"
        )
        ax.plot(group["time"], group["infectious"], linewidth=1.0, alpha=0.6, label=f"{label} (raw)")
        ax.plot(
            group["time"],
            group["rolling_infectious"],
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
            label=f"{label} (rolling)",
        )

    ax.set_title("Vectorized SIR infectious trajectories", fontsize=12)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infectious individuals (count)")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved infectious trajectory plot to %s", output_path)
    return output_path


def _load_config(path: Path) -> dict:
    """Load a JSON configuration file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    LOGGER.debug("Loaded configuration from %s", path)
    return payload


def _merge_args_with_config(args: argparse.Namespace, payload: dict) -> VectorizedSIRConfig:
    """Merge CLI arguments with JSON payload into a configuration object."""

    def _get_sequence(name: str, default: Sequence[float]) -> Sequence[float]:
        cli_value = getattr(args, name)
        if cli_value:
            return cli_value
        if name in payload:
            return payload[name]
        return default

    def _get_scalar(name: str, default: float) -> float:
        cli_value = getattr(args, name)
        if cli_value is not None:
            return cli_value
        if name in payload:
            return payload[name]
        return default

    def _get_path(name: str, default: str) -> Path:
        cli_value = getattr(args, name)
        if cli_value is not None:
            return Path(cli_value)
        if name in payload:
            return Path(payload[name])
        return Path(default)

    config = VectorizedSIRConfig(
        betas=_get_sequence("betas", [0.2]),
        gammas=_get_sequence("gammas", [0.1]),
        contact_multipliers=_get_sequence("contact_multipliers", [1.0]),
        time_step=float(_get_scalar("time_step", 0.25)),
        duration=float(_get_scalar("duration", 30.0)),
        population=float(_get_scalar("population", 1000.0)),
        initial_infected=float(_get_scalar("initial_infected", 10.0)),
        initial_recovered=float(_get_scalar("initial_recovered", 0.0)),
        rolling_window=int(_get_scalar("rolling_window", 5)),
        output_csv=_get_path("output_csv", "artifacts/vectorized_sir_results.csv"),
        output_plot=_get_path("output_plot", "artifacts/vectorized_sir_infectious.svg"),
        verbose=bool(_get_scalar("verbose", False)),
    )
    return config


def parse_arguments(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Vectorized SIR sweep runner producing CSV and SVG artifacts.",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON configuration file.")
    parser.add_argument("--betas", type=float, nargs="*", help="Transmission rates beta (per day).")
    parser.add_argument("--gammas", type=float, nargs="*", help="Recovery rates gamma (per day).")
    parser.add_argument(
        "--contact-multipliers",
        type=float,
        nargs="*",
        help="Dimensionless multipliers applied to each beta to emulate contact changes.",
    )
    parser.add_argument("--time-step", type=float, help="Integration step (days).")
    parser.add_argument("--duration", type=float, help="Simulation duration (days).")
    parser.add_argument("--population", type=float, help="Total population (individuals).")
    parser.add_argument("--initial-infected", type=float, help="Initial infectious count (individuals).")
    parser.add_argument("--initial-recovered", type=float, help="Initial recovered count (individuals).")
    parser.add_argument("--rolling-window", type=int, help="Rolling window (time steps) for infection averages.")
    parser.add_argument("--output-csv", type=Path, help="Destination CSV path for results.")
    parser.add_argument("--output-plot", type=Path, help="Destination SVG path for infectious plot.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point returning a process exit status."""

    args = parse_arguments(argv)
    payload = {}
    if args.config:
        payload = _load_config(args.config)
    config = _merge_args_with_config(args, payload)

    configure_logging(config.verbose or args.verbose)
    LOGGER.info("Starting vectorized SIR sweep")
    LOGGER.debug("Configuration: %s", config)

    results = run_vectorized_sweep(config)
    save_results(results, config.output_csv)
    plot_infectious_curves(results, config.output_plot)

    LOGGER.info("Vectorized sweep complete: %d rows", len(results))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

