"""Quantro Heart Model result analysis utilities (v1.0.0).

This module provides data-loading, filtering, rolling-average computation, and
visualization routines tailored to the CSV output produced by ``run.apl``.

The script can be executed as a CLI to generate rolling-average plots for a
specified model/mode combination. All computations and figures adhere to the
project's clarity and reproducibility standards.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

# Configure module-level logger with a sensible default format for reuse.
LOGGER = logging.getLogger("quantro.analysis")


def configure_logging(verbose: bool) -> None:
    """Configure logging verbosity.

    Parameters
    ----------
    verbose:
        When ``True`` the log level is set to ``DEBUG``; otherwise ``INFO``.
    """

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LOGGER.debug("Logging configured (verbose=%s).", verbose)


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load the project CSV results into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    csv_path:
        Path to ``results.csv`` or an equivalent CSV file.

    Returns
    -------
    pandas.DataFrame
        The parsed data with appropriate dtypes preserved.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the CSV cannot be parsed into a DataFrame.
    """

    LOGGER.debug("Attempting to load CSV data from %s", csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    try:
        data_frame = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - pandas supplies detailed message
        raise ValueError(f"Failed to read CSV '{csv_path}': {exc}") from exc

    LOGGER.info("Loaded %d rows from %s", len(data_frame), csv_path)
    return data_frame


def filter_results(
    data_frame: pd.DataFrame,
    model: Optional[str] = None,
    mode: Optional[str] = None,
) -> pd.DataFrame:
    """Filter the DataFrame by model and/or mode when provided.

    Parameters
    ----------
    data_frame:
        The input DataFrame containing at least ``model`` and ``mode`` columns.
    model:
        Optional model name to keep (case sensitive to remain explicit).
    mode:
        Optional mode identifier to keep (also case sensitive).

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame; original index is reset for tidy plotting.

    Raises
    ------
    KeyError
        If required columns are missing from ``data_frame``.
    ValueError
        If the filtered result is empty, indicating mismatched filters.
    """

    required_columns = {"model", "mode"}
    if not required_columns.issubset(data_frame.columns):
        missing = required_columns.difference(data_frame.columns)
        raise KeyError(
            "Input DataFrame must contain columns: " + ", ".join(sorted(missing))
        )

    filtered = data_frame.copy()

    if model is not None:
        LOGGER.debug("Applying model filter: %s", model)
        filtered = filtered[filtered["model"] == model]
    if mode is not None:
        LOGGER.debug("Applying mode filter: %s", mode)
        filtered = filtered[filtered["mode"] == mode]

    if filtered.empty:
        raise ValueError(
            "No data remaining after applying filters: "
            f"model={model!r}, mode={mode!r}."
        )

    filtered = filtered.reset_index(drop=True)
    LOGGER.info(
        "Filtered dataset to %d rows using model=%s mode=%s",
        len(filtered),
        model,
        mode,
    )
    return filtered


def compute_rolling_average(
    data_frame: pd.DataFrame,
    value_column: str,
    window_size: int,
) -> pd.Series:
    """Compute a rolling-window average for the specified column.

    Parameters
    ----------
    data_frame:
        DataFrame containing the target column.
    value_column:
        Column on which to compute the rolling mean.
    window_size:
        Window length :math:`n` (positive integer) for the rolling mean.

    Returns
    -------
    pandas.Series
        Rolling mean aligned with the original index.

    Raises
    ------
    KeyError
        If ``value_column`` is absent from ``data_frame``.
    ValueError
        If ``window_size`` is not a positive integer.
    """

    if value_column not in data_frame.columns:
        raise KeyError(f"Column '{value_column}' not found in DataFrame.")

    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")

    LOGGER.debug(
        "Computing rolling average for column %s with window size %d",
        value_column,
        window_size,
    )
    # ``min_periods=1`` retains early samples while the window warms up.
    return data_frame[value_column].rolling(window=window_size, min_periods=1).mean()


def plot_results(
    data_frame: pd.DataFrame,
    rolling_mean: pd.Series,
    x_axis: str,
    value_column: str,
    output_path: Path,
    title: str,
    window_size: int,
) -> None:
    """Generate a publication-ready plot comparing raw data to rolling mean.

    Parameters
    ----------
    data_frame:
        DataFrame containing at least ``x_axis`` and ``value_column`` columns.
    rolling_mean:
        Series produced by :func:`compute_rolling_average`.
    x_axis:
        Column name for the x-axis (e.g., ``lambda``).
    value_column:
        Target y-axis data column name.
    output_path:
        Destination file for the saved plot (PNG by convention).
    title:
        Plot title describing the context (model/mode/value combination).
    window_size:
        Size of the rolling window :math:`n` used for the averaged curve.
    """

    if x_axis not in data_frame.columns:
        raise KeyError(f"Column '{x_axis}' not found in DataFrame for plotting.")

    LOGGER.debug("Creating plot for %s vs %s", value_column, x_axis)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        data_frame[x_axis],
        data_frame[value_column],
        label="Raw data",
        marker="o",
        linestyle="-",
        linewidth=1.2,
    )
    ax.plot(
        data_frame[x_axis],
        rolling_mean,
        label=f"Rolling mean (window={window_size})",
        marker="s",
        linestyle="--",
        linewidth=1.4,
    )
    ax.set_xlabel(f"{x_axis} (dimensionless)")
    ax.set_ylabel(f"{value_column} (SI units where applicable)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved plot to %s", output_path)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for the analysis script."""

    parser = argparse.ArgumentParser(
        description="Analyze Quantro Heart Model CSV results and plot rolling averages.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results.csv"),
        help="Path to the CSV file generated by run.apl (default: results.csv).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SIR",
        help="Model name to filter by (default: SIR).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="Residual",
        help="Mode name to filter by (default: Residual).",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="val1",
        help="Column to analyze and plot (default: val1).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Rolling window size n for the average (default: 5).",
    )
    parser.add_argument(
        "--x-axis",
        type=str,
        default="lambda",
        help="Column to use for the x-axis (default: lambda).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/rolling_mean.png"),
        help="Output path for the generated plot (default: artifacts/rolling_mean.png).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging purposes.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the CLI script."""

    args = parse_arguments()
    configure_logging(args.verbose)
    LOGGER.debug("CLI arguments: %s", args)

    data_frame = load_results(args.input)
    filtered = filter_results(data_frame, model=args.model, mode=args.mode)

    # Sort by the x-axis to keep the visualization monotonic if the CSV is unordered.
    filtered = filtered.sort_values(by=args.x_axis).reset_index(drop=True)
    rolling_mean = compute_rolling_average(filtered, args.column, args.window)

    title = (
        f"{args.model} ({args.mode}) — {args.column} rolling mean (n={args.window})"
    )
    plot_results(
        data_frame=filtered,
        rolling_mean=rolling_mean,
        x_axis=args.x_axis,
        value_column=args.column,
        output_path=args.output,
        title=title,
        window_size=args.window,
    )

    # Provide a quick textual summary for reproducibility and logging.
    LOGGER.info(
        "Summary statistics for %s: min=%.4g median=%.4g max=%.4g",
        args.column,
        filtered[args.column].min(),
        filtered[args.column].median(),
        filtered[args.column].max(),
    )


if __name__ == "__main__":
    main()
