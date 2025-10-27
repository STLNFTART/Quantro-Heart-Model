"""Quantro Heart Model result analysis utilities (v1.2.1).

This module provides data-loading, filtering, rolling-average computation, and
visualization routines tailored to the CSV output produced by ``run.apl``.

The script can be executed as a CLI to generate rolling-average plots for a
specified model/mode combination. All computations and figures adhere to the
project's clarity and reproducibility standards.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

# Use the non-interactive Agg backend so batch jobs and tests run headlessly.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Configure module-level logger with a sensible default format for reuse.
LOGGER = logging.getLogger("quantro.analysis")


@dataclass(frozen=True)
class AnalysisParameters:
    """Typed container for a single rolling-average analysis job."""

    input_path: Path
    model: str = "SIR"
    mode: str = "Residual"
    column: str = "val1"
    window: int = 5
    window_list: Optional[Sequence[int]] = None
    min_periods: int = 1
    center: bool = False
    fig_width: float = 8.0
    fig_height: float = 5.0
    x_axis: str = "lambda"
    output: Path = field(default_factory=lambda: Path("artifacts/rolling_mean.png"))
    skip_default_window: bool = False
    verbose: bool = False
    label: str = "analysis"


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
        Path to a Quantro simulation CSV file (see ``data/example_results.csv``).

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
    *,
    min_periods: int = 1,
    center: bool = False,
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
    min_periods:
        Minimum required samples within each window to emit a statistic (>=1).
    center:
        When ``True`` the rolling window is centered, producing symmetric
        smoothing suitable for retrospective analysis.

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

    if min_periods <= 0:
        raise ValueError("min_periods must be a positive integer.")

    LOGGER.debug(
        "Computing rolling average for column %s with window size %d (min_periods=%d, center=%s)",
        value_column,
        window_size,
        min_periods,
        center,
    )
    return data_frame[value_column].rolling(
        window=window_size,
        min_periods=min_periods,
        center=center,
    ).mean()


def compute_rolling_averages(
    data_frame: pd.DataFrame,
    value_column: str,
    window_sizes: Sequence[int],
    *,
    min_periods: int = 1,
    center: bool = False,
) -> Dict[int, pd.Series]:
    """Compute rolling averages for multiple window sizes in one pass.

    Parameters
    ----------
    data_frame:
        DataFrame containing the target column.
    value_column:
        Column on which to compute each rolling mean.
    window_sizes:
        Iterable of rolling window lengths :math:`n_i` (positive integers).
    min_periods:
        Minimum non-null observations required to produce a value.
    center:
        Whether to label each rolling window at its midpoint.

    Returns
    -------
    dict[int, pandas.Series]
        Mapping of window size to rolling mean Series.

    Raises
    ------
    ValueError
        If ``window_sizes`` is empty after validation.
    """

    unique_windows: List[int] = []
    for size in window_sizes:
        if size <= 0:
            raise ValueError("All window sizes must be positive integers.")
        if size not in unique_windows:
            unique_windows.append(size)

    if not unique_windows:
        raise ValueError("At least one window size must be provided for analysis.")

    LOGGER.debug(
        "Computing %d rolling means for column %s: %s",
        len(unique_windows),
        value_column,
        unique_windows,
    )

    return {
        size: compute_rolling_average(
            data_frame,
            value_column,
            size,
            min_periods=min_periods,
            center=center,
        )
        for size in unique_windows
    }


def _resolve_window_sizes(
    default_window: int,
    additional_windows: Optional[Iterable[int]],
    skip_default: bool,
) -> List[int]:
    """Normalize window selections into a validated, de-duplicated list."""

    normalized: List[int] = []
    if additional_windows:
        for candidate in additional_windows:
            if candidate <= 0:
                raise ValueError("All window sizes must be positive integers.")
            if candidate not in normalized:
                normalized.append(candidate)
    if not skip_default:
        if default_window <= 0:
            raise ValueError("Default window must be a positive integer.")
        if default_window not in normalized:
            normalized.append(default_window)
    if not normalized:
        raise ValueError(
            "No rolling window sizes provided. Supply a default, additional list, "
            "or disable skipping the default window."
        )
    return normalized


def run_analysis(parameters: AnalysisParameters) -> Path:
    """Execute the full analysis pipeline and persist the resulting figure."""

    LOGGER.debug("Running analysis %s with parameters: %s", parameters.label, parameters)

    data_frame = load_results(parameters.input_path)
    filtered = filter_results(data_frame, model=parameters.model, mode=parameters.mode)
    filtered = filtered.sort_values(by=parameters.x_axis).reset_index(drop=True)

    window_sizes = _resolve_window_sizes(
        parameters.window,
        parameters.window_list,
        parameters.skip_default_window,
    )

    rolling_means = compute_rolling_averages(
        filtered,
        parameters.column,
        window_sizes,
        min_periods=parameters.min_periods,
        center=parameters.center,
    )

    LOGGER.info(
        "Computed rolling means for %s using windows %s (min_periods=%d, center=%s)",
        parameters.label,
        sorted(rolling_means),
        parameters.min_periods,
        parameters.center,
    )

    title = (
        f"{parameters.model} ({parameters.mode}) — {parameters.column} rolling mean "
        f"(n={', '.join(str(w) for w in sorted(rolling_means))})"
    )
    plot_results(
        data_frame=filtered,
        rolling_means=rolling_means,
        x_axis=parameters.x_axis,
        value_column=parameters.column,
        output_path=parameters.output,
        title=title,
        figsize=(parameters.fig_width, parameters.fig_height),
    )

    LOGGER.info(
        "Summary statistics for %s (%s): min=%.4g median=%.4g max=%.4g",
        parameters.column,
        parameters.label,
        filtered[parameters.column].min(),
        filtered[parameters.column].median(),
        filtered[parameters.column].max(),
    )

    return parameters.output


def plot_results(
    data_frame: pd.DataFrame,
    rolling_means: Dict[int, pd.Series],
    x_axis: str,
    value_column: str,
    output_path: Path,
    title: str,
    *,
    figsize: Sequence[float] = (8.0, 5.0),
) -> None:
    """Generate a publication-ready plot comparing raw data to rolling mean.

    Parameters
    ----------
    data_frame:
        DataFrame containing at least ``x_axis`` and ``value_column`` columns.
    rolling_means:
        Mapping of window size to Series produced by :func:`compute_rolling_averages`.
    x_axis:
        Column name for the x-axis (e.g., ``lambda``).
    value_column:
        Target y-axis data column name.
    output_path:
        Destination file for the saved plot (PNG by convention).
    title:
        Plot title describing the context (model/mode/value combination).
    figsize:
        Tuple ``(width, height)`` specifying matplotlib figure dimensions in inches.
    """

    if x_axis not in data_frame.columns:
        raise KeyError(f"Column '{x_axis}' not found in DataFrame for plotting.")

    LOGGER.debug("Creating plot for %s vs %s", value_column, x_axis)
    fig, ax = plt.subplots(figsize=tuple(figsize))
    ax.plot(
        data_frame[x_axis],
        data_frame[value_column],
        label="Raw data",
        marker="o",
        linestyle="-",
        linewidth=1.2,
    )
    for idx, (window_size, series) in enumerate(sorted(rolling_means.items())):
        ax.plot(
            data_frame[x_axis],
            series,
            label=f"Rolling mean (window={window_size})",
            marker="s",
            linestyle="--" if idx == 0 else "-.",
            linewidth=1.4 + 0.2 * idx,
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
        default=Path("data/example_results.csv"),
        help=(
            "Path to the CSV file generated by run.apl "
            "(default: data/example_results.csv)."
        ),
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
        help=(
            "Primary rolling window size n for the average (default: 5). "
            "Can be combined with --window-list to plot multiple smoothings."
        ),
    )
    parser.add_argument(
        "--skip-default-window",
        action="store_true",
        help=(
            "Skip plotting the --window value when providing --window-list, "
            "useful for custom window sweeps."
        ),
    )
    parser.add_argument(
        "--window-list",
        type=int,
        nargs="+",
        help="Additional rolling window sizes n_i to overlay on the plot.",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=1,
        help=(
            "Minimum samples required before emitting a rolling average "
            "(default: 1 to retain warm-up points)."
        ),
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center the rolling window on each sample for symmetric smoothing.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=8.0,
        help="Figure width in inches for the saved matplotlib output (default: 8).",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=5.0,
        help="Figure height in inches for the saved matplotlib output (default: 5).",
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

    parameters = AnalysisParameters(
        input_path=args.input,
        model=args.model,
        mode=args.mode,
        column=args.column,
        window=args.window,
        window_list=args.window_list,
        min_periods=args.min_periods,
        center=args.center,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        x_axis=args.x_axis,
        output=args.output,
        skip_default_window=args.skip_default_window,
        verbose=args.verbose,
        label=args.output.stem,
    )

    run_analysis(parameters)

    LOGGER.info(
        "Figure saved at %s with size %.2fx%.2f inches",
        parameters.output,
        parameters.fig_width,
        parameters.fig_height,
    )


if __name__ == "__main__":
    main()
