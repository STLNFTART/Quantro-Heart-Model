"""Batch runner for Quantro Heart Model analyses (v1.1.0)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from analyze_results import AnalysisParameters, configure_logging, run_analysis

LOGGER = logging.getLogger("quantro.batch")


@dataclass(frozen=True)
class BatchTask:
    """Container linking a human-readable name to analysis parameters."""

    name: str
    parameters: AnalysisParameters


def _resolve_path(base: Path, candidate: str) -> Path:
    """Resolve ``candidate`` relative to ``base`` when not absolute."""

    path = Path(candidate)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def load_batch_config(config_path: Path) -> List[BatchTask]:
    """Parse a JSON batch configuration into executable tasks."""

    if not config_path.exists():
        raise FileNotFoundError(f"Batch configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    tasks_data = payload.get("tasks")
    if not isinstance(tasks_data, list) or not tasks_data:
        raise ValueError("Configuration must contain a non-empty 'tasks' list.")

    tasks: List[BatchTask] = []
    for entry in tasks_data:
        if not isinstance(entry, dict):
            raise ValueError("Each task entry must be a JSON object.")

        name = entry.get("name") or entry.get("label")
        if not name:
            raise ValueError("Each task requires a 'name' for logging.")

        input_field = entry.get("input", "data/example_results.csv")
        output_field = entry.get("output", f"artifacts/{name}.svg")

        format_field = entry.get("output_format", entry.get("format"))
        resolved_format = "auto" if format_field is None else str(format_field).lower()

        parameters = AnalysisParameters(
            input_path=_resolve_path(config_path.parent, input_field),
            model=entry.get("model", "SIR"),
            mode=entry.get("mode", "Residual"),
            column=entry.get("column", "val1"),
            window=entry.get("window", 5),
            window_list=entry.get("window_list"),
            min_periods=entry.get("min_periods", 1),
            center=entry.get("center", False),
            fig_width=float(entry.get("fig_width", 8.0)),
            fig_height=float(entry.get("fig_height", 5.0)),
            x_axis=entry.get("x_axis", "lambda"),
            output=_resolve_path(config_path.parent, output_field),
            output_format=resolved_format,
            skip_default_window=entry.get("skip_default_window", False),
            verbose=entry.get("verbose", False),
            label=name,
        )

        tasks.append(BatchTask(name=name, parameters=parameters))

    return tasks


def execute_batch(tasks: Sequence[BatchTask]) -> List[Path]:
    """Run each task sequentially, returning the generated artifact paths."""

    results: List[Path] = []
    for task in tasks:
        LOGGER.info("Starting batch task: %s", task.name)
        output_path = run_analysis(task.parameters)
        results.append(output_path)
        LOGGER.info("Completed batch task: %s -> %s", task.name, output_path)
    return results


def parse_arguments(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the batch runner."""

    parser = argparse.ArgumentParser(
        description="Execute multiple Quantro Heart Model analyses from JSON config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/batch_analysis.json"),
        help="Path to the batch configuration JSON file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging the batch runner.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for executing batch analyses."""

    args = parse_arguments(argv)
    configure_logging(args.verbose)
    LOGGER.debug("Batch CLI arguments: %s", args)

    tasks = load_batch_config(args.config)
    execute_batch(tasks)


if __name__ == "__main__":
    main()
