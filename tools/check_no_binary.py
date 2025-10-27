"""Repository binary-file guard for Quantro Heart Model (v1.0.0).

This module provides utilities to detect disallowed binary files inside the
Git working tree. The script is intended to run locally or in CI before
submitting commits, helping Donte Lightfoot maintain text-only provenance per
host restrictions.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

LOGGER = logging.getLogger("quantro.binary_guard")

# Files smaller than this threshold are fully read to avoid missing binary data.
_SMALL_FILE_BYTES: int = 4096


@dataclass(frozen=True)
class ScanResult:
    """Container summarizing the binary-scan outcome for a single file."""

    path: Path
    is_binary: bool


def configure_logging(verbose: bool) -> None:
    """Set up a consistent logging format for CLI and library usage."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LOGGER.debug("Logging configured (verbose=%s)", verbose)


def _gather_tracked_files(repo_root: Path) -> List[Path]:
    """Return a list of tracked file paths relative to ``repo_root``.

    The function shells out to ``git ls-files`` to respect the repository's
    ignore rules, ensuring generated artifacts are skipped.
    """

    LOGGER.debug("Collecting tracked files from %s", repo_root)
    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        # Fallback for unit tests or environments where Git metadata is unavailable.
        LOGGER.debug("git ls-files unavailable (%s); falling back to glob search", exc)
        files = [
            path
            for path in repo_root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ]
        LOGGER.debug("Discovered %d files via glob fallback", len(files))
        return files

    files = [repo_root / line.strip() for line in completed.stdout.splitlines() if line.strip()]
    LOGGER.debug("Discovered %d tracked files", len(files))
    return files


def _is_binary_blob(file_path: Path) -> bool:
    """Heuristically detect if ``file_path`` contains binary content.

    The detector looks for the ``\0`` byte within the sampled prefix. Text
    files in this repository (APL scripts, Markdown, CSV) never contain NULL
    bytes, so this heuristic is robust and cheap.
    """

    try:
        size_bytes = file_path.stat().st_size
    except FileNotFoundError:
        LOGGER.debug("Skipping missing file: %s", file_path)
        return False

    read_size = _SMALL_FILE_BYTES if size_bytes <= _SMALL_FILE_BYTES else 1024
    with file_path.open("rb") as handle:
        sample = handle.read(read_size)
    contains_null = b"\0" in sample
    LOGGER.debug(
        "Sampled %d bytes from %s (size=%d) -> contains_null=%s",
        len(sample),
        file_path,
        size_bytes,
        contains_null,
    )
    return contains_null


def scan_repository(
    repo_root: Path,
    *,
    allow_suffixes: Sequence[str] | None = None,
    extra_paths: Iterable[Path] | None = None,
) -> List[ScanResult]:
    """Scan for binary files under ``repo_root``.

    Parameters
    ----------
    repo_root:
        Directory containing the Git repository.
    allow_suffixes:
        Optional iterable of filename suffixes (including the leading ``.``)
        that are permitted even if they trip the binary heuristic.
    extra_paths:
        Optional iterable of paths to include in the scan beyond tracked files.
    """

    allow_set = {suffix.lower() for suffix in allow_suffixes or ()}
    tracked_files = _gather_tracked_files(repo_root)
    if extra_paths:
        tracked_files.extend(repo_root / path for path in extra_paths)

    results: List[ScanResult] = []
    for file_path in tracked_files:
        suffix = file_path.suffix.lower()
        if allow_set and suffix in allow_set:
            LOGGER.debug("Skipping %s due to allowed suffix", file_path)
            continue
        is_binary = _is_binary_blob(file_path)
        results.append(ScanResult(path=file_path, is_binary=is_binary))
    return results


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the binary guard CLI."""

    parser = argparse.ArgumentParser(
        description="Detect disallowed binary files tracked by Git.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the repository root (defaults to current working directory).",
    )
    parser.add_argument(
        "--allow-suffix",
        action="append",
        default=[],
        help="Filename suffix to allow even if flagged as binary (may repeat).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point returning an exit status for shell integration."""

    args = parse_arguments(argv)
    configure_logging(args.verbose)
    repo_root = args.repo_root.resolve()

    LOGGER.info("Scanning repository at %s", repo_root)
    results = scan_repository(repo_root, allow_suffixes=args.allow_suffix)
    offenders = [result.path for result in results if result.is_binary]

    if offenders:
        LOGGER.error("Binary files detected: %s", ", ".join(str(path) for path in offenders))
        return 1

    LOGGER.info("No binary files detected among %d tracked paths.", len(results))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
