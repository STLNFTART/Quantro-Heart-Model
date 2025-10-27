"""Repository binary-file guard for Quantro Heart Model (v1.3.0).

This module provides utilities to detect disallowed binary files inside the
Git working tree. The script is intended to run locally or in CI before
submitting commits, helping Donte Lightfoot maintain text-only provenance per
host restrictions.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from codecs import getincrementaldecoder
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

LOGGER = logging.getLogger("quantro.binary_guard")

# Chunk size used when scanning large files. Incremental decoding keeps memory
# usage bounded while still inspecting the entire payload for binary markers.
_SCAN_CHUNK_BYTES: int = 4096


@dataclass(frozen=True)
class ScanResult:
    """Container summarizing the content-scan outcome for a single file."""

    path: Path
    is_binary: bool
    ascii_violation: bool = False


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

    The detector performs two lightweight checks:

    1. Look for ``\0`` bytes in the sampled prefix. Presence of a NULL byte is
       a strong indicator that the file is binary.
    2. Attempt to decode the sampled prefix as UTF-8. The repository relies on
       UTF-8 encoded sources (including APL glyphs), so any decoding failure is
       treated as a binary indicator. This guards against high-ASCII blobs that
       may not include NULLs but still violate hosting restrictions.
    """

    try:
        size_bytes = file_path.stat().st_size
    except FileNotFoundError:
        LOGGER.debug("Skipping missing file: %s", file_path)
        return False

    decoder = getincrementaldecoder("utf-8")()
    with file_path.open("rb") as handle:
        total_read = 0
        while True:
            chunk = handle.read(_SCAN_CHUNK_BYTES)
            if not chunk:
                break
            total_read += len(chunk)
            if b"\0" in chunk:
                LOGGER.debug(
                    "Chunk inspection detected NULL byte in %s at offset %d", file_path, total_read
                )
                return True
            try:
                decoder.decode(chunk)
            except UnicodeDecodeError as exc:
                LOGGER.debug(
                    "UTF-8 decode failure for %s at offset %d: %s",
                    file_path,
                    total_read,
                    exc,
                )
                return True

    LOGGER.debug(
        "Scanned %d bytes from %s (size=%d) -> utf8_valid=True",
        total_read,
        file_path,
        size_bytes,
    )
    return False


def _violates_ascii_policy(file_path: Path) -> bool:
    """Return ``True`` when ``file_path`` contains non-ASCII characters.

    The helper streams each file through Python's incremental ASCII decoder so
    large artifacts remain memory-efficient. Repositories that must retain the
    legacy APL glyph sources can opt out by passing the suffix via
    ``allow_suffixes``.
    """

    decoder = getincrementaldecoder("ascii")()
    try:
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(_SCAN_CHUNK_BYTES)
                if not chunk:
                    break
                try:
                    decoder.decode(chunk)
                except UnicodeDecodeError as exc:
                    LOGGER.debug(
                        "ASCII decode failure for %s at byte offset %d: %s",
                        file_path,
                        exc.start,
                        exc,
                    )
                    return True
    except FileNotFoundError:
        LOGGER.debug("Skipping missing file during ASCII scan: %s", file_path)
        return False
    return False


def scan_repository(
    repo_root: Path,
    *,
    allow_suffixes: Sequence[str] | None = None,
    extra_paths: Iterable[Path] | None = None,
    ascii_only: bool = False,
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
        ascii_violation = False
        if ascii_only and not is_binary:
            ascii_violation = _violates_ascii_policy(file_path)
        results.append(
            ScanResult(path=file_path, is_binary=is_binary, ascii_violation=ascii_violation)
        )
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
        "--ascii-only",
        action="store_true",
        help=(
            "Enable strict ASCII validation for tracked files. Use --allow-suffix to "
            "exclude known Unicode-heavy sources such as .apl models."
        ),
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
    results = scan_repository(
        repo_root,
        allow_suffixes=args.allow_suffix,
        ascii_only=args.ascii_only,
    )
    binary_offenders = [result.path for result in results if result.is_binary]
    ascii_offenders = [result.path for result in results if result.ascii_violation]

    if binary_offenders or ascii_offenders:
        if binary_offenders:
            LOGGER.error(
                "Binary files detected: %s",
                ", ".join(str(path) for path in binary_offenders),
            )
        if ascii_offenders:
            LOGGER.error(
                "Non-ASCII files detected in strict mode: %s",
                ", ".join(str(path) for path in ascii_offenders),
            )
        return 1

    mode_suffix = " (ASCII-only mode)" if args.ascii_only else ""
    LOGGER.info(
        "No binary files detected among %d tracked paths%s.",
        len(results),
        mode_suffix,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
