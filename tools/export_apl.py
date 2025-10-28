"""Export Unicode APL sources from ASCII-safe templates (v1.0.0).

This utility regenerates the original Dyalog APL scripts from the
ASCII-only templates stored under ``apl_sources``. Hosting providers
used for Donte Lightfoot's Quantro Heart Model prohibit direct storage
of Unicode-heavy APL glyphs, so we encode the sources using ``\\uXXXX``
escapes. Running this exporter reconstructs the authentic APL files for
local simulation work while keeping the repository compliant with the
text-only restriction.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger("quantro.export_apl")
DEFAULT_SOURCE_DIR = Path("apl_sources")
DEFAULT_OUTPUT_DIR = Path("generated_apl")


def configure_logging(verbose: bool) -> None:
    """Configure the root logger for deterministic CLI output."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LOGGER.debug("Logging initialised (verbose=%s)", verbose)


def decode_apl_template(encoded_text: str) -> str:
    """Decode ``\\uXXXX`` escape sequences into Unicode glyphs.

    Parameters
    ----------
    encoded_text:
        Raw text containing escape sequences emitted by
        :func:`str.encode` with the ``'unicode_escape'`` codec.

    Returns
    -------
    str
        Decoded Unicode string ready to write to disk.
    """

    return encoded_text.encode("utf-8").decode("unicode_escape")


def export_templates(
    source_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> list[Path]:
    """Export every ``*.apl.txt`` template into ``output_dir``.

    Parameters
    ----------
    source_dir:
        Directory containing ASCII-only templates.
    output_dir:
        Destination folder for reconstructed APL files.
    overwrite:
        When ``True`` existing files are replaced. Otherwise the exporter
        refuses to clobber user modifications to ensure reproducibility.
    """

    if not source_dir.exists():
        raise FileNotFoundError(
            f"APL template directory not found: {source_dir.resolve()}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []

    for template in sorted(source_dir.glob("*.apl.txt")):
        encoded = template.read_text(encoding="ascii")
        decoded = decode_apl_template(encoded)
        output_path = output_dir / template.name.replace(".apl.txt", ".apl")

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {output_path}. "
                "Pass --overwrite to replace it."
            )

        output_path.write_text(decoded, encoding="utf-8")
        exported.append(output_path)
        LOGGER.info("Exported %s -> %s", template.name, output_path)

    if not exported:
        LOGGER.warning("No APL templates found in %s", source_dir)

    return exported


def parse_arguments(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the exporter."""

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate Unicode APL scripts from ASCII-safe templates for "
            "the Quantro Heart Model."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing *.apl.txt templates (default: apl_sources).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for Unicode APL files (default: generated_apl).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting of existing APL files in the output directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for troubleshooting.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for exporting APL sources."""

    args = parse_arguments(argv)
    configure_logging(args.verbose)
    LOGGER.debug("Exporter arguments: %s", args)

    exported = export_templates(
        args.source_dir,
        args.output_dir,
        overwrite=args.overwrite,
    )

    LOGGER.info("Exported %d APL files to %s", len(exported), args.output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
