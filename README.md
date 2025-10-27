# Quantro-Heart-Model

## Quick run (no install)
Use an online APL REPL:
- Load `overlays.apl`, `integrators.apl`, `mm.apl`, `sir.apl`, `fhn.apl`, `nernst.apl`, `poiseuille.apl`
- Then paste `run.apl` and execute.

## Local run
- Dyalog APL: install, then `dyalog` → `)LOAD run.apl`
- GNU APL (built from source): `apl -s -f run.apl`

## Files
- overlays.apl — Residual/ParamMod/Control/TimeWarp
- integrators.apl — RK4 with per-step TimeWarp
- mm.apl / sir.apl / fhn.apl / nernst.apl / poiseuille.apl — model RHS
- run.apl — driver that prints CSV

## Overview
Quantro-Heart-Model is an open-source project developed by STLNFTART. The toolkit explores modular reaction and transport models relevant to cardiac research with configurable overlays for experimentation.

## Features
- Time-domain integration of multiple electrophysiology and hemodynamics models (Michaelis-Menten, SIR, FitzHugh-Nagumo, Nernst potential, Poiseuille flow).
- Overlay mechanisms supporting residual correction, parameter modulation, control inputs, and time-warp experimentation.
- Python analysis companion tooling for CSV post-processing with statistical summaries and publication-ready plots.

## Installation

```bash
# Clone the repository
git clone https://github.com/STLNFTART/Quantro-Heart-Model.git
cd Quantro-Heart-Model
```

## Usage

### APL simulation driver

```bash
# Run with GNU APL (example)
apl -s -f run.apl > simulation.log
```

### Python analysis workflow (v1.2.1)

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install pinned analysis/testing dependencies for reproducible results
pip install -r requirements-dev.txt

# Generate a rolling-average plot of SIR val1 across lambda values
python3 analyze_results.py \
  --input data/example_results.csv \
  --model SIR \
  --mode Residual \
  --column val1 \
  --window 5 \
  --window-list 9 15 \
  --min-periods 3 \
  --center \
  --fig-width 10 \
  --fig-height 6 \
  --output artifacts/sir_val1_residual_multi.png

# Execute the curated batch to build both SIR residual plots in one command
python3 batch_analysis.py --config configs/batch_analysis.json

# Run the built-in unit tests for verification
python3 -m unittest discover -s tests -v
```

The CLI stores plots under `artifacts/` by default to maintain traceability.
Binary PNG outputs are **not** committed to the repository so the branch stays
compatible with hosts that reject large binaries; regenerate them locally using
the commands above whenever visual validation is required.

> **Note:** The tracked `data/example_results.csv` file is a lightweight sample
> to keep the repository text-only. Replace it with your own simulation output
> (e.g., `run.apl > results.csv`) and point the CLI to that path via `--input`.

### Plot interpretation and benchmarking

- Inspect the generated PNG to compare raw data against the rolling mean.
- Adjust `--window` or supply `--window-list` to benchmark smoothing sensitivity across multiple horizons.
- Combine `--window-list` with `--skip-default-window` to isolate custom sweeps without the default span.
- Tune `--min-periods` and `--center` for retrospective vs. streaming analyses.
- Control figure layout with `--fig-width`/`--fig-height` to match publication requirements.
- Use `--verbose` to collect diagnostic logs with summary statistics.

### Binary-file compliance guard (v1.0.0)

```bash
# Verify the current branch does not contain disallowed binary blobs
python3 tools/check_no_binary.py --verbose
```

- The guard scans tracked files for NULL-byte signatures to enforce the
  hosting provider's text-only requirement.
- Integrate this command into pre-commit hooks or CI to block future commits
  from introducing `.png`, `.gsbak`, or similar binary artifacts.

## Contributing
Contributions are welcome! Please open issues or submit pull requests.

## License
This project is licensed under the MIT License.

## Contact
For questions, contact [STLNFTART](https://github.com/STLNFTART).
