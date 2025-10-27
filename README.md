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

### Python analysis workflow (v1.1.0)

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install pinned analysis/testing dependencies for reproducible results
pip install -r requirements-dev.txt

# Generate rolling-average plots of SIR val1 across lambda values
python3 analyze_results.py \
  --input results.csv \
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

# Run the built-in unit tests for verification
python3 -m unittest discover -s tests -v
```

The CLI stores plots under `artifacts/` by default to maintain traceability.

### Plot interpretation and benchmarking

- Inspect the generated PNG to compare raw data against the rolling mean.
- Adjust `--window` or supply `--window-list` to benchmark smoothing sensitivity across multiple horizons.
- Combine `--window-list` with `--skip-default-window` to isolate custom sweeps without the default span.
- Tune `--min-periods` and `--center` for retrospective vs. streaming analyses.
- Control figure layout with `--fig-width`/`--fig-height` to match publication requirements.
- Use `--verbose` to collect diagnostic logs with summary statistics.

## Contributing
Contributions are welcome! Please open issues or submit pull requests.

## License
This project is licensed under the MIT License.

## Contact
For questions, contact [STLNFTART](https://github.com/STLNFTART).
