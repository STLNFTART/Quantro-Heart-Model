# Quantro-Heart-Model

## Quick run (no install)
Use an online APL REPL:
1. Regenerate the Unicode sources from their ASCII-safe templates:
   `python3 tools/export_apl.py --output-dir generated_apl`
2. Load the exported `generated_apl/overlays.apl`, `generated_apl/integrators.apl`,
   `generated_apl/mm.apl`, `generated_apl/sir.apl`, `generated_apl/fhn.apl`,
   `generated_apl/nernst.apl`, `generated_apl/poiseuille.apl`.
3. Then paste `generated_apl/run.apl` and execute.

## Local run
- Dyalog APL: install, regenerate the sources with
  `python3 tools/export_apl.py --output-dir generated_apl`, then
  `dyalog` -> `)LOAD generated_apl/run`
- GNU APL (built from source): `python3 tools/export_apl.py --output-dir generated_apl`
  followed by `apl -s -f generated_apl/run.apl`

## Files
- `apl_sources/*.apl.txt` - ASCII-safe templates of the original APL glyph
  sources.
- `generated_apl/*.apl` (created locally) - Residual/ParamMod/Control/TimeWarp,
  integrators, and model RHS scripts reconstructed from the templates.
- `tools/export_apl.py` - CLI to rebuild Unicode `.apl` files prior to running
  simulations.

## Overview
Quantro-Heart-Model is an open-source project developed by STLNFTART. The toolkit explores modular reaction and transport models relevant to cardiac research with configurable overlays for experimentation.

## Why APL for Computational Physiology
APL was created as a formal mathematical notation; its array semantics let us express
axiomatic physiological models directly as executable mathematics.

### Axiomatic Rigor
Each subsystem (e.g., Nernst potential, FitzHugh-Nagumo, Poiseuille flow)
is written in the same symbolic form as its governing equations.
This minimizes the semantic distance between the model and its code.

### Vectorized Computation
APL's intrinsic array operations let the framework simulate thousands of
parameter sets or patient states simultaneously, turning ensemble
simulations into pure matrix algebra.

### Quantro Architecture
| Layer | Purpose |
|-------|----------|
| Residual | Error minimization, data-model alignment |
| ParamMod | Parameter modulation across population ensembles |
| Control  | Closed-loop physiological control |
| TimeWarp | Temporal scaling and synchronization of multi-organ models |

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
python3 tools/export_apl.py --output-dir generated_apl
apl -s -f generated_apl/run.apl > simulation.log
```

### Python analysis workflow (v1.3.0)

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install pinned analysis/testing dependencies for reproducible results
pip install -r requirements-dev.txt

# Generate a rolling-average plot of SIR val1 across lambda values (SVG output)
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
  --output artifacts/sir_val1_residual_multi.svg \
  --output-format auto

# Execute the curated batch to build both SIR residual plots in one command
python3 batch_analysis.py --config configs/batch_analysis.json

# Run the built-in unit tests for verification
python3 -m unittest discover -s tests -v
```

The CLI stores plots under `artifacts/` by default to maintain traceability.
Vector SVG outputs are text-only, so they can be safely versioned when
required. Raster formats are intentionally unsupported to prevent binary
artifacts from blocking commits.

> **Note:** The tracked `data/example_results.csv` file is a lightweight sample
> to keep the repository text-only. Replace it with your own simulation output
> (e.g., `generated_apl/run.apl > results.csv`) and point the CLI to that path via `--input`.

### Plot interpretation and benchmarking

- Inspect the generated SVG to compare raw data against the rolling mean.
- Adjust `--window` or supply `--window-list` to benchmark smoothing sensitivity across multiple horizons.
- Combine `--window-list` with `--skip-default-window` to isolate custom sweeps without the default span.
- Tune `--min-periods` and `--center` for retrospective vs. streaming analyses.
- Control figure layout with `--fig-width`/`--fig-height` to match publication requirements.
- Use `--verbose` to collect diagnostic logs with summary statistics.

### Vectorized SIR parameter sweeps (v1.0.0)

```bash
# Execute the vectorized sweep using the documented configuration
python3 vectorized_sweep.py --config configs/vectorized_sir.json --verbose

# Override specific parameters at the CLI (e.g., tighten the rolling window)
python3 vectorized_sweep.py \
  --config configs/vectorized_sir.json \
  --rolling-window 7 \
  --output-csv artifacts/vectorized_sir_results_window7.csv \
  --output-plot artifacts/vectorized_sir_infectious_window7.svg
```

- The sweep simulates every combination of `beta`, `gamma`, and contact
  multipliers in a single vectorized Euler loop.
- Outputs include a CSV containing all trajectories and an SVG overlay plot of
  infectious counts alongside their rolling averages.
- Tune `--time-step` and `--duration` to explore coarser or finer temporal
  resolutions for physiology studies.

### Regenerating APL sources (v1.0.0)

```bash
# Rebuild Unicode APL scripts from ASCII-safe templates
python3 tools/export_apl.py --output-dir generated_apl

# Overwrite existing exports when refreshing from git mainline
python3 tools/export_apl.py --output-dir generated_apl --overwrite
```

- Exported files live in `generated_apl/` and are ignored by git so pushes stay
  compliant with ASCII-only hosting rules.
- Keep regenerated files under version control only if your upstream allows
  Unicode APL glyphs.

### Binary-file compliance guard (v1.3.0)

```bash
# Verify the current branch does not contain disallowed binary blobs across full-file scans
python3 tools/check_no_binary.py --verbose

# Enable strict ASCII validation across the entire tree
python3 tools/check_no_binary.py --verbose --ascii-only
```

- The guard scans tracked files for NULL-byte signatures *and* enforces
  UTF-8 decoding so high-ASCII blobs are detected before they trigger the
  hosting provider's "binary not supported" error.
- Strict ASCII mode helps diagnose hosts that disallow extended Unicode.
- Integrate this command into pre-commit hooks or CI to block future commits
  from introducing `.png`, `.gsbak`, or similar binary artifacts.

## Contributing
Contributions are welcome! Please open issues or submit pull requests.

## License
This project is licensed under the MIT License.

## Contact
For questions, contact [STLNFTART](https://github.com/STLNFTART).
