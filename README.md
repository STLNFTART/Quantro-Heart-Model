
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
