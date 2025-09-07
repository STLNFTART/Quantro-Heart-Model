)LOAD overlays.apl
)LOAD integrators.apl
)LOAD mm.apl
)LOAD sir.apl

⍝ Globals used by RK4 and RHS
Mode ← 'Residual'  ⋄ α ← 0.1 ⋄ λ ← 1.0

⍝ CSV header
'model,mode,alpha,lambda,val1,val2,val3' ⎕←

⍝ === MM ===
f ← MM_f
x0 ← 1 0               ⍝ S P
θ  ← 1 0.5             ⍝ Vmax Km
dt ← 1e¯3 ⋄ steps ← 10000 ⋄ t0 ← 0
{
  Mode α λ ← ⍵
  (xT tT) ← RK4 (x0 t0 θ dt steps)
  'MM,',Mode,',',⍕α,',',⍕λ,',',⍕xT[1],',',⍕xT[2],',',⍕0 ⎕←
}¨ ('Residual' 'ParamMod' 'Control' 'TimeWarp') ∘., ⍳1 ⋄ ⍝ placeholders

⍝ === SIR ===
f ← SIR_f
N ← 1000 ⋄ x0 ← (N-1) 1 0  ⍝ S I R
θ ← 0.3 0.1 N
dt ← 1e¯2 ⋄ steps ← 10000 ⋄ t0 ← 0
{
  Mode α λ ← ⍵
  (xT tT) ← RK4 (x0 t0 θ dt steps)
  'SIR,',Mode,',',⍕α,',',⍕λ,',',⍕xT[1],',',⍕xT[2],',',⍕xT[3] ⎕←
}¨ ('Residual' 'ParamMod' 'Control' 'TimeWarp') ∘., 0.1 1.0

⍝ Done
'OK' ⎕←
