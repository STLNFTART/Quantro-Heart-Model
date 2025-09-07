⍝ RHS for SIR
SIR_f ← {
  (x θ t) ← ⍵ ⋄ (S I Rc) ← x ⋄ (β γ N) ← θ
  inf ← β×S×I ÷ N ⋄ rec ← γ×I
  dS ← -inf ⋄ dI ← inf - rec ⋄ dR ← rec

  ⍝ Residual overlay
  sR ← α × (1○(λ×t))
  dS ← dS + (Mode≡'Residual')×(sR×S)
  dI ← dI + (Mode≡'Residual')×(sR×I)
  dR ← dR + (Mode≡'Residual')×(sR×Rc)

  ⍝ Control overlay on I only
  dI ← dI + (Mode≡'Control')×U (t α λ)

  ⍝ ParamMod overlay: modulate β then recompute base dynamics
  βeff ← β × (1 + (Mode≡'ParamMod')×α×TANH (λ×I))
  inf2 ← βeff×S×I ÷ N
  dS ← (Mode≡'ParamMod')×(-inf2) + (Mode≢'ParamMod')×dS
  dI ← (Mode≡'ParamMod')×(inf2 - rec) + (Mode≢'ParamMod')×dI
  dR ← (Mode≡'ParamMod')×rec          + (Mode≢'ParamMod')×dR

  (dS , dI , dR)
}
