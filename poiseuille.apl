⍝ Poiseuille: x=[Q], θ=[ΔP μ L r], relax to Qss = π r^4 ΔP / (8 μ L)
POIS_f ← {
  (x θ t) ← ⍵ ⋄ Q ← x[1] ⋄ (dP μ L r) ← θ
  Qss ← ○1 × r*4 × dP ÷ (8 × μ × L)   ⍝ π=radians circle fn ○1
  k ← 5
  dQ ← k × (Qss - Q)
  sR ← α × (1○(λ×t))
  dQ ← dQ + (Mode≡'Residual')×(sR×Q)
  r  ← r × (1 + (Mode≡'ParamMod')×α×TANH (λ×Q))
  :If Mode≡'ParamMod' ⋄ Qss ← ○1 × r*4 × dP ÷ (8 × μ × L) ⋄ dQ ← k×(Qss - Q) ⋄ :EndIf
  dQ ← (Mode≡'Control')×(dQ + U (t α λ)) + (Mode≢'Control')×dQ
  (dQ)
}
