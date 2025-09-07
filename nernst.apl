⍝ Nernst: x=[E], θ=[T z Co Ci], relax to EN
NERNST_f ← {
  (x θ t) ← ⍵ ⋄ E ← x[1] ⋄ (T z Co Ci) ← θ
  Rgas ← 8.314462618 ⋄ F ← 96485.33212
  EN ← (Rgas×T) ÷ (z×F) × ⍟(Co÷Ci)
  k ← 10
  dE ← k × (EN - E)
  sR ← α × (1○(λ×t))
  dE ← dE + (Mode≡'Residual')×(sR×E)
  T  ← T × (1 + (Mode≡'ParamMod')×α×TANH (λ×E))
  :If Mode≡'ParamMod' ⋄ EN ← (Rgas×T) ÷ (z×F) × ⍟(Co÷Ci) ⋄ dE ← k×(EN - E) ⋄ :EndIf
  dE ← (Mode≡'Control')×(dE + U (t α λ)) + (Mode≢'Control')×dE
  (dE)
}
