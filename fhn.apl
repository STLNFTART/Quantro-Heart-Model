⍝ FitzHugh–Nagumo: x=[v w], θ=[a b c]
FHN_f ← {
  (x θ t) ← ⍵ ⋄ (v w) ← x ⋄ (a b c) ← θ
  dv ← v - (v*3)÷3 - w           ⍝ v - v^3/3 - w
  dw ← (v + a - b×w) ÷ c
  sR ← α × (1○(λ×t))
  dv ← dv + (Mode≡'Residual')×(sR×v)
  dw ← dw + (Mode≡'Residual')×(sR×w)
  a  ← a × (1 + (Mode≡'ParamMod')×α×TANH (λ×v))
  b  ← b × (1 + (Mode≡'ParamMod')×α×TANH (λ×w))
  dv ← (Mode≡'Control')×(dv + U (t α λ)) + (Mode≢'Control')×dv
  ⍝ Recompute with modulated params for ParamMod
  :If Mode≡'ParamMod' ⋄ dv ← v - (v*3)÷3 - w ⋄ dw ← (v + a - b×w) ÷ c ⋄ :EndIf
  (dv , dw)
}
