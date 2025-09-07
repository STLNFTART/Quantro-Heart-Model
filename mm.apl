⍝ RHS for Michaelis–Menten
MM_f ← {
  ⍝ args packed: (x θ t)
  (x θ t) ← ⍵ ⋄ (S P) ← x ⋄ (Vmax Km) ← θ
  Veff ← Vmax × (1 + (Mode≡'ParamMod')×α×TANH (λ×S))
  v ← Veff×S ÷ Km+S
  v ← v + (Mode≡'Control')  × U (t α λ)
  v ← v + (Mode≡'Residual') × (α × (1○(λ×t)) × S)
  (-v , v)
}
