⍝ RK4 integrator. Uses global f (RHS), Mode, α, λ.
RK4 ← {
  ⍝ right arg: (x0 t0 θ dt steps)
  (x t θ dt n) ← ⍵
  { ⍝ one step
    dti ← dt × (Mode≡'TimeWarp' : G t α λ ⋄ 1)
    k1 ← f (x θ t)
    k2 ← f ((x + 0.5×dti×k1) θ (t + 0.5×dti))
    k3 ← f ((x + 0.5×dti×k2) θ (t + 0.5×dti))
    k4 ← f ((x + dti×k3)      θ (t + dti))
    x  ← x + dti×(k1 + 2×k2 + 2×k3 + k4) ÷ 6
    t  ← t + dti
    x t
  } ⍣ n ⊢ x t
}
