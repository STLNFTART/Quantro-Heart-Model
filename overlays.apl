⍝ Overlays
TANH ← { ⍝ tanh(x) = (e^x - e^-x)/(e^x + e^-x)
  y←⍵ ⋄ ( *y - *-y ) ÷ ( *y + *-y )
}
R ← { ⍝ residual term: α⋅sin(λt)⋅x
  ⍝ args: x t α λ
  x t α λ ← ⍵ ⋄ α × (1○(λ×t)) × x
}
M ← { ⍝ param factor: 1 + α⋅tanh(λx)
  x t α λ ← ⍵ ⋄ 1 + α × TANH (λ×x)
}
U ← { ⍝ control input: α⋅cos(λt)
  t α λ ← ⍵ ⋄ α × (2○(λ×t))
}
G ← { ⍝ time-warp factor ≥ 0: 1 ⌈ 1 + α⋅sin(λt)
  t α λ ← ⍵ ⋄ 1 ⌈ 1 + α × (1○(λ×t))
}
