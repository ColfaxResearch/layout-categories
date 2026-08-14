# layout-categories

Companion software for the paper "Categorical Foundations for CuTe Layouts",
by Colfax Research. The paper and accompanying blog post are
[on our website](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/).

In the paper, we develop a robust algebra of morphisms in the categories
$\text{Tuple}$ and $\text{Nest}$, which encode flat layouts and layouts,
respectively. This repo implements that algebra — together with the
factorization category $\text{Fact}$, the refinement category $\text{Ref}$,
and spans/cospans over them — and demonstrates empirically that the
operations align with their counterparts in CuTe.

Contents:

- [`tract/`](tract/README.md) — the library: categories, morphism operations,
  layout ↔ morphism bridges (CuTe DSL and pycute backends), TikZ export, and
  the cross-validation test suites. See its README for install and test
  instructions.
- [`visualization/`](visualization/README.md) — Manim scenes animating the
  constructions. [`ANIMATION_AUDIT.md`](ANIMATION_AUDIT.md) tracks which
  library capabilities are animated.
- [`papers/`](papers/) — the paper PDF.
- [`COMPOSABILITY_NOTES.md`](COMPOSABILITY_NOTES.md) — notes on CuTe
  composability vs. tractability, with verified counterexamples.
- [`REDESIGN.md`](REDESIGN.md) — audit and phased refactor plan for the repo.
