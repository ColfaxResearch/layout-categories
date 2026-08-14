# tract

Companion library for the paper "Categorical Foundations for CuTe Layouts"
(Colfax Research). It implements the algebra of morphisms in the categories
$\text{Tuple}$ and $\text{Nest}$ — which encode flat layouts and layouts —
together with the factorization category $\text{Fact}$, the refinement
category $\text{Ref}$, and spans/cospans over them, and cross-validates the
operations against CuTe.

Project structure:
```
tract/
├── pyproject.toml
├── notebooks/              # Jupyter notebooks (01–05 concept series)
├── src/tract/
│   ├── fin.py              # FinMorphism: the category E_0 of pointed finite sets
│   ├── nested_tuple.py     # NestedTuple
│   ├── tuple_morphism.py   # TupleMorphism: the category Tuple (flat layouts)
│   ├── nest_morphism.py    # NestMorphism:  the category Nest  (layouts)
│   ├── fact_morphism.py    # FactMorphism:  the category Fact
│   ├── ref_morphism.py     # RefMorphism:   the category Ref
│   ├── spans.py            # Span/CoSpan(Tuple, Fact) and Span/CoSpan(Tuple, Ref)
│   ├── refinement.py       # mutual refinement, weak composition (pure)
│   ├── tuple_morph_tikz.py # TikZ export
│   └── backends/
│       ├── base.py         # shared layout ↔ morphism algorithms
│       ├── cute_dsl.py     # CuTe DSL backend (pip install "tract[cute]")
│       └── pycute.py       # pycute backend (no GPU; local CuTe clone)
└── tests/                  # pytest suites; cross-validation runs against
                            # every installed backend
```

Install (with the CuTe DSL backend):
```
cd tract
pip install ".[cute]"
```

The pycute backend instead needs NVIDIA's reference implementation from a
local clone: `pip install -e /path/to/CuTe`.

Run the full test suite:
```
pytest
```
Cross-validation tests are parameterized over the installed backends and
skip cleanly when a backend is absent.
