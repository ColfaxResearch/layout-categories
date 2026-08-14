# Layout Categories visualization

Manim scenes and reusable drawing primitives for explaining the categorical
model of CuTe layouts.

## Setup

From this directory, create the local environment and install the project:

```bash
uv sync
```

## Render a scene

```bash
uv run manim scenes/tuple_morphism_composition_curve.py TupleMorphismCurvedCompositionCollapse -pql

# Extract a flat layout from a tuple morphism
uv run manim scenes/tuple_morphism_to_flat_layout.py TupleMorphismToFlatLayout -pql

# Create and remove a gallery of tuple morphisms
uv run manim scenes/tuple_morphism_create_remove.py TupleMorphismCreateRemove -pql

# Weak composition, end to end: mutual refinement, pullback, pushforward, compose
uv run manim scenes/weak_composition_test.py WeakCompositionTest -pql

# The pullback and its mirror, the pushforward
uv run manim scenes/tuple_pullback_test.py TuplePullbackTest -pql
uv run manim scenes/tuple_pushforward_test.py TuplePushforwardTest -pql

# Draw Ref morphisms as banded trees between flat(X) and ρ(X)
uv run manim scenes/ref_morphism_create_test.py RefMorphismCreateTest -pql

# Compose Ref morphisms: dissolve the middle stack and collapse onto the graft
uv run manim scenes/ref_morphism_composition_test.py RefMorphismCompositionTest -pql

# Compose spans in Span(Tuple, Ref): pull the middle Tuple past the middle Ref
uv run manim scenes/ref_span_composition_test.py RefSpanMorphismCompositionTest -pql
```

`tuple_pullback_test.py` owns the drawing primitives every refinement animation
shares — cells, connectors, map arrows, the splitting gesture — and the
pushforward, weak composition and the benches below import them, so the family
cannot drift apart.

## Benches

Scenes that exist to develop one thing in isolation, running the same code the
finished scenes do:

| scene | what it isolates |
| --- | --- |
| `tuple_cell_split_test.py` | the gesture that splits a cell into its factors |
| `tuple_pullback_projection_test.py` | a domain mode that is projected away |
| `tuple_pullback_transposition_test.py` | a morphism that reorders modes |
| `decorative_variations_test.py` | mode colour, and a paper shadow per cell |
| `graphite_line_test.py` | drawing connectors in graphite rather than ink |

Rendered assets are written to `media/`, which is intentionally untracked.
Use `-ql` while iterating and `-qh` for final renders.

## Layout

- `src/layout_categories_viz/` — shared palette, typography, and future
  primitives for tuples, morphisms, layouts, and composition.
- `scenes/` — self-contained Manim scenes.
- `assets/` — checked-in source assets only.
- `media/` — generated video, image, and TeX output (gitignored).
