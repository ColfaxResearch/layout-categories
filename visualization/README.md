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

# Weak composition, end to end: mutual refinement, pullback, pushforward, compose
uv run manim scenes/weak_composition_test.py WeakCompositionTest -pql

# The pullback and its mirror, the pushforward
uv run manim scenes/tuple_pullback_test.py TuplePullbackTest -pql
uv run manim scenes/tuple_pushforward_test.py TuplePushforwardTest -pql

# Compose spans in Span(Tuple, Fact); single-beat variant is the same scene
# with single_beat = True
uv run manim scenes/span_composition_test.py SpanMorphismCompositionTest -pql
uv run manim scenes/span_composition_single_beat_test.py SpanMorphismCompositionSingleBeatTest -pql

# Draw and compose Ref morphisms as banded trees
uv run manim scenes/ref_morphism_create_test.py RefMorphismCreateTest -pql
uv run manim scenes/ref_morphism_composition_test.py RefMorphismCompositionTest -pql

# Compose spans in Span(Tuple, Ref)
uv run manim scenes/ref_span_composition_test.py RefSpanMorphismCompositionTest -pql
```

`ANIMATION_AUDIT.md` (repo root) lists every finished animation and the gaps.

## Layout

- `src/layout_categories_viz/` — the drawing library every scene builds on:
  - `style.py` — palette and typography
  - `scene_base.py` — `LayoutScene` (background, `clear_scene`)
  - `tuple_morphism.py`, `composition.py` — morphism stack diagrams and
    mapsto arrows; `animations.py` — the arrow draw/undraw gestures
  - `stacks.py` — cells, decks, the split gesture, column placement
  - `paths.py` — Bézier path matching and route builders
  - `trees` — `nested_tuple.py`, `nest_morphism.py`, `ref_trees.py`
  - `coalesce.py` — weld geometry and the shared coalesce choreography
- `scenes/` — one file per animation; each holds only its examples and beat
  sequencing. `span_common.py` carries the span-family shared geometry.
- `scenes/prototypes/` — look-and-feel experiments (graphite strokes,
  decorative variations), not animations of the library.
- `assets/` — checked-in source assets only.
- `media/` — generated video, image, and TeX output (gitignored).

## Benches

Scenes that exist to develop one thing in isolation, running the same code the
finished scenes do:

| scene | what it isolates |
| --- | --- |
| `tuple_cell_split_test.py` | the gesture that splits a cell into its factors |
| `tuple_pullback_projection_test.py` | a domain mode that is projected away |
| `tuple_pullback_transposition_test.py` | a morphism that reorders modes |
| `tuple_entry_coalesce_test.py` | the entry-weld gesture |

Rendered assets are written to `media/`, which is intentionally untracked.
Use `-ql` while iterating and `-qh` for final renders.
