# Layout-categories animation audit

This document compares the capabilities implemented in `tract` with the
current Manim scenes in `visualization/`. It is organized as completed
animations followed by the most substantial animation gaps.

## Done

- [x] Construct and remove flat tuple morphisms, including zero maps,
  permutations, and examples of different sizes.
  ([`TupleMorphismCreateRemove`](visualization/scenes/tuple_morphism_create_remove.py))
- [x] Compose flat tuple morphisms by routing through the shared intermediate
  tuple and collapsing the two-stage paths into `g ∘ f`.
  ([`TupleMorphismCurvedCompositionCollapse`](visualization/scenes/tuple_morphism_composition_curve.py))
- [x] Animate the glyph-level transition from `f`, `g` to `g ∘ f`.
  ([`tuple_morphism_composition_curve.py`](visualization/scenes/tuple_morphism_composition_curve.py))
- [x] Coalesce consecutive flat tuple-morphism runs, including the entry
  welding, correspondence lines, and product cells.
  ([`TupleMorphismCoalesce`](visualization/scenes/tuple_morphism_coalesce.py))
- [x] Compose a flat morphism and then coalesce the resulting composite.
  ([`TupleMorphismCompositionThenCoalesceTest`](visualization/scenes/tuple_morphism_composition_then_coalesce.py))
- [x] Explain the prefix-product calculation used to read a flat layout from
  a tuple morphism.
  ([`TupleMorphismToFlatLayout`](visualization/scenes/tuple_morphism_to_flat_layout.py))
- [x] Construct nested-tuple trees, including mirrored source/target tree
  orientations and multiple nesting depths.
  ([`NestedTupleTreeTest`](visualization/scenes/nested_tuple_tree_test.py))
- [x] Construct and remove nested morphisms between nested tuple trees.
  ([`NestMorphismCreateTest`](visualization/scenes/nest_morphism_create_test.py))
- [x] Cancel matching nested trees level by level.
  ([`NestTreeCancellationTest`](visualization/scenes/nest_tree_cancellation_test.py))
- [x] Compose nested morphisms by cancelling their shared tree and collapsing
  the resulting two-stage routes into `g ∘ f`, with three examples of
  increasing complexity.
  ([`NestMorphismCancellationCompositionTest`](visualization/scenes/nest_morphism_composition_test.py))
- [x] Prototype mutual refinement for flat tuples by splitting two
  factorizations into the refinements computed by `mutual_refinement()` and
  showing the resulting divisibility relation.
  ([`TupleMutualRefinementTest`](visualization/scenes/tuple_mutual_refinement_test.py))
- [x] Animate the pullback of a flat tuple morphism along a refinement of its
  codomain, deforming "morphism then refinement" into "refinement then
  morphism", in two stages: the middle stack refines in place, giving parallel
  arrows into the refinement, and its blocks then reorder into the domain
  refinement, so the crossing moves from the morphism's arrows to the
  pullback's. `f` is arbitrary — it may permute or reverse modes, miss codomain
  modes, or send domain modes to the basepoint — and `pullback_along` supplies
  the picture. Six examples of increasing size.
  ([`TuplePullbackTest`](visualization/scenes/tuple_pullback_test.py),
  [`pullback_along`](tract/src/tract/categories.py))
- [x] Animate the pushforward of a flat tuple morphism along a refinement of
  its domain, as the exact mirror: the same two stages with the fixed
  refinement on the left, the same treatment of basepoints and unhit modes, and
  the same six examples.
  ([`TuplePushforwardTest`](visualization/scenes/tuple_pushforward_test.py),
  [`pushforward_along`](tract/src/tract/categories.py))
- [x] Animate weak composition of flat tuple morphisms end to end: `f` and `g`
  open side by side and draw apart, their middle tuples grow the mutual
  refinement between them, both halves then refine and reorder at once — the
  pullback on the left and the pushforward on the right, sharing that
  refinement, with the two families of parallel arrows meeting at it exhibiting
  the inclusion — and the two-stage route finally collapses into the weak
  composite. Coalescence is deliberately omitted. Five examples of increasing
  size.
  ([`WeakCompositionTest`](visualization/scenes/weak_composition_test.py),
  [`weak_composite`](tract/src/tract/layout_utils.py))
- [x] Isolate the refinement gesture on its own — a stack of cells splitting in
  place, with the factors dealt out from under the cell they came from — as a
  bench for tuning it, running the same code the pullback and pushforward do.
  ([`TupleCellSplitTest`](visualization/scenes/tuple_cell_split_test.py))
- [x] Isolate the two cases a general morphism adds, as benches: a domain mode
  projected away, and a mode-reordering morphism.
  ([`TuplePullbackProjectionTest`](visualization/scenes/tuple_pullback_projection_test.py),
  [`TuplePullbackTranspositionTest`](visualization/scenes/tuple_pullback_transposition_test.py))
- [x] Draw Ref morphisms as general trees between flat(X) and the depth-1
  reduction ρ(X), with a banded layout that makes unintended overlaps
  impossible: the strip between the stacks splits into one vertical band per
  nesting level, a junction at depth k sits on band boundary c_k at the
  height of its bottom-most child (the bottom strand continuing level, the
  others descending onto it, as in a Fact fan), and every strand holds its
  height until its parent's band before bending. Depth-1 modes reduce exactly to the Fact
  fans. Five examples of increasing depth and width.
  ([`RefMorphismCreateTest`](visualization/scenes/ref_morphism_create_test.py),
  [`Ref_morphism`](tract/src/tract/ref_morphism.py))
- [x] Compose Ref morphisms by dissolving the middle stack (one connector per
  vanished cell) and collapsing stroke by stroke onto the grafted composite
  tree, matched through the graft's exact edge correspondence: a first-tree
  edge keeps its address under the graft prefix, a second-tree edge keeps
  its own, and each connector fused with its leaf strand lands on the one
  edge leaving the graft junction — so no strand is ever doubled and the
  final frame is the create scene's canonical drawing of the composite.
  Unrefined first-side modes add no junction and travel as three-piece Fact
  routes. The landing shows the graft (one junction where each middle cell
  stood), since composition in Ref remembers the tower of refinements where
  Fact flattens it. Three examples: a Fact pair, a one-junction graft, and
  junctions on both sides.
  ([`RefMorphismCompositionTest`](visualization/scenes/ref_morphism_composition_test.py),
  [`Ref_morphism.compose`](tract/src/tract/ref_morphism.py))
- [x] Compose spans in Span(Tuple, Ref): the Fact span scene's five-stack
  pullback gesture (the middle splits along b, blocks reorder into X′,
  then X and Y dissolve and the chain contracts), with the backward legs
  drawn as mirrored banded Ref trees. Trees enter through two bracketing
  beats — b's trees unbraid into one parallel strand per leaf before the
  split, and b′'s strands re-braid into its banded trees before the
  collapse — and the final contraction moves stroke for stroke through the
  graft's edge correspondence, landing on the canonical drawing of the
  composite backward leg. Two examples: the Fact span example lifted, and
  junctions on both legs with a junctioned mode surviving into b′.
  ([`RefSpanMorphismCompositionTest`](visualization/scenes/ref_span_composition_test.py),
  [`RefSpan_morphism.compose`](tract/src/tract/ref_span.py))

## To do

### Highest priority

- [ ] Animate the layout-to-morphism and morphism-to-layout correspondence,
  including tractability, flattening, and the round trip between a CuTe layout
  and its canonical morphism.
  ([`is_tractable`](tract/src/tract/layout_utils.py),
  [`compute_Tuple_morphism`](tract/src/tract/layout_utils.py),
  [`compute_layout`](tract/src/tract/layout_utils.py),
  [`compute_Nest_morphism`](tract/src/tract/layout_utils.py))

### High priority

- [ ] Lift weak composition from flat tuples to nested ones: the flat animation
  refines and reorders stacks of cells, and the nested case needs the same
  steps drawn on trees.
  ([`weak_composite`](tract/src/tract/layout_utils.py),
  [`NestedTupleTree`](visualization/src/layout_categories_viz/nested_tuple.py))
- [ ] Animate complements of morphisms: identify the unused codomain modes and
  construct the complementary morphism.
  ([`Tuple_morphism.complement`](tract/src/tract/categories.py),
  [`Nest_morphism.complement`](tract/src/tract/categories.py))
- [ ] Animate logical division and logical product, including the complement
  and concatenation steps from which they are built.
  ([`flat_divide`](tract/src/tract/categories.py),
  [`flat_product`](tract/src/tract/categories.py),
  [`logical_divide`](tract/src/tract/categories.py),
  [`logical_product`](tract/src/tract/categories.py))
- [ ] Animate sums/concatenations of morphisms with disjoint images, including
  the resulting nested source structure and wedge-like map.
  ([`Tuple_morphism.sum`](tract/src/tract/categories.py),
  [`Tuple_morphism.concat`](tract/src/tract/categories.py),
  [`Nest_morphism.concat`](tract/src/tract/categories.py))

### Medium priority

- [ ] Animate standalone nested-morphism coalescence and the distinction
  between nested-tree cancellation during composition and
  `Nest_morphism.coalesce()`.
  ([`Nest_morphism.coalesce`](tract/src/tract/categories.py))
- [ ] Animate flattening only the codomain of a nested morphism.
  ([`flatten_codomain`](tract/src/tract/categories.py))
- [ ] Animate tuple sorting and the resulting reordering morphism.
  ([`sort`](tract/src/tract/categories.py))
- [ ] Animate squeezing away unit modes, then factorization through a selected
  codomain subtuple.
  ([`squeeze`](tract/src/tract/categories.py),
  [`factorize`](tract/src/tract/categories.py))
- [ ] Add a separate strong-coalescence animation, showing how it differs from
  the currently animated weak coalescence.
  ([`strong_coalesce`](tract/src/tract/categories.py))
- [ ] Animate nested-tuple refinement primitives such as profiles, relative
  modes, and relative flattening.
  ([`profile`](tract/src/tract/categories.py),
  [`refines`](tract/src/tract/categories.py),
  [`relative_mode`](tract/src/tract/categories.py),
  [`relative_flattening`](tract/src/tract/categories.py))

### Lower priority

- [ ] Animate flat and nested layout concatenation.
  ([`flat_concatenate`](tract/src/tract/layout_utils.py),
  [`concatenate`](tract/src/tract/layout_utils.py))
- [ ] Animate layout flattening, stride sorting, and nullification of trivial
  or zero strides.
  ([`flatten_layout`](tract/src/tract/layout_utils.py),
  [`sort_flat_layout`](tract/src/tract/layout_utils.py),
  [`nullify_trivial_strides`](tract/src/tract/layout_utils.py),
  [`nullify_zero_strides`](tract/src/tract/layout_utils.py))
- [ ] Animate flat-layout complementation with respect to a total size.
  ([`flat_complement`](tract/src/tract/layout_utils.py))
- [ ] Add a standalone visualization of the foundational `Fin_morphism`
  operations (`compose`, `sum`, and `wedge`) that underlie the tuple and nest
  categories.
  ([`Fin_morphism`](tract/src/tract/categories.py))
- [ ] Animate TikZ generation/export as a presentation endpoint for nested
  morphisms, layouts, and mutual refinements. Static TikZ/image examples exist,
  but there is no Manim animation for the export process.
  ([`to_tikz`](tract/src/tract/categories.py),
  [`layout_to_tikz`](tract/src/tract/layout_utils.py))

## Prototypes

Not animations of the library, but experiments in how these scenes could look.

- Two decorative variations tried on one figure: a hue per mode threaded
  through a refinement, and a paper shadow under every cell.
  ([`DecorativeVariationsTest`](visualization/scenes/decorative_variations_test.py))
- Drawing connectors in graphite rather than ink: a wavering path, pressure
  varying along the stroke, and two passes plus a bloom, drawn start to end.
  ([`GraphiteLineTest`](visualization/scenes/graphite_line_test.py))
