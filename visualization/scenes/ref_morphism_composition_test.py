"""Prototype animation for composition in the category Ref.

A Ref morphism flat(X) -> rho(X) is drawn as one banded tree per top-level
mode (see ref_morphism_create_test.py); composition grafts the first
morphism's top-level modes into the second's leaves, so the animation
mirrors the Fact composition collapse:

    open:      [S] --a--> [T] --b--> [U]
    dissolve:  the middle stack shrinks away; one connector per vanished
               cell joins a's tree, across the gap, onto b's leaf strand
    collapse:  the outer stacks contract inward and every stroke deforms
               into its edge of the grafted composite tree

The collapse moves strokes edge by edge, matched through the graft's exact
edge correspondence: an edge of a's mode-j tree at tree address α lands at
address β_j + α of the composite (β_j the address of b's j-th leaf), an
internal edge of b's tree keeps its address, and the connector fused with
b's leaf strand lands on the one edge leaving the graft junction that
stands where the middle cell was.  An unrefined mode of a (a bare integer)
adds no junction when grafted, so there its whole three-piece route -- its
single strand, the connector, and b's leaf strand -- lands on the one
composite edge, exactly a Fact route.  Every on-screen stroke participates
in exactly one deformation, so no strand is ever doubled; the landing
geometry is the create scene's banded drawing of the composite between the
contracted stacks, and the final frame swaps in that canonical drawing
stroke for stroke.

Every source leaf survives (Ref morphisms have no basepoints), so as in
the Fact case nothing retracts: strokes are only ever joined and
contracted.  For depth-1 inputs the opening and dissolve reduce to the
Fact composition animation exactly; the landing differs on purpose,
because composition in Ref remembers the tower of refinements -- the
composite is the grafted tree (one junction where each middle cell stood),
where Fact would flatten the blocks into fans.  The library supplies every
stack and tree: RefMorphism.compose gives the grafted nest the collapse
lands on.  The second morphism's top-level modes are kept as tuples (as
every Fact-style presentation writes them), so each middle cell's graft
junction is present in the composite.
"""

from layout_categories_viz.scene_base import LayoutScene

from dataclasses import dataclass

import numpy as np
from manim import (
    Create,
    CubicBezier,
    FadeIn,
    LaggedStart,
    LEFT,
    RIGHT,
    ShrinkToCenter,
    Text,
    Transform,
    VGroup,
    VMobject,
    Write,
)
from tract import NestedTuple, RefMorphism

from layout_categories_viz.style import CODE_FONT, INK

# The composition scene owns the route-surgery helpers, the pullback scene
# the drawing primitives, and the create scene the banded trees, so this
# scene cannot drift from any of them.
from layout_categories_viz.ref_trees import ref_tree_with_paths
from layout_categories_viz.paths import (
    append_cubic_segments,
    matched_path_pair,
)
from layout_categories_viz import stacks
from layout_categories_viz.stacks import (
    CELL_H,
    LEFT_X,
    MID_X,
    RIGHT_X,
    SLOT_STEP,
    STROKE_WIDTH,
)

LABEL_BUFF = 0.55


def _joined_route(pieces) -> VMobject:
    """Join consecutive strand sections into one identical VMobject path.

    The generalization of three_segment_route to any number of pieces:
    consecutive pieces meet exactly (at junction anchors, or across an
    exact connector), so appending their cubics reproduces the strokes.
    """
    route = VMobject(stroke_color=INK, stroke_width=STROKE_WIDTH)
    route.start_new_path(pieces[0].points[0])
    for piece in pieces:
        append_cubic_segments(route, piece.points)
    return route


def _tree_forest(morphism: RefMorphism, source_cells, target_cells):
    """All of a morphism's trees, with per-mode edge and leaf-address data.

    Leaves anchor on the right edges of ``source_cells`` and each mode's
    root on the left edge of its cell in ``target_cells``; all modes share
    the band boundaries of the deepest.  Returns (segments, edges,
    leaf_addresses): the strands as one VGroup, per mode the strands keyed
    by child tree address, and per mode each leaf's address in leaf order.
    """
    global_depth = max((mode.depth() for mode in morphism.modes), default=1)
    segments = VGroup()
    edges = []
    leaf_addresses = []
    start = 0
    for index, mode in enumerate(morphism.modes):
        leaf_anchors = [
            source_cells[start + offset].get_right()
            for offset in range(mode.length())
        ]
        mode_segments, _, mode_edges, mode_leaf_addresses = ref_tree_with_paths(
            mode, leaf_anchors, target_cells[index].get_left(), global_depth
        )
        segments.add(*mode_segments)
        edges.append(mode_edges)
        leaf_addresses.append(mode_leaf_addresses)
        start += mode.length()
    return segments, edges, leaf_addresses


@dataclass(frozen=True)
class RefCompositionExample:
    """Two composable Ref morphisms, specified by their nested tuples."""

    first_nest: tuple
    second_nest: tuple

    def morphisms(self) -> tuple:
        """The morphisms a and b, validated by the library."""
        f = RefMorphism(NestedTuple(self.first_nest))
        g = RefMorphism(NestedTuple(self.second_nest))
        return f, g


EXAMPLES = (
    # Depth-1 modes on both sides: exactly the Fact composition scene's
    # first example, so the opening and dissolve must reproduce the Fact
    # animation and the landing shows the graft of its blocks.
    RefCompositionExample(
        first_nest=((2, 3), (2, 2), (5,)),
        second_nest=((6, 4), (5,)),
    ),
    # A junction on the first side grafted under a flat second side.
    RefCompositionExample(
        first_nest=(((2, 3), 4), (5,)),
        second_nest=((24, 5),),
    ),
    # Junctions on both sides: unrefined entries of a (bare integers, whose
    # routes stay three-piece Fact routes), a deep chain, and a second
    # morphism that regroups them.
    RefCompositionExample(
        first_nest=(2, (3, (5, 7)), 11),
        second_nest=((2, 105), (11,)),
    ),
)


class RefMorphismCompositionTest(LayoutScene):
    def construct(self):
        for index, example in enumerate(EXAMPLES):
            self._play_example(example)
            self.clear_scene(last=index == len(EXAMPLES) - 1)

    def _play_example(self, example: RefCompositionExample) -> None:
        f, g = example.morphisms()
        # Composing validates the pair; its grafted nest is the composite
        # tree the collapse lands on.
        composite = f.compose(g)

        # Which mode of b each middle cell is a leaf of, and its address
        # there: middle cell j is leaf j of b in global leaf order.
        mode_of_middle = tuple(
            k
            for k, mode in enumerate(g.modes)
            for _ in range(mode.length())
        )
        offset_of_middle = tuple(
            offset
            for mode in g.modes
            for offset in range(mode.length())
        )

        tallest = len(f.domain)
        baseline = -(tallest - 1) * SLOT_STEP / 2 + 0.25

        def column(values, x):
            return VGroup(
                *(
                    stacks.cell(
                        value, np.array([x, baseline + i * SLOT_STEP, 0.0])
                    )
                    for i, value in enumerate(values)
                )
            )

        source_cells = column(f.domain, LEFT_X)
        middle_cells = column(f.codomain, MID_X)
        target_cells = column(g.codomain, RIGHT_X)

        first_segments, first_edges, _ = _tree_forest(
            f, source_cells, middle_cells
        )
        second_segments, second_edges, second_leaf_addresses = _tree_forest(
            g, middle_cells, target_cells
        )

        # Morphism labels, and a normally typeset "b ∘ a" harvested for the
        # exact glyph destinations they slide into.
        label_y = baseline - CELL_H / 2 - LABEL_BUFF
        first_label = Text("a", color=INK, font=CODE_FONT, font_size=30)
        first_label.move_to(np.array([(LEFT_X + MID_X) / 2, label_y, 0.0]))
        second_label = Text("b", color=INK, font=CODE_FONT, font_size=30)
        second_label.move_to(np.array([(MID_X + RIGHT_X) / 2, label_y, 0.0]))
        composition_reference = Text(
            "b ∘ a", color=INK, font=CODE_FONT, font_size=30
        )
        composition_reference.move_to(np.array([MID_X, label_y, 0.0]))
        b_glyph, _, composition_glyph, _, a_glyph = tuple(composition_reference)
        composition_symbol = Text("∘", color=INK, font=CODE_FONT, font_size=30)
        composition_symbol.move_to(composition_glyph)

        # ---------------------------------------------------------- opening
        self.play(
            FadeIn(source_cells),
            FadeIn(middle_cells),
            FadeIn(target_cells),
            FadeIn(first_label),
            FadeIn(second_label),
        )
        self.play(
            LaggedStart(
                *(Create(segment) for segment in first_segments),
                lag_ratio=0.12,
                run_time=1.3,
            )
        )
        self.play(
            LaggedStart(
                *(Create(segment) for segment in second_segments),
                lag_ratio=0.12,
                run_time=1.3,
            )
        )
        self.wait(0.35)

        # One connector per middle cell, joining the shared endpoint of a's
        # mode-j strands, across the cell, onto b's leaf-j strand.
        connectors = []
        for j in range(len(f.codomain)):
            k = mode_of_middle[j]
            beta = second_leaf_addresses[k][offset_of_middle[j]]
            first_end = next(
                strand
                for address, strand in first_edges[j].items()
                if len(address) == 1
            ).get_end()
            second_start = second_edges[k][beta].get_start()
            span = second_start - first_end
            connectors.append(
                CubicBezier(
                    first_end,
                    first_end + span / 3,
                    first_end + 2 * span / 3,
                    second_start,
                    color=INK,
                    stroke_width=STROKE_WIDTH,
                )
            )

        # --------------------------------------------------------- dissolve
        self.play(
            *(ShrinkToCenter(cell) for cell in middle_cells),
            run_time=0.9,
        )
        self.play(
            *(Create(connector) for connector in connectors),
            run_time=1.35,
        )
        self.wait(0.25)

        # --------------------------------------------------------- collapse
        # The outer stacks contract inward until the composite occupies the
        # width of a single morphism, and every stroke deforms into its
        # edge of the grafted composite tree, matched by tree address.
        source_shift = RIGHT * (MID_X - LEFT_X) / 2
        target_shift = LEFT * (RIGHT_X - MID_X) / 2

        # The composite trees between the contracted stacks, built on the
        # shifted anchors the cells are about to move to.
        shifted_source = [
            cell.copy().shift(source_shift) for cell in source_cells
        ]
        shifted_target = [
            cell.copy().shift(target_shift) for cell in target_cells
        ]
        composite_segments, composite_edges, _ = _tree_forest(
            composite, shifted_source, shifted_target
        )

        # Edge correspondence through the graft.
        pairs = []
        for j, mode in enumerate(f.modes):
            k = mode_of_middle[j]
            beta = second_leaf_addresses[k][offset_of_middle[j]]
            leaf_strand = second_edges[k][beta]
            if isinstance(mode.data, int):
                # An unrefined mode adds no junction when grafted: its
                # whole three-piece route lands on the one composite edge.
                (single,) = first_edges[j].values()
                pairs.append(
                    (
                        _joined_route([single, connectors[j], leaf_strand]),
                        composite_edges[k][beta],
                    )
                )
                continue
            # The graft junction stands where the middle cell was: the
            # connector fuses onto b's leaf strand, and a's edges keep
            # their addresses under the graft prefix.
            pairs.append(
                (
                    _joined_route([connectors[j], leaf_strand]),
                    composite_edges[k][beta],
                )
            )
            for alpha, strand in first_edges[j].items():
                pairs.append((strand, composite_edges[k][beta + alpha]))
        for k, mode_edges in enumerate(second_edges):
            leaf_betas = set(second_leaf_addresses[k])
            for beta, strand in mode_edges.items():
                if beta not in leaf_betas:
                    pairs.append((strand, composite_edges[k][beta]))

        initial_strokes = VGroup()
        final_strokes = VGroup()
        for initial_piece, final_piece in pairs:
            initial, final = matched_path_pair(initial_piece, final_piece)
            initial_strokes.add(initial)
            final_strokes.add(final)

        # Hand off from the stage strands to the matched strokes with a
        # single frame of identical pixels.
        self.remove(*first_segments, *second_segments, *connectors)
        self.add(initial_strokes)
        self.wait(1 / 15)

        # The composed label forms in step with the move-in, so it stays
        # centered under the contracting diagram.
        self.play(
            Transform(initial_strokes, final_strokes),
            source_cells.animate.shift(source_shift),
            target_cells.animate.shift(target_shift),
            first_label.animate.move_to(a_glyph),
            second_label.animate.move_to(b_glyph),
            Write(composition_symbol),
            run_time=1.35,
        )
        # Land on the canonical drawing of the composite exactly: the same
        # strokes the create scene would draw between these stacks.
        self.remove(initial_strokes, final_strokes)
        self.add(composite_segments)

        # The picture now reads as the composite: one tree per composite mode.
        assert len(composite.modes) == len(target_cells)
        self.wait(1.0)
