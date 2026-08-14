"""Prototype animation for composition in the category Span(Tuple, Ref).

A span S <-a- X -f-> T has a Ref morphism for its backward leg and a tuple
morphism for its forward leg: the backward leg is drawn as a mirrored
banded tree (see ref_morphism_create_test.py; the root cell on the left
gathers its mode's leaves through junctions) and the forward leg as map
arrows.  Naming runs in order of appearance: boundary objects S, T, U,
middle objects X, Y, tuple morphisms f, g, Ref morphisms a, b, and the
pullback's outputs take primes: its middle object is X', its tuple
morphism f', its Ref morphism b'.  Two spans in a row read Ref, Tuple,
Ref, Tuple from right to left; composing them is a matter of pulling the
middle Tuple past the middle Ref and then composing the two adjacent pairs
that leaves.  This scene runs those steps in place on five evenly spaced
stacks:

    open:      [S] <-a- [X] -f-> [T] <-b- [Y] -g-> [U]
    unbraid:   b's trees relax into one parallel strand per leaf while f's
               arrows give up their carets
    stage 1:   [S] <-a- [X] -fans-> [Y] -parallel-> [Y] -g-> [U]
    stage 2:   [S] <-a- [X] <-b'- [X'] -f'-> [Y] -g-> [U]
    re-braid:  b''s strands braid back into its banded trees
    compose:   [S] <---a ∘ b'--- [X'] ---g ∘ f'---> [U]

Stages 1 and 2 are the Fact span scene's pullback gesture verbatim: the
middle stack splits in place along b (f's arrows fanning out, the strands
to Y becoming parallel and taking carets), then its blocks reorder into
the order of X'.  Trees enter through the two bracketing beats.  The
unbraid plays before the split because the gesture moves one strand per
leaf: each leaf's path through b's tree deforms into the straight strand
its split piece will carry, the trunk deliberately peeling apart the way
split cells do.  The re-braid is its mirror: after the reordering, each
X' leaf's strand deforms into its path through b''s banded trees, so the
chain reads Ref, Ref, Tuple, Tuple in canonical drawings.

The composition collapse then moves stroke for stroke, matched through the
graft's exact edge correspondence exactly as in the Ref composition scene
(a's edges keep their addresses, b''s edges land under their apex leaf's
address, and each bridge across a dissolved X cell fuses with a's leaf
strand onto the edge leaving the graft junction that stands where the
cell was; an unrefined entry of b' travels as a three-piece Fact route).
No strand is ever doubled, and the final frame is the create scene's
canonical drawing of the composite backward leg.  The special cases are
the Fact span scene's: an apex cell sent to the basepoint arrives in X'
with a strand and no arrow, a T cell outside f's image retires with its
block (junctions and all), and a Y cell sent to the basepoint by g takes
its route with it when Y dissolves.  The first backward leg's top-level
modes are kept as tuples, so each graft junction is present in the
composite.  The library supplies every stack and map:
``pullback_with_refinement`` gives X', b' and f', ``RefSpan_morphism.compose``
the picture being landed on.
"""

from dataclasses import dataclass

import numpy as np
from manim import (
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    LEFT,
    ORIGIN,
    RIGHT,
    Scene,
    ShrinkToCenter,
    Succession,
    Text,
    Transform,
    Uncreate,
    ValueTracker,
    VGroup,
    Write,
    smooth,
)
from tract import NestedTuple, Ref_morphism, RefSpan_morphism, Tuple_morphism

from layout_categories_viz.animations import (
    DrawMapstoTip,
    TailToTipMapsto,
    UndrawMapstoTip,
)
from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK

# The composition scene owns the route-surgery helpers, the pullback scene
# the drawing primitives and the two-stage gesture, the weak-composition
# scene the bridging, and the Ref create scene the banded trees, so this
# scene cannot drift from any of them.
from scenes.ref_morphism_create_test import ref_tree_with_paths
from scenes.tuple_morphism_composition_curve import _matched_path_pair
from scenes.tuple_pullback_test import (
    ARROW_INSET,
    CELL_H,
    LABEL_FONT_SIZE,
    MAX_STACK_HEIGHT,
    SLOT_STEP,
    TuplePullbackTest,
    _revealed,
)
from scenes.weak_composition_test import _bridge, _join_route

COLUMN_GAP = 3.0
COL_S, COL_X, COL_T, COL_Y, COL_U = ((index - 2) * COLUMN_GAP for index in range(5))
# The composite span occupies the footprint of a single span: the outer
# stacks come in by one column pitch each and the refined apex, already in
# the middle, stays put.
CONTRACTION = COLUMN_GAP


def _backward_forest(morphism: Ref_morphism, root_cells, leaf_cells):
    """A backward leg's mirrored banded trees, with edge and path data.

    The morphism points leaf_cells ↠ root_cells: leaves anchor on the LEFT
    edges of ``leaf_cells`` (the finer stack, on the right) and each mode's
    root on the RIGHT edge of its cell in ``root_cells`` (the coarser
    stack, on the left), so every strand runs leftward.  Returns
    (segments, paths, edges, leaf_addresses): the strands as one VGroup,
    per leaf (in global leaf order) its strands from leaf to root, per mode
    the strands keyed by child tree address, and per mode each leaf's
    address.
    """
    global_depth = max((mode.depth() for mode in morphism.modes), default=1)
    segments = VGroup()
    paths = []
    edges = []
    leaf_addresses = []
    start = 0
    for index, mode in enumerate(morphism.modes):
        leaf_anchors = [
            leaf_cells[start + offset].get_left()
            for offset in range(mode.length())
        ]
        mode_segments, mode_paths, mode_edges, mode_addresses = (
            ref_tree_with_paths(
                mode,
                leaf_anchors,
                root_cells[index].get_right(),
                global_depth,
            )
        )
        segments.add(*mode_segments)
        paths.extend(mode_paths)
        edges.append(mode_edges)
        leaf_addresses.append(mode_addresses)
        start += mode.length()
    return segments, paths, edges, leaf_addresses


def _rightward_route(path) -> "VGroup":
    """A leaf's root path joined into one path running root to leaf.

    Backward-leg strands run leftward (leaf to root); the pullback gesture
    and its bracketing beats move rightward strands, so the pieces are
    reversed individually and joined root-first.
    """
    pieces = []
    for strand in reversed(path):
        piece = strand.copy()
        piece.reverse_points()
        pieces.append(piece)
    return _join_route(*pieces)


@dataclass(frozen=True)
class RefSpanCompositionExample:
    """Two composable spans, specified leg by leg."""

    first_nest: tuple
    first_map: tuple
    middle: tuple
    second_nest: tuple
    second_map: tuple
    codomain: tuple

    def spans(self) -> tuple:
        """The spans f and g, validated by the library."""
        first_left = Ref_morphism(NestedTuple(self.first_nest))
        first = RefSpan_morphism(
            first_left,
            Tuple_morphism(first_left.domain, self.middle, self.first_map),
        )
        second_left = Ref_morphism(NestedTuple(self.second_nest))
        second = RefSpan_morphism(
            second_left,
            Tuple_morphism(second_left.domain, self.codomain, self.second_map),
        )
        return first, second


EXAMPLES = (
    # The Fact span scene's first example, lifted: both backward legs have
    # depth-1 modes, so the unbraid and re-braid are identities and the
    # gesture is the Fact one.  Exercises every special case: an apex cell
    # sent to the basepoint, a middle cell outside the first forward leg's
    # image, and a Y cell the second forward leg sends to the basepoint.
    RefSpanCompositionExample(
        first_nest=((4, 3), (2,)),
        first_map=(2, 1, 0),
        middle=(3, 4, 5),
        second_nest=((3,), (2, 2), (5,)),
        second_map=(1, 3, 0, 2),
        codomain=(3, 5, 2),
    ),
    # Junctions on both backward legs: a's first mode nests (4, 3), b's
    # middle mode ((2, 2),) is hit and survives into b' and the composite
    # graft, b's retiring block is flat, and both basepoint cases recur.
    RefSpanCompositionExample(
        first_nest=((2, (4, 3)), (5,)),
        first_map=(0, 2, 1, 0),
        middle=(3, 4, 2),
        second_nest=((3,), ((2, 2),), (2,)),
        second_map=(2, 1, 0, 3),
        codomain=(2, 3, 2),
    ),
)


class RefSpanMorphismCompositionTest(Scene):
    """Pull the middle Tuple past the middle Ref, then compose like pairs."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        for index, example in enumerate(EXAMPLES):
            self._play_example(example)
            TuplePullbackTest._clear_scene(self, last=index == len(EXAMPLES) - 1)

    def _play_example(self, example: RefSpanCompositionExample) -> None:
        first, second = example.spans()
        composite = first.compose(second)
        refinement, pulled = second.left.pullback_with_refinement(first.right)

        # Bookkeeping, all read off the library.  Y indices and X' leaves
        # are matched by f', which is one to one away from the basepoint.
        u_of_apex = tuple(
            k
            for k, mode in enumerate(first.left.modes)
            for _ in range(mode.length())
        )
        y_groups = []
        start = 0
        for mode in second.left.modes:
            y_groups.append(tuple(range(start, start + mode.length())))
            start += mode.length()
        xprime_groups = []
        start = 0
        for mode in refinement.modes:
            xprime_groups.append(tuple(range(start, start + mode.length())))
            start += mode.length()
        source_apex = {
            target - 1: i for i, target in enumerate(first.right.map) if target
        }
        destination = {
            pulled.map[leaf] - 1: leaf
            for leaf in range(len(refinement.domain))
            if pulled.map[leaf]
        }

        # Geometry: one bottom baseline, one uniform step, scaled to fit.
        slots = max(
            len(first.domain),
            len(first.apex),
            len(second.apex),
            len(refinement.domain),
            len(composite.codomain),
        )
        scale = min(1.0, MAX_STACK_HEIGHT / ((slots - 1) * SLOT_STEP + CELL_H))
        step = SLOT_STEP * scale

        def place(column, index):
            return np.array(
                (column, -((slots - 1) * step) / 2 + index * step + 0.45, 0.0)
            )

        def cell(value, center):
            return (
                TuplePullbackTest._cell(value, ORIGIN)
                .scale(scale)
                .move_to(center)
            )

        stack_label_y = place(0, 0)[1] - CELL_H * scale / 2 - 0.42
        leg_label_y = stack_label_y - 0.62

        def stack_label(text, column):
            return (
                Text(text, color=INK, font=CODE_FONT, font_size=LABEL_FONT_SIZE)
                .scale(scale)
                .move_to(np.array([column, stack_label_y, 0.0]))
            )

        def leg_label(text, x):
            return (
                Text(text, color=INK, font=CODE_FONT, font_size=28)
                .scale(scale)
                .move_to(np.array([x, leg_label_y, 0.0]))
            )

        segment = TuplePullbackTest._tree_segment
        attached = TuplePullbackTest._attached

        # --- Open: both spans, trees backward and arrows forward. ------------
        u_cells = [
            cell(value, place(COL_S, index))
            for index, value in enumerate(first.domain)
        ]
        x_cells = [
            cell(value, place(COL_X, index))
            for index, value in enumerate(first.apex)
        ]
        v_cells = [
            cell(value, place(COL_T, index))
            for index, value in enumerate(first.codomain)
        ]
        y_cells = [
            cell(value, place(COL_Y, index))
            for index, value in enumerate(second.apex)
        ]
        w_cells = [
            cell(value, place(COL_U, index))
            for index, value in enumerate(second.codomain)
        ]

        first_tree, first_paths, first_edges, first_leaf_addresses = (
            _backward_forest(first.left, u_cells, x_cells)
        )
        second_tree, second_paths, _, _ = _backward_forest(
            second.left, v_cells, y_cells
        )
        first_arrows = {
            i: TuplePullbackTest._segment_arrow(x_cells[i], v_cells[j - 1])
            for i, j in enumerate(first.right.map)
            if j
        }
        second_arrows = {
            m: TuplePullbackTest._segment_arrow(y_cells[m], w_cells[k - 1])
            for m, k in enumerate(second.right.map)
            if k
        }

        label_source = stack_label("S", COL_S)
        label_x = stack_label("X", COL_X)
        label_t = stack_label("T", COL_T)
        label_y = stack_label("Y", COL_Y)
        label_target = stack_label("U", COL_U)
        label_a = leg_label("a", (COL_S + COL_X) / 2)
        label_f = leg_label("f", (COL_X + COL_T) / 2)
        label_b = leg_label("b", (COL_T + COL_Y) / 2)
        label_g = leg_label("g", (COL_Y + COL_U) / 2)

        self.play(
            *(FadeIn(VGroup(*cells)) for cells in
              (u_cells, x_cells, v_cells, y_cells, w_cells)),
            *(FadeIn(label) for label in
              (label_source, label_x, label_t, label_y, label_target)),
        )
        self.play(
            LaggedStart(
                *(Create(strand) for strand in first_tree),
                lag_ratio=0.12,
                run_time=1.1,
            ),
            FadeIn(label_a),
        )
        self.play(
            LaggedStart(
                *(
                    TailToTipMapsto(arrow, run_time=1.0)
                    for arrow in first_arrows.values()
                ),
                lag_ratio=0.12,
            ),
            FadeIn(label_f),
        )
        self.play(
            LaggedStart(
                *(Create(strand) for strand in second_tree),
                lag_ratio=0.12,
                run_time=1.1,
            ),
            FadeIn(label_b),
        )
        self.play(
            LaggedStart(
                *(
                    TailToTipMapsto(arrow, run_time=1.0)
                    for arrow in second_arrows.values()
                ),
                lag_ratio=0.12,
            ),
            FadeIn(label_g),
        )
        self.wait(0.35)

        # --- Unbraid: b's trees relax into one parallel strand per leaf ------
        # while f's arrows give up their carets.  Each leaf's path through
        # its tree deforms into the straight strand its split piece will
        # carry, the trunk deliberately peeling apart the way split cells
        # do; for depth-1 modes only the strand's closing run changes
        # sides (the mirrored tree closes at the root, the fan at the
        # leaf).
        second_fans = {
            m: segment(
                v_cells[v_of_y].get_right(), y_cells[m].get_left()
            )
            for v_of_y, leaves in enumerate(y_groups)
            for m in leaves
        }
        unbraid_initial = VGroup()
        unbraid_final = VGroup()
        for m in sorted(second_fans):
            initial, final = _matched_path_pair(
                _rightward_route(second_paths[m]), second_fans[m]
            )
            unbraid_initial.add(initial)
            unbraid_final.add(final)
        self.remove(*second_tree)
        self.add(unbraid_initial)
        self.play(
            Transform(unbraid_initial, unbraid_final),
            *(UndrawMapstoTip(arrow.tip) for arrow in first_arrows.values()),
            run_time=0.6,
        )
        self.remove(unbraid_initial, unbraid_final)
        self.add(*(second_fans[m] for m in sorted(second_fans)))

        # --- Stage 1: the middle splits in place along the second backward ---
        # leg, exactly the Fact span gesture: f's arrows fan out into the
        # blocks; the strands to Y become parallel, cell for cell.
        alpha = ValueTracker(0.0)
        middle_cells, middle_fans, connectors = {}, {}, {}
        for mode, leaves in enumerate(y_groups):
            start_center = v_cells[mode].get_center()
            for index, leaf in enumerate(leaves):
                split = cell(second.apex[leaf], start_center)
                if not index:
                    coarse = cell(first.codomain[mode], start_center)[1]
                    split[1].set_opacity(0)
                    split.add(coarse)
                TuplePullbackTest._split(
                    split,
                    alpha,
                    start_center,
                    place(COL_T, leaf),
                    peeled=bool(index),
                )
                middle_cells[leaf] = split
                if mode in source_apex:
                    middle_fans[leaf] = attached(
                        x_cells[source_apex[mode]],
                        split,
                        reveal=(
                            (lambda: _revealed(alpha.get_value()))
                            if index
                            else None
                        ),
                    )
                connectors[leaf] = attached(split, y_cells[leaf])

        # Every replacement starts out coincident with what it replaces.
        # The middle is becoming a second copy of Y, cell for cell.
        self.remove(
            *v_cells,
            *second_fans.values(),
            *(arrow.shaft for arrow in first_arrows.values()),
            *(arrow.tail for arrow in first_arrows.values()),
        )
        self.add(*middle_fans.values(), *connectors.values())
        self.add(*TuplePullbackTest._deck(middle_cells))
        label_y_middle = stack_label("Y", COL_T)
        self.play(
            alpha.animate.set_value(1.0),
            FadeOut(label_t),
            FadeIn(label_y_middle),
            run_time=1.4,
            rate_func=smooth,
        )
        for split in middle_cells.values():
            split.clear_updaters()

        # The parallel connectors take carets: they are becoming f', whose
        # carets they keep through the reordering.
        tips = {
            leaf: TuplePullbackTest._arrow_tip(
                y_cells[leaf].get_left() + LEFT * ARROW_INSET
            )
            for leaf in middle_cells
        }
        self.add(*tips.values())
        self.play(*(DrawMapstoTip(tip) for tip in tips.values()), run_time=0.4)
        self.wait(0.6)

        # --- Stage 2: reorder the blocks into the order of X'.  A T cell -----
        # outside f's image has no place in X', so its block leaves,
        # junctions and all.  The morphism data migrates in place: f's
        # arrows have become b''s strands and b's strands have become f''s
        # arrows, so each label transforms where it stands.
        retiring = [leaf for leaf in middle_cells if leaf not in destination]
        for leaf in retiring:
            connectors[leaf].clear_updaters()
        self.add(*(middle_cells[leaf] for leaf in retiring))
        self.add(
            *TuplePullbackTest._deck(
                {
                    destination[leaf]: middle_cells[leaf]
                    for leaf in middle_cells
                    if leaf in destination
                }
            )
        )
        reorder = [
            middle_cells[leaf].animate.move_to(place(COL_T, destination[leaf]))
            for leaf in middle_cells
            if leaf in destination
        ] + [
            FadeOut(VGroup(middle_cells[leaf], connectors[leaf], tips[leaf]))
            for leaf in retiring
        ]

        # An apex cell sent to the basepoint has no block to reorder: it
        # carries its own value into X', arriving with a strand and no
        # arrow.
        arrival_cells, arrival_fans, arrivals = {}, {}, []
        for i, target in enumerate(first.right.map):
            if target:
                continue
            for leaf in xprime_groups[i]:
                joined = cell(refinement.domain[leaf], place(COL_T, leaf))
                fan = segment(x_cells[i].get_right(), joined.get_left())
                arrival_cells[leaf] = joined
                arrival_fans[leaf] = fan
                arrivals.append(
                    Succession(
                        FadeIn(joined, run_time=0.45),
                        Create(fan, run_time=0.95),
                    )
                )

        self.play(
            *reorder,
            *arrivals,
            FadeOut(label_y_middle),
            FadeIn(stack_label("X′", COL_T)),
            Transform(label_f, leg_label("b′", (COL_X + COL_T) / 2)),
            Transform(label_b, leg_label("f′", (COL_T + COL_Y) / 2)),
            run_time=1.4,
            rate_func=smooth,
        )
        for mobject in (*middle_fans.values(), *connectors.values()):
            mobject.clear_updaters()
        self.wait(0.6)

        # Re-key the middle by X' leaf.
        xprime_cells, left_strands, right_strands, right_tips = {}, {}, {}, {}
        for leaf in range(len(refinement.domain)):
            if leaf in arrival_cells:
                xprime_cells[leaf] = arrival_cells[leaf]
                left_strands[leaf] = arrival_fans[leaf]
            else:
                m = pulled.map[leaf] - 1
                xprime_cells[leaf] = middle_cells[m]
                left_strands[leaf] = middle_fans[m]
                right_strands[leaf] = connectors[m]
                right_tips[leaf] = tips[m]

        # --- Re-braid: b''s strands braid back into its banded trees, --------
        # each X' leaf's strand deforming into its path through the
        # canonical mirrored drawing of b', so the chain reads Ref, Ref,
        # Tuple, Tuple in canonical pictures.
        bprime_tree, bprime_paths, bprime_edges, bprime_leaf_addresses = (
            _backward_forest(
                refinement,
                x_cells,
                [xprime_cells[leaf] for leaf in range(len(refinement.domain))],
            )
        )
        rebraid_initial = VGroup()
        rebraid_final = VGroup()
        for leaf in range(len(refinement.domain)):
            initial, final = _matched_path_pair(
                left_strands[leaf], _rightward_route(bprime_paths[leaf])
            )
            rebraid_initial.add(initial)
            rebraid_final.add(final)
        self.remove(*left_strands.values())
        self.add(rebraid_initial)
        self.play(Transform(rebraid_initial, rebraid_final), run_time=0.6)
        self.remove(rebraid_initial, rebraid_final)
        self.add(*bprime_tree)
        self.wait(0.4)

        # The chain now reads Ref, Ref, Tuple, Tuple: compose the two
        # adjacent pairs.  A Y cell sent to the basepoint by g takes its
        # route with it; a Y cell nothing reached takes its arrow back.
        surviving, dying = {}, {}
        for leaf in right_strands:
            m = pulled.map[leaf] - 1
            if second.right.map[m]:
                surviving[leaf] = m
            else:
                dying[leaf] = m
        reached = {pulled.map[leaf] - 1 for leaf in right_strands}
        unreached_arrows = [
            arrow for m, arrow in second_arrows.items() if m not in reached
        ]

        self.play(
            *(ShrinkToCenter(cell) for cell in x_cells),
            *(ShrinkToCenter(cell) for cell in y_cells),
            FadeOut(label_x),
            FadeOut(label_y),
            *(
                UndrawMapstoTip(right_tips[leaf])
                for leaf in list(surviving) + list(dying)
            ),
            *(FadeOut(VGroup(right_strands[leaf])) for leaf in dying),
            *(Uncreate(arrow.shaft) for arrow in unreached_arrows),
            *(UndrawMapstoTip(arrow.tip) for arrow in unreached_arrows),
            run_time=0.9,
        )

        # One bridge across each dissolved X cell (joining b''s mode root
        # onto a's leaf strand) and each dissolved Y cell (joining f''s
        # strand onto g's shaft).
        apex_of_leaf = {
            leaf: i
            for i, leaves in enumerate(xprime_groups)
            for leaf in leaves
        }
        left_bridges = {}
        for i in range(len(first.apex)):
            root_end = next(
                strand
                for address, strand in bprime_edges[i].items()
                if len(address) == 1
            ).get_end()
            leaf_start = first_paths[i][0].points[0]
            left_bridges[i] = _bridge(root_end, leaf_start)
        right_bridges = {}
        for leaf, m in surviving.items():
            right_bridges[leaf] = _bridge(
                right_strands[leaf].get_end(),
                second_arrows[m].shaft.get_start(),
            )

        # The adjacent Ref labels slide together, and so do the Tuples',
        # landing mid-gap of the contracted span.
        left_reference = Text(
            "a ∘ b′", color=INK, font=CODE_FONT, font_size=28
        ).scale(scale)
        left_reference.move_to(
            np.array([(COL_S + CONTRACTION + COL_T) / 2, leg_label_y, 0.0])
        )
        right_reference = Text(
            "g ∘ f′", color=INK, font=CODE_FONT, font_size=28
        ).scale(scale)
        right_reference.move_to(
            np.array([(COL_T + COL_U - CONTRACTION) / 2, leg_label_y, 0.0])
        )
        # Text submobjects include space glyphs: "a ∘ b′" splits into
        # a,␣,∘,␣,b,′ and "g ∘ f′" into g,␣,∘,␣,f,′.
        left_symbol = Text("∘", color=INK, font=CODE_FONT, font_size=28)
        left_symbol.scale(scale).move_to(left_reference[2])
        right_symbol = Text("∘", color=INK, font=CODE_FONT, font_size=28)
        right_symbol.scale(scale).move_to(right_reference[2])

        self.play(
            *(Create(bridge) for bridge in left_bridges.values()),
            *(Create(bridge) for bridge in right_bridges.values()),
            run_time=1.35,
        )
        self.wait(0.25)

        # --- Contract onto the composite span. -------------------------------
        left_shift = RIGHT * CONTRACTION
        right_shift = LEFT * CONTRACTION

        # The composite backward trees between the contracted stacks, built
        # on the shifted anchors the S cells are about to move to, and the
        # edge correspondence through the graft.
        shifted_u = [cell.copy().shift(left_shift) for cell in u_cells]
        _, _, composite_edges, _ = _backward_forest(
            composite.left,
            shifted_u,
            [xprime_cells[leaf] for leaf in range(len(refinement.domain))],
        )

        initial_routes = VGroup()
        final_routes = VGroup()

        def add_pair(initial_piece, final_piece):
            initial, final = _matched_path_pair(initial_piece, final_piece)
            initial_routes.add(initial)
            final_routes.add(final)

        consumed = set()
        for i, mode in enumerate(refinement.modes):
            k = u_of_apex[i]
            leaf_address = first_leaf_addresses[k][
                i - sum(m.length() for m in first.left.modes[:k])
            ]
            leaf_edge = first_edges[k][leaf_address]
            consumed.add((k, leaf_address))
            if isinstance(mode.data, int):
                # An unrefined entry of b' adds no junction when grafted:
                # its whole three-piece route lands on the one composite
                # edge, exactly a Fact route.
                (single,) = bprime_edges[i].values()
                add_pair(
                    _join_route(single, left_bridges[i], leaf_edge),
                    composite_edges[k][leaf_address],
                )
                continue
            # The graft junction stands where the X cell was: the bridge
            # fuses onto a's leaf strand, and b''s edges keep their
            # addresses under the graft prefix.
            add_pair(
                _join_route(left_bridges[i], leaf_edge),
                composite_edges[k][leaf_address],
            )
            for beta, strand in bprime_edges[i].items():
                add_pair(strand, composite_edges[k][leaf_address + beta])
        for k, mode_edges in enumerate(first_edges):
            for alpha, strand in mode_edges.items():
                if (k, alpha) not in consumed:
                    add_pair(strand, composite_edges[k][alpha])

        surviving_arrows = {
            leaf: second_arrows[m] for leaf, m in surviving.items()
        }
        for leaf, arrow in surviving_arrows.items():
            glued = _join_route(
                right_strands[leaf], right_bridges[leaf], arrow.shaft
            )
            target = composite.right.map[leaf]
            final_shaft = segment(
                xprime_cells[leaf].get_right(),
                w_cells[target - 1].get_left() + right_shift,
            )
            add_pair(glued, final_shaft)

        self.remove(
            *bprime_tree,
            *first_tree,
            *left_bridges.values(),
            *right_strands.values(),
            *right_bridges.values(),
            *(arrow.shaft for arrow in surviving_arrows.values()),
            *(arrow.tail for arrow in surviving_arrows.values()),
        )
        self.add(initial_routes)
        self.wait(1 / 15)

        # The composed labels form in step with the move-in, so each label
        # stays centered under the portion of the diagram it names.
        self.play(
            Transform(initial_routes, final_routes),
            VGroup(*u_cells, label_source).animate.shift(left_shift),
            VGroup(
                *w_cells,
                label_target,
                *(arrow.tip for arrow in surviving_arrows.values()),
            ).animate.shift(right_shift),
            label_a.animate.move_to(left_reference[0]),
            label_f.animate.move_to(VGroup(*left_reference[4:6]).get_center()),
            Write(left_symbol),
            label_g.animate.move_to(right_reference[0]),
            label_b.animate.move_to(VGroup(*right_reference[4:6]).get_center()),
            Write(right_symbol),
            run_time=1.35,
        )
        self.remove(initial_routes)
        self.add(final_routes)
        self.wait(1.0)
