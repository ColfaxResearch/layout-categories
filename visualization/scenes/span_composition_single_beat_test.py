"""Prototype animation for composition in the category Span, single-beat.

An experimental variant of span_composition_test.py: the pullback plays
as ONE beat instead of two.  Every middle cell splits and its pieces
travel straight to their X' slots, so the split and the reordering are
a single motion; blocks with no place in X' fade unsplit, and the
basepoint arrivals join the same beat.  Everything else is unchanged.

A span S <-a- X -f-> T has a Fact morphism for its backward leg and a tuple
morphism for its forward leg: the backward leg is drawn as caret-free fans
(each S cell gathering its block of apex cells) and the forward leg as map
arrows.  Naming runs in order of appearance: boundary objects S, T, U, ...,
middle objects X, Y, ..., tuple morphisms f, g, ..., Fact morphisms
a, b, ..., and the pullback's outputs take primes: its middle object is
X', its tuple morphism f', its Fact morphism b'.  Two spans in
a row read Fact, Tuple, Fact, Tuple from right to left; composing them is a
matter of pulling the middle Tuple past the middle Fact and then composing
the two adjacent pairs that leaves.  This scene runs those steps in place on
five evenly spaced stacks:

    open:      [S] <-a- [X] -f-> [T] <-b- [Y] -g-> [U]
    pullback:  [S] <-a- [X] <-b'- [X'] -f'-> [Y] -g-> [U]   (one beat)
    compose:   [S] <---a ∘ b'--- [X'] ---g ∘ f'---> [U]

Stages 1 and 2 are the pullback animation verbatim, played in the middle of
the chain: the middle stack splits in place along b (f's arrows giving up
their carets and fanning out, the strands to Y becoming parallel and taking
carets), then its blocks reorder into the order of X'.  The morphism data
migrates in place -- f's arrows become b′'s fans and b's fans become f′'s
arrows -- so each leg label transforms where it stands.  After stage 2 the
chain reads Fact, Fact, Tuple, Tuple: the two Fact legs compose as X
dissolves and the two tuple legs compose as Y dissolves, exactly the two
composition collapses this family already has, and the outer stacks contract
onto the composite span.

The special cases are the pullback scene's: an apex cell sent to the
basepoint arrives in X' with a fan and no arrow, a T cell outside f's image
retires with its block, and a Y cell sent to the basepoint by g takes its
route with it when Y dissolves.  The library supplies every stack and map:
``pullback_with_refinement`` gives X', b′ and f′, ``Span_morphism.compose`` the
picture being landed on.
"""

from dataclasses import dataclass
from math import prod

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
from tract import Fact_morphism, Span_morphism, Tuple_morphism

from layout_categories_viz.animations import (
    DrawMapstoTip,
    TailToTipMapsto,
    UndrawMapstoTip,
)
from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK

# The composition scene owns the route-surgery helpers, the pullback scene
# the drawing primitives and the two-stage gesture, and the weak-composition
# scene the bridging, so this scene cannot drift from any of them.
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


@dataclass(frozen=True)
class SpanCompositionExample:
    """Two composable spans, specified leg by leg."""

    first_modes: tuple
    first_map: tuple
    middle: tuple
    second_modes: tuple
    second_map: tuple
    codomain: tuple

    def spans(self) -> tuple:
        """The spans f and g, validated by the library."""
        apex = tuple(x for mode in self.first_modes for x in mode)
        domain = tuple(prod(mode) for mode in self.first_modes)
        first = Span_morphism(
            Fact_morphism(apex, domain, self.first_modes),
            Tuple_morphism(apex, self.middle, self.first_map),
        )
        nadir = tuple(y for mode in self.second_modes for y in mode)
        second = Span_morphism(
            Fact_morphism(nadir, self.middle, self.second_modes),
            Tuple_morphism(nadir, self.codomain, self.second_map),
        )
        return first, second


EXAMPLES = (
    # Exercises every special case: an apex cell sent to the basepoint, a
    # middle cell outside the first forward leg's image, and a Y cell the
    # second forward leg sends to the basepoint.
    SpanCompositionExample(
        first_modes=((4, 3), (2,)),
        first_map=(2, 1, 0),
        middle=(3, 4, 5),
        second_modes=((3,), (2, 2), (5,)),
        second_map=(1, 3, 0, 2),
        codomain=(3, 5, 2),
    ),
    # A larger, generic example: every apex cell maps, every middle cell is
    # hit, every middle cell splits or passes through, and both maps cross.
    SpanCompositionExample(
        first_modes=((4, 6), (2, 10)),
        first_map=(2, 4, 1, 3),
        middle=(2, 4, 10, 6),
        second_modes=((2,), (2, 2), (2, 5), (3, 2)),
        second_map=(4, 1, 5, 7, 2, 3, 6),
        codomain=(2, 5, 3, 2, 2, 2, 2),
    ),
)


class SpanMorphismCompositionSingleBeatTest(Scene):
    """Pull the middle Tuple past the middle Fact, then compose like pairs."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        for index, example in enumerate(EXAMPLES):
            self._play_example(example)
            TuplePullbackTest._clear_scene(self, last=index == len(EXAMPLES) - 1)

    def _play_example(self, example: SpanCompositionExample) -> None:
        first, second = example.spans()
        composite = first.compose(second)
        refinement, pulled = second.left.pullback_with_refinement(first.right)

        # Bookkeeping, all read off the library.  Y indices and X' leaves are
        # matched by f₁′, which is one to one away from the basepoint.
        u_of_apex = tuple(
            k for k, mode in enumerate(first.left.modes) for _ in mode
        )
        y_groups = []
        start = 0
        for mode in second.left.modes:
            y_groups.append(tuple(range(start, start + len(mode))))
            start += len(mode)
        xprime_groups = []
        start = 0
        for mode in refinement.modes:
            xprime_groups.append(tuple(range(start, start + len(mode))))
            start += len(mode)
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

        # --- Open: both spans, fans backward and arrows forward. -------------
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

        first_fans = [
            segment(u_cells[u_of_apex[i]].get_right(), x_cells[i].get_left())
            for i in range(len(first.apex))
        ]
        first_arrows = {
            i: TuplePullbackTest._segment_arrow(x_cells[i], v_cells[j - 1])
            for i, j in enumerate(first.right.map)
            if j
        }
        second_fans = {
            m: segment(
                v_cells[v_of_y].get_right(), y_cells[m].get_left()
            )
            for v_of_y, leaves in enumerate(y_groups)
            for m in leaves
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
                *(Create(fan) for fan in first_fans),
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
                *(Create(second_fans[m]) for m in sorted(second_fans)),
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

        # --- The pullback in one beat: every middle cell splits and its ------
        # pieces travel straight to their X' slots, so the split and the
        # reordering are one motion.  f's arrows give up their carets and
        # fan out into the blocks as they go, the strands to Y cross into
        # f′, a T cell outside f's image fades with its block unsplit, and
        # the basepoint arrivals join the same beat.
        self.play(
            *(UndrawMapstoTip(arrow.tip) for arrow in first_arrows.values()),
            run_time=0.4,
        )

        alpha = ValueTracker(0.0)
        retiring_modes = [
            mode
            for mode in range(len(first.codomain))
            if mode not in source_apex
        ]
        middle_cells, middle_fans, connectors = {}, {}, {}
        for mode, leaves in enumerate(y_groups):
            if mode in retiring_modes:
                continue
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
                    place(COL_T, destination[leaf]),
                    peeled=bool(index),
                )
                middle_cells[leaf] = split
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

        # An apex cell sent to the basepoint has no block to receive: it
        # carries its own value into X', arriving with a fan and no arrow.
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

        # Every replacement starts out coincident with what it replaces; the
        # blocks with no place in X' keep their originals, to fade unsplit.
        self.remove(
            *(
                v_cells[mode]
                for mode in range(len(first.codomain))
                if mode not in retiring_modes
            ),
            *(
                second_fans[m]
                for mode, leaves in enumerate(y_groups)
                if mode not in retiring_modes
                for m in leaves
            ),
            *(arrow.shaft for arrow in first_arrows.values()),
            *(arrow.tail for arrow in first_arrows.values()),
        )
        self.add(*middle_fans.values(), *connectors.values())
        self.add(
            *TuplePullbackTest._deck(
                {
                    destination[leaf]: middle_cells[leaf]
                    for leaf in middle_cells
                }
            )
        )
        self.play(
            alpha.animate.set_value(1.0),
            *arrivals,
            *(FadeOut(v_cells[mode]) for mode in retiring_modes),
            *(
                FadeOut(second_fans[m])
                for mode in retiring_modes
                for m in y_groups[mode]
            ),
            FadeOut(label_t),
            FadeIn(stack_label("X′", COL_T)),
            Transform(label_f, leg_label("b′", (COL_X + COL_T) / 2)),
            Transform(label_b, leg_label("f′", (COL_T + COL_Y) / 2)),
            run_time=1.8,
            rate_func=smooth,
        )
        for mobject in (
            *middle_cells.values(),
            *middle_fans.values(),
            *connectors.values(),
        ):
            mobject.clear_updaters()

        # The crossed connectors take carets: they are f′.
        tips = {
            leaf: TuplePullbackTest._arrow_tip(
                y_cells[leaf].get_left() + LEFT * ARROW_INSET
            )
            for leaf in middle_cells
        }
        self.add(*tips.values())
        self.play(*(DrawMapstoTip(tip) for tip in tips.values()), run_time=0.4)
        self.wait(0.6)

        # The chain now reads Fact, Fact, Tuple, Tuple: re-key the middle by
        # X' leaf and compose the two adjacent pairs.
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

        # --- Compose the Facts and the Tuples: X and Y dissolve at once. -----
        # A Y cell sent to the basepoint by f₂ takes its route with it; a Y
        # cell nothing reached takes its arrow back.
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

        # One private copy of the shared U-ward strand per X' leaf, and a
        # bridge across each vanished cell on both sides.
        left_pieces, left_bridges = {}, {}
        for leaf in xprime_cells:
            apex_index = next(
                i for i, leaves in enumerate(xprime_groups) if leaf in leaves
            )
            shared = first_fans[apex_index].copy()
            left_pieces[leaf] = shared
            left_bridges[leaf] = _bridge(
                shared.get_end(), left_strands[leaf].get_start()
            )
        right_bridges = {}
        for leaf, m in surviving.items():
            right_bridges[leaf] = _bridge(
                right_strands[leaf].get_end(),
                second_arrows[m].shaft.get_start(),
            )

        # The adjacent Fact labels slide together, and so do the Tuples',
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

        self.remove(*first_fans)
        self.add(*left_pieces.values())
        self.play(
            *(Create(bridge) for bridge in left_bridges.values()),
            *(Create(bridge) for bridge in right_bridges.values()),
            run_time=1.35,
        )
        self.wait(0.25)

        # --- Contract onto the composite span. -------------------------------
        left_shift = RIGHT * CONTRACTION
        right_shift = LEFT * CONTRACTION

        initial_routes = VGroup()
        final_routes = VGroup()
        for leaf in xprime_cells:
            glued = _join_route(
                left_pieces[leaf], left_bridges[leaf], left_strands[leaf]
            )
            owner = composite_left_owner(composite, leaf)
            final_segment = segment(
                u_cells[owner].get_right() + left_shift,
                xprime_cells[leaf].get_left(),
            )
            initial, final = _matched_path_pair(glued, final_segment)
            initial_routes.add(initial)
            final_routes.add(final)

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
            initial, final = _matched_path_pair(glued, final_shaft)
            initial_routes.add(initial)
            final_routes.add(final)

        self.remove(
            *left_pieces.values(),
            *left_bridges.values(),
            *left_strands.values(),
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


def composite_left_owner(composite, leaf: int) -> int:
    """Which composite-domain cell an X' leaf belongs to."""
    running = 0
    for k, mode in enumerate(composite.left.modes):
        running += len(mode)
        if leaf < running:
            return k
    raise ValueError(f"Leaf {leaf} outside the composite domain.")
