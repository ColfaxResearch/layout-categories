"""Prototype animation for composition in the category CoSpan.

A cospan S -f-> X <-a- T has a tuple morphism for its forward leg and a Fact
morphism for its backward leg: the forward leg is drawn as map arrows and the
backward leg as caret-free fans (each T cell gathering, off to its left, the
block of nadir cells that multiply to it).  Naming runs in order of
appearance: boundary objects S, T, U, ..., middle objects X, Y, ..., tuple
morphisms f, g, ..., Fact morphisms a, b, ..., and the pushforward's outputs
take primes: its middle object is Y', its tuple morphism g', its Fact
morphism a'.  Two cospans in a row read Tuple, Fact, Tuple, Fact from left
to right; composing them is a matter of pushing the middle Fact past the
middle Tuple and then composing the two adjacent pairs that leaves.  This
scene runs those steps in place on five evenly spaced stacks:

    open:      [S] -f-> [X] -a-fans-> [T] -g-> [Y] -b-fans-> [U]
    stage 1:   [S] -f-> [X] -parallel-> [X] -fans-> [Y] -b-fans-> [U]
    stage 2:   [S] -f-> [X] -g'-> [Y'] -a'-fans-> [Y] -b-fans-> [U]
    compose:   [S] ---g' ∘ f---> [Y'] ---b ∘ a'---> [U]

Stages 1 and 2 are the pushforward played as the pullback scene's gesture,
mirrored: the middle stack splits in place along a (g's arrows giving up
their carets and fanning out toward Y, the strands to X becoming parallel
and taking carets), then its blocks reorder into the order of Y'.  The
morphism data migrates in place -- a's fans become g''s arrows and g's
arrows become a''s fans -- so each leg label transforms where it stands.
After stage 2 the chain reads Tuple, Tuple, Fact, Fact: the two tuple legs
compose as X dissolves and the two Fact legs compose as Y dissolves, and the
outer stacks contract onto the composite cospan.

The special cases mirror the span scene's: a Y cell outside g's image
arrives in Y' with a fan and no arrow, a T cell sent to the basepoint by g
retires with its block, and an S cell whose route runs through it loses that
route when X dissolves.  The library supplies every stack and map:
``pushforward_with_refinement`` gives Y', a' and g',
``CoSpan_morphism.compose`` the picture being landed on.
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
from tract import CoSpan_morphism, Fact_morphism, Tuple_morphism

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
# The composite cospan occupies the footprint of a single cospan: the outer
# stacks come in by one column pitch each and the refined nadir, already in
# the middle, stays put.
CONTRACTION = COLUMN_GAP


@dataclass(frozen=True)
class CoSpanCompositionExample:
    """Two composable cospans, specified leg by leg."""

    domain: tuple
    first_map: tuple
    first_modes: tuple
    second_map: tuple
    second_modes: tuple

    def cospans(self) -> tuple:
        """The cospans, validated by the library."""
        nadir = tuple(x for mode in self.first_modes for x in mode)
        middle = tuple(prod(mode) for mode in self.first_modes)
        first = CoSpan_morphism(
            Tuple_morphism(self.domain, nadir, self.first_map),
            Fact_morphism(nadir, middle, self.first_modes),
        )
        second_nadir = tuple(y for mode in self.second_modes for y in mode)
        codomain = tuple(prod(mode) for mode in self.second_modes)
        second = CoSpan_morphism(
            Tuple_morphism(middle, second_nadir, self.second_map),
            Fact_morphism(second_nadir, codomain, self.second_modes),
        )
        return first, second


EXAMPLES = (
    # Exercises every special case: an S cell sent to the basepoint, nadir
    # cells outside f's image, a T cell sent to the basepoint by g (its
    # block retires), and a Y cell outside g's image (an arrival).
    CoSpanCompositionExample(
        domain=(2, 3, 9),
        first_map=(2, 1, 0),
        first_modes=((3,), (2, 2), (5,)),
        second_map=(0, 2, 1),
        second_modes=((5,), (4, 7)),
    ),
    # A larger, generic example: every cell maps, every cell is hit, every
    # middle cell splits or passes through, and both maps cross.
    CoSpanCompositionExample(
        domain=(2, 2, 5, 2, 2, 2, 3),
        first_map=(4, 1, 5, 7, 2, 3, 6),
        first_modes=((2,), (2, 2), (2, 5), (3, 2)),
        second_map=(2, 4, 1, 3),
        second_modes=((10, 2), (6, 4)),
    ),
)


class CoSpanMorphismCompositionTest(Scene):
    """Push the middle Fact past the middle Tuple, then compose like pairs."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        for index, example in enumerate(EXAMPLES):
            self._play_example(example)
            TuplePullbackTest._clear_scene(self, last=index == len(EXAMPLES) - 1)

    def _play_example(self, example: CoSpanCompositionExample) -> None:
        first, second = example.cospans()
        composite = first.compose(second)
        refinement, pushed = first.right.pushforward_with_refinement(second.left)

        # Bookkeeping, all read off the library.  Nadir cells and Y' leaves
        # are matched by g', which is one to one away from the basepoint.
        x_groups = []
        start = 0
        for mode in first.right.modes:
            x_groups.append(tuple(range(start, start + len(mode))))
            start += len(mode)
        yprime_groups = []
        start = 0
        for mode in refinement.modes:
            yprime_groups.append(tuple(range(start, start + len(mode))))
            start += len(mode)
        u_of_y = tuple(
            k for k, mode in enumerate(second.right.modes) for _ in mode
        )
        # Which nadir cell each Y' leaf came from, and where each nadir
        # cell's middle copy is headed.
        destination = {
            i: pushed.map[i] - 1
            for i in range(len(first.nadir))
            if pushed.map[i]
        }
        leaf_to_x = {leaf: i for i, leaf in destination.items()}

        # Geometry: one bottom baseline, one uniform step, scaled to fit.
        slots = max(
            len(first.domain),
            len(first.nadir),
            len(second.nadir),
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

        # --- Open: both cospans, arrows forward and fans backward. -----------
        s_cells = [
            cell(value, place(COL_S, index))
            for index, value in enumerate(first.domain)
        ]
        x_cells = [
            cell(value, place(COL_X, index))
            for index, value in enumerate(first.nadir)
        ]
        t_cells = [
            cell(value, place(COL_T, index))
            for index, value in enumerate(first.codomain)
        ]
        y_cells = [
            cell(value, place(COL_Y, index))
            for index, value in enumerate(second.nadir)
        ]
        u_cells = [
            cell(value, place(COL_U, index))
            for index, value in enumerate(composite.codomain)
        ]

        first_arrows = {
            s: TuplePullbackTest._segment_arrow(s_cells[s], x_cells[i - 1])
            for s, i in enumerate(first.left.map)
            if i
        }
        first_fans = {
            i: segment(x_cells[i].get_right(), t_cells[mode].get_left())
            for mode, leaves in enumerate(x_groups)
            for i in leaves
        }
        second_arrows = {
            t: TuplePullbackTest._segment_arrow(t_cells[t], y_cells[j - 1])
            for t, j in enumerate(second.left.map)
            if j
        }
        second_fans = {
            j: segment(y_cells[j].get_right(), u_cells[u_of_y[j]].get_left())
            for j in range(len(second.nadir))
        }

        label_source = stack_label("S", COL_S)
        label_x = stack_label("X", COL_X)
        label_t = stack_label("T", COL_T)
        label_y = stack_label("Y", COL_Y)
        label_target = stack_label("U", COL_U)
        label_f = leg_label("f", (COL_S + COL_X) / 2)
        label_a = leg_label("a", (COL_X + COL_T) / 2)
        label_g = leg_label("g", (COL_T + COL_Y) / 2)
        label_b = leg_label("b", (COL_Y + COL_U) / 2)

        self.play(
            *(FadeIn(VGroup(*cells)) for cells in
              (s_cells, x_cells, t_cells, y_cells, u_cells)),
            *(FadeIn(label) for label in
              (label_source, label_x, label_t, label_y, label_target)),
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
                *(Create(first_fans[i]) for i in sorted(first_fans)),
                lag_ratio=0.12,
                run_time=1.1,
            ),
            FadeIn(label_a),
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
        self.play(
            LaggedStart(
                *(Create(second_fans[j]) for j in sorted(second_fans)),
                lag_ratio=0.12,
                run_time=1.1,
            ),
            FadeIn(label_b),
        )
        self.wait(0.35)

        # --- Stage 1: the middle splits in place along the first backward ----
        # leg.  g's arrows give up their carets and fan out toward Y; the
        # strands to X become parallel, cell for cell.
        self.play(
            *(UndrawMapstoTip(arrow.tip) for arrow in second_arrows.values()),
            run_time=0.4,
        )

        alpha = ValueTracker(0.0)
        middle_cells, middle_strands, y_strands = {}, {}, {}
        for mode, leaves in enumerate(x_groups):
            start_center = t_cells[mode].get_center()
            image = second.left.map[mode]
            for index, i in enumerate(leaves):
                split = cell(first.nadir[i], start_center)
                if not index:
                    coarse = cell(first.codomain[mode], start_center)[1]
                    split[1].set_opacity(0)
                    split.add(coarse)
                TuplePullbackTest._split(
                    split,
                    alpha,
                    start_center,
                    place(COL_T, i),
                    peeled=bool(index),
                )
                middle_cells[i] = split
                # One strand of a's fan per nadir cell already exists, so
                # every parallel replacement has its own coincident
                # predecessor: no reveal is needed on this side.
                middle_strands[i] = attached(x_cells[i], split)
                if image:
                    y_strands[i] = attached(
                        split,
                        y_cells[image - 1],
                        reveal=(
                            (lambda: _revealed(alpha.get_value()))
                            if index
                            else None
                        ),
                    )

        # Every replacement starts out coincident with what it replaces.  The
        # middle is becoming a second copy of X, cell for cell.
        self.remove(
            *t_cells,
            *first_fans.values(),
            *(arrow.shaft for arrow in second_arrows.values()),
            *(arrow.tail for arrow in second_arrows.values()),
        )
        self.add(*middle_strands.values(), *y_strands.values())
        self.add(*TuplePullbackTest._deck(middle_cells))
        label_x_middle = stack_label("X", COL_T)
        self.play(
            alpha.animate.set_value(1.0),
            FadeOut(label_t),
            FadeIn(label_x_middle),
            run_time=1.4,
            rate_func=smooth,
        )
        for split in middle_cells.values():
            split.clear_updaters()

        # The parallel strands take carets: they are becoming g', whose
        # carets they keep through the reordering, following their cells.
        tips = {
            i: TuplePullbackTest._arrow_tip(
                split.get_left() + LEFT * ARROW_INSET
            )
            for i, split in middle_cells.items()
        }
        self.add(*tips.values())
        self.play(*(DrawMapstoTip(tip) for tip in tips.values()), run_time=0.4)
        # Only now do the carets start following their cells: an updater
        # rebuilding the caret each frame would fight the draw above.
        for i, tip in tips.items():
            tip.add_updater(
                lambda mobject, split=middle_cells[i]: mobject.become(
                    TuplePullbackTest._arrow_tip(
                        split.get_left() + LEFT * ARROW_INSET
                    )
                )
            )
        self.wait(0.6)

        # --- Stage 2: reorder the blocks into the order of Y'.  A T cell -----
        # sent to the basepoint by g has no place in Y', so its block
        # leaves.  The morphism data migrates in place: a's fans have become
        # g''s arrows and g's arrows have become a''s fans, so each label
        # transforms where it stands.
        retiring = [i for i in middle_cells if i not in destination]
        for i in retiring:
            middle_strands[i].clear_updaters()
            tips[i].clear_updaters()
        self.add(*(middle_cells[i] for i in retiring))
        self.add(
            *TuplePullbackTest._deck(
                {
                    destination[i]: middle_cells[i]
                    for i in middle_cells
                    if i in destination
                }
            )
        )
        reorder = [
            middle_cells[i].animate.move_to(place(COL_T, destination[i]))
            for i in middle_cells
            if i in destination
        ] + [
            FadeOut(VGroup(middle_cells[i], middle_strands[i], tips[i]))
            for i in retiring
        ]

        # A Y cell outside g's image has no block to receive: it carries its
        # own value into Y', arriving with a fan and no arrow.
        arrival_cells, arrival_fans, arrivals = {}, {}, []
        hit_y = {j - 1 for j in second.left.map if j}
        for j in range(len(second.nadir)):
            if j in hit_y:
                continue
            for leaf in yprime_groups[j]:
                joined = cell(refinement.domain[leaf], place(COL_T, leaf))
                fan = segment(joined.get_right(), y_cells[j].get_left())
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
            FadeOut(label_x_middle),
            FadeIn(stack_label("Y′", COL_T)),
            Transform(label_a, leg_label("g′", (COL_X + COL_T) / 2)),
            Transform(label_g, leg_label("a′", (COL_T + COL_Y) / 2)),
            run_time=1.4,
            rate_func=smooth,
        )
        for mobject in (
            *middle_strands.values(),
            *y_strands.values(),
            *(tips[i] for i in tips if i not in retiring),
        ):
            mobject.clear_updaters()
        self.wait(0.6)

        # The chain now reads Tuple, Tuple, Fact, Fact: re-key the middle by
        # Y' leaf and compose the two adjacent pairs.
        yprime_cells, left_strands, left_tips, right_strands = {}, {}, {}, {}
        for leaf in range(len(refinement.domain)):
            if leaf in arrival_cells:
                yprime_cells[leaf] = arrival_cells[leaf]
                right_strands[leaf] = arrival_fans[leaf]
            else:
                i = leaf_to_x[leaf]
                yprime_cells[leaf] = middle_cells[i]
                left_strands[leaf] = middle_strands[i]
                left_tips[leaf] = tips[i]
                right_strands[leaf] = y_strands[i]

        # --- Compose the Tuples and the Facts: X and Y dissolve at once. -----
        # An S cell whose nadir cell lost its block loses its route; a nadir
        # cell nothing reached takes its arrow back.
        x_of_s = {
            s: i - 1 for s, i in enumerate(first.left.map) if i
        }
        surviving, dying = {}, {}
        for s, i in x_of_s.items():
            if i in destination:
                surviving[s] = destination[i]
            else:
                dying[s] = i
        reached = set(x_of_s.values())
        unreached = [
            leaf
            for leaf in left_strands
            if leaf_to_x[leaf] not in reached
        ]

        self.play(
            *(ShrinkToCenter(cell) for cell in x_cells),
            *(ShrinkToCenter(cell) for cell in y_cells),
            FadeOut(label_x),
            FadeOut(label_y),
            *(
                UndrawMapstoTip(first_arrows[s].tip)
                for s in first_arrows
            ),
            *(FadeOut(VGroup(first_arrows[s].shaft)) for s in dying),
            *(Uncreate(left_strands[leaf]) for leaf in unreached),
            *(UndrawMapstoTip(left_tips[leaf]) for leaf in unreached),
            run_time=0.9,
        )

        # A bridge across each vanished cell on both sides, and one private
        # copy of the shared U-ward strand per Y' leaf.
        left_bridges = {}
        for s, leaf in surviving.items():
            left_bridges[s] = _bridge(
                first_arrows[s].shaft.get_end(),
                left_strands[leaf].get_start(),
            )
        right_pieces, right_bridges = {}, {}
        for leaf in yprime_cells:
            j = next(
                j for j, leaves in enumerate(yprime_groups) if leaf in leaves
            )
            shared = second_fans[j].copy()
            right_pieces[leaf] = shared
            right_bridges[leaf] = _bridge(
                right_strands[leaf].get_end(), shared.get_start()
            )

        self.remove(*second_fans.values())
        self.add(*right_pieces.values())
        self.play(
            *(Create(bridge) for bridge in left_bridges.values()),
            *(Create(bridge) for bridge in right_bridges.values()),
            run_time=1.35,
        )
        self.wait(0.25)

        # --- Contract onto the composite cospan. -----------------------------
        left_shift = RIGHT * CONTRACTION
        right_shift = LEFT * CONTRACTION

        # The composed labels form in step with the move-in, so each label
        # stays centered under the portion of the diagram it names.
        left_reference = Text(
            "g′ ∘ f", color=INK, font=CODE_FONT, font_size=28
        ).scale(scale)
        left_reference.move_to(
            np.array([(COL_S + CONTRACTION + COL_T) / 2, leg_label_y, 0.0])
        )
        right_reference = Text(
            "b ∘ a′", color=INK, font=CODE_FONT, font_size=28
        ).scale(scale)
        right_reference.move_to(
            np.array([(COL_T + COL_U - CONTRACTION) / 2, leg_label_y, 0.0])
        )
        # Text submobjects include space glyphs: "g′ ∘ f" splits into
        # g,′,␣,∘,␣,f and "b ∘ a′" into b,␣,∘,␣,a,′.
        left_symbol = Text("∘", color=INK, font=CODE_FONT, font_size=28)
        left_symbol.scale(scale).move_to(left_reference[3])
        right_symbol = Text("∘", color=INK, font=CODE_FONT, font_size=28)
        right_symbol.scale(scale).move_to(right_reference[2])

        initial_routes = VGroup()
        final_routes = VGroup()
        for s, leaf in surviving.items():
            glued = _join_route(
                first_arrows[s].shaft, left_bridges[s], left_strands[leaf]
            )
            final_shaft = segment(
                s_cells[s].get_right() + left_shift,
                yprime_cells[leaf].get_left(),
            )
            initial, final = _matched_path_pair(glued, final_shaft)
            initial_routes.add(initial)
            final_routes.add(final)
        for leaf in yprime_cells:
            glued = _join_route(
                right_strands[leaf], right_bridges[leaf], right_pieces[leaf]
            )
            owner = composite_right_owner(composite, leaf)
            final_segment = segment(
                yprime_cells[leaf].get_right(),
                u_cells[owner].get_left() + right_shift,
            )
            initial, final = _matched_path_pair(glued, final_segment)
            initial_routes.add(initial)
            final_routes.add(final)

        self.remove(
            *(first_arrows[s].shaft for s in surviving),
            *(first_arrows[s].tail for s in surviving),
            *left_bridges.values(),
            *left_strands.values(),
            *right_strands.values(),
            *right_bridges.values(),
            *right_pieces.values(),
        )
        self.add(initial_routes)
        self.wait(1 / 15)

        self.play(
            Transform(initial_routes, final_routes),
            VGroup(*s_cells, label_source).animate.shift(left_shift),
            VGroup(*u_cells, label_target).animate.shift(right_shift),
            label_f.animate.move_to(left_reference[5]),
            label_a.animate.move_to(VGroup(*left_reference[0:2]).get_center()),
            Write(left_symbol),
            label_b.animate.move_to(right_reference[0]),
            label_g.animate.move_to(VGroup(*right_reference[4:6]).get_center()),
            Write(right_symbol),
            run_time=1.35,
        )
        self.remove(initial_routes)
        self.add(final_routes)
        self.wait(1.0)


def composite_right_owner(composite, leaf: int) -> int:
    """Which composite-codomain cell a Y' leaf belongs to."""
    running = 0
    for k, mode in enumerate(composite.right.modes):
        running += len(mode)
        if leaf < running:
            return k
    raise ValueError(f"Leaf {leaf} outside the composite codomain.")
