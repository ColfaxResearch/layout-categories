"""Prototype animation for composition in the category Span.

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
    stage 1:   [S] <-a- [X] -fans-> [Y] -parallel-> [Y] -g-> [U]
    stage 2:   [S] <-a- [X] <-b'- [X'] -f'-> [Y] -g-> [U]
    compose:   [S] <---a ∘ b'--- [X'] ---g ∘ f'---> [U]

Stages 1 and 2 are the pullback animation verbatim, played in the middle of
the chain: the middle stack splits in place along b (f's arrows giving up
their carets and fanning out, the strands to Y becoming parallel and taking
carets), then its blocks reorder into the order of X'.  The morphism data
migrates in place -- f's arrows become b′'s fans and b's fans become f′'s
arrows -- so each leg label transforms where it stands.  With ``single_beat``
set (see span_composition_single_beat_test.py) the pullback plays as ONE
beat instead of two: every middle cell splits and its pieces travel straight
to their X' slots, so the split and the reordering are a single motion;
blocks with no place in X' fade unsplit, and the basepoint arrivals join the
same beat.  After the pullback the chain reads Fact, Fact, Tuple, Tuple: the
two Fact legs compose as X dissolves and the two tuple legs compose as Y
dissolves, exactly the two composition collapses this family already has,
and the outer stacks contract onto the composite span.

The special cases are the pullback scene's: an apex cell sent to the
basepoint arrives in X' with a fan and no arrow, a T cell outside f's image
retires with its block, and a Y cell sent to the basepoint by g takes its
route with it when Y dissolves.  The library supplies every stack and map:
``pullback_with_refinement`` gives X', b′ and f′, ``SpanMorphism.compose`` the
picture being landed on.
"""

from layout_categories_viz.scene_base import LayoutScene

from dataclasses import dataclass
from math import prod

from manim import (
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    LEFT,
    RIGHT,
    ShrinkToCenter,
    Transform,
    Uncreate,
    ValueTracker,
    VGroup,
    Write,
    smooth,
)
from tract import FactMorphism, SpanMorphism, TupleMorphism

from layout_categories_viz.animations import (
    DrawMapstoTip,
    TailToTipMapsto,
    UndrawMapstoTip,
)

# The composition scene owns the route-surgery helpers, the pullback scene
# the drawing primitives and the two-stage gesture, and the weak-composition
# scene the bridging, so this scene cannot drift from any of them.
from layout_categories_viz.paths import matched_path_pair
from layout_categories_viz import stacks
from layout_categories_viz.stacks import ARROW_INSET
from layout_categories_viz.paths import bridge, join_route
from scenes.span_common import (
    COL_S,
    COL_T,
    COL_U,
    COL_X,
    COL_Y,
    CONTRACTION,
    build_basepoint_arrivals,
    build_middle_splits,
    composition_labels,
    owner_of_leaf,
    span_geometry,
)


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
        first = SpanMorphism(
            FactMorphism(apex, domain, self.first_modes),
            TupleMorphism(apex, self.middle, self.first_map),
        )
        nadir = tuple(y for mode in self.second_modes for y in mode)
        second = SpanMorphism(
            FactMorphism(nadir, self.middle, self.second_modes),
            TupleMorphism(nadir, self.codomain, self.second_map),
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


class SpanMorphismCompositionTest(LayoutScene):
    """Pull the middle Tuple past the middle Fact, then compose like pairs."""

    # With single_beat set the pullback plays as one beat instead of two:
    # the split and the reordering are a single motion.
    single_beat: bool = False

    def construct(self) -> None:
        for index, example in enumerate(EXAMPLES):
            self._play_example(example)
            self.clear_scene(last=index == len(EXAMPLES) - 1)

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

        geometry = span_geometry(
            len(first.domain),
            len(first.apex),
            len(second.apex),
            len(refinement.domain),
            len(composite.codomain),
        )
        place = geometry.place
        cell = geometry.cell
        stack_label = geometry.stack_label
        leg_label = geometry.leg_label

        segment = stacks.tree_segment

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
            i: stacks.segment_arrow(x_cells[i], v_cells[j - 1])
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
            m: stacks.segment_arrow(y_cells[m], w_cells[k - 1])
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

        # --- The pullback.  Two beats: the middle splits in place along the --
        # second backward leg (f₁'s arrows give up their carets and fan out
        # into the blocks, the strands to Y become parallel and take
        # carets), then its blocks reorder into the order of X'.  One beat
        # (``single_beat``): every middle cell splits and its pieces travel
        # straight to their X' slots, so the split and the reordering are
        # one motion; a T cell outside f's image fades with its block
        # unsplit, and the basepoint arrivals join the same beat.
        self.play(
            *(UndrawMapstoTip(arrow.tip) for arrow in first_arrows.values()),
            run_time=0.4,
        )

        alpha = ValueTracker(0.0)
        # T cells outside f's image, whose blocks have no place in X'.
        retiring_modes = [
            mode
            for mode in range(len(first.codomain))
            if mode not in source_apex
        ]
        middle_cells, middle_fans, connectors = build_middle_splits(
            first=first,
            second=second,
            y_groups=y_groups,
            v_cells=v_cells,
            x_cells=x_cells,
            y_cells=y_cells,
            source_apex=source_apex,
            alpha=alpha,
            cell=cell,
            place=place,
            skip_modes=retiring_modes if self.single_beat else (),
            slot_of_leaf=destination.__getitem__ if self.single_beat else None,
        )

        # An apex cell sent to the basepoint has no block to receive: it
        # carries its own value into X', arriving with a fan and no arrow.
        arrival_cells, arrival_fans, arrivals = build_basepoint_arrivals(
            first=first,
            refinement=refinement,
            xprime_groups=xprime_groups,
            x_cells=x_cells,
            cell=cell,
            place=place,
        )

        if self.single_beat:
            # Every replacement starts out coincident with what it replaces;
            # the blocks with no place in X' keep their originals, to fade
            # unsplit.
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
                *stacks.deck(
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
                leaf: stacks.arrow_tip(
                    y_cells[leaf].get_left() + LEFT * ARROW_INSET
                )
                for leaf in middle_cells
            }
            self.add(*tips.values())
            self.play(
                *(DrawMapstoTip(tip) for tip in tips.values()), run_time=0.4
            )
        else:
            # Every replacement starts out coincident with what it replaces.
            # The middle is becoming a second copy of Y, cell for cell.
            self.remove(
                *v_cells,
                *second_fans.values(),
                *(arrow.shaft for arrow in first_arrows.values()),
                *(arrow.tail for arrow in first_arrows.values()),
            )
            self.add(*middle_fans.values(), *connectors.values())
            self.add(*stacks.deck(middle_cells))
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

            # The parallel connectors take carets: they are becoming f₁′,
            # whose carets they keep through the reordering.
            tips = {
                leaf: stacks.arrow_tip(
                    y_cells[leaf].get_left() + LEFT * ARROW_INSET
                )
                for leaf in middle_cells
            }
            self.add(*tips.values())
            self.play(
                *(DrawMapstoTip(tip) for tip in tips.values()), run_time=0.4
            )
            self.wait(0.6)

            # --- Stage 2: reorder the blocks into the order of X'.  A V ------
            # cell outside f's image has no place in X', so its block
            # leaves.  The morphism data migrates in place: f's arrows have
            # become b′'s fans and b's fans have become f′'s arrows, so each
            # label transforms where it stands.
            retiring = [
                leaf for leaf in middle_cells if leaf not in destination
            ]
            for leaf in retiring:
                connectors[leaf].clear_updaters()
            self.add(*(middle_cells[leaf] for leaf in retiring))
            self.add(
                *stacks.deck(
                    {
                        destination[leaf]: middle_cells[leaf]
                        for leaf in middle_cells
                        if leaf in destination
                    }
                )
            )
            reorder = [
                middle_cells[leaf].animate.move_to(
                    place(COL_T, destination[leaf])
                )
                for leaf in middle_cells
                if leaf in destination
            ] + [
                FadeOut(
                    VGroup(middle_cells[leaf], connectors[leaf], tips[leaf])
                )
                for leaf in retiring
            ]

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
            left_bridges[leaf] = bridge(
                shared.get_end(), left_strands[leaf].get_start()
            )
        right_bridges = {}
        for leaf, m in surviving.items():
            right_bridges[leaf] = bridge(
                right_strands[leaf].get_end(),
                second_arrows[m].shaft.get_start(),
            )

        # The adjacent Fact labels slide together, and so do the Tuples',
        # landing mid-gap of the contracted span.  "a ∘ b′" splits into
        # a,␣,∘,␣,b,′ and "g ∘ f′" into g,␣,∘,␣,f,′.
        left_reference, right_reference, left_symbol, right_symbol = (
            composition_labels(
                "a ∘ b′",
                "g ∘ f′",
                scale=geometry.scale,
                leg_label_y=geometry.leg_label_y,
                left_symbol_index=2,
                right_symbol_index=2,
            )
        )

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
            glued = join_route(
                left_pieces[leaf], left_bridges[leaf], left_strands[leaf]
            )
            owner = owner_of_leaf(composite.left.modes, leaf)
            final_segment = segment(
                u_cells[owner].get_right() + left_shift,
                xprime_cells[leaf].get_left(),
            )
            initial, final = matched_path_pair(glued, final_segment)
            initial_routes.add(initial)
            final_routes.add(final)

        surviving_arrows = {
            leaf: second_arrows[m] for leaf, m in surviving.items()
        }
        for leaf, arrow in surviving_arrows.items():
            glued = join_route(
                right_strands[leaf], right_bridges[leaf], arrow.shaft
            )
            target = composite.right.map[leaf]
            final_shaft = segment(
                xprime_cells[leaf].get_right(),
                w_cells[target - 1].get_left() + right_shift,
            )
            initial, final = matched_path_pair(glued, final_shaft)
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
