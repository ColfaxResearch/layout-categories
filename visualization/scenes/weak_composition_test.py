"""Prototype animation for the weak composition algorithm.

``f : S -> T`` and ``g : U -> V`` are not composable on the nose: ``T`` and ``U``
are different factorizations.  ``weak_composite`` mutually refines them, pulls
``f`` back along the refinement of ``T``, pushes ``g`` forward along the
refinement of ``U``, and composes the three resulting stages through the shared
refinement.  This scene runs those steps in place, on five evenly spaced stacks:

    open:      [S] --f--> [T]   [U] --g--> [V]      (f and g, side by side)
    apart:     [S] --f--> [T]        [U] --g--> [V]  (making room between them)
    refine:    [S] --f--> [T] --> [W] <-- [U] --g--> [V]
    stage 1:   [S] --fans--> [T'] --> [W] <-- [U'] --fans--> [V]
    stage 2:   [S] --fans--> [S'] --f'--> [W] --g'--> [V'] --fans--> [V]
    compose:   [S] --fans--> [S'] --g' o f'--> [V'] --fans--> [V]

``W`` is the mutual refinement: ``mutual_refinement(T, U)`` returns ``T'`` and
``U'``, and the examples are chosen so those have the same leaves, which is what
lets one stack stand for both.  The inclusion between them is then the identity,
exhibited by the two families of parallel arrows that meet at ``W`` after stage 1.

The two halves are the pullback and the pushforward animations played at once,
sharing ``W`` as the refinement they are taken along: on the left ``T`` refines in
place and then reorders into ``S'``, on the right ``U`` refines in place and then
reorders into ``V'``, with the special cases handled as in those scenes -- a
domain mode of ``f`` sent to the basepoint appears in ``S'`` with a fan and no
arrow, a codomain mode outside ``g``'s image appears in ``V'`` the same way, and
a block that nothing is pushed through leaves.

Finally the two-stage route ``S' -> W -> V'`` collapses into the weak composite
and ``W`` retires.  The coalescence ``weak_composite`` ends with is omitted.
"""

from layout_categories_viz.scene_base import LayoutScene

from dataclasses import dataclass

import numpy as np
from manim import (
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    ORIGIN,
    RIGHT,
    ShrinkToCenter,
    Succession,
    Text,
    Transform,
    ValueTracker,
    VGroup,
    smooth,
)
from tract import (
    NestedTuple,
    NestMorphism,
    mutual_refinement,
    weak_composite,
)

from layout_categories_viz.animations import DrawMapstoTip, UndrawMapstoTip
from layout_categories_viz.paths import bridge, join_route, matched_path_pair
from layout_categories_viz.style import CODE_FONT, INK
from layout_categories_viz.tuple_morphism import MapstoArrow

# The library owns the drawing primitives every refinement animation shares,
# so they cannot drift apart.
from layout_categories_viz import stacks
from layout_categories_viz.stacks import (
    ARROW_INSET,
    CELL_H,
    LABEL_FONT_SIZE,
    MAX_STACK_HEIGHT,
    SLOT_STEP,
    leaf_groups,
    make_place,
    revealed,
)
from scenes.tuple_mutual_refinement_test import TupleMorphismRefinementTest


@dataclass(frozen=True)
class WeakCompositionExample:
    """``f : S -> T`` and ``g : U -> V``, in the one-based map convention."""

    domain: tuple
    first_codomain: tuple
    first_map: tuple
    second_domain: tuple
    codomain: tuple
    second_map: tuple


EXAMPLES = (
    # f the transposition of (8, 8), g the three-cycle of (4, 4, 4): the middle
    # tuples factor 64 differently, and W = (4, 2, 2, 4) refines both.
    WeakCompositionExample(
        domain=(8, 8),
        first_codomain=(8, 8),
        first_map=(2, 1),
        second_domain=(4, 4, 4),
        codomain=(4, 4, 4),
        second_map=(3, 1, 2),
    ),
    # T = (4, 6) against U = (2, 4, 3), of different lengths and with distinct
    # entries throughout, so every block is easy to follow: W = (2, 2, 2, 3)
    # splits T into blocks of two and two and U into one, two and one.  f is the
    # transposition and g the three-cycle, and neither projects anything away or
    # leaves a mode unhit.
    WeakCompositionExample(
        domain=(6, 4),
        first_codomain=(4, 6),
        first_map=(2, 1),
        second_domain=(2, 4, 3),
        codomain=(4, 3, 2),
        second_map=(3, 1, 2),
    ),
    # Five shared leaves: T = (48, 3, 2) against U = (4, 3, 12, 2), so
    # W = (4, 3, 4, 3, 2) splits T's first mode three ways and U's third mode in
    # two.  f and g are three- and four-cycles.
    WeakCompositionExample(
        domain=(2, 48, 3),
        first_codomain=(48, 3, 2),
        first_map=(3, 1, 2),
        second_domain=(4, 3, 12, 2),
        codomain=(12, 4, 2, 3),
        second_map=(2, 4, 1, 3),
    ),
    # Six shared leaves: T = (12, 4, 3, 2) against U = (3, 4, 2, 12), each of
    # four modes, refined into blocks of 2/2/1/1 and 1/1/1/3 by
    # W = (3, 4, 2, 2, 3, 2).
    WeakCompositionExample(
        domain=(4, 3, 2, 12),
        first_codomain=(12, 4, 3, 2),
        first_map=(2, 3, 4, 1),
        second_domain=(3, 4, 2, 12),
        codomain=(4, 2, 12, 3),
        second_map=(4, 1, 2, 3),
    ),
    # Seven shared leaves: T = (12, 3, 2, 4) against U = (3, 2, 24, 2), refined
    # into blocks of 3/1/1/2 and 1/1/4/1 by W = (3, 2, 2, 3, 2, 2, 2).  Both
    # halves reorder every block they have.
    WeakCompositionExample(
        domain=(3, 4, 12, 2),
        first_codomain=(12, 3, 2, 4),
        first_map=(2, 4, 1, 3),
        second_domain=(3, 2, 24, 2),
        codomain=(2, 2, 3, 24),
        second_map=(3, 1, 4, 2),
    ),
)

COLUMN_GAP = 3.0
X_S, X_T, X_W, X_U, X_V = ((index - 2) * COLUMN_GAP for index in range(5))
# f and g open one column pitch apart -- as close as any two stacks ever get --
# and then draw apart by this much each to open the gap the mutual refinement
# grows in.
OPENING_SHIFT = COLUMN_GAP / 2
# How far either side of W the composite ends up: one column pitch apart, so
# its arrows span what f's and g's did.
COMPOSITE_HALF_WIDTH = COLUMN_GAP / 2


class WeakCompositionTest(LayoutScene):
    """Refine, pull back, push forward, and compose through the refinement."""

    def construct(self) -> None:
        for index, example in enumerate(EXAMPLES):
            self._show_weak_composite(example)
            self.clear_scene(last=index == len(EXAMPLES) - 1)

    def _show_weak_composite(self, example: WeakCompositionExample) -> None:
        S = NestedTuple(example.domain)
        T = NestedTuple(example.first_codomain)
        U = NestedTuple(example.second_domain)
        V = NestedTuple(example.codomain)
        f = NestMorphism(S, T, example.first_map)
        g = NestMorphism(U, V, example.second_map)
        Tprime, Uprime = mutual_refinement(T, U)
        if Tprime.flatten() != Uprime.flatten():
            raise ValueError(
                "one stack stands for the mutual refinement, so this scene "
                "needs T' and U' to have the same leaves"
            )
        fprime = f.pullback_along(Tprime)
        gprime = g.pushforward_along(Uprime)
        Sprime, Vprime = fprime.domain, gprime.codomain
        weak_composite(f, g)  # validates the whole construction

        shared = Tprime.flatten()
        s_groups = leaf_groups(Sprime, S)
        v_groups = leaf_groups(Vprime, V)

        # Geometry: one bottom baseline, one uniform step, scaled to fit.
        slots = max(
            len(shared), Sprime.length(), Vprime.length(), S.length(), V.length()
        )
        scale = min(1.0, MAX_STACK_HEIGHT / ((slots - 1) * SLOT_STEP + CELL_H))
        step = SLOT_STEP * scale
        baseline = -((slots - 1) * step) / 2

        place = make_place(baseline, step)

        def cell(value, center):
            return (
                stacks.cell(value, ORIGIN)
                .scale(scale)
                .move_to(center)
            )

        def label(text, anchor):
            return (
                Text(
                    text, color=INK, font=CODE_FONT, font_size=LABEL_FONT_SIZE
                )
                .scale(scale)
                .next_to(anchor, DOWN, buff=0.3)
            )

        segment = stacks.tree_segment

        # --- Open: f and g, with a gap between them for the refinement. ------
        s_cells = [
            cell(S.entry(mode + 1), place(X_S, mode))
            for mode in range(S.length())
        ]
        t_cells = [
            cell(T.entry(mode + 1), place(X_T, mode))
            for mode in range(T.length())
        ]
        u_cells = [
            cell(U.entry(mode + 1), place(X_U, mode))
            for mode in range(U.length())
        ]
        v_cells = [
            cell(V.entry(mode + 1), place(X_V, mode))
            for mode in range(V.length())
        ]
        f_arrows = {
            mode: stacks.segment_arrow(
                s_cells[mode], t_cells[target - 1]
            )
            for mode, target in enumerate(example.first_map)
            if target
        }
        g_arrows = {
            mode: stacks.segment_arrow(
                u_cells[mode], v_cells[target - 1]
            )
            for mode, target in enumerate(example.second_map)
            if target
        }
        label_s = label("S", s_cells[0])
        label_t = label("T", t_cells[0])
        label_u = label("U", u_cells[0])
        label_v = label("V", v_cells[0])

        # Everything is built where it ends up, then brought in closer together:
        # the two morphisms open side by side and draw apart to make room.
        opening = (
            *s_cells,
            *t_cells,
            *u_cells,
            *v_cells,
            *f_arrows.values(),
            *g_arrows.values(),
            label_s,
            label_t,
            label_u,
            label_v,
        )
        left_half = VGroup(
            *s_cells, *t_cells, *f_arrows.values(), label_s, label_t
        )
        right_half = VGroup(
            *u_cells, *v_cells, *g_arrows.values(), label_u, label_v
        )
        left_half.shift(RIGHT * OPENING_SHIFT)
        right_half.shift(LEFT * OPENING_SHIFT)

        self.play(
            *(FadeIn(mobject) for mobject in opening), run_time=1.0
        )
        self.wait(0.7)

        # --- Apart: the room the mutual refinement needs. --------------------
        self.play(
            left_half.animate.shift(LEFT * OPENING_SHIFT),
            right_half.animate.shift(RIGHT * OPENING_SHIFT),
            run_time=0.9,
            rate_func=smooth,
        )
        self.wait(0.4)

        # --- Refine: grow the mutual refinement between T and U. -------------
        w_cells = []
        refinement_segs = []
        for source, target, source_value, _, factor in (
            TupleMorphismRefinementTest._refinement_steps(
                T.flatten(), U.flatten()
            )
        ):
            entry = cell(factor, place(X_W, len(w_cells)))
            # Each segment grows from the stack it divides toward the middle, so
            # the two sides of the refinement meet at the cell they share.
            growing = []
            if source_value is not None:
                growing.append(
                    segment(t_cells[source].get_right(), entry.get_left())
                )
            # Built toward U, as the connector that replaces it is, but
            # traced backwards so it grows from U in toward the middle.
            growing.append(
                segment(entry.get_right(), u_cells[target].get_left())
                .reverse_points()
            )
            self.play(
                FadeIn(entry, shift=DOWN * 0.08),
                *(Create(connector) for connector in growing),
                run_time=0.8,
                rate_func=smooth,
            )
            w_cells.append(entry)
            refinement_segs.extend(growing)
        self.wait(0.9)

        # --- Stage 1: both middle stacks refine in place, neither reorders. ---
        self.play(
            *(
                UndrawMapstoTip(arrow.tip)
                for arrow in (*f_arrows.values(), *g_arrows.values())
            ),
            run_time=0.4,
        )

        # For the left half the outer cell of a T mode is the S mode f sends to
        # it; for the right half it is the V mode g sends that U mode to.
        left_outer = [None] * T.length()
        for mode, target in enumerate(example.first_map):
            if target:
                left_outer[target - 1] = s_cells[mode]
        right_outer = [
            v_cells[target - 1] if target else None
            for target in example.second_map
        ]

        alpha = ValueTracker(0.0)
        left = self._refine_in_place(
            coarse=T,
            refined=Tprime,
            coarse_cells=t_cells,
            column=X_T,
            outer_cells=left_outer,
            shared_cells=w_cells,
            shared_on_right=True,
            alpha=alpha,
            place=place,
            cell=cell,
        )
        right = self._refine_in_place(
            coarse=U,
            refined=Uprime,
            coarse_cells=u_cells,
            column=X_U,
            outer_cells=right_outer,
            shared_cells=w_cells,
            shared_on_right=False,
            alpha=alpha,
            place=place,
            cell=cell,
        )

        # Every replacement starts out coincident with what it replaces.
        self.remove(
            *f_arrows.values(),
            *g_arrows.values(),
            *refinement_segs,
            *t_cells,
            *u_cells,
        )
        for half in (left, right):
            self.add(
                *half["fans"].values(),
                *half["inner"].values(),
                *stacks.deck(half["cells"]),
            )

        label_tprime = label("T'", cell(shared[0], place(X_T, 0)))
        label_uprime = label("U'", cell(shared[0], place(X_U, 0)))
        self.play(
            alpha.animate.set_value(1.0),
            FadeOut(label_t),
            FadeOut(label_u),
            FadeIn(label_tprime),
            FadeIn(label_uprime),
            run_time=1.4,
            rate_func=smooth,
        )
        for half in (left, right):
            for split in half["cells"].values():
                split.clear_updaters()

        # The connectors either side of W are parallel now.  They take tips, and
        # together they exhibit the inclusion of the mutual refinement: T' into
        # W on the left, W into U' on the right.
        left["tips"] = {
            leaf: stacks.arrow_tip(
                w_cells[leaf].get_left() + LEFT * ARROW_INSET
            )
            for leaf in left["cells"]
        }
        right["tips"] = {
            leaf: stacks.arrow_tip(
                right["cells"][leaf].get_left() + LEFT * ARROW_INSET
            )
            for leaf in right["cells"]
        }
        self.add(*left["tips"].values(), *right["tips"].values())
        self.play(
            *(
                DrawMapstoTip(tip)
                for tip in (*left["tips"].values(), *right["tips"].values())
            ),
            run_time=0.4,
        )
        # The right half's tips land on cells that are about to move.
        for leaf, tip in right["tips"].items():
            tip.add_updater(
                lambda mobject, leaf=leaf: mobject.become(
                    stacks.arrow_tip(
                        right["cells"][leaf].get_left() + LEFT * ARROW_INSET
                    )
                )
            )
        self.wait(0.9)

        # --- Stage 2: both middle stacks reorder. ----------------------------
        left_destination = {
            fprime.map[index] - 1: index
            for index in range(Sprime.length())
            if fprime.map[index]
        }
        right_destination = {
            leaf: target - 1
            for leaf, target in enumerate(gprime.map)
            if target
        }
        reorder = []
        retiring = []
        for half, column, destination in (
            (left, X_T, left_destination),
            (right, X_U, right_destination),
        ):
            for leaf, split in half["cells"].items():
                if leaf in destination:
                    reorder.append(
                        split.animate.move_to(place(column, destination[leaf]))
                    )
                else:
                    retiring.append(self._retire(half, leaf))
            # Draw each stack bottom to top for the reordering too, by the slots
            # its cells are heading for, so blocks passing one another always
            # cover in the same direction, as they did while splitting.
            self.add(
                *stacks.deck(
                    {
                        destination[leaf]: split
                        for leaf, split in half["cells"].items()
                        if leaf in destination
                    }
                )
            )

        # A mode f projects away has no block in W to reorder, and a mode of V
        # outside g's image has none pushed into it: both appear in the slot the
        # reordering leaves free, with a fan drawing in and no arrow.
        arrivals = []
        left_joined = []
        right_joined = []
        for mode, target in enumerate(example.first_map):
            if target:
                continue
            for leaf in s_groups[mode]:
                joined = cell(Sprime.flatten()[leaf], place(X_T, leaf))
                connector = segment(
                    s_cells[mode].get_right(), joined.get_left()
                )
                left_joined.extend((joined, connector))
                arrivals.append(self._arrival(joined, connector))
        image = set(example.second_map)
        for mode in range(V.length()):
            if mode + 1 in image:
                continue
            for leaf in v_groups[mode]:
                joined = cell(Vprime.flatten()[leaf], place(X_U, leaf))
                # Built toward V, like every other fan of the refinement of V,
                # but traced backwards so it draws in toward the cell.
                connector = segment(
                    joined.get_right(), v_cells[mode].get_left()
                ).reverse_points()
                right_joined.extend((joined, connector))
                arrivals.append(self._arrival(joined, connector))

        label_sprime = label("S'", cell(Sprime.flatten()[0], place(X_T, 0)))
        label_vprime = label("V'", cell(Vprime.flatten()[0], place(X_U, 0)))
        self.play(
            *reorder,
            *(FadeOut(group) for group in retiring),
            *arrivals,
            FadeOut(label_tprime),
            FadeOut(label_uprime),
            FadeIn(label_sprime),
            FadeIn(label_vprime),
            run_time=1.4,
            rate_func=smooth,
        )
        for half in (left, right):
            for mobject in (
                *half["fans"].values(),
                *half["inner"].values(),
                *half["tips"].values(),
            ):
                mobject.clear_updaters()
        self.wait(1.2)

        # --- Compose the two stages through W. -------------------------------
        # A route survives only where f' and g' both carry the leaf.  Both
        # halves key their cells by that leaf of W: the left one has come to
        # rest in its slot of S', the right one in its slot of V'.
        routes = [
            fprime.map[index] - 1
            for index in range(Sprime.length())
            if fprime.map[index] and gprime.map[fprime.map[index] - 1]
        ]
        routed = set(routes)
        bridges = [
            bridge(
                left["inner"][leaf].get_end(),
                right["inner"][leaf].get_start(),
            )
            for leaf in routes
        ]
        stray = [
            self._retire(half, leaf)
            for half, destination in (
                (left, left_destination),
                (right, right_destination),
            )
            for leaf in half["cells"]
            if leaf in destination and leaf not in routed
        ]
        self.play(
            ShrinkToCenter(VGroup(*w_cells)),
            *(UndrawMapstoTip(left["tips"][leaf]) for leaf in routes),
            *(Create(bridge) for bridge in bridges),
            *(FadeOut(group) for group in stray),
            run_time=1.0,
        )

        # Contract the halves toward the middle and deform each route into the
        # single arrow of the weak composite.
        left_shift = np.array((X_W - COMPOSITE_HALF_WIDTH - X_T, 0.0, 0.0))
        right_shift = np.array((X_W + COMPOSITE_HALF_WIDTH - X_U, 0.0, 0.0))
        curved = VGroup()
        contracted = VGroup()
        for leaf, bridge in zip(routes, bridges):
            route = join_route(
                left["inner"][leaf], bridge, right["inner"][leaf]
            )
            arrow = stacks.segment_arrow(
                left["cells"][leaf].copy().shift(left_shift),
                right["cells"][leaf].copy().shift(right_shift),
            )
            initial, final = matched_path_pair(route, arrow.shaft)
            curved.add(
                MapstoArrow.from_parts(
                    arrow.tail.copy(), initial, right["tips"][leaf].copy()
                )
            )
            contracted.add(
                MapstoArrow.from_parts(arrow.tail, final, arrow.tip)
            )
        for leaf in routes:
            self.remove(
                left["inner"][leaf],
                left["tips"][leaf],
                right["inner"][leaf],
                right["tips"][leaf],
            )
        self.remove(*bridges)
        self.add(curved)

        left_half = VGroup(
            *s_cells,
            *left["cells"].values(),
            *left["fans"].values(),
            *left_joined,
            label_s,
            label_sprime,
        )
        right_half = VGroup(
            *v_cells,
            *right["cells"].values(),
            *right["fans"].values(),
            *right_joined,
            label_v,
            label_vprime,
        )
        self.play(
            Transform(curved, contracted),
            left_half.animate.shift(left_shift),
            right_half.animate.shift(right_shift),
            run_time=1.4,
        )
        self.remove(curved)
        self.add(contracted)
        self.wait(2.2)

    # ------------------------------------------------------------------ steps
    def _refine_in_place(
        self,
        *,
        coarse,
        refined,
        coarse_cells,
        column,
        outer_cells,
        shared_cells,
        shared_on_right,
        alpha,
        place,
        cell,
    ):
        """Split one middle stack in place, as stage 1 of either half does.

        ``shared_on_right`` picks the side ``W`` is on: the pullback's middle
        stack has it on the right and its outer connectors are ``f``'s arrows
        turning into fans, the pushforward's has it on the left and its outer
        connectors are ``g``'s.  ``outer_cells[mode]`` is the cell the outer
        connector of that middle mode joins, or ``None`` if it has none.
        """
        attached = stacks.attached
        values = refined.flatten()
        cells, fans, inner = {}, {}, {}
        for mode, leaves in enumerate(leaf_groups(refined, coarse)):
            start = coarse_cells[mode].get_center()
            outer = outer_cells[mode]
            for index, leaf in enumerate(leaves):
                # The first cell of a mode stands in for the cell it came from,
                # and sheds its value; the rest are stacked under it, already
                # carrying their own.
                split = cell(values[leaf], start)
                if not index:
                    coarse_label = cell(coarse.entry(mode + 1), start)[1]
                    split[1].set_opacity(0)
                    split.add(coarse_label)
                stacks.split_cell(
                    split,
                    alpha,
                    start,
                    place(column, leaf),
                    peeled=bool(index),
                )
                cells[leaf] = split
                # Toward W: one connector per leaf, already one to one.
                inner[leaf] = (
                    attached(split, shared_cells[leaf])
                    if shared_on_right
                    else attached(shared_cells[leaf], split)
                )
                # Away from W: the arrow of f or g, splitting into a fan, whose
                # copies coincide until the cells separate.
                if outer is not None:
                    reveal = (
                        (lambda: revealed(alpha.get_value()))
                        if index
                        else None
                    )
                    fans[leaf] = (
                        attached(outer, split, reveal=reveal)
                        if shared_on_right
                        else attached(split, outer, reveal=reveal)
                    )
        return {"cells": cells, "fans": fans, "inner": inner, "tips": {}}

    @staticmethod
    def _retire(half, leaf) -> VGroup:
        """Everything that leaves with one cell of a middle stack."""
        parts = [half["cells"][leaf], half["inner"][leaf], half["tips"][leaf]]
        if leaf in half["fans"]:
            parts.append(half["fans"][leaf])
        for part in parts:
            part.clear_updaters()
        return VGroup(*parts)

    @staticmethod
    def _arrival(joined, connector) -> Succession:
        """A cell appearing in its slot with its fan drawing in toward it."""
        return Succession(
            FadeIn(joined, run_time=0.45),
            Create(connector, run_time=0.95),
        )

