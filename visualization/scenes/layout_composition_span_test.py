"""Prototype animation for layout composition in span language.

Two tractable layouts A = S : D_A and B = U : D_B are composed by running
weak composition at the level of the span data each layout *is*:

    A  =  the RefSpan morphism   ρ(S) <-nest(S)- flat(S) --f--> flat(T)
    B  =  the Tuple morphism     flat(U) --g--> flat(V)

A's backward Ref leg is the nesting of its shape, drawn as the layout
depiction's mirrored banded tree; its forward Tuple leg is read off the
strides, each stride naming a position in the sorted factorization
flat(T).  B needs less: the composite inherits A's shape, so B's own
nesting is scaffolding — it is discarded at the encode, leaving only the
flat Tuple morphism g.  The scene runs the algorithm in place, on six
columns:

    open:     [ρ(S)]=[S]:[D_A]        [ρ(U)]=[U]:[D_B]
    encode:   [ρ(S)]=[S] --f--> [T]        [U] --g--> [V]
                                        (strides unfold, B's tree retires)
    bridge:   [ρ(S)]=[S] --f--> [T] --> [W] <-- [U] --g--> [V]
    refine:   [ρ(S)]=[S] -fans-> [T'] --> [W] <-- [U'] -fans-> [V]
    reorder:  [ρ(S)]=[S] -fans-> [S'] -f'-> [W] -g'-> [V'] -fans-> [V]
    compose:  [ρ(S)]=[S] -fans-> [S'] --g' ∘ f'--> [V'] -fans-> [V]
    read off: [ρ(S)]=[S] -fans-> [S'] : [D']

``W`` is the mutual refinement, packaged as the bridge b, ι, c: the two
middle beats are the weak-composition scene's gestures, but every stack
and map is read off one RefSpan composition (the pullback of f along b —
the fans grafted onto A's tree are the backward legs composing in Ref)
and one RefCoSpan composition (the pushforward of g along c).  Unlike the
morphism-level scene, dead routes keep their cells: an S' mode whose
route dies is a domain mode of the composite with stride 0, and a V'
mode nothing reaches still counts toward the prefix products.

The read-off is the layout-extraction scene's collapse verbatim: V and
its fans retire (they are c and V, scaffolding of B), the prefix products
of flat(V') appear beside V', and every composite arrow extends into a
grabber and pulls its stride home beside S'.  What remains — ρ(S), A's
tree, the grafted fans, S', colons, strides — is the nested layout
depiction of B ∘ A, which slides to center.  The final coalescence back
to the profile of S is omitted, as in the weak-composition scene.
"""

from dataclasses import dataclass

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    ShrinkToCenter,
    Text,
    Transform,
    ValueTracker,
    VGroup,
    smooth,
)
from tract import (
    NestedTuple,
    NestMorphism,
    RefCoSpanMorphism,
    RefMorphism,
    RefSpanMorphism,
    TupleMorphism,
    mutual_refinement,
    weak_composite,
)
from tract.backends.base import flat_layout_components

from layout_categories_viz import stacks
from layout_categories_viz.animations import (
    DrawMapstoTip,
    TailToTipMapsto,
    UndrawMapstoTip,
)
from layout_categories_viz.layouts import fitted_cell, layout_colon, layout_diagram
from layout_categories_viz.paths import bridge, join_route, matched_path_pair
from layout_categories_viz.scene_base import LayoutScene  # noqa: F401
from layout_categories_viz.stacks import (
    ARROW_INSET,
    CELL_H,
    LABEL_FONT_SIZE,
    MAX_STACK_HEIGHT,
    SLOT_STEP,
    leaf_groups,
    make_place,
)
from layout_categories_viz.style import CODE_FONT, INK, SHAPE_FILL, STRIDE_FILL
from layout_categories_viz.tuple_morphism import MapstoArrow
from scenes.layout_from_tuple_morphism_test import GrabberPull, GrabberReach
from scenes.tuple_morphism_to_flat_layout import prefix_products
from scenes.tuple_mutual_refinement_test import TupleMorphismRefinementTest
from scenes.weak_composition_test import WeakCompositionTest


@dataclass(frozen=True)
class LayoutCompositionExample:
    """Layouts A and B, given as the span data they encode.

    ``shape_a`` is A's nested shape S; ``first_codomain``/``first_map`` are
    the forward leg f: flat(S) -> flat(T) its strides present.  ``shape_b``
    is B's nested shape (discarded at the encode beyond its flattening U)
    and ``codomain``/``second_map`` the Tuple morphism g: flat(U) -> flat(V).
    Both layouts' strides are computed, never restated.
    """

    shape_a: tuple
    first_codomain: tuple
    first_map: tuple
    shape_b: tuple
    codomain: tuple
    second_map: tuple


EXAMPLES = (
    # A = ((6,4)):((4,1)) against B = ((2,4),3):(12,1,4): T = (4,6) and
    # U = (2,4,3) factor 24 differently, W = (2,2,2,3) refines both, and
    # every mode is mapped and hit, so every block survives to the read-off.
    LayoutCompositionExample(
        shape_a=((6, 4),),
        first_codomain=(4, 6),
        first_map=(2, 1),
        shape_b=((2, 4), 3),
        codomain=(4, 3, 2),
        second_map=(3, 1, 2),
    ),
    # A = ((3,5),48):((48,0),1) against B = ((4,3),(12,2)):(504,0,3,252):
    # T = (48,3,2) against U = (4,3,12,2), W = (4,3,4,3,2).  Every special
    # case at once: f sends a mode to the basepoint (its stride 0 rides
    # through to the composite), g sends a mode to the basepoint (its
    # block retires, killing a route — that S' mode keeps its cell and
    # picks up stride 0), g misses codomain modes (they still count toward
    # the prefix products), and one T entry is outside f's image, so its
    # block retires.
    LayoutCompositionExample(
        shape_a=((3, 5), 48),
        first_codomain=(48, 3, 2),
        first_map=(2, 0, 1),
        shape_b=((4, 3), (12, 2)),
        codomain=(3, 12, 7, 2, 4),
        second_map=(5, 0, 2, 4),
    ),
)

# Six columns: A's roots, A's shape (the apex flat(S)), the factorization
# flat(T), the mutual refinement W, B's shape flat(U), and flat(V).  B's
# roots stand where W will grow, and retire before it does.  The layouts'
# stride columns live inside the S-T and U-V gaps.
X_ROOT, X_S, X_T, X_W, X_U, X_V = -6.2, -4.5, -2.1, 0.0, 2.1, 4.5
PAIR_GAP = 1.2
# How far either side of W the composite lands, exactly as in the
# weak-composition scene.
COMPOSITE_HALF_WIDTH = 1.5


class LayoutCompositionSpanTest(WeakCompositionTest):
    """Encode, bridge, compose in RefSpan and RefCoSpan, read the layout off."""

    def construct(self) -> None:
        for index, example in enumerate(EXAMPLES):
            self._play_example(example)
            self.clear_scene(last=index == len(EXAMPLES) - 1)

    def _play_example(self, example: LayoutCompositionExample) -> None:
        # --- The span data, read off the library. -----------------------------
        S = NestedTuple(example.shape_a)
        forward_a = TupleMorphism(
            S.flatten(), example.first_codomain, example.first_map
        )
        shape_b = NestedTuple(example.shape_b)
        forward_b = TupleMorphism(
            shape_b.flatten(), example.codomain, example.second_map
        )
        T = NestedTuple(forward_a.codomain)
        U = NestedTuple(forward_b.domain)

        Tprime, Uprime = mutual_refinement(T, U)
        if Tprime.flatten() != Uprime.flatten():
            raise ValueError(
                "one stack stands for the mutual refinement, so this scene "
                "needs T' and U' to have the same leaves"
            )
        leg_b = RefMorphism(Tprime.relative_flattening(T))
        leg_c = RefMorphism(Uprime.relative_flattening(U))
        inclusion = TupleMorphism(
            Tprime.flatten(), Uprime.flatten(),
            tuple(range(1, Tprime.length() + 1)),
        )

        # The whole algorithm: one RefSpan composition, one RefCoSpan
        # composition, and the layout read off the outer legs.
        left_span = RefSpanMorphism(RefMorphism(S), forward_a).compose(
            RefSpanMorphism(leg_b, inclusion)
        )
        total = RefCoSpanMorphism(left_span.right, leg_c).compose(
            RefCoSpanMorphism(
                forward_b, RefMorphism.identity(forward_b.codomain)
            )
        )
        composite = total.left  # flat(S') -> flat(V')
        shape_prime = left_span.left.nest  # S', the composite layout's shape

        # Cross-check against the weak composition of the Nest morphisms.
        expected = weak_composite(
            NestMorphism(S, T, forward_a.map),
            NestMorphism(U, NestedTuple(forward_b.codomain), forward_b.map),
        )
        assert expected.map == composite.map
        assert expected.domain == shape_prime

        # The two halves the beats animate, exactly the compositions' insides.
        refinement_s, pulled = leg_b.pullback_with_refinement(forward_a)
        refinement_v, pushed = leg_c.pushforward_with_refinement(forward_b)
        strides_a = flat_layout_components(forward_a)[1]
        strides_b = flat_layout_components(forward_b)[1]
        strides = flat_layout_components(composite)[1]

        shared = Tprime.flatten()
        s_groups = leaf_groups(refinement_s.nest, NestedTuple(forward_a.domain))
        v_groups = leaf_groups(refinement_v.nest, NestedTuple(forward_b.codomain))

        # Geometry: one bottom baseline, one uniform step, scaled to fit.
        slots = max(
            len(shared),
            len(pulled.domain),
            len(pushed.codomain),
            len(forward_a.domain),
            len(forward_b.codomain),
            len(forward_b.domain),
        )
        scale = min(1.0, MAX_STACK_HEIGHT / ((slots - 1) * SLOT_STEP + CELL_H))
        step = SLOT_STEP * scale
        place = make_place(-((slots - 1) * step) / 2, step)

        def cell(value, center):
            # Every non-stride entry shares the shape fill: mid-stream
            # stacks are factorizations of shape entries, and the layout
            # read-off should find the shape side already in its color.
            return (
                stacks.cell(value, np.zeros(3), fill=SHAPE_FILL)
                .scale(scale)
                .move_to(center)
            )

        def label(text, anchor):
            return (
                Text(text, color=INK, font=CODE_FONT, font_size=LABEL_FONT_SIZE)
                .scale(scale)
                .next_to(anchor, DOWN, buff=0.3)
            )

        segment = stacks.tree_segment

        # --- Open: the two layouts, as their nested depictions. ---------------
        diagram_a = layout_diagram(
            S,
            S.sub(strides_a),
            place,
            scale=scale,
            shape_x=X_S,
            stride_x=X_S + PAIR_GAP,
            root_x=X_ROOT,
        )
        diagram_b = layout_diagram(
            shape_b,
            shape_b.sub(strides_b),
            place,
            scale=scale,
            shape_x=X_U,
            stride_x=X_U + PAIR_GAP,
            root_x=X_W,
        )
        s_cells = list(diagram_a.shape_cells)
        u_cells = list(diagram_b.shape_cells)
        label_a = label("A", VGroup(diagram_a.root_cells[0], s_cells[0]))
        label_b = label("B", VGroup(diagram_b.root_cells[0], u_cells[0]))

        self.play(
            FadeIn(diagram_a.cells()),
            FadeIn(diagram_a.colons),
            FadeIn(diagram_b.cells()),
            FadeIn(diagram_b.colons),
            FadeIn(label_a),
            FadeIn(label_b),
            run_time=1.0,
        )
        self.play(
            LaggedStart(
                *(
                    Create(strand)
                    for tree in (*diagram_a.trees, *diagram_b.trees)
                    for strand in tree
                ),
                lag_ratio=0.12,
                run_time=1.2,
            )
        )
        self.wait(0.7)

        # --- Encode A: the strides unfold into the factorization flat(T). ----
        # Each stride names a position in flat(T); the stride column gives
        # way to the factorization stack and the forward leg's arrows.
        t_cells = [
            cell(value, place(X_T, index))
            for index, value in enumerate(forward_a.codomain)
        ]
        f_arrows = {
            mode: stacks.segment_arrow(s_cells[mode], t_cells[target - 1])
            for mode, target in enumerate(forward_a.map)
            if target
        }
        label_t = label("T", t_cells[0])
        self.play(
            FadeOut(diagram_a.colons),
            FadeOut(VGroup(*diagram_a.stride_cells)),
            FadeOut(label_a),
            run_time=0.6,
        )
        self.play(
            *(FadeIn(t_cell) for t_cell in t_cells),
            LaggedStart(
                *(
                    TailToTipMapsto(arrow, run_time=0.85)
                    for arrow in f_arrows.values()
                ),
                lag_ratio=0.12,
            ),
            FadeIn(label_t),
        )
        self.wait(0.4)

        # --- Encode B: the same unfolding — and the nesting is discarded. ----
        # The composite inherits A's shape, so B's tree and roots are
        # scaffolding: they retire with the stride column, leaving only the
        # Tuple morphism g.
        v_cells = [
            cell(value, place(X_V, index))
            for index, value in enumerate(forward_b.codomain)
        ]
        g_arrows = {
            mode: stacks.segment_arrow(u_cells[mode], v_cells[target - 1])
            for mode, target in enumerate(forward_b.map)
            if target
        }
        label_u = label("U", u_cells[0])
        label_v = label("V", v_cells[0])
        self.play(
            FadeOut(diagram_b.colons),
            FadeOut(VGroup(*diagram_b.stride_cells)),
            FadeOut(VGroup(*diagram_b.root_cells)),
            FadeOut(VGroup(*diagram_b.strands())),
            FadeOut(label_b),
            FadeIn(label_u),
            run_time=0.6,
        )
        self.play(
            *(FadeIn(v_cell) for v_cell in v_cells),
            LaggedStart(
                *(
                    TailToTipMapsto(arrow, run_time=0.85)
                    for arrow in g_arrows.values()
                ),
                lag_ratio=0.12,
            ),
            FadeIn(label_v),
        )
        self.wait(0.7)

        # --- Bridge: grow the mutual refinement between flat(T) and flat(U). --
        w_cells = []
        refinement_segs = []
        for source, target, source_value, _, factor in (
            TupleMorphismRefinementTest._refinement_steps(
                T.flatten(), U.flatten()
            )
        ):
            entry = cell(factor, place(X_W, len(w_cells)))
            growing = []
            if source_value is not None:
                growing.append(
                    segment(t_cells[source].get_right(), entry.get_left())
                )
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

        # --- Refine: both middle stacks split along b and c. ------------------
        self.play(
            *(
                UndrawMapstoTip(arrow.tip)
                for arrow in (*f_arrows.values(), *g_arrows.values())
            ),
            run_time=0.4,
        )

        left_outer = [None] * T.length()
        for mode, target in enumerate(forward_a.map):
            if target:
                left_outer[target - 1] = s_cells[mode]
        right_outer = [
            v_cells[target - 1] if target else None
            for target in forward_b.map
        ]

        alpha = ValueTracker(0.0)
        left = self._refine_in_place(
            coarse=T,
            refined=leg_b.nest,
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
            refined=leg_c.nest,
            coarse_cells=u_cells,
            column=X_U,
            outer_cells=right_outer,
            shared_cells=w_cells,
            shared_on_right=False,
            alpha=alpha,
            place=place,
            cell=cell,
        )

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

        left["tips"] = {
            leaf: stacks.arrow_tip(w_cells[leaf].get_left() + LEFT * ARROW_INSET)
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
        for leaf, tip in right["tips"].items():
            tip.add_updater(
                lambda mobject, leaf=leaf: mobject.become(
                    stacks.arrow_tip(
                        right["cells"][leaf].get_left() + LEFT * ARROW_INSET
                    )
                )
            )
        self.wait(0.9)

        # --- Reorder: the pullback of f and the pushforward of g. -------------
        left_destination = {
            pulled.map[index] - 1: index
            for index in range(len(pulled.domain))
            if pulled.map[index]
        }
        right_destination = {
            leaf: target - 1
            for leaf, target in enumerate(pushed.map)
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
            self.add(
                *stacks.deck(
                    {
                        destination[leaf]: split
                        for leaf, split in half["cells"].items()
                        if leaf in destination
                    }
                )
            )

        arrivals = []
        left_joined = []
        right_joined = []
        arrival_left = {}
        arrival_right = {}
        for mode, target in enumerate(forward_a.map):
            if target:
                continue
            for leaf in s_groups[mode]:
                joined = cell(pulled.domain[leaf], place(X_T, leaf))
                connector = segment(s_cells[mode].get_right(), joined.get_left())
                left_joined.extend((joined, connector))
                arrival_left[leaf] = joined
                arrivals.append(self._arrival(joined, connector))
        image = set(forward_b.map)
        for mode in range(len(forward_b.codomain)):
            if mode + 1 in image:
                continue
            for leaf in v_groups[mode]:
                joined = cell(pushed.codomain[leaf], place(X_U, leaf))
                connector = segment(
                    joined.get_right(), v_cells[mode].get_left()
                ).reverse_points()
                right_joined.extend((joined, connector))
                arrival_right[leaf] = joined
                arrivals.append(self._arrival(joined, connector))

        label_s = label("S", s_cells[0])
        label_sprime = label("S'", cell(pulled.domain[0], place(X_T, 0)))
        label_vprime = label("V'", cell(pushed.codomain[0], place(X_U, 0)))
        self.play(
            *reorder,
            *(FadeOut(group) for group in retiring),
            *arrivals,
            FadeOut(label_tprime),
            FadeOut(label_uprime),
            FadeIn(label_s),
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

        # --- Compose through W. ------------------------------------------------
        # Unlike the morphism-level scene, dead routes keep their cells: an
        # S' mode whose route dies has stride 0, and a V' mode nothing
        # reaches still counts toward the prefix products.  Only the
        # connectors and tips of dead routes retire with W.
        routes = [
            pulled.map[index] - 1
            for index in range(len(pulled.domain))
            if pulled.map[index] and pushed.map[pulled.map[index] - 1]
        ]
        routed = set(routes)
        bridges = [
            bridge(
                left["inner"][leaf].get_end(),
                right["inner"][leaf].get_start(),
            )
            for leaf in routes
        ]
        stray = []
        for half, destination in (
            (left, left_destination),
            (right, right_destination),
        ):
            for leaf in half["cells"]:
                if leaf in destination and leaf not in routed:
                    stray.append(VGroup(half["inner"][leaf], half["tips"][leaf]))
        self.play(
            ShrinkToCenter(VGroup(*w_cells)),
            *(UndrawMapstoTip(left["tips"][leaf]) for leaf in routes),
            *(Create(bridge_line) for bridge_line in bridges),
            *(FadeOut(group) for group in stray),
            run_time=1.0,
        )

        sprime_x = X_W - COMPOSITE_HALF_WIDTH
        vprime_x = X_W + COMPOSITE_HALF_WIDTH
        left_shift = np.array((sprime_x - X_T, 0.0, 0.0))
        right_shift = np.array((vprime_x - X_U, 0.0, 0.0))
        curved = VGroup()
        contracted = VGroup()
        for leaf, bridge_line in zip(routes, bridges):
            route = join_route(
                left["inner"][leaf], bridge_line, right["inner"][leaf]
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
            contracted.add(MapstoArrow.from_parts(arrow.tail, final, arrow.tip))
        for leaf in routes:
            self.remove(
                left["inner"][leaf],
                left["tips"][leaf],
                right["inner"][leaf],
                right["tips"][leaf],
            )
        self.remove(*bridges)
        self.add(curved)

        # Cells and fans retired at the reorder must not ride along: a
        # faded mobject re-added by a group animation comes back at full
        # opacity, resurrecting the retired block over a live one.
        left_alive = [leaf for leaf in left["cells"] if leaf in left_destination]
        right_alive = [
            leaf for leaf in right["cells"] if leaf in right_destination
        ]
        left_half = VGroup(
            *diagram_a.root_cells,
            *diagram_a.strands(),
            *s_cells,
            *(left["cells"][leaf] for leaf in left_alive),
            *(left["fans"][leaf] for leaf in left_alive if leaf in left["fans"]),
            *left_joined,
            label_s,
            label_sprime,
        )
        right_half = VGroup(
            *v_cells,
            *(right["cells"][leaf] for leaf in right_alive),
            *(
                right["fans"][leaf]
                for leaf in right_alive
                if leaf in right["fans"]
            ),
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
        self.wait(1.0)

        # --- Read off the layout of the composite span. ------------------------
        # V and its fans retire — they are c and V, scaffolding of B — and
        # the prefix products of flat(V') appear beside V' behind colons.
        self.play(
            FadeOut(VGroup(*v_cells)),
            FadeOut(
                VGroup(
                    *(
                        right["fans"][leaf]
                        for leaf in right_alive
                        if leaf in right["fans"]
                    )
                )
            ),
            FadeOut(VGroup(*right_joined[1::2])),
            FadeOut(label_v),
            run_time=0.6,
        )
        products = prefix_products(pushed.codomain)
        product_cells = [
            fitted_cell(
                value,
                place(vprime_x + PAIR_GAP, index),
                fill=STRIDE_FILL,
                scale=scale,
            )
            for index, value in enumerate(products)
        ]
        target_colons = [
            layout_colon(place(vprime_x + PAIR_GAP / 2, index), scale=scale)
            for index in range(len(products))
        ]
        self.play(
            LaggedStart(
                *(
                    FadeIn(VGroup(colon, product), shift=RIGHT * 0.12)
                    for colon, product in zip(target_colons, product_cells)
                ),
                lag_ratio=0.25,
            ),
            run_time=1.2,
        )
        self.wait(0.25)

        # The V' entries and colons fall away (with any product no mode picks
        # up); every composite arrow extends into a grabber and pulls its
        # stride home beside S'; a stride-0 mode has no arrow, so its 0
        # appears only once every pulled stride is seated, with the colons.
        used = set(composite.map) - {0}
        unused_products = [
            product_cells[index]
            for index in range(len(products))
            if index + 1 not in used
        ]
        vprime_cells = [
            right["cells"][leaf] for leaf in right["cells"]
            if leaf in right_destination
        ] + list(arrival_right.values())
        self.play(
            FadeOut(VGroup(*vprime_cells)),
            FadeOut(VGroup(*target_colons)),
            FadeOut(label_vprime),
            *(FadeOut(product) for product in unused_products),
            run_time=0.6,
        )

        source_colons = [
            layout_colon(place(sprime_x + PAIR_GAP / 2, index), scale=scale)
            for index in range(len(composite.domain))
        ]
        stride_cells = {}
        reaches = {}
        pulls = {}
        for arrow, leaf in zip(contracted, routes):
            index = left_destination[leaf]
            target = composite.map[index]
            carried = product_cells[target - 1]
            destination = place(sprime_x + PAIR_GAP, index)
            reaches[target] = GrabberReach(arrow, carried, run_time=0.85)
            pulls[target] = GrabberPull(arrow, carried, destination)
            stride_cells[index] = carried
        basepoint_strides = [
            fitted_cell(
                strides[index],
                place(sprime_x + PAIR_GAP, index),
                fill=STRIDE_FILL,
                scale=scale,
            )
            for index in range(len(composite.domain))
            if index not in stride_cells
        ]
        self.play(
            LaggedStart(
                *(reaches[target] for target in sorted(reaches)), lag_ratio=0.12
            )
        )
        self.play(
            LaggedStart(
                *(pulls[target] for target in sorted(pulls)), lag_ratio=0.12
            ),
            run_time=1.8,
        )
        self.remove(*contracted)
        self.play(
            LaggedStart(
                *(FadeIn(colon) for colon in source_colons), lag_ratio=0.1
            ),
            *(FadeIn(stride) for stride in basepoint_strides),
            run_time=0.6,
        )

        # What remains is the nested layout depiction of B ∘ A — ρ(S), A's
        # tree with the grafted fans, S', colons, strides — which slides to
        # center.
        remaining = VGroup(
            left_half,
            *stride_cells.values(),
            *basepoint_strides,
            *source_colons,
        )
        self.play(
            remaining.animate.shift(RIGHT * -remaining.get_center()[0]),
            run_time=1.0,
        )
        self.wait(2.2)
