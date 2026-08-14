"""Prototype animation for the pushforward of a flat tuple morphism.

The mirror image of the pullback.  Instead of refining the codomain and pulling
the refinement back onto the domain, we refine the *domain* and push it forward
onto the codomain.  ``U'`` (far left) and ``V`` (far right) are fixed.

    open:     [U'] --refinement--> [U] --g--> [V]
    stage 1:  [U'] --parallel arrows--> [U'] --fans--> [V]
    stage 2:  [U'] --g'--> [V'] --fans--> [V]

``g : U -> V`` is an arbitrary flat tuple morphism: it may permute modes, may
reverse their order, may miss codomain modes, and may send domain modes to the
basepoint.  ``g.pushforward_along(U')`` supplies ``V'`` and ``g'``, so the
picture is read off the library rather than assumed.

Stage 1 refines, exactly as in the pullback.  Each middle cell splits in place,
with no reordering, into its own factors in ``U'``: the first factor keeps the
cell's slot and the rest are peeled off it upward, so a mode's split pushes
everything above it up.  The middle stack is then ``U'``, cell for cell alongside
the stack on the left, so the connectors between them are necessarily parallel;
they take tips and are the identity of ``U'``.  On the right, each arrow of ``g``
has given up its tip and split into the fan of the mode it leaves, so those fans
cross exactly as ``g`` did.

Stage 2 reorders.  The blocks of the middle stack are permuted into the order of
``V'``: the block a codomain mode is hit by descends or rises into that mode's
own slots.  Every connector stays attached to its own cells -- tips to the middle
cells they land on, tails to the ``U'`` cells they leave -- so the fans on the
right uncross into the refinement of ``V`` while the parallel arrows on the left
cross into ``g'``.  A codomain mode outside the image of ``g`` has no block to
receive: its cell appears in the slot the reordering leaves free for it, and the
fan exhibiting it as its own refinement draws in toward it from ``V``, with no
arrow to follow.  A domain mode sent to the basepoint pushes nothing forward, so
its block leaves.

Every cell is stacked with equal spacing (both compact and spread) and every
stack shares one bottom baseline.  Arrows and segments share one geometry: an
arrow is a refinement segment carrying a caret, so replacing one by the other
changes nothing but the tip, and a cell peeled off another stays invisible until
it has moved clear, since coincident strokes would double up and darken.
"""

from dataclasses import dataclass
from math import prod

import numpy as np
from manim import (
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    ORIGIN,
    Scene,
    Succession,
    Text,
    ValueTracker,
    VGroup,
    smooth,
)
from tract import NestedTuple, Nest_morphism

from layout_categories_viz.animations import DrawMapstoTip, UndrawMapstoTip
from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK

# The pullback scene owns the drawing primitives the two share, so the mirror
# images cannot drift apart.
from scenes.tuple_pullback_test import (
    ARROW_INSET,
    CELL_H,
    LABEL_FONT_SIZE,
    LEFT_X,
    MAX_STACK_HEIGHT,
    MID_X,
    RIGHT_X,
    SLOT_STEP,
    TuplePullbackTest,
    _groups,
    _revealed,
)


@dataclass(frozen=True)
class PushforwardExample:
    """One pushforward: a refinement of ``U``, and the morphism out of ``U``.

    ``modes[i]`` lists the factors that ``U``'s ``i``-th mode splits into, so
    ``U`` is the tuple of their products and ``U'`` groups the factors.
    ``mapping`` is ``g`` in the repository's one-based convention, with ``0``
    for the basepoint.  ``codomain`` is only needed when ``g`` leaves a mode of
    ``V`` unhit: otherwise every ``V`` mode is the ``U`` mode sent to it.
    """

    modes: tuple
    mapping: tuple
    codomain: tuple | None = None

    @property
    def domain(self) -> tuple:
        return tuple(prod(factors) for factors in self.modes)

    @property
    def refinement(self) -> tuple:
        # A mode that does not split is a plain entry, not a one-tuple.
        return tuple(
            factors[0] if len(factors) == 1 else tuple(factors)
            for factors in self.modes
        )

    def resolved_codomain(self) -> tuple:
        if self.codomain is not None:
            return self.codomain
        values = [None] * max(self.mapping)
        for mode, target in enumerate(self.mapping):
            if target:
                values[target - 1] = self.domain[mode]
        if any(value is None for value in values):
            raise ValueError("a mapping that misses a mode needs a codomain")
        return tuple(values)


def _identity(count: int) -> tuple:
    return tuple(range(1, count + 1))


EXAMPLES = (
    # U = V = (6,): one mode, split in two.  Nothing to reorder.
    PushforwardExample(((2, 3),), _identity(1)),
    # U = V = (6, 12): blocks of two and three cells.
    PushforwardExample(((2, 3), (2, 2, 3)), _identity(2)),
    # U = V = (8, 8) with g the transposition: the weak-composition example.
    PushforwardExample(((4, 2), (2, 4)), (2, 1)),
    # U = (6, 4, 6) -> V = (4, 6, 6) over the three-cycle (3, 1, 2): blocks of
    # two, one and two cells, and a first mode that does not split.
    PushforwardExample(((2, 3), (4,), (3, 2)), (3, 1, 2)),
    # U = (12, 5, 8, 6) -> V = (8, 6, 5, 12) over (4, 3, 1, 2): four blocks of
    # three, one, two and two cells, reversed in pairs.
    PushforwardExample(((2, 2, 3), (5,), (2, 4), (3, 2)), (4, 3, 1, 2)),
    # U = (6, 5, 4) -> V = (4, 7, 6) over (3, 0, 1): g reverses two modes, sends
    # 5 to the basepoint, and misses V's middle mode 7.
    PushforwardExample(((2, 3), (5,), (2, 2)), (3, 0, 1), codomain=(4, 7, 6)),
)


class TuplePushforwardTest(Scene):
    """Deform 'refinement then morphism' into 'morphism then refinement'."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        for index, example in enumerate(EXAMPLES):
            self._show_pushforward(example)
            self._clear_scene(last=index == len(EXAMPLES) - 1)

    def _show_pushforward(self, example: PushforwardExample) -> None:
        U = NestedTuple(example.domain)
        V = NestedTuple(example.resolved_codomain())
        Uprime = NestedTuple(example.refinement)
        g = Nest_morphism(U, V, example.mapping)
        gprime = g.pushforward_along(Uprime)
        Vprime = gprime.codomain
        u_groups = _groups(Uprime, U)
        v_groups = _groups(Vprime, V)

        # Scale the figure so the taller of the two spread stacks fits.
        slots = max(Uprime.length(), Vprime.length())
        scale = min(
            1.0, MAX_STACK_HEIGHT / ((slots - 1) * SLOT_STEP + CELL_H)
        )
        step = SLOT_STEP * scale
        baseline = -((slots - 1) * step) / 2

        def place(column, index):
            return np.array((column, baseline + index * step, 0.0))

        def cell(value, center):
            return (
                TuplePullbackTest._cell(value, ORIGIN)
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

        segment = TuplePullbackTest._tree_segment
        attached = TuplePullbackTest._attached

        # --- Open: [U'] --refinement--> [U] --g--> [V]. ----------------------
        up_cells = [
            cell(value, place(LEFT_X, leaf))
            for leaf, value in enumerate(Uprime.flatten())
        ]
        u_cells = [
            cell(U.entry(mode + 1), place(MID_X, mode))
            for mode in range(U.length())
        ]
        v_cells = [
            cell(V.entry(mode + 1), place(RIGHT_X, mode))
            for mode in range(V.length())
        ]
        refinement_segs = [
            segment(up_cells[leaf].get_right(), u_cells[mode].get_left())
            for mode, leaves in enumerate(u_groups)
            for leaf in leaves
        ]
        g_arrows = {
            mode: TuplePullbackTest._segment_arrow(
                u_cells[mode], v_cells[target - 1]
            )
            for mode, target in enumerate(example.mapping)
            if target
        }
        label_up = label("U'", up_cells[0])
        label_u = label("U", u_cells[0])
        label_v = label("V", v_cells[0])

        self.play(
            FadeIn(
                VGroup(
                    *up_cells,
                    *u_cells,
                    *v_cells,
                    *refinement_segs,
                    *g_arrows.values(),
                    label_up,
                    label_u,
                    label_v,
                )
            ),
            run_time=1.0,
        )
        self.wait(0.8)

        # The arrows of g give up their tips: they are becoming fans.
        self.play(
            *(UndrawMapstoTip(arrow.tip) for arrow in g_arrows.values()),
            run_time=0.4,
        )

        # The V' slot each U' leaf is pushed into.
        destination = {
            leaf: target - 1
            for leaf, target in enumerate(gprime.map)
            if target
        }

        # --- Stage 1: split every middle cell in place, no reordering. --------
        alpha = ValueTracker(0.0)
        cells = {}
        arrows = {}
        fans = {}
        for mode, leaves in enumerate(u_groups):
            start_center = u_cells[mode].get_center()
            target = example.mapping[mode]
            for index, leaf in enumerate(leaves):
                # The first cell of a mode stands in for the U cell it came
                # from, and sheds its value; the rest are stacked under it,
                # already carrying their own.
                split = cell(Uprime.flatten()[leaf], start_center)
                if not index:
                    coarse = cell(U.entry(mode + 1), start_center)[1]
                    split[1].set_opacity(0)
                    split.add(coarse)
                TuplePullbackTest._split(
                    split,
                    alpha,
                    start_center,
                    place(MID_X, leaf),
                    peeled=bool(index),
                )
                cells[leaf] = split
                arrows[leaf] = attached(up_cells[leaf], split)
                if target:
                    fans[leaf] = attached(
                        split,
                        v_cells[target - 1],
                        reveal=(
                            (lambda: _revealed(alpha.get_value()))
                            if index
                            else None
                        ),
                    )

        # Every replacement starts out coincident with what it replaces.
        self.remove(*refinement_segs, *g_arrows.values(), *u_cells)
        self.add(
            *arrows.values(),
            *fans.values(),
            *TuplePullbackTest._deck(cells),
        )

        label_uprime = label("U'", cell(Uprime.flatten()[0], place(MID_X, 0)))
        self.play(
            alpha.animate.set_value(1.0),
            FadeOut(label_u),
            FadeIn(label_uprime),
            run_time=1.4,
            rate_func=smooth,
        )
        for split in cells.values():
            split.clear_updaters()

        # The parallel connectors on the left take tips: they are the identity
        # of U', and they keep those tips through the reordering, where they
        # follow the cells they land on.
        tips = {
            leaf: TuplePullbackTest._arrow_tip(
                cells[leaf].get_left() + LEFT * ARROW_INSET
            )
            for leaf in cells
        }
        self.add(*tips.values())
        self.play(
            *(DrawMapstoTip(tip) for tip in tips.values()), run_time=0.4
        )
        for leaf, tip in tips.items():
            tip.add_updater(
                lambda mobject, leaf=leaf: mobject.become(
                    TuplePullbackTest._arrow_tip(
                        cells[leaf].get_left() + LEFT * ARROW_INSET
                    )
                )
            )
        self.wait(0.9)

        # --- Stage 2: reorder the blocks into the order of V'. ----------------
        # A domain mode sent to the basepoint pushes nothing forward, so its
        # block leaves with the arrow that reaches it.
        retiring = [leaf for leaf in cells if leaf not in destination]
        for leaf in retiring:
            arrows[leaf].clear_updaters()
            tips[leaf].clear_updaters()
        # Draw the stack bottom to top for the reordering too, by the slots the
        # cells are heading for, so blocks passing one another always cover in
        # the same direction -- as in the pullback, and as while splitting.
        self.add(*(cells[leaf] for leaf in retiring))
        self.add(
            *TuplePullbackTest._deck(
                {
                    destination[leaf]: cells[leaf]
                    for leaf in cells
                    if leaf in destination
                }
            )
        )
        reorder = [
            cells[leaf].animate.move_to(place(MID_X, destination[leaf]))
            for leaf in cells
            if leaf in destination
        ] + [
            FadeOut(VGroup(cells[leaf], arrows[leaf], tips[leaf]))
            for leaf in retiring
        ]

        # A codomain mode outside the image of g has no block to receive:
        # nothing in U is pushed into it, so it carries its own value over into
        # V'.  The reordering leaves its slot free, and it appears there while
        # the fan exhibiting it as its own refinement draws in toward it.
        image = set(example.mapping)
        arrivals = []
        for mode in range(V.length()):
            if mode + 1 in image:
                continue
            for leaf in v_groups[mode]:
                joined = cell(Vprime.flatten()[leaf], place(MID_X, leaf))
                # Built toward V, like every other fan of the refinement
                # of V, but traced backwards so it draws in toward the cell.
                connector = segment(
                    joined.get_right(), v_cells[mode].get_left()
                ).reverse_points()
                arrivals.append(
                    Succession(
                        FadeIn(joined, run_time=0.45),
                        Create(connector, run_time=0.95),
                    )
                )

        label_vprime = label("V'", cell(Vprime.flatten()[0], place(MID_X, 0)))
        self.play(
            *reorder,
            *arrivals,
            FadeOut(label_uprime),
            FadeIn(label_vprime),
            run_time=1.4,
            rate_func=smooth,
        )
        for mobject in (*arrows.values(), *fans.values(), *tips.values()):
            mobject.clear_updaters()
        self.wait(1.8)

    def _clear_scene(self, *, last) -> None:
        self.play(*(FadeOut(m) for m in self.mobjects), run_time=0.6)
        self.clear()
        if not last:
            self.wait(0.2)
