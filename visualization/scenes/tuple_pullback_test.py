"""Prototype animation for the pullback of a flat tuple morphism.

The pullback is shown as a deformation from 'morphism then refinement' into
'refinement then morphism'.  ``S`` (far left) and ``T'`` (far right) are fixed.

    open:     [S] --f--> [T] --refinement--> [T']
    stage 1:  [S] --fans--> [T'] --parallel arrows--> [T']
    stage 2:  [S] --fans--> [S'] --f'--> [T']

``f : S -> T`` is an arbitrary flat tuple morphism: it may permute modes, may
reverse their order, may miss codomain modes, and may send domain modes to the
basepoint.  ``f.pullback_along(T')`` supplies ``S'`` and ``f'``, so the picture
is read off the library rather than assumed.

Stage 1 refines.  Each middle cell splits in place, with no reordering, into its
own factors in ``T'``: the first factor keeps the cell's slot and the rest are
peeled off it upward, so a mode's split pushes everything above it up.  The
middle stack is then ``T'``, cell for cell alongside the stack on the right, so
the connectors between them are necessarily parallel; they take tips and are the
identity of ``T'``.  On the left, each arrow of ``f`` has given up its tip and
split into the fan of the mode it lands on, so those fans cross exactly as ``f``
did.

Stage 2 reorders.  The blocks of the middle stack are permuted into the order of
``S'``: the block a domain mode maps onto descends or rises into that mode's own
slots.  Every connector stays attached to its own cells -- tails to the middle
cells they leave, tips to the ``T'`` cells they land on -- so the fans on the
left uncross into the refinement of ``S`` while the parallel arrows on the right
cross into ``f'``.  A domain mode sent to the basepoint has no block to reorder:
its cell appears in the slot the reordering leaves free for it, and the fan
exhibiting it as its own refinement draws in toward it, with no arrow to follow.
A codomain mode outside the image of ``f`` has no place in ``S'``, so its block
leaves.

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
    CubicBezier,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    Line,
    RIGHT,
    RoundedRectangle,
    ORIGIN,
    Scene,
    Succession,
    Text,
    UP,
    ValueTracker,
    VGroup,
    smooth,
)
from tract import NestedTuple, Nest_morphism

from layout_categories_viz.animations import DrawMapstoTip, UndrawMapstoTip
from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK, PANEL
from layout_categories_viz.tuple_morphism import MapstoArrow


@dataclass(frozen=True)
class PullbackExample:
    """One pullback: a refinement of ``T``, and the morphism into ``T``.

    ``modes[j]`` lists the factors that ``T``'s ``j``-th mode splits into, so
    ``T`` is the tuple of their products and ``T'`` groups the factors.
    ``mapping`` is ``f`` in the repository's one-based convention, with ``0``
    for the basepoint.  ``domain`` is only needed when ``mapping`` has a zero:
    otherwise every ``S`` mode is the ``T`` mode it is sent to.
    """

    modes: tuple
    mapping: tuple
    domain: tuple | None = None

    @property
    def codomain(self) -> tuple:
        return tuple(prod(factors) for factors in self.modes)

    @property
    def refinement(self) -> tuple:
        # A mode that does not split is a plain entry, not a one-tuple.
        return tuple(
            factors[0] if len(factors) == 1 else tuple(factors)
            for factors in self.modes
        )

    def resolved_domain(self) -> tuple:
        if self.domain is not None:
            return self.domain
        if not all(self.mapping):
            raise ValueError("a mapping with a basepoint needs an explicit domain")
        return tuple(self.codomain[target - 1] for target in self.mapping)


def _identity(count: int) -> tuple:
    return tuple(range(1, count + 1))


EXAMPLES = (
    # S = T = (6,): one mode, split in two.  Nothing to reorder.
    PullbackExample(((2, 3),), _identity(1)),
    # S = T = (6, 12): blocks of two and three cells.
    PullbackExample(((2, 3), (2, 2, 3)), _identity(2)),
    # S = T = (8, 8) with f the transposition: the weak-composition example.
    PullbackExample(((4, 2), (2, 4)), (2, 1)),
    # S = (6, 6, 4) -> T = (6, 4, 6) over the three-cycle (3, 1, 2): blocks of
    # two, one and two cells, and a middle mode that does not split.
    PullbackExample(((2, 3), (4,), (3, 2)), (3, 1, 2), domain=(6, 6, 4)),
    # S = (6, 8, 12, 5) -> T = (12, 5, 8, 6) over (4, 3, 1, 2): four blocks of
    # three, one, two and two cells, reversed in pairs.
    PullbackExample(
        ((2, 2, 3), (5,), (2, 4), (3, 2)), (4, 3, 1, 2), domain=(6, 8, 12, 5)
    ),
    # S = (4, 7, 6) -> T = (6, 5, 4) over (3, 0, 1): f reverses two modes, sends
    # 7 to the basepoint, and misses T's middle mode 5.
    PullbackExample(((2, 3), (5,), (2, 2)), (3, 0, 1), domain=(4, 7, 6)),
)

CELL_H = 0.7
ROW_GAP = 0.2  # uniform gap between adjacent cells in any stack
FONT_SIZE = 30
LABEL_FONT_SIZE = 30
STROKE_WIDTH = 3.6
TIP_LENGTH = 0.2
TIP_WIDTH = 0.18
ARROW_INSET = 0.08  # matches MapstoArrow's endpoint_inset default
# Every connector ends with a straight horizontal run, so a caret drawn at its
# end is approached horizontally however far the connector has climbed.  The run
# sits at the end a tip would: a fan diverging out of one cell still separates at
# once, and one converging into a cell merges over this last stretch.
ARROW_RUN = 0.35
TAIL_LENGTH = 0.22  # the bar of a |-> arrow, drawn invisibly here
MAX_STACK_HEIGHT = 6.3  # taller examples are scaled down to fit the frame
# A cell peeled off a splitting cell is invisible until it has moved clear of
# it: coincident strokes would otherwise double up and darken at that instant.
REVEAL = 0.3

LEFT_X = -3.6
MID_X = 0.0
RIGHT_X = 3.6
SLOT_STEP = CELL_H + ROW_GAP


def _groups(refined: NestedTuple, coarse: NestedTuple) -> tuple:
    """Leaf indices of ``refined`` under each mode of ``coarse``, zero-based."""
    return tuple(
        tuple(
            range(
                refined.sublength(index, coarse),
                refined.sublength(index, coarse)
                + refined.relative_mode(index, coarse).length(),
            )
        )
        for index in range(1, coarse.length() + 1)
    )


def _revealed(value: float) -> float:
    """How much of a peeled copy is showing, once the motion has begun."""
    return min(1.0, value / REVEAL)


class TuplePullbackTest(Scene):
    """Deform 'morphism then refinement' into 'refinement then morphism'."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        for index, example in enumerate(EXAMPLES):
            self._show_pullback(example)
            self._clear_scene(last=index == len(EXAMPLES) - 1)

    def _show_pullback(self, example: PullbackExample) -> None:
        S = NestedTuple(example.resolved_domain())
        T = NestedTuple(example.codomain)
        Tprime = NestedTuple(example.refinement)
        f = Nest_morphism(S, T, example.mapping)
        fprime = f.pullback_along(Tprime)
        Sprime = fprime.domain
        t_groups = _groups(Tprime, T)
        s_groups = _groups(Sprime, S)

        # Scale the figure so the taller of the two spread stacks fits.
        slots = max(Tprime.length(), Sprime.length())
        scale = min(
            1.0, MAX_STACK_HEIGHT / ((slots - 1) * SLOT_STEP + CELL_H)
        )
        step = SLOT_STEP * scale
        baseline = -((slots - 1) * step) / 2

        def place(column, index):
            return np.array((column, baseline + index * step, 0.0))

        def cell(value, center):
            return self._cell(value, ORIGIN).scale(scale).move_to(center)

        def label(text, anchor):
            return (
                Text(
                    text, color=INK, font=CODE_FONT, font_size=LABEL_FONT_SIZE
                )
                .scale(scale)
                .next_to(anchor, DOWN, buff=0.3)
            )

        # --- Open: [S] --f--> [T] --refinement--> [T']. ----------------------
        s_cells = [
            cell(S.entry(mode + 1), place(LEFT_X, mode))
            for mode in range(S.length())
        ]
        t_cells = [
            cell(T.entry(mode + 1), place(MID_X, mode))
            for mode in range(T.length())
        ]
        tp_cells = [
            cell(value, place(RIGHT_X, leaf))
            for leaf, value in enumerate(Tprime.flatten())
        ]
        f_arrows = {
            mode: self._segment_arrow(s_cells[mode], t_cells[target - 1])
            for mode, target in enumerate(example.mapping)
            if target
        }
        refinement_segs = [
            self._tree_segment(
                t_cells[mode].get_right(), tp_cells[leaf].get_left()
            )
            for mode, leaves in enumerate(t_groups)
            for leaf in leaves
        ]
        label_s = label("S", s_cells[0])
        label_t = label("T", t_cells[0])

        self.play(
            FadeIn(
                VGroup(
                    *s_cells,
                    *t_cells,
                    *tp_cells,
                    *f_arrows.values(),
                    *refinement_segs,
                    label_s,
                    label_t,
                )
            ),
            run_time=1.0,
        )
        self.wait(0.8)

        # The arrows of f give up their tips: they are becoming fans.
        self.play(
            *(UndrawMapstoTip(arrow.tip) for arrow in f_arrows.values()),
            run_time=0.4,
        )

        # The domain mode landing on each codomain mode, and the S' slot each
        # T' leaf ends up in.
        source_mode = {
            target - 1: mode
            for mode, target in enumerate(example.mapping)
            if target
        }
        destination = {
            fprime.map[leaf] - 1: leaf
            for leaf in range(Sprime.length())
            if fprime.map[leaf]
        }

        # --- Stage 1: split every middle cell in place, no reordering. --------
        alpha = ValueTracker(0.0)
        cells = {}
        fans = {}
        connectors = {}
        for mode, leaves in enumerate(t_groups):
            start_center = t_cells[mode].get_center()
            for index, leaf in enumerate(leaves):
                # The first cell of a mode stands in for the T cell it came
                # from, and sheds its value; the rest are stacked under it,
                # already carrying their own.
                split = cell(Tprime.flatten()[leaf], start_center)
                if not index:
                    coarse = cell(T.entry(mode + 1), start_center)[1]
                    split[1].set_opacity(0)
                    split.add(coarse)
                self._split(
                    split,
                    alpha,
                    start_center,
                    place(MID_X, leaf),
                    peeled=bool(index),
                )
                cells[leaf] = split
                if mode in source_mode:
                    fans[leaf] = self._attached(
                        s_cells[source_mode[mode]],
                        split,
                        reveal=(
                            (lambda: _revealed(alpha.get_value()))
                            if index
                            else None
                        ),
                    )
                connectors[leaf] = self._attached(split, tp_cells[leaf])

        # Every replacement starts out coincident with what it replaces.
        self.remove(*f_arrows.values(), *refinement_segs, *t_cells)
        self.add(*fans.values(), *connectors.values(), *self._deck(cells))

        label_tprime = label("T'", cell(Tprime.flatten()[0], place(MID_X, 0)))
        self.play(
            alpha.animate.set_value(1.0),
            FadeOut(label_t),
            FadeIn(label_tprime),
            run_time=1.4,
            rate_func=smooth,
        )
        for split in cells.values():
            split.clear_updaters()

        # The parallel connectors on the right take tips: they are the identity
        # of T', and they keep those tips through the reordering.
        tips = {
            leaf: self._arrow_tip(
                tp_cells[leaf].get_left() + LEFT * ARROW_INSET
            )
            for leaf in cells
        }
        self.add(*tips.values())
        self.play(
            *(DrawMapstoTip(tip) for tip in tips.values()), run_time=0.4
        )
        self.wait(0.9)

        # --- Stage 2: reorder the blocks into the order of S'. ----------------
        # A codomain mode outside the image of f has no place in S', so its
        # block leaves with its arrow.
        retiring = [leaf for leaf in cells if leaf not in destination]
        for leaf in retiring:
            connectors[leaf].clear_updaters()
        # Draw the stack bottom to top for the reordering too, by the slots the
        # cells are heading for: then blocks passing one another always cover in
        # the same direction, as they did while splitting.  Nothing overlaps at
        # this instant, so the reordering costs nothing to impose.
        self.add(*(cells[leaf] for leaf in retiring))
        self.add(
            *self._deck(
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
            FadeOut(VGroup(cells[leaf], connectors[leaf], tips[leaf]))
            for leaf in retiring
        ]

        # A domain mode sent to the basepoint has no block to reorder: nothing
        # in T refines into it, so it carries its own value over into S'.  The
        # reordering leaves its slot free, and it appears there while the fan
        # exhibiting it as its own refinement draws in toward it.
        arrivals = []
        for mode, target in enumerate(example.mapping):
            if target:
                continue
            for leaf in s_groups[mode]:
                joined = cell(Sprime.flatten()[leaf], place(MID_X, leaf))
                arrivals.append(
                    Succession(
                        FadeIn(joined, run_time=0.45),
                        Create(
                            self._tree_segment(
                                s_cells[mode].get_right(), joined.get_left()
                            ),
                            run_time=0.95,
                        ),
                    )
                )

        label_sprime = label("S'", cell(Sprime.flatten()[0], place(MID_X, 0)))
        self.play(
            *reorder,
            *arrivals,
            FadeOut(label_tprime),
            FadeIn(label_sprime),
            run_time=1.4,
            rate_func=smooth,
        )
        for mobject in (*fans.values(), *connectors.values()):
            mobject.clear_updaters()
        self.wait(1.8)

    def _clear_scene(self, *, last) -> None:
        self.play(*(FadeOut(m) for m in self.mobjects), run_time=0.6)
        self.clear()
        if not last:
            self.wait(0.2)

    # ------------------------------------------------------------------ utils
    @classmethod
    def _segment_arrow(cls, source_cell, target_cell):
        """A map arrow whose shaft is exactly a refinement segment.

        Sharing the segment's curvature is what lets an arrow be replaced by the
        fan it splits into without any snap: only the caret differs.
        """
        shaft = cls._tree_segment(
            source_cell.get_right(), target_cell.get_left()
        )
        tip = cls._arrow_tip(target_cell.get_left() + LEFT * ARROW_INSET)
        start = shaft.get_start()
        tail = Line(
            start + DOWN * TAIL_LENGTH / 2,
            start + UP * TAIL_LENGTH / 2,
            color=INK,
            stroke_width=STROKE_WIDTH,
        ).set_opacity(0)
        return MapstoArrow.from_parts(tail, shaft, tip)

    @classmethod
    def _attached(cls, source, target, *, reveal=None):
        """A connector that follows both of the cells it joins.

        ``reveal`` is a callable returning how much of the connector is showing.
        A connector that starts out coincident with one already on screen uses it
        to fade in, since coincident strokes would double up and darken.
        """
        connector = cls._tree_segment(source.get_right(), target.get_left())
        if reveal is not None:
            # Stroke only: a connector is an open bezier, and giving it fill
            # opacity would flood the area its curve encloses.
            connector.set_stroke(opacity=0)

        def update(mobject):
            mobject.become(
                cls._tree_segment(source.get_right(), target.get_left())
            )
            if reveal is not None:
                mobject.set_stroke(opacity=reveal())

        connector.add_updater(update)
        return connector

    @staticmethod
    def _split(split, alpha, start, end, *, peeled):
        """Send a cell to its slot in T', shedding the coarse value.

        A cell that is peeled off another is one of a stack of cells sitting
        under it, carrying its own value from the outset: it is drawn behind (see
        ``_deck``), so it is simply hidden by the cell in front until it slides
        clear of it.  The cell that stands in for the one they all came from is
        the only one that has a coarse value to shed.
        """
        box, factor_label, *carried = split

        def update(mobject):
            value = alpha.get_value()
            mobject.move_to(start + (end - start) * value)
            if not peeled:
                factor_label.set_opacity(value)
                carried[0].set_opacity(1.0 - value)

        split.add_updater(update)

    @staticmethod
    def _deck(cells):
        """A stack's cells in drawing order, lowest in front.

        The cells of a mode start out stacked under the one they come from, so
        every cell has to be drawn behind the cells below it: then a cell rising
        into its slot slides out from underneath rather than over the top.
        """
        return tuple(cells[leaf] for leaf in sorted(cells, reverse=True))

    @staticmethod
    def _cell(value, center):
        box = RoundedRectangle(
            corner_radius=0.08,
            width=CELL_H,
            height=CELL_H,
            stroke_color=INK,
            stroke_width=1.8,
            fill_color=PANEL,
            fill_opacity=1,
        )
        label = Text(str(value), color=INK, font=CODE_FONT, font_size=FONT_SIZE)
        label.move_to(box)
        return VGroup(box, label).move_to(center)

    @staticmethod
    def _arrow_tip(tip_point):
        """A rightward caret matching MapstoArrow's tip, at ``tip_point``."""
        return VGroup(
            Line(
                tip_point,
                tip_point - RIGHT * TIP_LENGTH + UP * TIP_WIDTH / 2,
                color=INK,
                stroke_width=STROKE_WIDTH,
            ),
            Line(
                tip_point,
                tip_point - RIGHT * TIP_LENGTH - UP * TIP_WIDTH / 2,
                color=INK,
                stroke_width=STROKE_WIDTH,
            ),
        )

    @staticmethod
    def _tree_segment(start, end):
        # Inset both ends by ARROW_INSET, exactly like a morphism arrow's tail
        # and tip, so segment <-> arrow is only a matter of the tip.  The bend
        # arrives horizontally at the start of the closing run, which the caret
        # of an arrow then sits at the far end of.
        start = start.copy()
        end = end.copy()
        direction = RIGHT if end[0] >= start[0] else LEFT
        start = start + direction * ARROW_INSET
        end = end - direction * ARROW_INSET
        span = abs(end[0] - start[0])
        run = min(ARROW_RUN, 0.45 * span)
        turn = end - direction * run
        handle = direction * 0.42 * abs(turn[0] - start[0])
        segment = CubicBezier(
            start,
            start + handle,
            turn - handle,
            turn,
            color=INK,
            stroke_width=STROKE_WIDTH,
        )
        segment.add_line_to(end)
        return segment
