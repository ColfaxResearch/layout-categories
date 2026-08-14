"""Isolated test scene for two decorative variations.

Both are tried on the same figure -- the opening of a pullback, and the split
that refines its middle stack -- so they can be judged against the plain drawing
the other scenes use.  Neither changes what any animation means; they are there
to make the same statement more pleasant to look at.

1. Mode colour.  Each mode of the coarse tuple is given a hue from the palette
   in ``style.py`` (defined there long ago and never used).  The cell takes it as
   a wash of fill and a tint in its stroke, and so do the factors it splits into
   and every connector that carries it -- so a mode is a coloured thread running
   through the refinement, and a reordering becomes legible as colour changing
   places.  Values stay in ink, so nothing turns decorative at the expense of
   being read.

2. Paper shadow.  Every cell sits on a soft shadow, offset down and to the right.
   It costs nothing but gives the stack depth, and it pays off exactly where the
   splitting gesture already suggests it: the cells stacked under a cell are
   hidden by it, but their shadows peep out from under its edge before they slide
   clear -- so the deck looks like a deck.
"""

from math import prod

import numpy as np
from manim import (
    DOWN,
    FadeIn,
    FadeOut,
    RIGHT,
    Scene,
    Text,
    ValueTracker,
    interpolate_color,
    smooth,
)
from tract import NestedTuple, Nest_morphism

from layout_categories_viz.animations import DrawMapstoTip, UndrawMapstoTip

from layout_categories_viz.style import (
    BACKGROUND,
    CODE_FONT,
    INK,
    MODE_COLORS,
    PANEL,
)
from scenes.tuple_pullback_test import (
    ARROW_INSET,
    LEFT_X,
    MID_X,
    RIGHT_X,
    SLOT_STEP,
    TuplePullbackTest,
    _groups,
)


# The figure both variations are tried on: T = (6, 12) refined into blocks of
# two and three, with f the identity, so the eye is free for the decoration.
MODES = ((2, 3), (2, 2, 3))
MAPPING = (1, 2)

FILL_WASH = 0.16  # how much of a mode's hue its cells are washed with
STROKE_TINT = 0.45  # and how far their outlines are drawn toward it
CONNECTOR_TINT = 0.55
SHADOW_OFFSET = np.array((0.055, -0.055, 0.0))
SHADOW_OPACITY = 0.13


def _hue(mode: int):
    return MODE_COLORS[mode % len(MODE_COLORS)]


class DecorativeVariationsTest(Scene):
    """Plain, then coloured, then coloured and shadowed."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        variations = (
            ("as it is drawn now", False, False),
            ("a hue per mode, threaded through the refinement", True, False),
            ("and every cell on a paper shadow", True, True),
        )
        for index, (caption, colour, shadow) in enumerate(variations):
            self._show(caption=caption, colour=colour, shadow=shadow)
            self._clear_scene(last=index == len(variations) - 1)

    def _show(self, *, caption: str, colour: bool, shadow: bool) -> None:
        T = NestedTuple(tuple(prod(factors) for factors in MODES))
        Tprime = NestedTuple(
            tuple(
                factors[0] if len(factors) == 1 else tuple(factors)
                for factors in MODES
            )
        )
        S = NestedTuple(T.data)
        Nest_morphism(S, T, MAPPING)  # validates the figure
        groups = _groups(Tprime, T)
        owner = {leaf: mode for mode, leaves in enumerate(groups) for leaf in leaves}
        values = Tprime.flatten()

        baseline = -((Tprime.length() - 1) * SLOT_STEP) / 2

        def place(column, index):
            return np.array((column, baseline + index * SLOT_STEP, 0.0))

        def cell(value, center, mode):
            drawn = TuplePullbackTest._cell(value, center)
            if colour:
                box = drawn[0]
                box.set_fill(
                    interpolate_color(PANEL, _hue(mode), FILL_WASH), opacity=1
                )
                box.set_stroke(
                    interpolate_color(INK, _hue(mode), STROKE_TINT)
                )
            return drawn

        def connector(start, end, mode, *, arrow=False):
            drawn = (
                TuplePullbackTest._segment_arrow(start, end)
                if arrow
                else TuplePullbackTest._tree_segment(
                    start.get_right(), end.get_left()
                )
            )
            if colour:
                drawn.set_stroke(
                    interpolate_color(INK, _hue(mode), CONNECTOR_TINT)
                )
            return drawn

        def shade(drawn_cell):
            """A soft shadow that follows a cell, drawn just behind it."""
            shadow_shape = drawn_cell[0].copy()
            shadow_shape.set_stroke(width=0)
            shadow_shape.set_fill(INK, opacity=SHADOW_OPACITY)
            shadow_shape.shift(SHADOW_OFFSET)
            shadow_shape.add_updater(
                lambda mobject, source=drawn_cell: mobject.move_to(
                    source.get_center() + SHADOW_OFFSET
                )
            )
            return shadow_shape

        def show(drawn_cell):
            """Add a cell, on its shadow if it has one."""
            if shadow:
                self.add(shade(drawn_cell))
            self.add(drawn_cell)

        # --- The opening of a pullback. --------------------------------------
        s_cells = [
            cell(S.entry(mode + 1), place(LEFT_X, mode), mode)
            for mode in range(S.length())
        ]
        t_cells = [
            cell(T.entry(mode + 1), place(MID_X, mode), mode)
            for mode in range(T.length())
        ]
        tp_cells = [
            cell(value, place(RIGHT_X, leaf), owner[leaf])
            for leaf, value in enumerate(values)
        ]
        f_arrows = [
            connector(s_cells[mode], t_cells[target - 1], mode, arrow=True)
            for mode, target in enumerate(MAPPING)
            if target
        ]
        refinement = [
            connector(t_cells[mode], tp_cells[leaf], mode)
            for mode, leaves in enumerate(groups)
            for leaf in leaves
        ]
        note = Text(
            caption, color=INK, font=CODE_FONT, font_size=20
        ).to_edge(DOWN, buff=0.45)

        self.add(*f_arrows, *refinement)
        for drawn in (*s_cells, *t_cells, *tp_cells):
            show(drawn)
        self.play(
            *(
                FadeIn(mobject)
                for mobject in (
                    *s_cells,
                    *t_cells,
                    *tp_cells,
                    *f_arrows,
                    *refinement,
                    note,
                )
            ),
            run_time=1.0,
        )
        self.wait(1.0)

        # The arrows of f give up their tips, as in the pullback: they are
        # becoming the fan that exhibits S' as a refinement of S.
        self.play(
            *(UndrawMapstoTip(arrow.tip) for arrow in f_arrows), run_time=0.4
        )

        # --- The split that refines it, with the cells stacked underneath. ----
        alpha = ValueTracker(0.0)
        splits = {}
        for mode, leaves in enumerate(groups):
            start = t_cells[mode].get_center()
            for index, leaf in enumerate(leaves):
                split = cell(values[leaf], start, mode)
                if not index:
                    carried = cell(T.entry(mode + 1), start, mode)[1]
                    split[1].set_opacity(0)
                    split.add(carried)
                TuplePullbackTest._split(
                    split, alpha, start, place(MID_X, leaf), peeled=bool(index)
                )
                splits[leaf] = split

        fans = {
            leaf: TuplePullbackTest._attached(
                s_cells[owner[leaf]], splits[leaf]
            )
            for leaf in splits
        }
        arrows = {
            leaf: TuplePullbackTest._attached(splits[leaf], tp_cells[leaf])
            for leaf in splits
        }
        if colour:
            for leaf in splits:
                tint = interpolate_color(INK, _hue(owner[leaf]), CONNECTOR_TINT)
                fans[leaf].set_stroke(tint)
                arrows[leaf].set_stroke(tint)

        self.remove(*f_arrows, *refinement, *t_cells)
        self.add(*fans.values(), *arrows.values())
        for drawn in TuplePullbackTest._deck(splits):
            show(drawn)

        self.play(
            alpha.animate.set_value(1.0), run_time=1.4, rate_func=smooth
        )
        for split in splits.values():
            split.clear_updaters()

        # The connectors on the right are parallel now, so they take their tips.
        tips = {
            leaf: TuplePullbackTest._arrow_tip(
                tp_cells[leaf].get_left() - RIGHT * ARROW_INSET
            )
            for leaf in splits
        }
        if colour:
            for leaf, tip in tips.items():
                tip.set_stroke(
                    interpolate_color(INK, _hue(owner[leaf]), CONNECTOR_TINT)
                )
        self.add(*tips.values())
        self.play(
            *(DrawMapstoTip(tip) for tip in tips.values()), run_time=0.4
        )
        self.wait(1.8)

    def _clear_scene(self, *, last) -> None:
        self.play(*(FadeOut(m) for m in self.mobjects), run_time=0.6)
        self.clear()
        if not last:
            self.wait(0.2)
