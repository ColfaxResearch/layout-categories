"""Isolated test scene for the cell-splitting gesture.

This is the one motion at the heart of stage 1 of both the pullback and the
pushforward: a stack of cells refines *in place*, each cell separating into its
own factors with no cell ever changing places.  Everything else those scenes draw
-- the morphisms, the fans, the reordering -- is stripped away, so the gesture can
be judged on its own.

It is the gesture those scenes actually use, not a variant: the cells come from
``TuplePullbackTest._cell`` and the motion from ``TuplePullbackTest._split``, so
whatever is tuned here changes what they draw.  As it stands, the cells of a mode
all start on the cell they come from and each travels to its slot while the coarse
value it carried fades out and its own factor fades in.  The first cell of a mode
keeps that cell's slot and stands in for it, so it is opaque throughout; the rest
are stacked under it, carrying their own values from the outset and drawn behind,
so each is hidden by the cell in front of it until it slides clear -- as if the
factors had been stacked under the cell all along.  A mode's split pushes
everything above it up, which is why a cell high in the stack can travel several
slots without splitting at all.

The knobs: ``SPLIT_RUN_TIME`` here, ``REVEAL`` in the pullback scene, the rate
function the motion is played with, and ``TuplePullbackTest._split`` itself.
"""

from math import prod

import numpy as np
from manim import (
    DOWN,
    FadeIn,
    FadeOut,
    ORIGIN,
    Scene,
    Text,
    ValueTracker,
    smooth,
)
from tract import NestedTuple

from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK
from scenes.tuple_pullback_test import (
    CELL_H,
    LABEL_FONT_SIZE,
    MAX_STACK_HEIGHT,
    SLOT_STEP,
    TuplePullbackTest,
    _groups,
)


# Each example is a tuple of modes; a mode is the factor tuple its cell splits
# into.  So the stack starts as the products and ends as the factors.
EXAMPLES = (
    ((2, 3),),  # one cell, split in two
    ((2, 3, 5),),  # one cell, split in three
    ((2, 3), (2, 2, 3)),  # blocks of two and three
    ((2, 3), (7,), (2, 2, 3)),  # a middle mode that does not split
    ((2, 2, 3), (5,), (2, 4), (3, 2)),  # eight leaves out of four modes
)

SPLIT_RUN_TIME = 1.4  # as stage 1 of the pullback and the pushforward plays it
# One stack alone leaves the frame empty at the size the real scenes use, so the
# cells grow to fill the height available, up to this factor.  The gesture is
# scale invariant, so anything tuned here transfers back unchanged.
ZOOM = 1.8


class TupleCellSplitTest(Scene):
    """Refine a stack of cells in place, and nothing else."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        for index, modes in enumerate(EXAMPLES):
            self._show_split(modes)
            self._clear_scene(last=index == len(EXAMPLES) - 1)

    def _show_split(self, modes) -> None:
        coarse = NestedTuple(tuple(prod(factors) for factors in modes))
        refined = NestedTuple(
            tuple(
                factors[0] if len(factors) == 1 else tuple(factors)
                for factors in modes
            )
        )
        values = refined.flatten()

        scale = min(
            ZOOM,
            MAX_STACK_HEIGHT / ((refined.length() - 1) * SLOT_STEP + CELL_H),
        )
        step = SLOT_STEP * scale
        baseline = -((refined.length() - 1) * step) / 2

        def place(index):
            return np.array((0.0, baseline + index * step, 0.0))

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

        coarse_cells = [
            cell(coarse.entry(mode + 1), place(mode))
            for mode in range(coarse.length())
        ]
        label_coarse = label("T", coarse_cells[0])
        self.play(
            *(FadeIn(mobject) for mobject in (*coarse_cells, label_coarse)),
            run_time=0.8,
        )
        self.wait(0.8)

        alpha = ValueTracker(0.0)
        splits = {}
        for mode, leaves in enumerate(_groups(refined, coarse)):
            start = coarse_cells[mode].get_center()
            for index, leaf in enumerate(leaves):
                # The first cell of a mode stands in for the cell it came from,
                # and sheds its value; the rest are stacked under it, already
                # carrying their own.
                split = cell(values[leaf], start)
                if not index:
                    carried = cell(coarse.entry(mode + 1), start)[1]
                    split[1].set_opacity(0)
                    split.add(carried)
                TuplePullbackTest._split(
                    split, alpha, start, place(leaf), peeled=bool(index)
                )
                splits[leaf] = split

        # The cells that stand in for the coarse ones are coincident with them,
        # and the rest are stacked under them: lowest in front.
        self.remove(*coarse_cells)
        self.add(*TuplePullbackTest._deck(splits))

        label_refined = label("T'", cell(values[0], place(0)))
        self.play(
            alpha.animate.set_value(1.0),
            FadeOut(label_coarse),
            FadeIn(label_refined),
            run_time=SPLIT_RUN_TIME,
            rate_func=smooth,
        )
        for split in splits.values():
            split.clear_updaters()
        self.wait(1.6)

    def _clear_scene(self, *, last) -> None:
        self.play(*(FadeOut(m) for m in self.mobjects), run_time=0.6)
        self.clear()
        if not last:
            self.wait(0.2)
