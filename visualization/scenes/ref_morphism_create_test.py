"""Prototype animation for drawing morphisms in the category Ref.

A Ref morphism is presented by a nested tuple X and points
flat(X) -> rho(X): its source stack holds the leaves of X and its target
stack one cell per top-level mode, holding that mode's product.  Where a
Fact morphism is drawn as a depth-1 fan (each codomain cell gathering its
flat block of domain cells), a Ref morphism is drawn as a general tree:
each codomain cell is the root of its mode's tree, leaves anchor on the
source cells, and every internal grouping of the nested tuple is a junction
where its children's strands merge before continuing rootward.

The layout is banded so that strands can only meet on purpose.  Writing D
for the maximum depth of X's modes, the strip between the stacks is split
into D vertical bands with boundaries c_D = source column, ...,
c_0 = target column, one band per level of nesting; the root band is
double-width, since it hosts the largest descents (from a subtree's rows
to its codomain cell's slot), and the rest split the remainder equally.  A junction at depth k
sits on the boundary c_k, at the height of its bottom-most child, so a
merge reads the way a Fact fan does: the bottom strand continues level and
the others descend onto it.  Every strand runs horizontally at its own
height until it reaches its parent's band [c_(k+1), c_k], and bends only
inside that band.  This makes unintended overlaps impossible: sibling
subtrees occupy disjoint consecutive runs of source rows, a junction's
height is the bottom of its subtree's rows, and a bend's height moves
monotonically between its endpoints, so every merge
stays inside its own subtree's horizontal strip and every pass-through
strand travels at a row no other subtree's strip contains.  Strands of one
merge meet exactly at their junction, and nowhere else.

When X is a Fact morphism (depth 1), D = 1 and the single band spans the
whole strip, so the horizontal run is empty and the bend is exactly the
Fact fan segment: this scheme reduces to the implemented Fact picture.

Junctions carry no cells or values -- like Fact fans, the tree is drawn
purely as strands sharing endpoints.  The bends reuse the pullback scene's
segment geometry (bezier with a horizontal closing run, cell-side insets),
with the insets suppressed at junction endpoints so strands meet exactly.

The scene loops through examples of increasing depth and width to stress
the layout: a pure Fact case, one junction, a deep chain, a single bushy
mode, and a wide mixed-depth morphism.  The library validates every
example: stacks and labels are read off RefMorphism, never restated.
"""

from layout_categories_viz.scene_base import LayoutScene

from dataclasses import dataclass

import numpy as np
from manim import (
    Create,
    FadeIn,
    LaggedStart,
    ORIGIN,
    Text,
    UP,
    VGroup,
)
from tract import NestedTuple, RefMorphism

from layout_categories_viz.style import CODE_FONT, INK

# The library owns the drawing primitives every refinement animation shares,
# so the Ref tree cannot drift from the fans it generalizes.
from layout_categories_viz import stacks
from layout_categories_viz.ref_trees import ref_tree_segments, ref_tree_with_paths
from layout_categories_viz.stacks import (
    make_place,
    CELL_H,
    LABEL_FONT_SIZE,
    MAX_STACK_HEIGHT,
    SLOT_STEP,
)

SOURCE_X = -2.8
TARGET_X = 2.8
LABEL_BUFF = 0.55


@dataclass(frozen=True)
class RefExample:
    """A Ref morphism, specified by its nested tuple alone."""

    nest: tuple

    def morphism(self) -> RefMorphism:
        """The morphism, validated by the library."""
        return RefMorphism(NestedTuple(self.nest))


EXAMPLES = (
    # Depth 2 with flat modes: exactly a Fact morphism, so the trees must
    # degenerate to the Fact fans.
    RefExample(nest=((2, 3), (2, 2), (5,))),
    # One junction: the first mode groups (2, 3) before merging with 4.
    RefExample(nest=(((2, 3), 4), 5)),
    # A deep chain: junctions at every level between root and leaves.
    RefExample(nest=(2, (3, (5, 7)), 11)),
    # A single bushy mode of depth 4: every strand passes a junction.
    RefExample(nest=((((2, 2), 3), (4, (5, 6))),)),
    # Wide and mixed: an unrefined entry, a three-block mode, and a nested
    # tail, stressing the shared-baseline layout and scaling.
    RefExample(nest=(7, ((2, 2), (3, 3), 2), (2, (2, 2)))),
)


class RefMorphismCreateTest(LayoutScene):
    """Draw Ref morphisms as general trees between their two stacks."""

    def construct(self) -> None:
        for index, example in enumerate(EXAMPLES):
            self._play_example(example)
            self.clear_scene(last=index == len(EXAMPLES) - 1)

    def _play_example(self, example: RefExample) -> None:
        f = example.morphism()

        slots = max(len(f.domain), len(f.codomain), 1)
        # The title reserves headroom, so tall stacks scale into a shorter
        # window than MAX_STACK_HEIGHT and sit slightly below center.
        usable_height = MAX_STACK_HEIGHT - 0.9
        scale = min(1.0, usable_height / ((slots - 1) * SLOT_STEP + CELL_H))
        step = SLOT_STEP * scale

        place = make_place(-((slots - 1) * step) / 2 - 0.05, step)

        def cell(value, center):
            return (
                stacks.cell(value, ORIGIN)
                .scale(scale)
                .move_to(center)
            )

        source_cells = [
            cell(value, place(SOURCE_X, index))
            for index, value in enumerate(f.domain)
        ]
        target_cells = [
            cell(value, place(TARGET_X, index))
            for index, value in enumerate(f.codomain)
        ]

        # One tree per top-level mode, its leaves anchored on the mode's
        # consecutive run of source cells.  All modes share the band
        # boundaries of the deepest one.
        global_depth = max((mode.depth() for mode in f.modes), default=1)
        trees = []
        start = 0
        for index, mode in enumerate(f.modes):
            leaf_anchors = [
                source_cells[start + offset].get_right()
                for offset in range(mode.length())
            ]
            trees.append(
                ref_tree_segments(
                    mode,
                    leaf_anchors,
                    target_cells[index].get_left(),
                    global_depth,
                )
            )
            start += mode.length()

        title = Text(
            f"X = {f.nest}", color=INK, font=CODE_FONT, font_size=27
        ).to_edge(UP, buff=0.38)
        label_y = place(0, 0)[1] - CELL_H * scale / 2 - LABEL_BUFF
        source_label = (
            Text("flat(X)", color=INK, font=CODE_FONT, font_size=LABEL_FONT_SIZE)
            .scale(scale)
            .move_to(np.array([SOURCE_X, label_y, 0.0]))
        )
        target_label = (
            Text("ρ(X)", color=INK, font=CODE_FONT, font_size=LABEL_FONT_SIZE)
            .scale(scale)
            .move_to(np.array([TARGET_X, label_y, 0.0]))
        )

        self.play(
            FadeIn(VGroup(*source_cells)),
            FadeIn(VGroup(*target_cells)),
            FadeIn(title),
            FadeIn(source_label),
            FadeIn(target_label),
        )
        self.play(
            LaggedStart(
                *(
                    Create(segment)
                    for tree in trees
                    for segment in tree
                ),
                lag_ratio=0.12,
                run_time=1.6,
            )
        )
        self.wait(1.0)
