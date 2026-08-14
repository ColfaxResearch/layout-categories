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
example: stacks and labels are read off Ref_morphism, never restated.
"""

from dataclasses import dataclass

import numpy as np
from manim import (
    Create,
    FadeIn,
    LaggedStart,
    LEFT,
    ORIGIN,
    RIGHT,
    Scene,
    Text,
    UP,
    VGroup,
    VMobject,
)
from tract import NestedTuple, Ref_morphism

from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK

# The pullback scene owns the drawing primitives every refinement animation
# shares, so the Ref tree cannot drift from the fans it generalizes.
from scenes.tuple_pullback_test import (
    ARROW_INSET,
    ARROW_RUN,
    CELL_H,
    LABEL_FONT_SIZE,
    MAX_STACK_HEIGHT,
    SLOT_STEP,
    STROKE_WIDTH,
    TuplePullbackTest,
)

SOURCE_X = -2.8
TARGET_X = 2.8
LABEL_BUFF = 0.55


def _strand(start, band_x, end, *, inset_start, inset_end):
    """One child-to-parent strand: a horizontal run to the parent's band,
    then a bend across the band.

    The bend is tuple_pullback_test._tree_segment's geometry -- a cubic
    arriving horizontally at a closing run -- with the cell-side insets
    made optional so strands can share a junction endpoint the way tree
    edges share their root-side endpoint.  When the strand starts on the
    band boundary the horizontal run is empty and the strand is exactly a
    bend, which for a depth-1 mode spans the whole strip: the Fact fan.
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    direction = RIGHT if end[0] >= start[0] else LEFT
    if inset_start:
        start = start + direction * ARROW_INSET
    if inset_end:
        end = end - direction * ARROW_INSET

    # The band boundary can sit behind an inset cell edge (the Fact case,
    # where the band starts at the source column): never run backward.
    # Mirrored trees (root on the left, as a span's backward leg) run their
    # strands leftward, so "behind" follows the strand's direction.
    if direction[0] >= 0:
        bend_x = max(band_x, start[0])
    else:
        bend_x = min(band_x, start[0])
    bend_start = np.array([bend_x, start[1], 0.0])
    strand = VMobject(stroke_color=INK, stroke_width=STROKE_WIDTH)
    strand.start_new_path(start)
    if abs(bend_start[0] - start[0]) > 1e-6:
        strand.add_line_to(bend_start)

    span = abs(end[0] - bend_start[0])
    if inset_end:
        # A cell arrival keeps the closing run an arrow's caret would sit
        # at, exactly _tree_segment's geometry (and the Fact fan's).
        run = min(ARROW_RUN, 0.45 * span)
        handle_fraction = 0.42
    else:
        # A junction arrival carries no caret, so the whole band is spent
        # on the bend, with symmetric handles: the gentlest cubic the band
        # allows.  Its arrival tangent is still horizontal, so the
        # continuing strand leaves the junction smoothly.
        run = 0.0
        handle_fraction = 0.5
    turn = end - direction * run
    handle = direction * handle_fraction * abs(turn[0] - bend_start[0])
    strand.add_cubic_bezier_curve_to(bend_start + handle, turn - handle, turn)
    # With no closing run the bend already ends at the endpoint; a
    # zero-length closing line would be a degenerate curve, which the
    # arc-length route matching cannot partition.
    if run > 1e-9:
        strand.add_line_to(end)
    return strand


def ref_tree_with_paths(
    mode: NestedTuple, leaf_anchors, root_anchor, global_depth: int
):
    """The banded tree of one top-level mode, with each leaf's root path.

    ``leaf_anchors`` are the right edges of the mode's source cells in leaf
    order, ``root_anchor`` the left edge of its codomain cell, and
    ``global_depth`` the maximum depth D over all of the morphism's modes,
    shared so every mode agrees on the band boundaries c_D, ..., c_0.  A
    junction at depth k sits on c_k at the height of its bottom-most child;
    each strand runs horizontally at its own height and bends only inside
    its parent's band [c_(k+1), c_k].

    Returns (segments, paths, edges, leaf_addresses): the strands as one
    VGroup; per leaf (in leaf order) the list of strands from that leaf to
    the root, so a leaf's route through the tree can be glued end to end
    (consecutive strands of a path meet exactly at their shared junction
    anchor); the strands again, keyed by the tree address of the child they
    leave (the index path from the mode's top-level data down to that
    node); and per leaf its tree address.  Addresses are what composition
    scenes match on: grafting a tree under leaf address b puts its address-a
    edge at address b + a.
    """
    depth = max(global_depth, 1)
    source_x = leaf_anchors[0][0] if len(leaf_anchors) else root_anchor[0]

    # The root band is double-width: it hosts the largest descents (from a
    # subtree's rows to its codomain cell's slot), so it gets the most
    # horizontal room.  For depth 1 the root band is the whole strip either
    # way, keeping the Fact picture unchanged.
    total_weight = depth + 1

    def boundary_x(level):
        weight = 0 if level == 0 else min(level, depth) + 1
        return root_anchor[0] + (source_x - root_anchor[0]) * (
            weight / total_weight
        )

    leaves = iter(enumerate(leaf_anchors))
    segments = VGroup()
    paths = [[] for _ in leaf_anchors]
    edges = {}
    leaf_addresses = [None] * len(leaf_anchors)

    def anchor_of(data, level, address):
        """The meeting point of a subtree's strands and its leaf indices,
        recursing on children.

        Returns (None, []) for a subtree with no leaves (an empty
        factorization of 1), which contributes no strands.
        """
        if isinstance(data, int):
            index, anchor = next(leaves)
            leaf_addresses[index] = address
            return np.array(anchor, dtype=float), [index]
        children = [
            anchor_of(child, level + 1, address + (position,))
            for position, child in enumerate(data)
        ]
        anchored = [
            (child, position, child_anchor, indices)
            for position, (child, (child_anchor, indices)) in enumerate(
                zip(data, children)
            )
            if child_anchor is not None
        ]
        if not anchored:
            return None, []
        if level == 0:
            anchor = np.array(root_anchor, dtype=float)
        else:
            anchor = np.array(
                [
                    boundary_x(level),
                    min(
                        child_anchor[1]
                        for _, _, child_anchor, _ in anchored
                    ),
                    0.0,
                ]
            )
        for child, position, child_anchor, indices in anchored:
            strand = _strand(
                child_anchor,
                boundary_x(level + 1),
                anchor,
                inset_start=isinstance(child, int),
                inset_end=level == 0,
            )
            segments.add(strand)
            edges[address + (position,)] = strand
            # Children recurse before their strand is built, so each leaf's
            # path accumulates strands in leaf-to-root order.
            for index in indices:
                paths[index].append(strand)
        return anchor, [
            index for _, _, _, indices in anchored for index in indices
        ]

    data = mode.data if isinstance(mode.data, tuple) else (mode.data,)
    anchor_of(data, 0, ())
    return segments, paths, edges, leaf_addresses


def ref_tree_segments(
    mode: NestedTuple, leaf_anchors, root_anchor, global_depth: int
) -> VGroup:
    """The banded tree of one top-level mode, as strands from leaves to root.

    See ref_tree_with_paths for the layout; this returns the strands alone.
    """
    segments, _, _, _ = ref_tree_with_paths(
        mode, leaf_anchors, root_anchor, global_depth
    )
    return segments


@dataclass(frozen=True)
class RefExample:
    """A Ref morphism, specified by its nested tuple alone."""

    nest: tuple

    def morphism(self) -> Ref_morphism:
        """The morphism, validated by the library."""
        return Ref_morphism(NestedTuple(self.nest))


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


class RefMorphismCreateTest(Scene):
    """Draw Ref morphisms as general trees between their two stacks."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        for index, example in enumerate(EXAMPLES):
            self._play_example(example)
            TuplePullbackTest._clear_scene(self, last=index == len(EXAMPLES) - 1)

    def _play_example(self, example: RefExample) -> None:
        f = example.morphism()

        slots = max(len(f.domain), len(f.codomain), 1)
        # The title reserves headroom, so tall stacks scale into a shorter
        # window than MAX_STACK_HEIGHT and sit slightly below center.
        usable_height = MAX_STACK_HEIGHT - 0.9
        scale = min(1.0, usable_height / ((slots - 1) * SLOT_STEP + CELL_H))
        step = SLOT_STEP * scale

        def place(column, index):
            return np.array(
                (column, -((slots - 1) * step) / 2 + index * step - 0.05, 0.0)
            )

        def cell(value, center):
            return (
                TuplePullbackTest._cell(value, ORIGIN)
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
