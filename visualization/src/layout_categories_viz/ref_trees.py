"""Banded tree layout for Ref morphisms.

A Ref morphism is drawn as a general tree between its two stacks: each
codomain cell is the root of its mode's tree, leaves anchor on the source
cells, and every internal grouping of the nested tuple is a junction where
its children's strands merge before continuing rootward.  See
``ref_tree_with_paths`` for the banded layout that keeps strands from
meeting anywhere but their junctions.
"""

import numpy as np
from manim import LEFT, RIGHT, VGroup, VMobject
from tract import NestedTuple

from .style import INK
from .stacks import ARROW_INSET, ARROW_RUN, STROKE_WIDTH


def _strand(start, band_x, end, *, inset_start, inset_end):
    """One child-to-parent strand: a horizontal run to the parent's band,
    then a bend across the band.

    The bend is the stack primitives' tree_segment geometry -- a cubic
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
        # at, exactly tree_segment's geometry (and the Fact fan's).
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
