"""Vertical shape:stride depiction of a nested layout.

A layout L = S : D is drawn as two row-aligned vertical stacks: the leaves
of the shape tuple S in a left column and the leaves of the stride tuple D
in a right column, with a colon between each pair.  Shape cells and stride
cells carry distinct fills (SHAPE_FILL and STRIDE_FILL in style.py) so the
two halves of the layout read apart at a glance.

The nesting of the layout is the inverse of the Ref morphism of S,
flat(S) -> rho(S), drawn in full to the left of the layout: the shape
stack doubles as the morphism's source flat(S), and its target rho(S)
stands as a stack of product cells on a root column further left, one per
top-level mode.  Where the Ref picture gathers leaves rightward into their
codomain products, here the trees fan out of the root cells into the shape
stack.  The trees are the banded ref trees themselves, mirrored — same
junctions, same band boundaries — so the depiction cannot drift from the
morphism it inverts.  Since D is congruent to S, one tree carries the
nesting of both stacks.
"""

from dataclasses import dataclass

import numpy as np
from manim import Text, VGroup
from tract import NestedTuple, RefMorphism

from .ref_trees import ref_tree_segments
from .stacks import cell as stack_cell
from .style import CODE_FONT, MUTED, SHAPE_FILL, STRIDE_FILL

ROOT_X = -3.4
SHAPE_X = -0.8
STRIDE_X = 0.8
COLON_FONT_SIZE = 30


def fitted_cell(value, center, *, fill, scale: float = 1.0):
    """A stack cell whose label shrinks into the box when it outgrows it.

    Strides outgrow shapes fast; a wide value shrinks into its cell rather
    than spilling over the colon and its neighbor.
    """
    group = stack_cell(value, np.zeros(3), fill=fill)
    box, label = group
    max_width = box.width * 0.78
    if label.width > max_width:
        label.scale_to_fit_width(max_width)
    return group.scale(scale).move_to(center)


def layout_colon(center, *, scale: float = 1.0) -> Text:
    """The colon drawn between a paired shape and stride entry."""
    return (
        Text(":", color=MUTED, font=CODE_FONT, font_size=COLON_FONT_SIZE)
        .scale(scale)
        .move_to(center)
    )


@dataclass(frozen=True)
class LayoutDiagram:
    """The mobjects of one layout depiction, in drawing order."""

    shape: NestedTuple
    stride: NestedTuple
    root_cells: tuple  # the target rho(S), one product cell per mode
    shape_cells: tuple
    stride_cells: tuple
    colons: VGroup
    trees: tuple  # one VGroup of strands per top-level mode

    def cells(self) -> VGroup:
        return VGroup(*self.root_cells, *self.shape_cells, *self.stride_cells)

    def strands(self):
        return tuple(strand for tree in self.trees for strand in tree)


def layout_diagram(
    shape,
    stride,
    place,
    *,
    scale: float = 1.0,
    shape_x: float = SHAPE_X,
    stride_x: float = STRIDE_X,
    root_x: float = ROOT_X,
) -> LayoutDiagram:
    """Build the depiction of the layout ``shape : stride``.

    ``place(column, index)`` is the figure's slot-placement function (see
    stacks.make_place); leaf ``index`` of both tuples sits in its row, first
    leaf at the bottom.  The full Ref morphism of the shape stands to the
    left: its target rho(S) as shape-filled product cells on the column
    ``root_x``, mode ``index`` in slot ``index`` exactly as the forward Ref
    picture places its codomain, with each mode's tree rooted on its
    product cell.
    """
    shape = shape if isinstance(shape, NestedTuple) else NestedTuple(shape)
    stride = stride if isinstance(stride, NestedTuple) else NestedTuple(stride)
    if not shape.is_congruent_to(stride):
        raise ValueError(
            f"Shape {shape} and stride {stride} are not congruent"
        )
    # The nesting datum is exactly this morphism, drawn inverted below.
    ref = RefMorphism(shape)

    shape_cells = tuple(
        fitted_cell(value, place(shape_x, index), fill=SHAPE_FILL, scale=scale)
        for index, value in enumerate(shape.flatten())
    )
    stride_cells = tuple(
        fitted_cell(
            value, place(stride_x, index), fill=STRIDE_FILL, scale=scale
        )
        for index, value in enumerate(stride.flatten())
    )
    colons = VGroup(
        *(
            layout_colon(place((shape_x + stride_x) / 2, index), scale=scale)
            for index in range(len(shape_cells))
        )
    )

    # The target rho(S) carries the shape fill: its cells are products of
    # shape entries, so both columns read as the shape side of the layout.
    root_cells = tuple(
        fitted_cell(value, place(root_x, index), fill=SHAPE_FILL, scale=scale)
        for index, value in enumerate(ref.codomain)
    )

    # One mirrored tree per top-level mode, leaves anchored on the shape
    # cells' left edges: the Ref morphism read backward, from rho(S) on the
    # root column out to flat(S).  All modes share the deepest mode's band
    # boundaries, as in the forward picture.
    global_depth = max((mode.depth() for mode in ref.modes), default=1)
    trees = []
    start = 0
    for index, mode in enumerate(ref.modes):
        leaf_anchors = [
            shape_cells[start + offset].get_left()
            for offset in range(mode.length())
        ]
        trees.append(
            ref_tree_segments(
                mode,
                leaf_anchors,
                root_cells[index].get_right(),
                global_depth,
            )
        )
        start += mode.length()

    return LayoutDiagram(
        shape=shape,
        stride=stride,
        root_cells=root_cells,
        shape_cells=shape_cells,
        stride_cells=stride_cells,
        colons=colons,
        trees=tuple(trees),
    )
