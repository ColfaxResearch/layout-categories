"""Shared cell/deck/split/arrow drawing primitives for stack animations.

Every refinement animation draws the same picture: stacks of square cells on
a shared bottom baseline, refinement segments and map arrows sharing one
bezier geometry, cells splitting in place and peeling off one another.  The
primitives live here so the scenes that share them cannot drift apart.
"""

import numpy as np
from manim import (
    CubicBezier,
    DOWN,
    LEFT,
    Line,
    RIGHT,
    RoundedRectangle,
    Text,
    UP,
    VGroup,
)
from tract import NestedTuple

from .style import CODE_FONT, INK, PANEL
from .tuple_morphism import MapstoArrow

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


def leaf_groups(refined: NestedTuple, coarse: NestedTuple) -> tuple:
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


def revealed(value: float) -> float:
    """How much of a peeled copy is showing, once the motion has begun."""
    return min(1.0, value / REVEAL)


def make_place(baseline: float, step: float):
    """The slot-placement function of a figure with the given geometry.

    Every stack scene places cell ``index`` of the stack in column ``column``
    at ``(column, baseline + index * step)``: one bottom baseline and one
    uniform step, shared by every stack in the figure.
    """

    def place(column, index):
        return np.array((column, baseline + index * step, 0.0))

    return place


def cell(value, center):
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


def arrow_tip(tip_point):
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


def tree_segment(start, end):
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


def segment_arrow(source_cell, target_cell):
    """A map arrow whose shaft is exactly a refinement segment.

    Sharing the segment's curvature is what lets an arrow be replaced by the
    fan it splits into without any snap: only the caret differs.
    """
    shaft = tree_segment(source_cell.get_right(), target_cell.get_left())
    tip = arrow_tip(target_cell.get_left() + LEFT * ARROW_INSET)
    start = shaft.get_start()
    tail = Line(
        start + DOWN * TAIL_LENGTH / 2,
        start + UP * TAIL_LENGTH / 2,
        color=INK,
        stroke_width=STROKE_WIDTH,
    ).set_opacity(0)
    return MapstoArrow.from_parts(tail, shaft, tip)


def attached(source, target, *, reveal=None):
    """A connector that follows both of the cells it joins.

    ``reveal`` is a callable returning how much of the connector is showing.
    A connector that starts out coincident with one already on screen uses it
    to fade in, since coincident strokes would double up and darken.
    """
    connector = tree_segment(source.get_right(), target.get_left())
    if reveal is not None:
        # Stroke only: a connector is an open bezier, and giving it fill
        # opacity would flood the area its curve encloses.
        connector.set_stroke(opacity=0)

    def update(mobject):
        mobject.become(
            tree_segment(source.get_right(), target.get_left())
        )
        if reveal is not None:
            mobject.set_stroke(opacity=reveal())

    connector.add_updater(update)
    return connector


def split_cell(split, alpha, start, end, *, peeled):
    """Send a cell to its slot in T', shedding the coarse value.

    A cell that is peeled off another is one of a stack of cells sitting
    under it, carrying its own value from the outset: it is drawn behind (see
    ``deck``), so it is simply hidden by the cell in front until it slides
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


def deck(cells):
    """A stack's cells in drawing order, lowest in front.

    The cells of a mode start out stacked under the one they come from, so
    every cell has to be drawn behind the cells below it: then a cell rising
    into its slot slides out from underneath rather than over the top.
    """
    return tuple(cells[leaf] for leaf in sorted(cells, reverse=True))
