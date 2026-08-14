"""Shared geometry and construction helpers for the span-family scenes.

``span_composition_test``, ``cospan_composition_test``, and
``ref_span_composition_test`` all draw the same five-stack picture: stacks
of square cells on one bottom baseline, one uniform step, scaled to fit,
with stack labels below the baseline and leg labels below those.  The
common geometry and the construction blocks that are shared verbatim live
here so the three scenes cannot drift apart.  Blocks that genuinely differ
(the cospan's mirrored gesture, the Ref scenes' banded trees) stay in
their scenes.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from manim import Create, FadeIn, ORIGIN, Succession, Text

from layout_categories_viz import stacks
from layout_categories_viz.stacks import (
    CELL_H,
    LABEL_FONT_SIZE,
    MAX_STACK_HEIGHT,
    SLOT_STEP,
    make_place,
    revealed,
)
from layout_categories_viz.style import CODE_FONT, INK

COLUMN_GAP = 3.0
COL_S, COL_X, COL_T, COL_Y, COL_U = (
    (index - 2) * COLUMN_GAP for index in range(5)
)
# The composite (co)span occupies the footprint of a single one: the outer
# stacks come in by one column pitch each and the refined middle stack,
# already in the middle, stays put.
CONTRACTION = COLUMN_GAP


@dataclass(frozen=True)
class SpanGeometry:
    """One bottom baseline, one uniform step, scaled to fit."""

    scale: float
    step: float
    place: Callable
    cell: Callable
    stack_label: Callable
    leg_label: Callable
    stack_label_y: float
    leg_label_y: float


def span_geometry(*stack_lengths) -> SpanGeometry:
    """The five-stack figure's geometry, from the lengths of its stacks."""
    slots = max(stack_lengths)
    scale = min(1.0, MAX_STACK_HEIGHT / ((slots - 1) * SLOT_STEP + CELL_H))
    step = SLOT_STEP * scale

    place = make_place(-((slots - 1) * step) / 2 + 0.45, step)

    def cell(value, center):
        return (
            stacks.cell(value, ORIGIN)
            .scale(scale)
            .move_to(center)
        )

    stack_label_y = place(0, 0)[1] - CELL_H * scale / 2 - 0.42
    leg_label_y = stack_label_y - 0.62

    def stack_label(text, column):
        return (
            Text(text, color=INK, font=CODE_FONT, font_size=LABEL_FONT_SIZE)
            .scale(scale)
            .move_to(np.array([column, stack_label_y, 0.0]))
        )

    def leg_label(text, x):
        return (
            Text(text, color=INK, font=CODE_FONT, font_size=28)
            .scale(scale)
            .move_to(np.array([x, leg_label_y, 0.0]))
        )

    return SpanGeometry(
        scale=scale,
        step=step,
        place=place,
        cell=cell,
        stack_label=stack_label,
        leg_label=leg_label,
        stack_label_y=stack_label_y,
        leg_label_y=leg_label_y,
    )


def composition_labels(
    left_text,
    right_text,
    *,
    scale,
    leg_label_y,
    left_symbol_index,
    right_symbol_index,
):
    """The composed leg labels, landing mid-gap of the contracted span.

    Returns ``(left_reference, right_reference, left_symbol, right_symbol)``.
    The references are normally typeset labels supplying the exact glyph
    destinations; Text submobjects include space glyphs, so the caller names
    which glyph of each reference is the ∘.
    """

    def reference(text, x):
        label = Text(
            text, color=INK, font=CODE_FONT, font_size=28
        ).scale(scale)
        label.move_to(np.array([x, leg_label_y, 0.0]))
        return label

    left_reference = reference(left_text, (COL_S + CONTRACTION + COL_T) / 2)
    right_reference = reference(right_text, (COL_T + COL_U - CONTRACTION) / 2)

    def symbol(glyph):
        mark = Text("∘", color=INK, font=CODE_FONT, font_size=28)
        mark.scale(scale).move_to(glyph)
        return mark

    return (
        left_reference,
        right_reference,
        symbol(left_reference[left_symbol_index]),
        symbol(right_reference[right_symbol_index]),
    )


def owner_of_leaf(modes, leaf: int) -> int:
    """Which coarse cell a refined leaf belongs to, by mode lengths."""
    running = 0
    for k, mode in enumerate(modes):
        running += len(mode)
        if leaf < running:
            return k
    raise ValueError(f"Leaf {leaf} outside the refined stack.")


def build_middle_splits(
    *,
    first,
    second,
    y_groups,
    v_cells,
    x_cells,
    y_cells,
    source_apex,
    alpha,
    cell,
    place,
    skip_modes=(),
    slot_of_leaf=None,
):
    """Split cells for the pullback gesture, with fans and connectors.

    Each middle cell splits in place along the second backward leg: a fan
    from its apex cell replaces the map arrow and a parallel connector to
    its Y cell replaces the backward strand.  ``skip_modes`` leaves the
    named T cells unsplit and ``slot_of_leaf`` overrides where each split
    piece travels (both used by the single-beat gesture).
    """
    if slot_of_leaf is None:
        slot_of_leaf = lambda leaf: leaf  # noqa: E731
    middle_cells, middle_fans, connectors = {}, {}, {}
    for mode, leaves in enumerate(y_groups):
        if mode in skip_modes:
            continue
        start_center = v_cells[mode].get_center()
        for index, leaf in enumerate(leaves):
            split = cell(second.apex[leaf], start_center)
            if not index:
                coarse = cell(first.codomain[mode], start_center)[1]
                split[1].set_opacity(0)
                split.add(coarse)
            stacks.split_cell(
                split,
                alpha,
                start_center,
                place(COL_T, slot_of_leaf(leaf)),
                peeled=bool(index),
            )
            middle_cells[leaf] = split
            if mode in source_apex:
                middle_fans[leaf] = stacks.attached(
                    x_cells[source_apex[mode]],
                    split,
                    reveal=(
                        (lambda: revealed(alpha.get_value()))
                        if index
                        else None
                    ),
                )
            connectors[leaf] = stacks.attached(split, y_cells[leaf])
    return middle_cells, middle_fans, connectors


def build_basepoint_arrivals(
    *,
    first,
    refinement,
    xprime_groups,
    x_cells,
    cell,
    place,
):
    """X' cells for apex cells sent to the basepoint, with their fans.

    An apex cell sent to the basepoint has no block to receive: it carries
    its own value into X', arriving with a fan and no arrow.  Returns the
    cells and fans keyed by X' leaf, and the arrival animations.
    """
    arrival_cells, arrival_fans, arrivals = {}, {}, []
    for i, target in enumerate(first.right.map):
        if target:
            continue
        for leaf in xprime_groups[i]:
            joined = cell(refinement.domain[leaf], place(COL_T, leaf))
            fan = stacks.tree_segment(
                x_cells[i].get_right(), joined.get_left()
            )
            arrival_cells[leaf] = joined
            arrival_fans[leaf] = fan
            arrivals.append(
                Succession(
                    FadeIn(joined, run_time=0.45),
                    Create(fan, run_time=0.95),
                )
            )
    return arrival_cells, arrival_fans, arrivals
