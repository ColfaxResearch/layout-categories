"""Manim primitives for flat tuple morphisms.

A tuple morphism is shown as two vertical stacks.  The ``i``-th source entry
has an arrow to target entry ``mapping[i]``; mapping value zero denotes the
basepoint (``*``), so its source entry intentionally has no arrow.
"""

from collections.abc import Sequence

import numpy as np
from manim import CubicBezier, DOWN, RIGHT, UP, Line, Text, RoundedRectangle, VGroup

from .style import CODE_FONT, INK, PANEL


class MapstoArrow(VGroup):
    """A ``|→`` arrow whose tail, shaft, and caret tip are separate mobjects."""

    def __init__(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        bar_length: float = 0.22,
        tip_length: float = 0.2,
        tip_width: float = 0.18,
        endpoint_inset: float = 0.08,
        horizontal_run: float = 0.55,
        bend_handle: float = 0.35,
    ) -> None:
        super().__init__()
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        direction = end - start
        direction /= np.linalg.norm(direction)
        horizontal_direction = np.array(
            (1.0 if direction[0] >= 0 else -1.0, 0.0, 0.0)
        )
        normal = np.array((0.0, 1.0, 0.0))
        tail_center = start + horizontal_direction * endpoint_inset
        tip_point = end - horizontal_direction * endpoint_inset
        available_span = abs(tip_point[0] - tail_center[0])
        minimum_middle_span = min(0.6, available_span)
        straight_run = min(
            horizontal_run,
            max(0.0, (available_span - minimum_middle_span) / 2),
        )
        curve_start = tail_center + horizontal_direction * straight_run
        curve_end = tip_point - horizontal_direction * straight_run
        middle_span = abs(curve_end[0] - curve_start[0])
        middle_handle = min(bend_handle, middle_span / 2)

        self.tail = Line(
            tail_center - normal * bar_length / 2,
            tail_center + normal * bar_length / 2,
            stroke_width=3.6,
            color=INK,
        )
        # Use three cubic segments so the rendered path—not merely its endpoint
        # tangent—is exactly horizontal beside both cells.  The middle cubic
        # joins those runs with matching horizontal tangents.  Keeping the
        # shaft as a CubicBezier with appended segments preserves the topology
        # expected by the arrow and composition animations.
        self.shaft = CubicBezier(
            tail_center,
            tail_center + horizontal_direction * straight_run / 3,
            tail_center + horizontal_direction * 2 * straight_run / 3,
            curve_start,
            stroke_width=3.6,
            color=INK,
        )
        middle_curve = CubicBezier(
            curve_start,
            curve_start + horizontal_direction * middle_handle,
            curve_end - horizontal_direction * middle_handle,
            curve_end,
        )
        final_run = CubicBezier(
            curve_end,
            curve_end + horizontal_direction * straight_run / 3,
            curve_end + horizontal_direction * 2 * straight_run / 3,
            tip_point,
        )
        self.shaft.append_points(middle_curve.points)
        self.shaft.append_points(final_run.points)
        self.tip = VGroup(
            Line(
                tip_point,
                tip_point
                - horizontal_direction * tip_length
                + normal * tip_width / 2,
                stroke_width=3.6,
                color=INK,
            ),
            Line(
                tip_point,
                tip_point
                - horizontal_direction * tip_length
                - normal * tip_width / 2,
                stroke_width=3.6,
                color=INK,
            ),
        )
        self.add(self.tail, self.shaft, self.tip)

    @classmethod
    def from_parts(cls, tail, shaft, tip) -> "MapstoArrow":
        """Group pre-existing three arrow parts without altering their geometry."""
        arrow = cls.__new__(cls)
        VGroup.__init__(arrow, tail, shaft, tip)
        arrow.tail = tail
        arrow.shaft = shaft
        arrow.tip = tip
        return arrow


class TupleMorphismDiagram(VGroup):
    """A two-column diagram of a morphism between flat tuples.

    Parameters use the repository's one-based map convention: ``0`` means
    ``*`` and values from ``1`` through ``len(codomain)`` select target modes.
    Nonzero entries must be distinct, matching ``FinMorphism``.
    """

    def __init__(
        self,
        domain: Sequence[int],
        codomain: Sequence[int],
        mapping: Sequence[int],
        *,
        column_gap: float = 4.2,
        row_gap: float = 0.22,
        entry_width: float = 0.9,
        entry_height: float = 0.58,
        label_font_size: float = 28,
        source_label: str = "source",
        target_label: str = "target",
        arrow_endpoint_inset: float = 0.08,
        arrow_horizontal_run: float = 0.55,
        arrow_bend_handle: float = 0.35,
    ) -> None:
        super().__init__()
        self.domain = tuple(domain)
        self.codomain = tuple(codomain)
        self.mapping = tuple(mapping)
        self._validate()

        self.source_entries = self.make_entries(
            self.domain, entry_width, entry_height, label_font_size
        )
        self.target_entries = self.make_entries(
            self.codomain, entry_width, entry_height, label_font_size
        )
        # Tuple coordinates are read bottom to top: the first component sits
        # at the shared base and later components rise above it.
        self.source_entries.arrange(UP, buff=row_gap)
        self.target_entries.arrange(UP, buff=row_gap)
        self.target_entries.next_to(self.source_entries, RIGHT, buff=column_gap)
        # Tuple entries conventionally grow upward from their shared base.
        self.target_entries.align_to(self.source_entries, direction=(0, -1, 0))

        self.arrows = VGroup()
        for source_index, target_index in enumerate(self.mapping):
            if target_index == 0:
                continue
            arrow = self.mapsto_arrow(
                self.source_entries[source_index].get_right(),
                self.target_entries[target_index - 1].get_left(),
                endpoint_inset=arrow_endpoint_inset,
                horizontal_run=arrow_horizontal_run,
                bend_handle=arrow_bend_handle,
            )
            self.arrows.add(arrow)

        self.source_heading = Text(
            source_label, color=INK, font=CODE_FONT, font_size=24
        )
        self.target_heading = Text(
            target_label, color=INK, font=CODE_FONT, font_size=24
        )
        self.source_heading.next_to(self.source_entries, direction=(0, 1, 0), buff=0.3)
        self.target_heading.next_to(self.target_entries, direction=(0, 1, 0), buff=0.3)

        # Arrows are added first so entries remain readable at intersections.
        self.add(
            self.arrows,
            self.source_entries,
            self.target_entries,
            self.source_heading,
            self.target_heading,
        )

    def _validate(self) -> None:
        if len(self.mapping) != len(self.domain):
            raise ValueError("mapping must contain one entry for each domain mode")
        if any(not isinstance(value, int) for value in self.mapping):
            raise TypeError("mapping entries must be integers")
        if any(value < 0 or value > len(self.codomain) for value in self.mapping):
            raise ValueError("mapping entries must be 0 or valid one-based target indices")
        nonzero = [value for value in self.mapping if value]
        if len(nonzero) != len(set(nonzero)):
            raise ValueError("nonzero mapping entries must be distinct")
        for source, target_index in zip(self.domain, self.mapping):
            if target_index and source != self.codomain[target_index - 1]:
                raise ValueError("mapped source and target entries must have equal values")

    @staticmethod
    def make_entries(
        values: Sequence[int], width: float, height: float, font_size: float
    ) -> VGroup:
        entries = VGroup()
        for value in values:
            box = RoundedRectangle(
                corner_radius=0.08,
                width=width,
                height=height,
                stroke_color=INK,
                stroke_width=1.8,
                fill_color=PANEL,
                fill_opacity=1,
            )
            label = Text(str(value), color=INK, font=CODE_FONT, font_size=font_size)
            label.move_to(box)
            entries.add(VGroup(box, label))
        return entries

    @staticmethod
    def mapsto_arrow(
        start: np.ndarray,
        end: np.ndarray,
        *,
        bar_length: float = 0.22,
        tip_length: float = 0.2,
        tip_width: float = 0.18,
        endpoint_inset: float = 0.08,
        horizontal_run: float = 0.55,
        bend_handle: float = 0.35,
    ) -> MapstoArrow:
        """Build a ``|→`` arrow with independently animatable components."""
        return MapstoArrow(
            start,
            end,
            bar_length=bar_length,
            tip_length=tip_length,
            tip_width=tip_width,
            endpoint_inset=endpoint_inset,
            horizontal_run=horizontal_run,
            bend_handle=bend_handle,
        )

    # Backwards-compatible aliases for the former private names.
    _make_entries = make_entries
    _mapsto_arrow = mapsto_arrow

    def source_entry(self, index: int) -> VGroup:
        """Return a source entry using the public, one-based convention."""
        return self.source_entries[index - 1]

    def target_entry(self, index: int) -> VGroup:
        """Return a target entry using the public, one-based convention."""
        return self.target_entries[index - 1]
