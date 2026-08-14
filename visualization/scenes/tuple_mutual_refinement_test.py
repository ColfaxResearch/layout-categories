"""Prototype animation for mutual refinement of flat tuples."""

from manim import (
    Create,
    CubicBezier,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    ORIGIN,
    RIGHT,
    Scene,
    Text,
    UP,
    VGroup,
    config,
    smooth,
)
from math import gcd

from tract import NestedTuple, mutual_refinement

from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK
from layout_categories_viz.tuple_morphism import TupleMorphismDiagram


# Three examples that grow in size and complexity. Each is mutually
# refinable, and each yields more refinement steps than the last.
EXAMPLES = (
    ((6, 6), (2, 6, 6)),
    ((12, 12, 6), (2, 6, 6, 6, 6)),
    ((8, 16, 8), (2, 2, 4, 2, 2, 4, 2, 4, 8)),
)


class TupleMorphismRefinementTest(Scene):
    """Refine two flat factorizations until one divides the other."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND

        for source, target in EXAMPLES:
            # Validate mutual refinability before drawing the example.
            mutual_refinement(NestedTuple(source), NestedTuple(target))
            self._show_refinement(source, target)

    def _show_refinement(self, source, target) -> None:
        frame_height = config.frame_height
        cell_size = min(0.12 * frame_height, 0.075 * config.frame_width)

        # Shrink the cells for taller examples so the middle column of
        # factors still clears the title at the top of the frame.
        steps = list(self._refinement_steps(source, target))
        coefficient = (
            0.5
            + 1.28 * (len(steps) - 1)
            - (1.28 * len(source) - 0.28) / 2
        )
        if coefficient > 0:
            cell_size = min(cell_size, 4.45 / coefficient)

        label_font_size = 28 * cell_size / (0.085 * frame_height)

        title = Text(
            "mutual refinement",
            color=INK,
            font=CODE_FONT,
            font_size=28,
        ).to_edge(UP, buff=0.35)
        initial_source = self._stack(
            source,
            x=-2.15,
            cell_size=cell_size,
            label_font_size=label_font_size,
        )
        initial_target = self._stack(
            target,
            x=2.15,
            cell_size=cell_size,
            label_font_size=label_font_size,
        )
        self._place_pair(initial_source, initial_target)
        source_label = Text(
            "T", color=INK, font=CODE_FONT, font_size=25
        ).next_to(initial_source, DOWN, buff=0.34)
        target_label = Text(
            "U", color=INK, font=CODE_FONT, font_size=25
        ).next_to(initial_target, DOWN, buff=0.34)

        self.play(
            FadeIn(title, shift=DOWN * 0.12),
            FadeIn(initial_source),
            FadeIn(initial_target),
            FadeIn(source_label),
            FadeIn(target_label),
            run_time=0.9,
        )
        self.wait(0.55)

        middle_entries = VGroup()
        middle_connectors = VGroup()
        for source_index, target_index, source_value, target_value, factor in (
            steps
        ):
            target_entry = initial_target[target_index]
            source_entry = (
                initial_source[source_index]
                if source_value is not None
                else None
            )
            middle_entry = self._entry(
                factor,
                center=ORIGIN,
                cell_size=cell_size,
                label_font_size=label_font_size,
            )
            middle_entry.move_to(
                (
                    0.0,
                    initial_source.get_bottom()[1]
                    + cell_size / 2
                    + len(middle_entries) * 1.28 * cell_size,
                    0.0,
                )
            )
            connectors = []
            if source_entry is not None:
                connectors.append(
                    self._tree_segment(
                        source_entry.get_right(), middle_entry.get_left()
                    )
                )
            connectors.append(
                self._tree_segment(
                    target_entry.get_left(), middle_entry.get_right()
                )
            )
            self.play(
                FadeIn(middle_entry, shift=DOWN * 0.12),
                *(Create(connector) for connector in connectors),
                run_time=0.8,
                rate_func=smooth,
            )
            middle_entries.add(middle_entry)
            middle_connectors.add(*connectors)

        self.wait(2.0)

        self.play(
            FadeOut(
                VGroup(
                    title,
                    initial_source,
                    initial_target,
                    source_label,
                    target_label,
                    middle_entries,
                    middle_connectors,
                )
            ),
            run_time=0.8,
        )
        self.clear()
        self.wait(0.15)

    @staticmethod
    def _tree_segment(start, end):
        """Create a smooth, unarrowed tree edge between two tuple cells."""
        start = start.copy()
        end = end.copy()
        direction = RIGHT if end[0] >= start[0] else LEFT
        handle = direction * 0.42 * abs(end[0] - start[0])
        return CubicBezier(
            start,
            start + handle,
            end - handle,
            end,
            color=INK,
            stroke_width=3.6,
        )

    @staticmethod
    def _refinement_steps(source, target):
        """Yield the factor-by-factor gcd steps used by mutual_refinement."""
        source = list(source)
        target = list(target)
        source_index = target_index = 0
        while source_index < len(source) and target_index < len(target):
            source_value = source[source_index]
            target_value = target[target_index]
            factor = gcd(source_value, target_value)
            yield (
                source_index,
                target_index,
                source_value,
                target_value,
                factor,
            )
            if source_value == factor:
                source_index += 1
            else:
                source[source_index] //= factor
            if target_value == factor:
                target_index += 1
            else:
                target[target_index] //= factor

        while target_index < len(target):
            factor = target[target_index]
            yield (None, target_index, None, factor, factor)
            target_index += 1

    @staticmethod
    def _entry(value, *, center, cell_size, label_font_size):
        return TupleMorphismDiagram._make_entries(
            (value,), cell_size, cell_size, label_font_size
        )[0].move_to(center)

    @staticmethod
    def _place_pair(left, right) -> None:
        """Place two stacks side by side with a deliberate, shared baseline."""
        gap = 0.28 * config.frame_width
        baseline_y = -1.55
        left.move_to(
            (-gap / 2 - left.width / 2, baseline_y, 0.0)
        )
        right.move_to(
            (gap / 2 + right.width / 2, baseline_y, 0.0)
        )
        right.align_to(left, DOWN)

    @classmethod
    def _stack(cls, values, *, x, cell_size, label_font_size):
        entries = VGroup(
            *(
                cls._entry(
                    value,
                    center=ORIGIN,
                    cell_size=cell_size,
                    label_font_size=label_font_size,
                )
                for value in values
            )
        )
        entries.arrange(UP, buff=0.28 * cell_size)
        entries.move_to((x, 0.0, 0.0))
        return entries
