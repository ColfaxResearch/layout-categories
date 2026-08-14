"""Create and remove a small gallery of tuple morphisms."""

from layout_categories_viz.scene_base import LayoutScene

from dataclasses import dataclass

from manim import (
    DOWN,
    FadeIn,
    FadeOut,
    LaggedStart,
    ORIGIN,
    Text,
    VGroup,
    config,
)

from layout_categories_viz import (
    TailToTipMapsto,
    TipToTailUnmapsto,
    TupleMorphismDiagram,
)
from layout_categories_viz.style import CODE_FONT, INK


# Arrow-shape tuning knobs.  Increasing HORIZONTAL_RUN keeps each curve
# horizontal for longer near its cells; decreasing it lengthens the diagonal
# middle.  ENDPOINT_INSET controls the small gap at each cell boundary.
ARROW_HORIZONTAL_RUN = 0.2
ARROW_ENDPOINT_INSET = 0.08
ARROW_BEND_HANDLE = 0.8


@dataclass(frozen=True)
class MorphismExample:
    domain: tuple[int, ...]
    codomain: tuple[int, ...]
    mapping: tuple[int, ...]


EXAMPLES = (
    MorphismExample((4,), (4,), (1,)),
    MorphismExample(
        (2, 3, 5),
        (7, 11),
        (0, 0, 0),
    ),
    MorphismExample((2, 5), (2, 3, 5, 7), (1, 3)),
    MorphismExample(
        (4, 2, 6, 3),
        (2, 3, 4, 5),
        (3, 1, 0, 2),
    ),
    MorphismExample(
        (4, 2, 6, 5, 3, 7),
        (2, 3, 4, 5, 6),
        (3, 1, 5, 4, 2, 0),
    ),
    MorphismExample(
        (13, 2, 19, 7, 5, 23, 3),
        (2, 3, 5, 7, 11, 13),
        (6, 1, 0, 4, 3, 0, 2),
    ),
)


class TupleMorphismCreateRemove(LayoutScene):
    """Construct each tuple morphism, pause, and erase it again."""

    def construct(self) -> None:

        for example in EXAMPLES:
            self._show_example(example)

    def _show_example(self, example: MorphismExample) -> None:
        frame_width = config.frame_width
        frame_height = config.frame_height
        max_tuple_length = max(len(example.domain), len(example.codomain))
        row_gap_ratio = 0.32
        cell_size = min(
            0.11 * frame_height,
            0.07 * frame_width,
            0.66
            * frame_height
            / (max_tuple_length + row_gap_ratio * (max_tuple_length - 1)),
        )
        row_gap = row_gap_ratio * cell_size
        entry_font_size = 28 * cell_size / (0.085 * frame_height)
        column_gap = 0.23 * frame_width
        layout_font_size = int(3.75 * frame_height)

        diagram = TupleMorphismDiagram(
            example.domain,
            example.codomain,
            example.mapping,
            column_gap=column_gap,
            row_gap=row_gap,
            entry_width=cell_size,
            entry_height=cell_size,
            label_font_size=entry_font_size,
            source_label="S",
            target_label="T",
            arrow_endpoint_inset=ARROW_ENDPOINT_INSET,
            arrow_horizontal_run=ARROW_HORIZONTAL_RUN,
            arrow_bend_handle=ARROW_BEND_HANDLE,
        )
        diagram.source_heading.scale(1.25)
        diagram.target_heading.scale(1.25)
        diagram.source_heading.next_to(diagram.source_entries[0], DOWN, buff=0.24)
        diagram.target_heading.next_to(diagram.target_entries[0], DOWN, buff=0.24)
        for arrow in diagram.arrows:
            arrow.tail.set_opacity(0)

        morphism_label = Text(
            "f",
            color=INK,
            font=CODE_FONT,
            font_size=layout_font_size,
        )
        morphism_label.next_to(
            VGroup(diagram.source_heading, diagram.target_heading),
            DOWN,
            buff=0.22,
        )
        VGroup(diagram, morphism_label).move_to(ORIGIN)

        entries = tuple(diagram.source_entries) + tuple(diagram.target_entries)
        self.play(
            FadeIn(diagram.source_heading),
            FadeIn(diagram.target_heading),
            FadeIn(morphism_label),
            run_time=0.6,
        )
        self.play(
            LaggedStart(
                *(FadeIn(entry, scale=0.9) for entry in entries),
                lag_ratio=0.1,
            ),
            run_time=1.0,
        )

        if diagram.arrows:
            self.add(diagram.arrows)
            self.play(
                LaggedStart(
                    *(TailToTipMapsto(arrow) for arrow in diagram.arrows),
                    lag_ratio=0.15,
                ),
                run_time=1.0,
            )

        self.wait(1.6)

        removals = [
            *(TipToTailUnmapsto(arrow) for arrow in reversed(diagram.arrows)),
            *(FadeOut(entry, scale=0.9) for entry in reversed(entries)),
            FadeOut(diagram.source_heading),
            FadeOut(diagram.target_heading),
            FadeOut(morphism_label),
        ]
        self.play(LaggedStart(*removals, lag_ratio=0.06), run_time=1.15)
        self.clear()
        self.wait(0.2)
