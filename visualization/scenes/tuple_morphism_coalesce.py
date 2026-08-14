"""Animate disjoint local coalesces of tuple-morphism entries."""

from layout_categories_viz.scene_base import LayoutScene

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
from tract import TupleMorphism

from layout_categories_viz import TailToTipMapsto, TupleMorphismDiagram
from layout_categories_viz.coalesce import (
    coalesce_partitions,
    play_coalesce_collapse,
)
from layout_categories_viz.paths import (
    ARROW_BEND_HANDLE,
    ARROW_ENDPOINT_INSET,
    ARROW_HORIZONTAL_RUN,
)
from layout_categories_viz.style import CODE_FONT, INK


EXAMPLES = (
    {
        "domain": (2, 3, 7, 5),
        "codomain": (2, 3, 5, 7, 11),
        "mapping": (1, 2, 4, 3),
    },
    {
        # Two mapped runs separated by the untouched 7 mode.
        "domain": (2, 3, 7, 11, 5),
        "codomain": (2, 3, 11, 5, 7),
        "mapping": (1, 2, 5, 3, 4),
    },
    {
        # Three mapped runs, visibly separated by untouched 7 and 13 modes.
        "domain": (2, 3, 7, 5, 11, 13, 17, 19),
        "codomain": (2, 3, 13, 5, 11, 7, 17, 19),
        "mapping": (1, 2, 6, 4, 5, 3, 7, 8),
    },
)


class TupleMorphismCoalesce(LayoutScene):
    """Collapse each consecutive order-preserving block to its product."""

    def construct(self) -> None:

        for example in EXAMPLES:
            self._play_example(**example)

    @staticmethod
    def _place_labels(diagram, morphism_label) -> None:
        diagram.source_heading.scale(1.25)
        diagram.target_heading.scale(1.25)
        diagram.source_heading.next_to(diagram.source_entries[0], DOWN, buff=0.24)
        diagram.target_heading.next_to(diagram.target_entries[0], DOWN, buff=0.24)
        morphism_label.next_to(
            VGroup(diagram.source_heading, diagram.target_heading),
            DOWN,
            buff=0.22,
        )
        VGroup(diagram, morphism_label).move_to(ORIGIN)

    def _play_example(self, *, domain, codomain, mapping) -> None:
        morphism = TupleMorphism(domain, codomain, mapping)
        if any(value == 1 for value in (*domain, *codomain)):
            raise ValueError("This scene handles coalesce examples without squeezing")

        coalesced = morphism.coalesce()
        domain_classes, _ = coalesce_partitions(morphism)
        if sum(len(group) > 1 for group in domain_classes) < 1:
            raise ValueError("Each scene example must contain a coalescing run")

        frame_width = config.frame_width
        frame_height = config.frame_height
        max_tuple_length = max(len(morphism.domain), len(morphism.codomain))
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
            morphism.domain,
            morphism.codomain,
            morphism.map,
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
        morphism_label = Text(
            "f", color=INK, font=CODE_FONT, font_size=layout_font_size
        )
        self._place_labels(diagram, morphism_label)

        for arrow in diagram.arrows:
            arrow.tail.set_opacity(0)

        entries = tuple(diagram.source_entries) + tuple(diagram.target_entries)
        self.play(
            FadeIn(diagram.source_heading),
            FadeIn(diagram.target_heading),
            FadeIn(morphism_label),
            LaggedStart(
                *(FadeIn(entry, scale=0.9) for entry in entries),
                lag_ratio=0.08,
            ),
            run_time=1.1,
        )
        self.add(diagram.arrows)
        self.play(
            LaggedStart(
                *(TailToTipMapsto(arrow) for arrow in diagram.arrows),
                lag_ratio=0.14,
            ),
            run_time=1.35,
        )
        self.wait(1.0)

        (
            surviving_source_entries,
            surviving_target_entries,
            surviving_arrows,
        ) = play_coalesce_collapse(
            self,
            morphism,
            coalesced,
            source_entries=diagram.source_entries,
            target_entries=diagram.target_entries,
            arrows=diagram.arrows,
            column_gap=column_gap,
            row_gap=row_gap,
            entry_width=cell_size,
            entry_height=cell_size,
            label_font_size=entry_font_size,
        )

        visible_result = VGroup(
            *surviving_source_entries,
            *surviving_target_entries,
            *surviving_arrows,
            diagram.source_heading,
            diagram.target_heading,
            morphism_label,
        )
        self.play(FadeOut(visible_result), run_time=0.85)
        self.clear()
        self.wait(0.2)
