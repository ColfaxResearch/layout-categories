"""Compose two large tuple morphisms, then coalesce their composite."""

from manim import FadeOut, VGroup, config
from tract import TupleMorphism

from layout_categories_viz.coalesce import play_coalesce_collapse
from scenes.tuple_morphism_composition_curve import (
    TupleMorphismCurvedCompositionCollapse,
)


EXAMPLE = {
    "domain": (2, 3, 17, 5, 7, 29, 19, 23),
    "intermediate": (7, 13, 2, 19, 17, 29, 5, 23, 3, 11),
    "codomain": (13, 2, 3, 11, 5, 7, 17, 19, 23),
    "first_mapping": (3, 9, 5, 7, 1, 6, 4, 8),
    "second_mapping": (6, 1, 2, 8, 7, 0, 5, 9, 3, 4),
}


class TupleMorphismCompositionThenCoalesceTest(
    TupleMorphismCurvedCompositionCollapse
):
    """Animate ``f``, ``g``, ``g ∘ f``, and finally ``coalesce(g ∘ f)``."""

    def construct(self) -> None:

        first = TupleMorphism(
            EXAMPLE["domain"],
            EXAMPLE["intermediate"],
            EXAMPLE["first_mapping"],
        )
        second = TupleMorphism(
            EXAMPLE["intermediate"],
            EXAMPLE["codomain"],
            EXAMPLE["second_mapping"],
        )
        composite = first.compose(second)
        expected_composite = TupleMorphism(
            EXAMPLE["domain"],
            EXAMPLE["codomain"],
            (2, 3, 7, 5, 6, 0, 8, 9),
        )
        if composite.map != expected_composite.map:
            raise ValueError("The configured composition changed unexpectedly")
        expected_coalesced = (
            (6, 17, 35, 29, 437),
            (13, 6, 11, 35, 17, 437),
            (2, 5, 4, 0, 6),
        )
        coalesced = composite.coalesce()
        if (
            coalesced.domain,
            coalesced.codomain,
            coalesced.map,
        ) != expected_coalesced:
            raise ValueError("The configured coalescence changed unexpectedly")

        frame_width = config.frame_width
        frame_height = config.frame_height
        max_tuple_length = max(
            len(first.domain),
            len(first.codomain),
            len(second.codomain),
        )
        row_gap_ratio = 0.29
        cell_size = min(
            0.07 * frame_width,
            0.095 * frame_height,
            0.70
            * frame_height
            / (
                max_tuple_length
                + row_gap_ratio * (max_tuple_length - 1)
            ),
        )
        row_gap = row_gap_ratio * cell_size
        label_font_size = 28 * cell_size / (0.085 * frame_height)
        column_gap = 0.17 * frame_width

        state = self._play_example(
            **EXAMPLE,
            column_gap=column_gap,
            row_gap=row_gap,
            entry_width=cell_size,
            entry_height=cell_size,
            label_font_size=label_font_size,
            retain_composite=True,
        )
        self._coalesce_composite(composite, coalesced, state)

    def _coalesce_composite(self, morphism, coalesced, state) -> None:
        source_entries = state["source_entries"]
        target_entries = state["target_entries"]

        # The contracted composite's actual on-screen column gap, not the
        # two-stage diagram's construction parameter.
        column_gap = (
            target_entries[0].get_left()[0]
            - source_entries[0].get_right()[0]
        )
        (
            surviving_source_entries,
            surviving_target_entries,
            surviving_arrows,
        ) = play_coalesce_collapse(
            self,
            morphism,
            coalesced,
            source_entries=source_entries,
            target_entries=target_entries,
            arrows=state["arrows"],
            column_gap=column_gap,
            row_gap=state["row_gap"],
            entry_width=state["entry_width"],
            entry_height=state["entry_height"],
            label_font_size=state["label_font_size"],
        )

        visible_result = VGroup(
            *surviving_source_entries,
            *surviving_target_entries,
            *surviving_arrows,
            state["first_label"],
            state["second_label"],
            state["composition_symbol"],
        )
        self.play(FadeOut(visible_result), run_time=0.85)
