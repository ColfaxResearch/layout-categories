"""Composition diagrams for flat tuple morphisms."""

from collections.abc import Sequence

from manim import DOWN, RIGHT, UP, Text, VGroup

from .style import CODE_FONT, INK
from .tuple_morphism import TupleMorphismDiagram


def compose_tuple_maps(
    first_mapping: Sequence[int], second_mapping: Sequence[int]
) -> tuple[int, ...]:
    """Return the one-based map for ``g ∘ f``; zero is the basepoint ``*``."""
    return tuple(
        0 if intermediate_index == 0 else second_mapping[intermediate_index - 1]
        for intermediate_index in first_mapping
    )


class TupleMorphismCompositionDiagram(VGroup):
    """Three tuple stacks displaying composable morphisms ``f`` and ``g``.

    The class deliberately does not draw the composite arrow.  Keeping the two
    stages separate makes the intermediate mode routing legible; a caller can
    then present the composite as its own ``TupleMorphismDiagram``.
    """

    def __init__(
        self,
        domain: Sequence[int],
        intermediate: Sequence[int],
        codomain: Sequence[int],
        first_mapping: Sequence[int],
        second_mapping: Sequence[int],
        *,
        column_gap: float = 2.55,
        row_gap: float = 0.22,
        entry_width: float = 0.9,
        entry_height: float = 0.58,
        label_font_size: float = 28,
        intermediate_offset: Sequence[float] = (0.0, 0.0, 0.0),
        arrow_endpoint_inset: float = 0.08,
        arrow_horizontal_run: float = 0.55,
        arrow_bend_handle: float = 0.35,
    ) -> None:
        super().__init__()
        self.domain = tuple(domain)
        self.intermediate = tuple(intermediate)
        self.codomain = tuple(codomain)
        self.first_mapping = tuple(first_mapping)
        self.second_mapping = tuple(second_mapping)

        # Reuse the flat-morphism validation rules for both stages.
        TupleMorphismDiagram(self.domain, self.intermediate, self.first_mapping)
        TupleMorphismDiagram(self.intermediate, self.codomain, self.second_mapping)
        self.composite_mapping = compose_tuple_maps(
            self.first_mapping, self.second_mapping
        )

        self.source_entries = TupleMorphismDiagram.make_entries(
            self.domain, entry_width, entry_height, label_font_size
        )
        self.intermediate_entries = TupleMorphismDiagram.make_entries(
            self.intermediate, entry_width, entry_height, label_font_size
        )
        self.target_entries = TupleMorphismDiagram.make_entries(
            self.codomain, entry_width, entry_height, label_font_size
        )
        for entries in (
            self.source_entries,
            self.intermediate_entries,
            self.target_entries,
        ):
            # Tuple coordinates are read bottom to top.
            entries.arrange(UP, buff=row_gap)
        self.intermediate_entries.next_to(self.source_entries, RIGHT, buff=column_gap)
        self.target_entries.next_to(self.intermediate_entries, RIGHT, buff=column_gap)
        for entries in (self.intermediate_entries, self.target_entries):
            entries.align_to(self.source_entries, DOWN)
        self.intermediate_entries.shift(intermediate_offset)

        self.first_arrows = self._make_arrows(
            self.source_entries,
            self.intermediate_entries,
            self.first_mapping,
            endpoint_inset=arrow_endpoint_inset,
            horizontal_run=arrow_horizontal_run,
            bend_handle=arrow_bend_handle,
        )
        self.second_arrows = self._make_arrows(
            self.intermediate_entries,
            self.target_entries,
            self.second_mapping,
            endpoint_inset=arrow_endpoint_inset,
            horizontal_run=arrow_horizontal_run,
            bend_handle=arrow_bend_handle,
        )
        self.source_heading = self._heading("source", self.source_entries)
        self.intermediate_heading = self._heading("intermediate", self.intermediate_entries)
        self.target_heading = self._heading("target", self.target_entries)
        self.first_label = Text("f", color=INK, font=CODE_FONT, font_size=30)
        self.second_label = Text("g", color=INK, font=CODE_FONT, font_size=30)
        self.first_label.move_to(
            (self.source_entries.get_top() + self.intermediate_entries.get_top()) / 2
            + DOWN * 0.65
        )
        self.second_label.move_to(
            (self.intermediate_entries.get_top() + self.target_entries.get_top()) / 2
            + DOWN * 0.65
        )

        self.add(
            self.first_arrows,
            self.second_arrows,
            self.source_entries,
            self.intermediate_entries,
            self.target_entries,
            self.source_heading,
            self.intermediate_heading,
            self.target_heading,
            self.first_label,
            self.second_label,
        )

    @staticmethod
    def _heading(label: str, entries: VGroup) -> Text:
        heading = Text(label, color=INK, font=CODE_FONT, font_size=22)
        heading.next_to(entries, direction=(0, 1, 0), buff=0.28)
        return heading

    @staticmethod
    def _make_arrows(
        source: VGroup,
        target: VGroup,
        mapping: Sequence[int],
        *,
        endpoint_inset: float = 0.08,
        horizontal_run: float = 0.55,
        bend_handle: float = 0.35,
    ) -> VGroup:
        arrows = VGroup()
        for source_index, target_index in enumerate(mapping):
            if target_index:
                arrows.add(
                    TupleMorphismDiagram.mapsto_arrow(
                        source[source_index].get_right(),
                        target[target_index - 1].get_left(),
                        endpoint_inset=endpoint_inset,
                        horizontal_run=horizontal_run,
                        bend_handle=bend_handle,
                    )
                )
        return arrows

    def source_entry(self, index: int) -> VGroup:
        return self.source_entries[index - 1]

    def intermediate_entry(self, index: int) -> VGroup:
        return self.intermediate_entries[index - 1]

    def target_entry(self, index: int) -> VGroup:
        return self.target_entries[index - 1]
