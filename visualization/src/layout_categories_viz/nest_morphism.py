"""Manim diagrams for morphisms between nested tuples."""

import numpy as np
from manim import DOWN, LEFT, RIGHT, Text, VGroup
from tract import NestMorphism

from .nested_tuple import NestedTupleTree
from .style import CODE_FONT, INK
from .tuple_morphism import TupleMorphismDiagram


class NestMorphismDiagram(VGroup):
    """A source tree, flattened tuple morphism, and mirrored target tree."""

    def __init__(
        self,
        domain,
        codomain,
        mapping,
        *,
        column_gap=3.2,
        cell_size=0.62,
        level_gap=1.15,
        leaf_gap=0.82,
        label_font_size=27,
        source_label="S",
        target_label="T",
        arrow_endpoint_inset=0.08,
        arrow_horizontal_run=0.2,
        arrow_bend_handle=0.8,
    ) -> None:
        super().__init__()
        morphism = NestMorphism(domain, codomain, tuple(mapping))
        self.domain = morphism.domain.data
        self.codomain = morphism.codomain.data
        self.mapping = tuple(mapping)

        self.source_tree = NestedTupleTree(
            self.domain,
            direction=RIGHT,
            cell_size=cell_size,
            level_gap=level_gap,
            leaf_gap=leaf_gap,
            label_font_size=label_font_size,
        )
        self.target_tree = NestedTupleTree(
            self.codomain,
            direction=LEFT,
            cell_size=cell_size,
            level_gap=level_gap,
            leaf_gap=leaf_gap,
            label_font_size=label_font_size,
        )
        self.source_tree.shift(
            np.array((-column_gap / 2, 0.0, 0.0))
            - self.source_tree.leaf_entries[0].get_center()
        )
        self.target_tree.shift(
            np.array((column_gap / 2, 0.0, 0.0))
            - self.target_tree.leaf_entries[0].get_center()
        )

        self.arrows = VGroup()
        for source_index, target_index in enumerate(self.mapping):
            if target_index == 0:
                continue
            arrow = TupleMorphismDiagram.mapsto_arrow(
                self.source_tree.leaf_entries[source_index].get_right(),
                self.target_tree.leaf_entries[target_index - 1].get_left(),
                endpoint_inset=arrow_endpoint_inset,
                horizontal_run=arrow_horizontal_run,
                bend_handle=arrow_bend_handle,
            )
            arrow.tail.set_opacity(0)
            self.arrows.add(arrow)

        self.source_heading = Text(
            source_label,
            color=INK,
            font=CODE_FONT,
            font_size=24,
        ).next_to(self.source_tree, DOWN, buff=0.24)
        self.target_heading = Text(
            target_label,
            color=INK,
            font=CODE_FONT,
            font_size=24,
        ).next_to(self.target_tree, DOWN, buff=0.24)

        self.add(
            self.arrows,
            self.source_tree,
            self.target_tree,
            self.source_heading,
            self.target_heading,
        )
