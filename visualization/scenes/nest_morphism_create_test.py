"""Create example morphisms between nested tuples."""

from dataclasses import dataclass

from manim import Create, FadeIn, FadeOut, LaggedStart, Scene, Text, VGroup, config

from layout_categories_viz import NestMorphismDiagram, TailToTipMapsto
from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK


@dataclass(frozen=True)
class NestMorphismExample:
    domain: tuple
    codomain: tuple
    mapping: tuple[int, ...]


EXAMPLES = (
    NestMorphismExample(
        ((2, 3), (5, 7)),
        (11, (2, 3), (5, 7)),
        (2, 3, 4, 5),
    ),
    NestMorphismExample(
        (2, (3, (5, 7)), 11),
        ((5, 7), (2, 11), 3),
        (3, 5, 1, 2, 4),
    ),
)


class NestMorphismCreateTest(Scene):
    """Build nested source and target trees, then their flattened morphism."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        for example in EXAMPLES:
            self._show_example(example)

    def _show_example(self, example) -> None:
        frame_height = config.frame_height
        cell_size = 0.082 * frame_height
        diagram = NestMorphismDiagram(
            example.domain,
            example.codomain,
            example.mapping,
            column_gap=0.25 * config.frame_width,
            cell_size=cell_size,
            level_gap=1.7 * cell_size,
            leaf_gap=1.32 * cell_size,
            label_font_size=28 * cell_size / (0.085 * frame_height),
        )
        diagram.source_heading.scale(1.25)
        diagram.target_heading.scale(1.25)

        morphism_label = Text(
            "f",
            color=INK,
            font=CODE_FONT,
            font_size=int(3.75 * frame_height),
        ).next_to(
            VGroup(diagram.source_heading, diagram.target_heading),
            direction=(0, -1, 0),
            buff=0.22,
        )
        visible = VGroup(diagram, morphism_label)
        scale_factor = min(
            1.0,
            0.86 * config.frame_height / visible.height,
            0.92 * config.frame_width / visible.width,
        )
        visible.scale(scale_factor).move_to((0, 0, 0))

        self.play(
            FadeIn(diagram.source_heading),
            FadeIn(diagram.target_heading),
            FadeIn(morphism_label),
            *(
                FadeIn(node, scale=0.9)
                for node in (
                    *diagram.source_tree.nodes,
                    *diagram.target_tree.nodes,
                )
            ),
            run_time=0.9,
        )

        source_edges = sorted(
            diagram.source_tree.edges,
            key=lambda edge: edge.get_start()[0],
        )
        target_edges = sorted(
            diagram.target_tree.edges,
            key=lambda edge: edge.get_start()[0],
        )
        self.add(diagram.arrows)
        self.play(
            LaggedStart(
                *(Create(edge) for edge in source_edges),
                *(TailToTipMapsto(arrow) for arrow in diagram.arrows),
                *(Create(edge) for edge in target_edges),
                lag_ratio=0.09,
            ),
            run_time=2.2,
        )
        self.wait(2.0)
        self.play(
            FadeOut(visible),
            run_time=0.85,
        )
        self.clear()
        self.wait(0.2)
