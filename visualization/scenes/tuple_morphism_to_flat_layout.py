"""Build the target prefix products of a tuple morphism, one at a time."""

from manim import (
    DOWN,
    FadeIn,
    FadeOut,
    LaggedStart,
    ORIGIN,
    PURPLE,
    RIGHT,
    ReplacementTransform,
    Scene,
    SurroundingRectangle,
    Text,
    TransformFromCopy,
    UP,
    VGroup,
    YELLOW,
    YELLOW_D,
    config,
)

from layout_categories_viz import TailToTipMapsto, TupleMorphismDiagram
from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK, MUTED, PANEL


DOMAIN = (4, 2, 6, 5, 3, 7)
CODOMAIN = (2, 3, 4, 5, 6)
MAPPING = (3, 1, 5, 4, 2, 0)


def prefix_products(factors: tuple[int, ...]) -> tuple[int, ...]:
    """Return exclusive prefix products, including the empty product first."""
    products = []
    product = 1
    for factor in factors:
        products.append(product)
        product *= factor
    return tuple(products)


class TupleMorphismToFlatLayout(Scene):
    """Read prefix products by gathering preceding target factors."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND

        self._gathering_prefix_demo()
        self.wait(0.6)

    def _make_diagram(self, *, workroom: float) -> tuple[TupleMorphismDiagram, Text]:
        frame_width = config.frame_width
        frame_height = config.frame_height
        cell_size = min(0.085 * frame_height, 0.055 * frame_width)
        column_gap = 0.23 * frame_width
        layout_font_size = int(3.75 * frame_height)

        diagram = TupleMorphismDiagram(
            DOMAIN,
            CODOMAIN,
            MAPPING,
            column_gap=column_gap,
            entry_width=cell_size,
            entry_height=cell_size,
            source_label="S",
            target_label="T",
        )
        diagram.source_heading.scale(1.25)
        diagram.target_heading.scale(1.25)
        diagram.source_heading.next_to(diagram.source_entries[0], DOWN, buff=0.24)
        diagram.target_heading.next_to(diagram.target_entries[0], DOWN, buff=0.24)
        for arrow in diagram.arrows:
            arrow.tail.set_opacity(0)

        morphism_label = Text(
            "f", color=INK, font=CODE_FONT, font_size=layout_font_size
        )
        morphism_label.next_to(
            VGroup(diagram.source_heading, diagram.target_heading), DOWN, buff=0.22
        )
        VGroup(diagram, morphism_label).move_to(ORIGIN).shift(RIGHT * workroom)
        return diagram, morphism_label

    def _introduce_morphism(
        self, diagram: TupleMorphismDiagram, morphism_label: Text, title: Text
    ) -> None:
        title.to_edge(UP, buff=0.3)
        self.play(
            FadeIn(title, shift=DOWN * 0.12),
            FadeIn(diagram.source_entries),
            FadeIn(diagram.target_entries),
            FadeIn(diagram.source_heading),
            FadeIn(diagram.target_heading),
            FadeIn(morphism_label),
        )
        self.add(diagram.arrows)
        self.play(
            LaggedStart(
                *(TailToTipMapsto(arrow, run_time=0.85) for arrow in diagram.arrows),
                lag_ratio=0.12,
            )
        )
        self.wait(0.25)

    @staticmethod
    def _result_label(value: int, entry) -> Text:
        label = Text(str(value), color=PURPLE, font=CODE_FONT, font_size=26)
        label.next_to(entry, RIGHT, buff=0.42)
        return label

    def _gathering_prefix_demo(self) -> None:
        """Gather copies of every lower factor, then collapse them to a value."""
        diagram, morphism_label = self._make_diagram(workroom=-1.3)
        title = Text(
            "gather everything below",
            color=INK,
            font=CODE_FONT,
            font_size=25,
        )
        self._introduce_morphism(diagram, morphism_label, title)

        products = prefix_products(CODOMAIN)
        result_labels = tuple(
            self._result_label(product, entry)
            for entry, product in zip(diagram.target_entries, products)
        )
        result_column_x = (
            diagram.target_entries.get_right()[0]
            + 0.42
            + max(label.width for label in result_labels) / 2
        )
        for label in result_labels:
            label.set_x(result_column_x)

        workbench_x = diagram.target_entries.get_right()[0] + 2.45
        workbench_y = diagram.target_entries.get_center()[1] + 0.15
        for index, (entry, product, label) in enumerate(
            zip(diagram.target_entries, products, result_labels)
        ):
            highlight_entry = entry[0].animate.set_stroke(
                YELLOW_D, width=2.4
            ).set_fill(YELLOW, opacity=0.18)
            self.play(highlight_entry, run_time=0.4)
            if index == 0:
                equation = Text(
                    "empty product = 1",
                    color=PURPLE,
                    font=CODE_FONT,
                    font_size=22,
                )
                equation.move_to((workbench_x, workbench_y, 0))
                self.play(FadeIn(equation, shift=UP * 0.12), run_time=1.1)
                window = None
            else:
                contributors = VGroup(*diagram.target_entries[:index])
                window = SurroundingRectangle(
                    contributors,
                    color=PURPLE,
                    buff=0.11,
                    corner_radius=0.08,
                    stroke_width=2.4,
                )

                factor_tokens = VGroup(
                    *(
                        Text(
                            str(factor),
                            color=INK,
                            font=CODE_FONT,
                            font_size=24,
                        )
                        for factor in CODOMAIN[:index]
                    )
                )
                operators = VGroup(
                    *(
                        Text("×", color=MUTED, font=CODE_FONT, font_size=21)
                        for _ in range(index - 1)
                    )
                )
                expression_parts = VGroup()
                for token_index, token in enumerate(factor_tokens):
                    expression_parts.add(token)
                    if token_index < len(operators):
                        expression_parts.add(operators[token_index])
                equals = Text(
                    f"= {product}", color=PURPLE, font=CODE_FONT, font_size=24
                )
                expression_parts.add(equals)
                expression_parts.arrange(RIGHT, buff=0.12)
                expression_parts.move_to((workbench_x, workbench_y, 0))
                equation = expression_parts

                self.play(FadeIn(window), run_time=0.5)
                self.play(
                    *(
                        TransformFromCopy(source_entry[1], token)
                        for source_entry, token in zip(
                            diagram.target_entries[:index], factor_tokens
                        )
                    ),
                    FadeIn(operators),
                    FadeIn(equals),
                    run_time=1.3,
                )

            self.play(ReplacementTransform(equation, label), run_time=1.1)
            restore_entry = entry[0].animate.set_stroke(
                INK, width=1.8
            ).set_fill(PANEL, opacity=1)
            if window is not None:
                self.play(FadeOut(window), restore_entry, run_time=0.4)
            else:
                self.play(restore_entry, run_time=0.4)
