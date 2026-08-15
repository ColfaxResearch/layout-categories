"""Assemble the flat layout a tuple morphism encodes.

``L_f`` reads off a morphism ``f : S -> T`` directly: mode ``i`` of the layout
has shape ``S[i]`` and stride the product of everything in ``T`` below the
entry ``f`` sends it to — the prefix products ``tuple_morphism_to_flat_layout``
gathers one at a time.  A mode sent to the basepoint contributes stride ``0``.

The scene draws the morphism, writes the exclusive prefix products beside the
target stack in one pass, and then builds ``L(f) = shape : stride`` along the
bottom mode by mode: each shape entry drops out of its source cell, and each
stride entry travels back from the prefix product of its target — or appears
as ``0`` from nowhere, since a basepoint mode has no arrow to bring it home.

The example keeps one mode of each kind in play: two mapped modes that pick up
honest strides, the identity stride ``1``, and one projected-away mode.  The
result ``(4, 2, 5, 3) : (6, 1, 0, 2)`` is checked against
``flat_layout_components`` before anything is animated.
"""

from layout_categories_viz.scene_base import LayoutScene

from manim import (
    DOWN,
    FadeIn,
    LaggedStart,
    ORIGIN,
    PURPLE,
    RIGHT,
    Text,
    TransformFromCopy,
    UP,
    VGroup,
    YELLOW,
    YELLOW_D,
)
from tract import TupleMorphism
from tract.backends.base import flat_layout_components

from layout_categories_viz import TailToTipMapsto, TupleMorphismDiagram
from layout_categories_viz.style import CODE_FONT, INK, MUTED, PANEL
from scenes.tuple_morphism_to_flat_layout import prefix_products


DOMAIN = (4, 2, 5, 3)
CODOMAIN = (2, 3, 4)
MAPPING = (3, 1, 0, 2)
STRIDES = (6, 1, 0, 2)

TOKEN_FONT_SIZE = 30
PRODUCT_FONT_SIZE = 26


class TupleMorphismLayoutTest(LayoutScene):
    """Read the layout of a tuple morphism off its diagram."""

    def construct(self) -> None:
        morphism = TupleMorphism(DOMAIN, CODOMAIN, MAPPING)
        if flat_layout_components(morphism) != (DOMAIN, STRIDES):
            raise ValueError("The configured layout changed unexpectedly")

        diagram = self._introduce_morphism()
        products = self._show_prefix_products(diagram)
        self._assemble_layout(diagram, products)
        self.wait(0.6)

    def _introduce_morphism(self) -> TupleMorphismDiagram:
        diagram = TupleMorphismDiagram(
            DOMAIN,
            CODOMAIN,
            MAPPING,
            column_gap=3.2,
            entry_width=0.62,
            entry_height=0.62,
            source_label="S",
            target_label="T",
        )
        diagram.move_to(ORIGIN).shift(UP * 0.9)

        title = Text(
            "the layout of a tuple morphism",
            color=INK,
            font=CODE_FONT,
            font_size=25,
        ).to_edge(UP, buff=0.3)
        self.play(
            FadeIn(title, shift=DOWN * 0.12),
            FadeIn(diagram.source_entries),
            FadeIn(diagram.target_entries),
            FadeIn(diagram.source_heading),
            FadeIn(diagram.target_heading),
        )
        self.add(diagram.arrows)
        self.play(
            LaggedStart(
                *(TailToTipMapsto(arrow, run_time=0.85) for arrow in diagram.arrows),
                lag_ratio=0.12,
            )
        )
        self.wait(0.25)
        return diagram

    def _show_prefix_products(self, diagram: TupleMorphismDiagram) -> VGroup:
        """Write the exclusive prefix product beside each target entry."""
        products = VGroup(
            *(
                Text(
                    str(product),
                    color=PURPLE,
                    font=CODE_FONT,
                    font_size=PRODUCT_FONT_SIZE,
                )
                for product in prefix_products(CODOMAIN)
            )
        )
        for label, entry in zip(products, diagram.target_entries):
            label.next_to(entry, RIGHT, buff=0.42)
        self.play(
            LaggedStart(
                *(FadeIn(label, shift=RIGHT * 0.12) for label in products),
                lag_ratio=0.25,
            ),
            run_time=1.2,
        )
        self.wait(0.25)
        return products

    def _assemble_layout(
        self, diagram: TupleMorphismDiagram, products: VGroup
    ) -> None:
        """Fill ``L(f) = shape : stride`` one domain mode at a time."""
        shape_tokens, stride_tokens, skeleton = self._layout_line()
        self.play(FadeIn(skeleton), run_time=0.5)

        arrow_of_mode = {}
        arrow_index = 0
        for mode_index, target_index in enumerate(MAPPING):
            if target_index != 0:
                arrow_of_mode[mode_index] = diagram.arrows[arrow_index]
                arrow_index += 1

        for mode_index, target_index in enumerate(MAPPING):
            entry = diagram.source_entries[mode_index]
            highlight = entry[0].animate.set_stroke(YELLOW_D, width=2.4).set_fill(
                YELLOW, opacity=0.18
            )
            self.play(highlight, run_time=0.4)
            self.play(
                TransformFromCopy(entry[1], shape_tokens[mode_index]),
                run_time=0.9,
            )
            if target_index == 0:
                # No arrow, so nothing carries a stride back: it appears as 0.
                self.play(FadeIn(stride_tokens[mode_index]), run_time=0.9)
            else:
                self.play(
                    TransformFromCopy(
                        products[target_index - 1], stride_tokens[mode_index]
                    ),
                    run_time=0.9,
                )
            restore = entry[0].animate.set_stroke(INK, width=1.8).set_fill(
                PANEL, opacity=1
            )
            self.play(restore, run_time=0.3)
        self.wait(0.25)

    def _layout_line(self) -> tuple[VGroup, VGroup, VGroup]:
        """Build the bottom line's tokens: values, and the fixed punctuation."""

        def number(value: int, color) -> Text:
            return Text(
                str(value), color=color, font=CODE_FONT, font_size=TOKEN_FONT_SIZE
            )

        def mark(text: str) -> Text:
            return Text(text, color=MUTED, font=CODE_FONT, font_size=TOKEN_FONT_SIZE)

        shape_tokens = VGroup(*(number(value, INK) for value in DOMAIN))
        stride_tokens = VGroup(*(number(value, PURPLE) for value in STRIDES))
        skeleton = VGroup(mark("L(f) ="), mark("("))
        line = VGroup(skeleton[0], skeleton[1])
        for index, token in enumerate(shape_tokens):
            line.add(token)
            separator = mark(",") if index < len(shape_tokens) - 1 else mark(") : (")
            skeleton.add(separator)
            line.add(separator)
        for index, token in enumerate(stride_tokens):
            line.add(token)
            separator = mark(",") if index < len(stride_tokens) - 1 else mark(")")
            skeleton.add(separator)
            line.add(separator)
        # Center-arranging would float the commas at digit mid-height; the
        # shared bottom edge stands in for a common baseline.
        line.arrange(RIGHT, buff=0.16, aligned_edge=DOWN)
        line.to_edge(DOWN, buff=0.9)
        return shape_tokens, stride_tokens, skeleton
