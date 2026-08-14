"""Compare left-to-right and right-to-left nested-tuple tree drawings."""

from layout_categories_viz.scene_base import LayoutScene

from manim import (
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    LaggedStart,
    LEFT,
    RIGHT,
    Text,
    UP,
    VGroup,
)

from layout_categories_viz import NestedTupleTree
from layout_categories_viz.style import CODE_FONT, INK


EXAMPLES = (
    ((2, 3), (5, 7)),
    (2, (3, (5, 7)), 11),
)


def _format_nested_tuple(value):
    if isinstance(value, int):
        return str(value)
    return "(" + ", ".join(_format_nested_tuple(item) for item in value) + ")"


class NestedTupleTreeTest(LayoutScene):
    """Create mirrored nested-tuple forests for source and target use."""

    def construct(self) -> None:
        for example in EXAMPLES:
            self._show_example(example)

    def _show_example(self, example) -> None:
        left_tree = NestedTupleTree(example, direction=RIGHT)
        right_tree = NestedTupleTree(example, direction=LEFT)
        left_tree.move_to(LEFT * 3.6)
        right_tree.move_to(RIGHT * 3.6)

        title = Text(
            _format_nested_tuple(example),
            color=INK,
            font=CODE_FONT,
            font_size=27,
        ).to_edge(UP, buff=0.38)
        left_heading = Text(
            "left → right",
            color=INK,
            font=CODE_FONT,
            font_size=20,
        ).next_to(left_tree, DOWN, buff=0.38)
        right_heading = Text(
            "right → left",
            color=INK,
            font=CODE_FONT,
            font_size=20,
        ).next_to(right_tree, DOWN, buff=0.38)

        self.play(
            FadeIn(title),
            FadeIn(left_heading),
            FadeIn(right_heading),
            LaggedStart(
                *(
                    FadeIn(node, scale=0.9)
                    for node in (
                        *left_tree.nodes_by_depth[0],
                        *right_tree.nodes_by_depth[0],
                    )
                ),
                lag_ratio=0.1,
            ),
            run_time=0.9,
        )

        for depth in range(1, len(left_tree.nodes_by_depth)):
            edges = (
                *left_tree.edges_by_depth[depth],
                *right_tree.edges_by_depth[depth],
            )
            nodes = (
                *left_tree.nodes_by_depth[depth],
                *right_tree.nodes_by_depth[depth],
            )
            self.play(
                LaggedStart(
                    *(Create(edge) for edge in edges),
                    lag_ratio=0.08,
                ),
                LaggedStart(
                    *(FadeIn(node, scale=0.9) for node in nodes),
                    lag_ratio=0.08,
                ),
                run_time=0.8,
            )

        self.wait(2.0)
        visible = VGroup(
            left_tree,
            right_tree,
            title,
            left_heading,
            right_heading,
        )
        self.play(FadeOut(visible), run_time=0.75)
        self.clear()
        self.wait(0.2)
