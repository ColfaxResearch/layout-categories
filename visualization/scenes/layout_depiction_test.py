"""Depict nested layouts as paired vertical shape and stride tuples.

A layout L = S : D is drawn as two row-aligned vertical stacks — the shape
leaves on the left in light blue, the stride leaves on the right in light
red (SHAPE_FILL / STRIDE_FILL in style.py), a colon between each pair —
with the nesting drawn as the full Ref morphism of S, inverted, to the
left: the shape stack doubles as the source flat(S), the target rho(S)
stands unlabelled in shape-filled product cells further left, and the
banded ref trees, mirrored, fan out of the product cells into the shape
leaves.  Since D is congruent to S, the one tree carries the nesting of
both stacks.

The scene loops through examples of increasing depth: a flat layout (each
tree a single level strand, the identity Ref morphism read backward), one
junction, a nested tail beside an unnested mode, a deep chain, and a wide
mixed-depth layout.  The library validates every example: cells and trees
are read off layout_diagram, which checks shape/stride congruence and
builds its trees from the RefMorphism of the shape.
"""

from layout_categories_viz.scene_base import LayoutScene

from dataclasses import dataclass

from manim import (
    Create,
    FadeIn,
    LaggedStart,
    Text,
    UP,
)
from tract import NestedTuple

from layout_categories_viz.layouts import layout_diagram
from layout_categories_viz.style import CODE_FONT, INK
from layout_categories_viz.stacks import (
    CELL_H,
    LABEL_FONT_SIZE,
    MAX_STACK_HEIGHT,
    SLOT_STEP,
    make_place,
)

LABEL_BUFF = 0.55


@dataclass(frozen=True)
class LayoutExample:
    """A layout, as its congruent shape and stride nests."""

    shape: tuple
    stride: tuple


EXAMPLES = (
    # Flat: no junctions, every tree a single level strand — the identity
    # Ref morphism read backward.
    LayoutExample(shape=(4, 2, 5), stride=(1, 4, 8)),
    # One junction: the first mode groups (2, 3).
    LayoutExample(shape=((2, 3), 4), stride=((1, 2), 6)),
    # A nested tail beside an unnested mode.
    LayoutExample(shape=(8, (2, (2, 2))), stride=(1, (8, (16, 32)))),
    # A deep chain: junctions at every level between root and leaves.
    LayoutExample(shape=(((2, 2), 3), 5), stride=(((1, 2), 4), 12)),
    # Wide and mixed depth, stressing the shared band boundaries.
    LayoutExample(
        shape=(7, (2, 2, 3), (2, (2, 2))),
        stride=(1, (7, 14, 28), (84, (168, 336))),
    ),
)


class LayoutDepictionTest(LayoutScene):
    """Draw layouts as shape and stride stacks with their nesting tree."""

    def construct(self) -> None:
        for index, example in enumerate(EXAMPLES):
            self._play_example(example)
            self.clear_scene(last=index == len(EXAMPLES) - 1)

    def _play_example(self, example: LayoutExample) -> None:
        slots = len(NestedTuple(example.shape).flatten())
        # The title reserves headroom, so tall stacks scale into a shorter
        # window than MAX_STACK_HEIGHT and sit slightly below center.
        usable_height = MAX_STACK_HEIGHT - 0.9
        scale = min(1.0, usable_height / ((slots - 1) * SLOT_STEP + CELL_H))
        step = SLOT_STEP * scale
        place = make_place(-((slots - 1) * step) / 2 - 0.05, step)

        diagram = layout_diagram(
            example.shape, example.stride, place, scale=scale
        )

        title = Text(
            f"L = {diagram.shape}:{diagram.stride}",
            color=INK,
            font=CODE_FONT,
            font_size=27,
        ).to_edge(UP, buff=0.38)
        label_y = place(0, 0)[1] - CELL_H * scale / 2 - LABEL_BUFF

        def column_label(text: str, cells) -> Text:
            label = Text(
                text, color=INK, font=CODE_FONT, font_size=LABEL_FONT_SIZE
            ).scale(scale)
            return label.move_to(
                [cells[0].get_center()[0], label_y, 0.0]
            )

        shape_label = column_label("S", diagram.shape_cells)
        stride_label = column_label("D", diagram.stride_cells)

        self.play(
            FadeIn(diagram.cells()),
            FadeIn(diagram.colons),
            FadeIn(title),
            FadeIn(shape_label),
            FadeIn(stride_label),
        )
        # Strands are built child-to-parent; the inverse Ref morphism reads
        # root-to-leaves, so they draw left to right, root strands first.
        strands = [
            strand
            for tree in diagram.trees
            for strand in reversed(list(tree))
        ]
        for strand in strands:
            strand.reverse_points()
        self.play(
            LaggedStart(
                *(Create(strand) for strand in strands),
                lag_ratio=0.12,
                run_time=1.6,
            )
        )
        self.wait(1.0)
