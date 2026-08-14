"""Prototype animation for composition in the category Fact.

A Fact morphism (u_1,...,u_p) -> (t_1,...,t_n) factors each codomain entry:
its picture is a fan-in, each codomain cell gathering the block of domain
cells that multiply to it.  Composition concatenates blocks of blocks, so the
animation mirrors the tuple-morphism composition collapse:

    open:      [S] --a--> [T] --b--> [U]
    dissolve:  the middle stack shrinks away; each domain cell's two-segment
               route through its vanished middle cell is joined into one
    collapse:  the outer stacks contract inward and every route deforms into
               the composite fan, block within block becoming one block

Every domain cell survives (Fact morphisms have no basepoints), so unlike the
tuple case nothing retracts: routes are only ever joined and contracted.  The
drawing primitives are the pullback scene's -- a fan segment is a refinement
segment, and a stack is cells on one baseline with uniform slots -- so the
Fact picture cannot drift from the refinement pictures it is a special case
of.  Fans carry no carets: a Fact morphism is drawn purely as its refinement
segments, which converge on a cell sharing their closing run, exactly as tree
edges share their root-side endpoint.
"""

from layout_categories_viz.scene_base import LayoutScene

from dataclasses import dataclass
from math import prod

import numpy as np
from manim import (
    Create,
    CubicBezier,
    FadeIn,
    LaggedStart,
    LEFT,
    RIGHT,
    ShrinkToCenter,
    Text,
    Transform,
    VGroup,
    Write,
)
from tract import FactMorphism

from layout_categories_viz.style import CODE_FONT, INK

# The composition scene owns the route-surgery helpers and the pullback scene
# owns the drawing primitives, so this scene cannot drift from either family.
from layout_categories_viz.paths import (
    matched_path_pair,
    three_segment_route,
)
from layout_categories_viz import stacks
from layout_categories_viz.stacks import (
    CELL_H,
    LEFT_X,
    MID_X,
    RIGHT_X,
    SLOT_STEP,
    STROKE_WIDTH,
)

LABEL_BUFF = 0.55


@dataclass(frozen=True)
class FactCompositionExample:
    """Two composable Fact morphisms, specified by their modes alone."""

    first_modes: tuple
    second_modes: tuple

    def morphisms(self) -> tuple:
        """The morphisms a and b, validated by the library."""
        first_domain = tuple(x for mode in self.first_modes for x in mode)
        middle = tuple(prod(mode) for mode in self.first_modes)
        target = tuple(prod(mode) for mode in self.second_modes)
        f = FactMorphism(first_domain, middle, self.first_modes)
        g = FactMorphism(middle, target, self.second_modes)
        return f, g


EXAMPLES = (
    FactCompositionExample(
        first_modes=((2, 3), (2, 2), (5,)),
        second_modes=((6, 4), (5,)),
    ),
    FactCompositionExample(
        first_modes=((2, 2), (3,), (7,)),
        second_modes=((4, 3, 7),),
    ),
)


class FactMorphismCompositionTest(LayoutScene):
    def construct(self):
        for index, example in enumerate(EXAMPLES):
            self._play_example(example)
            self.clear_scene(last=index == len(EXAMPLES) - 1)

    def _play_example(self, example: FactCompositionExample) -> None:
        f, g = example.morphisms()
        # Composing validates the pair; its blocks are the composite fans.
        composite = f.compose(g)

        # Which middle cell a domain cell feeds, and which target cell a
        # middle cell feeds: composition is exactly the join of the two.
        middle_of_domain = tuple(
            j for j, mode in enumerate(f.modes) for _ in mode
        )
        target_of_middle = tuple(
            k for k, mode in enumerate(g.modes) for _ in mode
        )

        tallest = len(f.domain)
        baseline = -(tallest - 1) * SLOT_STEP / 2 + 0.25

        def column(values, x):
            return VGroup(
                *(
                    stacks.cell(
                        value, np.array([x, baseline + i * SLOT_STEP, 0.0])
                    )
                    for i, value in enumerate(values)
                )
            )

        source_cells = column(f.domain, LEFT_X)
        middle_cells = column(f.codomain, MID_X)
        target_cells = column(g.codomain, RIGHT_X)

        # Fan segments, one per source cell, converging on the cell of the
        # block it belongs to; one caret per receiving cell, shared by its
        # fan the way tree edges share their root-side endpoint.
        first_segments = VGroup(
            *(
                stacks.tree_segment(
                    source_cells[i].get_right(), middle_cells[j].get_left()
                )
                for i, j in enumerate(middle_of_domain)
            )
        )
        second_segments = VGroup(
            *(
                stacks.tree_segment(
                    middle_cells[j].get_right(), target_cells[k].get_left()
                )
                for j, k in enumerate(target_of_middle)
            )
        )

        # Morphism labels, and a normally typeset "b ∘ a" harvested for the
        # exact glyph destinations they slide into.
        label_y = baseline - CELL_H / 2 - LABEL_BUFF
        first_label = Text("a", color=INK, font=CODE_FONT, font_size=30)
        first_label.move_to(np.array([(LEFT_X + MID_X) / 2, label_y, 0.0]))
        second_label = Text("b", color=INK, font=CODE_FONT, font_size=30)
        second_label.move_to(np.array([(MID_X + RIGHT_X) / 2, label_y, 0.0]))
        composition_reference = Text(
            "b ∘ a", color=INK, font=CODE_FONT, font_size=30
        )
        composition_reference.move_to(np.array([MID_X, label_y, 0.0]))
        b_glyph, _, composition_glyph, _, a_glyph = tuple(composition_reference)
        composition_symbol = Text("∘", color=INK, font=CODE_FONT, font_size=30)
        composition_symbol.move_to(composition_glyph)

        # ---------------------------------------------------------- opening
        self.play(
            FadeIn(source_cells),
            FadeIn(middle_cells),
            FadeIn(target_cells),
            FadeIn(first_label),
            FadeIn(second_label),
        )
        self.play(
            LaggedStart(
                *(Create(segment) for segment in first_segments),
                lag_ratio=0.12,
                run_time=1.3,
            )
        )
        self.play(
            LaggedStart(
                *(Create(segment) for segment in second_segments),
                lag_ratio=0.12,
                run_time=1.3,
            )
        )
        self.wait(0.35)

        # Every domain cell's route continues through its middle cell: its
        # own fan segment, a connector across the cell, and a private copy of
        # the block's continuation (coincident copies render as one stroke).
        routes = []
        for i, j in enumerate(middle_of_domain):
            first_segment = first_segments[i]
            continuation = second_segments[j].copy()
            first_end = first_segment.get_end()
            second_start = continuation.get_start()
            span = second_start - first_end
            connector = CubicBezier(
                first_end,
                first_end + span / 3,
                first_end + 2 * span / 3,
                second_start,
                color=INK,
                stroke_width=STROKE_WIDTH,
            )
            routes.append((first_segment, connector, continuation))

        # --------------------------------------------------------- dissolve
        self.play(
            *(ShrinkToCenter(cell) for cell in middle_cells),
            run_time=0.9,
        )
        self.play(
            *(Create(connector) for _, connector, _ in routes),
            run_time=1.35,
        )
        self.wait(0.25)

        # --------------------------------------------------------- collapse
        # The outer stacks contract inward until the composite occupies the
        # width of a single morphism, and each glued route deforms into the
        # composite fan segment between its shifted cells.
        source_shift = RIGHT * (MID_X - LEFT_X) / 2
        target_shift = LEFT * (RIGHT_X - MID_X) / 2

        initial_routes = VGroup()
        final_routes = VGroup()
        for i, (first_segment, connector, continuation) in enumerate(routes):
            glued = three_segment_route(first_segment, connector, continuation)
            k = target_of_middle[middle_of_domain[i]]
            final_segment = stacks.tree_segment(
                source_cells[i].get_right() + source_shift,
                target_cells[k].get_left() + target_shift,
            )
            initial, final = matched_path_pair(glued, final_segment)
            initial_routes.add(initial)
            final_routes.add(final)

        # Hand off from the stage segments to the glued routes with a single
        # frame of identical pixels.
        self.remove(
            *first_segments,
            *second_segments,
            *(connector for _, connector, _ in routes),
        )
        self.add(initial_routes)
        self.wait(1 / 15)

        # The composed label forms in step with the move-in, so it stays
        # centered under the contracting diagram.
        self.play(
            Transform(initial_routes, final_routes),
            source_cells.animate.shift(source_shift),
            target_cells.animate.shift(target_shift),
            first_label.animate.move_to(a_glyph),
            second_label.animate.move_to(b_glyph),
            Write(composition_symbol),
            run_time=1.35,
        )
        # Land on the canonical geometry exactly.
        self.remove(initial_routes)
        self.add(final_routes)

        # The picture now reads as the composite: one fan per composite mode.
        assert len(composite.modes) == len(target_cells)
        self.wait(1.0)
