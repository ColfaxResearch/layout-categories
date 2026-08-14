"""Compose Nest morphisms by cancelling their common nested tuple tree."""

from layout_categories_viz.scene_base import LayoutScene

from dataclasses import dataclass

import numpy as np
from manim import (
    CubicBezier,
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    ShrinkToCenter,
    Text,
    Transform,
    UpdateFromAlphaFunc,
    VGroup,
    Write,
    config,
    smooth,
)
from tract import NestMorphism

from layout_categories_viz import NestMorphismDiagram, TailToTipMapsto
from layout_categories_viz.animations import UncreateMapstoTip
from layout_categories_viz.coalesce import interpolate_mapsto_arrow
from layout_categories_viz.style import CODE_FONT, INK
from layout_categories_viz.tuple_morphism import TupleMorphismDiagram
from layout_categories_viz.paths import (
    matched_path_pair,
    three_segment_route,
)
from scenes.nest_tree_cancellation_test import (
    SimplifiedNestTreeCancellationTest,
)


@dataclass(frozen=True)
class NestMorphismCompositionExample:
    domain: tuple
    intermediate: tuple
    codomain: tuple
    first_mapping: tuple[int, ...]
    second_mapping: tuple[int, ...]


EXAMPLES = (
    NestMorphismCompositionExample(
        domain=(2, 3),
        intermediate=(2, 3),
        codomain=(3, 2),
        first_mapping=(1, 2),
        second_mapping=(2, 1),
    ),
    NestMorphismCompositionExample(
        domain=(2, (3, 5), 7),
        intermediate=((2, 3), (5, 7)),
        codomain=((5, 7), (2, 3)),
        first_mapping=(1, 2, 3, 4),
        second_mapping=(3, 4, 1, 2),
    ),
    NestMorphismCompositionExample(
        domain=(((2, 5), (7, 11)), (13, (17, (19, 3)))),
        intermediate=(((2, 3), (5, 7)), ((11, 13), (17, 19))),
        codomain=(((19, 17), (13, 11)), ((7, 5), (3, 2))),
        first_mapping=(1, 3, 4, 5, 6, 7, 8, 2),
        second_mapping=(8, 7, 6, 5, 4, 3, 2, 1),
    ),
)

# Kept for the original single-example composition scene below.
DOMAIN = EXAMPLES[1].domain
INTERMEDIATE = EXAMPLES[1].intermediate
CODOMAIN = EXAMPLES[1].codomain
FIRST_MAPPING = EXAMPLES[1].first_mapping
SECOND_MAPPING = EXAMPLES[1].second_mapping


class NestMorphismCompositionTest(LayoutScene):
    """Overlap and cancel equal middle trees, leaving the composite."""

    def construct(self) -> None:
        frame_height = config.frame_height
        cell_size = 0.074 * frame_height
        diagram_options = dict(
            column_gap=0.19 * config.frame_width,
            cell_size=cell_size,
            level_gap=1.65 * cell_size,
            leaf_gap=1.30 * cell_size,
            label_font_size=28 * cell_size / (0.085 * frame_height),
        )

        first = NestMorphismDiagram(
            DOMAIN,
            INTERMEDIATE,
            FIRST_MAPPING,
            source_label="S",
            target_label="T",
            **diagram_options,
        )
        second = NestMorphismDiagram(
            INTERMEDIATE,
            CODOMAIN,
            SECOND_MAPPING,
            source_label="T",
            target_label="U",
            **diagram_options,
        )
        first.shift(
            (-0.28 - first.target_tree.get_right()[0], 0.0, 0.0)
        )
        second.shift(
            (0.28 - second.source_tree.get_left()[0], 0.0, 0.0)
        )

        label_font_size = int(3.25 * frame_height)
        first_label = Text(
            "f", color=INK, font=CODE_FONT, font_size=label_font_size
        ).next_to(
            VGroup(first.source_heading, first.target_heading),
            direction=(0, -1, 0),
            buff=0.18,
        )
        second_label = Text(
            "g", color=INK, font=CODE_FONT, font_size=label_font_size
        ).next_to(
            VGroup(second.source_heading, second.target_heading),
            direction=(0, -1, 0),
            buff=0.18,
        )
        visible = VGroup(first, second, first_label, second_label)
        scale_factor = min(
            1.0,
            0.88 * config.frame_height / visible.height,
            0.94 * config.frame_width / visible.width,
        )
        visible.scale(scale_factor).move_to((0, 0, 0))

        all_nodes = (
            *first.source_tree.nodes,
            *first.target_tree.nodes,
            *second.source_tree.nodes,
            *second.target_tree.nodes,
        )
        self.play(
            FadeIn(first.source_heading),
            FadeIn(first.target_heading),
            FadeIn(second.source_heading),
            FadeIn(second.target_heading),
            FadeIn(first_label),
            FadeIn(second_label),
            *(FadeIn(node, scale=0.9) for node in all_nodes),
            run_time=0.9,
        )

        tree_edges = (
            *first.source_tree.edges,
            *first.target_tree.edges,
            *second.source_tree.edges,
            *second.target_tree.edges,
        )
        connections = [
            *(Create(edge) for edge in tree_edges),
            *(TailToTipMapsto(arrow) for arrow in (*first.arrows, *second.arrows)),
        ]
        connections.sort(
            key=lambda animation: animation.mobject.get_left()[0]
        )
        self.add(first.arrows, second.arrows)
        self.play(
            LaggedStart(*connections, lag_ratio=0.055),
            run_time=2.5,
        )
        self.wait(0.8)

        aligned_second_arrows = []
        for source_index, target_index in enumerate(SECOND_MAPPING):
            arrow = TupleMorphismDiagram.mapsto_arrow(
                first.target_tree.leaf_entries[source_index].get_right(),
                second.target_tree.leaf_entries[target_index - 1].get_left(),
                endpoint_inset=0.08 * scale_factor,
                horizontal_run=0.2 * scale_factor,
                bend_handle=0.8 * scale_factor,
            )
            arrow.tail.set_opacity(0)
            aligned_second_arrows.append(arrow)

        overlap_animations = []
        for displayed, destination in zip(
            second.source_tree.nodes,
            first.target_tree.nodes,
        ):
            overlap_animations.append(Transform(displayed, destination.copy()))
        for displayed, destination in zip(
            second.source_tree.edges,
            first.target_tree.edges,
        ):
            overlap_animations.append(Transform(displayed, destination.copy()))
        overlap_animations.append(
            Transform(second.source_heading, first.target_heading.copy())
        )
        for displayed, destination in zip(
            second.arrows,
            aligned_second_arrows,
        ):
            initial = displayed.copy()

            def update_arrow(
                arrow,
                alpha,
                initial=initial,
                destination=destination,
            ):
                arrow.become(
                    interpolate_mapsto_arrow(initial, destination, alpha)
                )

            overlap_animations.append(
                UpdateFromAlphaFunc(
                    displayed,
                    update_arrow,
                    rate_func=smooth,
                )
            )
        self.play(*overlap_animations, run_time=1.35)
        self.wait(0.55)

        first_morphism = NestMorphism(
            DOMAIN, INTERMEDIATE, FIRST_MAPPING
        )
        second_morphism = NestMorphism(
            INTERMEDIATE, CODOMAIN, SECOND_MAPPING
        )
        composite = first_morphism.compose(second_morphism)
        final_arrows = []
        for source_index, target_index in enumerate(composite.map):
            if target_index == 0:
                continue
            arrow = TupleMorphismDiagram.mapsto_arrow(
                first.source_tree.leaf_entries[source_index].get_right(),
                second.target_tree.leaf_entries[target_index - 1].get_left(),
                endpoint_inset=0.08 * scale_factor,
                horizontal_run=0.2 * scale_factor,
                bend_handle=0.8 * scale_factor,
            )
            arrow.tail.set_opacity(0)
            final_arrows.append(arrow)

        composite_label = Text(
            "g ∘ f",
            color=INK,
            font=CODE_FONT,
            font_size=label_font_size * scale_factor,
        ).next_to(
            VGroup(first.source_heading, second.target_heading),
            direction=(0, -1, 0),
            buff=0.18,
        )
        finish_animations = [
            FadeOut(first.target_tree, scale=0.88),
            FadeOut(second.source_tree, scale=0.88),
            FadeOut(first.target_heading),
            FadeOut(second.source_heading),
            FadeOut(first_label),
            FadeOut(second_label),
            FadeIn(composite_label),
            FadeOut(second.arrows),
        ]
        for displayed, destination in zip(first.arrows, final_arrows):
            initial = displayed.copy()

            def update_composite_arrow(
                arrow,
                alpha,
                initial=initial,
                destination=destination,
            ):
                arrow.become(
                    interpolate_mapsto_arrow(initial, destination, alpha)
                )

            finish_animations.append(
                UpdateFromAlphaFunc(
                    displayed,
                    update_composite_arrow,
                    rate_func=smooth,
                )
            )
        self.play(*finish_animations, run_time=1.4)
        self.wait(2.2)

        result = VGroup(
            first.source_tree,
            second.target_tree,
            first.arrows,
            first.source_heading,
            second.target_heading,
            composite_label,
        )
        self.play(FadeOut(result), run_time=0.85)
        self.clear()
        self.wait(0.2)


class NestMorphismCancellationCompositionTest(
    SimplifiedNestTreeCancellationTest
):
    """Compose three increasingly complex Nest morphism examples."""

    @staticmethod
    def _arrow_transform(displayed, destination):
        initial = displayed.copy()

        def update_arrow(arrow, alpha):
            arrow.become(
                interpolate_mapsto_arrow(initial, destination, alpha)
            )

        return UpdateFromAlphaFunc(
            displayed,
            update_arrow,
            rate_func=smooth,
        )

    def construct(self) -> None:

        for example in EXAMPLES:
            self._show_example(example)

    def _show_example(self, example: NestMorphismCompositionExample) -> None:
        """Show one composition, cancelling its shared tree level by level."""
        frame_height = config.frame_height
        cell_size = 0.074 * frame_height
        arrow_options = dict(
            endpoint_inset=0.08,
            horizontal_run=0.2,
            bend_handle=0.8,
        )
        diagram_options = dict(
            column_gap=0.19 * config.frame_width,
            cell_size=cell_size,
            level_gap=1.65 * cell_size,
            leaf_gap=1.30 * cell_size,
            label_font_size=28 * cell_size / (0.085 * frame_height),
            arrow_endpoint_inset=arrow_options["endpoint_inset"],
            arrow_horizontal_run=arrow_options["horizontal_run"],
            arrow_bend_handle=arrow_options["bend_handle"],
        )

        first = NestMorphismDiagram(
            example.domain,
            example.intermediate,
            example.first_mapping,
            source_label="S",
            target_label="T",
            **diagram_options,
        )
        second = NestMorphismDiagram(
            example.intermediate,
            example.codomain,
            example.second_mapping,
            source_label="T",
            target_label="U",
            **diagram_options,
        )
        first.shift((-0.28 - first.target_tree.get_right()[0], 0.0, 0.0))
        second.shift((0.28 - second.source_tree.get_left()[0], 0.0, 0.0))

        label_font_size = int(3.25 * frame_height)
        first_label = Text(
            "f", color=INK, font=CODE_FONT, font_size=label_font_size
        ).next_to(
            VGroup(first.source_heading, first.target_heading),
            direction=(0, -1, 0),
            buff=0.18,
        )
        second_label = Text(
            "g", color=INK, font=CODE_FONT, font_size=label_font_size
        ).next_to(
            VGroup(second.source_heading, second.target_heading),
            direction=(0, -1, 0),
            buff=0.18,
        )
        visible = VGroup(first, second, first_label, second_label)
        scale_factor = min(
            1.0,
            0.88 * config.frame_height / visible.height,
            0.94 * config.frame_width / visible.width,
        )
        visible.scale(scale_factor).move_to((0, 0, 0))
        scaled_arrow_options = {
            key: value * scale_factor for key, value in arrow_options.items()
        }

        all_nodes = (
            *first.source_tree.nodes,
            *first.target_tree.nodes,
            *second.source_tree.nodes,
            *second.target_tree.nodes,
        )
        self.play(
            FadeIn(first.source_heading),
            FadeIn(first.target_heading),
            FadeIn(second.source_heading),
            FadeIn(second.target_heading),
            FadeIn(first_label),
            FadeIn(second_label),
            *(FadeIn(node, scale=0.9) for node in all_nodes),
            run_time=0.9,
        )

        connections = [
            *(
                Create(edge)
                for edge in (
                    *first.source_tree.edges,
                    *first.target_tree.edges,
                    *second.source_tree.edges,
                    *second.target_tree.edges,
                )
            ),
            *(
                TailToTipMapsto(arrow)
                for arrow in (*first.arrows, *second.arrows)
            ),
        ]
        connections.sort(key=lambda animation: animation.mobject.get_left()[0])
        self.add(first.arrows, second.arrows)
        self.play(
            LaggedStart(*connections, lag_ratio=0.055),
            run_time=2.5,
        )
        self.wait(0.65)

        previous_results = VGroup()
        depth_count = len(first.target_tree.nodes_by_depth)
        for depth in range(depth_count):
            left_nodes = tuple(first.target_tree.nodes_by_depth[depth])
            right_nodes = tuple(second.source_tree.nodes_by_depth[depth])
            center = (
                left_nodes[0].get_center() + right_nodes[0].get_center()
            ) / 2
            left_shift = center - left_nodes[0].get_center()
            right_shift = center - right_nodes[0].get_center()

            arrow_animations = []
            for displayed, source_index, target_index in zip(
                first.arrows,
                range(len(example.first_mapping)),
                example.first_mapping,
            ):
                destination = TupleMorphismDiagram.mapsto_arrow(
                    first.source_tree.leaf_entries[source_index].get_right()
                    + left_shift,
                    first.target_tree.leaf_entries[
                        target_index - 1
                    ].get_left()
                    + left_shift,
                    **scaled_arrow_options,
                )
                destination.tail.set_opacity(0)
                arrow_animations.append(
                    self._arrow_transform(displayed, destination)
                )
            for displayed, source_index, target_index in zip(
                second.arrows,
                range(len(example.second_mapping)),
                example.second_mapping,
            ):
                destination = TupleMorphismDiagram.mapsto_arrow(
                    second.source_tree.leaf_entries[
                        source_index
                    ].get_right()
                    + right_shift,
                    second.target_tree.leaf_entries[
                        target_index - 1
                    ].get_left()
                    + right_shift,
                    **scaled_arrow_options,
                )
                destination.tail.set_opacity(0)
                arrow_animations.append(
                    self._arrow_transform(displayed, destination)
                )

            preceding_edges = (
                ()
                if depth == 0
                else (
                    *first.target_tree.edges_by_depth[depth],
                    *second.source_tree.edges_by_depth[depth],
                )
            )
            left_remaining_nodes = tuple(
                node
                for level in first.target_tree.nodes_by_depth[depth + 1 :]
                for node in level
            )
            right_remaining_nodes = tuple(
                node
                for level in second.source_tree.nodes_by_depth[depth + 1 :]
                for node in level
            )
            left_remaining_edges = tuple(
                edge
                for level in first.target_tree.edges_by_depth[depth + 1 :]
                for edge in level
            )
            right_remaining_edges = tuple(
                edge
                for level in second.source_tree.edges_by_depth[depth + 1 :]
                for edge in level
            )
            diagram_motion_animations = [
                first.source_tree.animate(rate_func=smooth).shift(left_shift),
                first.source_heading.animate(rate_func=smooth).shift(left_shift),
                first_label.animate(rate_func=smooth).shift(left_shift),
                second.target_tree.animate(rate_func=smooth).shift(right_shift),
                second.target_heading.animate(rate_func=smooth).shift(right_shift),
                second_label.animate(rate_func=smooth).shift(right_shift),
            ]
            if depth == 0:
                diagram_motion_animations.extend(
                    (
                        FadeOut(first.target_heading),
                        FadeOut(second.source_heading),
                    )
                )
            previous_results = self._weld_common_level(
                left_nodes,
                right_nodes,
                preceding_edges,
                previous_results,
                left_remaining_nodes,
                right_remaining_nodes,
                left_remaining_edges,
                right_remaining_edges,
                extra_animations=(
                    *arrow_animations,
                    *diagram_motion_animations,
                ),
            )
            if depth == 0:
                # FadeOut is the visual transition; remove the headings from
                # the scene immediately afterward so later composition steps
                # cannot make the shared labels visible again.
                self.remove(first.target_heading, second.source_heading)
            if depth == depth_count - 1:
                self.wait(0.28)

        # The cancelled middle cells now serve as the shared column of two
        # composable tuple morphisms. Retract their tips and draw the short
        # connectors, as in the tuple-morphism composition animation.
        first_arrow_iterator = iter(first.arrows)
        first_arrows_by_source = {
            source_index: next(first_arrow_iterator)
            for source_index, target_index in enumerate(example.first_mapping)
            if target_index
        }
        second_arrow_iterator = iter(second.arrows)
        second_arrows_by_source = {
            source_index: next(second_arrow_iterator)
            for source_index, target_index in enumerate(example.second_mapping)
            if target_index
        }
        routes = []
        for source_index, intermediate_index in enumerate(
            example.first_mapping
        ):
            if intermediate_index == 0:
                continue
            target_index = example.second_mapping[intermediate_index - 1]
            if target_index == 0:
                continue
            routes.append(
                (
                    first_arrows_by_source[source_index],
                    second_arrows_by_source[intermediate_index - 1],
                    source_index,
                    target_index - 1,
                )
            )
        connectors = []
        for first_arrow, second_arrow, _, _ in routes:
            first_end = first_arrow.shaft.get_end()
            second_start = second_arrow.shaft.get_start()
            connector_span = np.array(
                (second_start[0] - first_end[0], 0.0, 0.0)
            )
            connectors.append(
                CubicBezier(
                    first_end,
                    first_end + connector_span / 3,
                    first_end + 2 * connector_span / 3,
                    second_start,
                    stroke_width=3.6,
                    color=INK,
                )
            )
        self.play(
            ShrinkToCenter(previous_results),
            *(UncreateMapstoTip(first_arrow) for first_arrow, _, _, _ in routes),
            *(Create(connector) for connector in connectors),
            run_time=1.0,
        )

        # Build the same curved two-stage routes used by the tuple scene.
        curved_arrows = VGroup()
        contracted_arrows = VGroup()
        # Preserve the tuple-morphism width from the example. Whole-tree
        # centers are not suitable anchors here: nested trees can have very
        # different shapes, which makes the leaf columns collapse toward one
        # another even when the underlying morphism has a normal width.
        source_leaf = first.source_tree.leaf_entries[0].get_center()
        first_target_leaf = first.target_tree.leaf_entries[0].get_center()
        second_source_leaf = second.source_tree.leaf_entries[0].get_center()
        target_leaf = second.target_tree.leaf_entries[0].get_center()
        leaf_span = max(
            first_target_leaf[0] - source_leaf[0],
            target_leaf[0] - second_source_leaf[0],
        )
        composite_center_x = (source_leaf[0] + target_leaf[0]) / 2
        source_shift = np.array(
            (
                composite_center_x - leaf_span / 2 - source_leaf[0],
                0.0,
                0.0,
            )
        )
        target_shift = np.array(
            (
                composite_center_x + leaf_span / 2 - target_leaf[0],
                0.0,
                0.0,
            )
        )
        for (
            first_arrow,
            second_arrow,
            source_index,
            target_index,
        ), connector in zip(routes, connectors):
            route_shaft = three_segment_route(
                first_arrow.shaft,
                connector,
                second_arrow.shaft,
            )
            contracted = TupleMorphismDiagram.mapsto_arrow(
                first.source_tree.leaf_entries[source_index].get_right()
                + source_shift,
                second.target_tree.leaf_entries[target_index].get_left()
                + target_shift,
                **scaled_arrow_options,
            )
            contracted.tail.set_opacity(0)
            initial_shaft, final_shaft = matched_path_pair(
                route_shaft,
                contracted.shaft,
            )
            curved_arrows.add(
                first_arrow.from_parts(
                    first_arrow.tail.copy(),
                    initial_shaft,
                    second_arrow.tip.copy(),
                )
            )
            contracted_arrows.add(contracted.from_parts(
                contracted.tail,
                final_shaft,
                contracted.tip,
            ))

        self.remove(
            *first.arrows,
            *second.arrows,
            *(arrow.tail for arrow in (*first.arrows, *second.arrows)),
            *(arrow.shaft for arrow in (*first.arrows, *second.arrows)),
            *(arrow.tip for arrow in (*first.arrows, *second.arrows)),
            *connectors,
        )
        self.add(curved_arrows)

        # Use the same glyph-level label transition as tuple-morphism
        # composition. The reference is hidden; its glyph centers preserve
        # the font's baseline and kerning for the moving f and g labels.
        composition_symbol = Text(
            "∘",
            color=INK,
            font=CODE_FONT,
            font_size=label_font_size * scale_factor,
        )
        composition_reference = Text(
            "g ∘ f",
            color=INK,
            font=CODE_FONT,
            font_size=label_font_size * scale_factor,
        ).next_to(
            VGroup(
                first.source_heading.copy().shift(source_shift),
                second.target_heading.copy().shift(target_shift),
            ),
            direction=(0, -1, 0),
            buff=0.18,
        )
        g_glyph, _, composition_glyph, _, f_glyph = tuple(
            composition_reference
        )
        composition_symbol.move_to(composition_glyph)
        self.play(
            Transform(curved_arrows, contracted_arrows),
            first.source_tree.animate.shift(source_shift),
            second.target_tree.animate.shift(target_shift),
            first.source_heading.animate.shift(source_shift),
            second.target_heading.animate.shift(target_shift),
            first_label.animate.move_to(f_glyph),
            second_label.animate.move_to(g_glyph),
            Write(composition_symbol),
            run_time=1.35,
        )
        self.remove(curved_arrows)
        self.add(contracted_arrows)
        self.wait(2.2)

        result = VGroup(
            first.source_tree,
            second.target_tree,
            contracted_arrows,
            first.source_heading,
            second.target_heading,
            first_label,
            second_label,
            composition_symbol,
        )
        self.play(FadeOut(result), run_time=0.85)
        self.clear()
        self.wait(0.2)
