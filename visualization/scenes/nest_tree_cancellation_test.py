"""Isolated study of cancelling equal, oppositely oriented Nest trees."""

import numpy as np
from manim import (
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    LEFT,
    Line,
    PI,
    RIGHT,
    ReplacementTransform,
    Scene,
    Text,
    Transform,
    Uncreate,
    UpdateFromAlphaFunc,
    VGroup,
    config,
    smooth,
)

from layout_categories_viz import NestedTupleTree
from layout_categories_viz.coalesce import (
    CELL_CORNER_RADIUS,
    selective_corner_box,
)
from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK


NESTED_TUPLE = (((2, 3), (5, 7)), ((2, 5), (3, 7)))


class NestTreeCancellationTest(Scene):
    """Weld matching tree levels into their common cells, root to leaves."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        cell_size = 0.069 * config.frame_height
        font_size = 28 * cell_size / (0.085 * config.frame_height)
        tree_options = dict(
            cell_size=cell_size,
            level_gap=1.68 * cell_size,
            leaf_gap=1.23 * cell_size,
            label_font_size=font_size,
        )
        target_tree = NestedTupleTree(
            NESTED_TUPLE,
            direction=LEFT,
            **tree_options,
        )
        source_tree = NestedTupleTree(
            NESTED_TUPLE,
            direction=RIGHT,
            **tree_options,
        )
        target_tree.shift(LEFT * (0.72 + target_tree.get_right()[0]))
        source_tree.shift(RIGHT * (0.72 - source_tree.get_left()[0]))

        target_heading = self._heading(target_tree)
        source_heading = self._heading(source_tree)
        visible = VGroup(
            target_tree,
            source_tree,
            target_heading,
            source_heading,
        )
        scale_factor = min(
            1.0,
            0.90 * config.frame_height / visible.height,
            0.92 * config.frame_width / visible.width,
        )
        visible.scale(scale_factor).move_to((0, 0, 0))

        self.play(
            FadeIn(target_heading),
            FadeIn(source_heading),
            *(
                FadeIn(node, scale=0.9)
                for node in (*target_tree.nodes, *source_tree.nodes)
            ),
            run_time=0.9,
        )
        self.play(
            LaggedStart(
                *(
                    Create(edge)
                    for edge in sorted(
                        (*target_tree.edges, *source_tree.edges),
                        key=lambda edge: edge.get_left()[0],
                    )
                ),
                lag_ratio=0.05,
            ),
            run_time=1.6,
        )
        self.wait(0.65)

        previous_results = VGroup()
        depth_count = len(target_tree.nodes_by_depth)
        for depth in range(depth_count):
            left_nodes = tuple(target_tree.nodes_by_depth[depth])
            right_nodes = tuple(source_tree.nodes_by_depth[depth])
            preceding_edges = (
                ()
                if depth == 0
                else (
                    *target_tree.edges_by_depth[depth],
                    *source_tree.edges_by_depth[depth],
                )
            )
            left_remaining_nodes = tuple(
                node
                for level in target_tree.nodes_by_depth[depth + 1 :]
                for node in level
            )
            right_remaining_nodes = tuple(
                node
                for level in source_tree.nodes_by_depth[depth + 1 :]
                for node in level
            )
            left_remaining_edges = tuple(
                edge
                for level in target_tree.edges_by_depth[depth + 1 :]
                for edge in level
            )
            right_remaining_edges = tuple(
                edge
                for level in source_tree.edges_by_depth[depth + 1 :]
                for edge in level
            )
            results = self._weld_common_level(
                left_nodes,
                right_nodes,
                preceding_edges,
                previous_results,
                left_remaining_nodes,
                right_remaining_nodes,
                left_remaining_edges,
                right_remaining_edges,
            )
            previous_results = results
            if depth == depth_count - 1:
                self.wait(0.28)

        final_heading = Text(
            "T",
            color=INK,
            font=CODE_FONT,
            font_size=24 * scale_factor,
        ).next_to(previous_results, direction=(0, -1, 0), buff=0.34)
        self.play(
            ReplacementTransform(
                VGroup(target_heading, source_heading),
                final_heading,
            ),
            run_time=0.7,
        )
        self.wait(2.0)
        self.play(
            FadeOut(VGroup(previous_results, final_heading)),
            run_time=0.75,
        )
        self.clear()
        self.wait(0.2)

    @staticmethod
    def _heading(tree):
        return Text(
            "T",
            color=INK,
            font=CODE_FONT,
            font_size=24,
        ).next_to(tree, direction=(0, -1, 0), buff=0.34)

    @staticmethod
    def _horizontal_box_frame(
        initial_box,
        initial_center,
        destination,
        is_left,
        alpha,
        position_alpha=None,
    ):
        if position_alpha is None:
            position_alpha = alpha
        interior_radius = CELL_CORNER_RADIUS * (1.0 - alpha)
        frame = selective_corner_box(
            initial_box.get_height(),
            initial_box.get_width(),
            interior_radius if is_left else CELL_CORNER_RADIUS,
            CELL_CORNER_RADIUS if is_left else interior_radius,
            template=initial_box,
        )
        frame.rotate(-PI / 2)
        frame.move_to(
            initial_center
            + position_alpha * (destination - initial_center)
        )
        return frame

    @staticmethod
    def _horizontal_hull(left_entry, right_entry, center):
        template = left_entry[0]
        hull = selective_corner_box(
            template.get_height(),
            left_entry[0].get_width() + right_entry[0].get_width(),
            CELL_CORNER_RADIUS,
            CELL_CORNER_RADIUS,
            template=template,
        )
        hull.rotate(-PI / 2)
        return hull.move_to(center)

    @staticmethod
    def _horizontal_common_box(destination_entry):
        template = destination_entry[0]
        box = selective_corner_box(
            template.get_height(),
            template.get_width(),
            CELL_CORNER_RADIUS,
            CELL_CORNER_RADIUS,
            template=template,
        )
        box.rotate(-PI / 2)
        return box.move_to(template)

    def _weld_common_level(
        self,
        left_nodes,
        right_nodes,
        preceding_edges,
        previous_results,
        left_remaining_nodes,
        right_remaining_nodes,
        left_remaining_edges,
        right_remaining_edges,
    ):
        if preceding_edges or previous_results:
            self.play(
                *(Uncreate(edge) for edge in preceding_edges),
                *(
                    (FadeOut(previous_results, scale=0.88),)
                    if previous_results
                    else ()
                ),
                run_time=0.42,
            )

        approach_animations = []
        prepared = []
        cell_width = left_nodes[0][0].get_width()
        left_shift = (
            -cell_width / 2 - left_nodes[0].get_center()[0]
        ) * RIGHT
        right_shift = (
            cell_width / 2 - right_nodes[0].get_center()[0]
        ) * RIGHT
        approach_animations.extend(
            node.animate(rate_func=smooth).shift(left_shift)
            for node in left_remaining_nodes
        )
        approach_animations.extend(
            node.animate(rate_func=smooth).shift(right_shift)
            for node in right_remaining_nodes
        )
        approach_animations.extend(
            edge.animate(rate_func=smooth).shift(left_shift)
            for edge in left_remaining_edges
        )
        approach_animations.extend(
            edge.animate(rate_func=smooth).shift(right_shift)
            for edge in right_remaining_edges
        )
        for left_entry, right_entry in zip(left_nodes, right_nodes):
            center = np.array(
                (
                    0.0,
                    (left_entry.get_center()[1] + right_entry.get_center()[1])
                    / 2,
                    0.0,
                )
            )
            half_width = left_entry[0].get_width() / 2
            destinations = (
                center + LEFT * half_width,
                center + RIGHT * half_width,
            )
            for entry, destination, is_left in (
                (left_entry, destinations[0], True),
                (right_entry, destinations[1], False),
            ):
                initial_box = entry[0].copy()
                initial_center = entry[0].get_center().copy()

                def update_box(
                    displayed,
                    alpha,
                    initial_box=initial_box,
                    initial_center=initial_center,
                    destination=destination,
                    is_left=is_left,
                ):
                    displayed.become(
                        self._horizontal_box_frame(
                            initial_box,
                            initial_center,
                            destination,
                            is_left,
                            alpha,
                        )
                    )

                approach_animations.extend(
                    (
                        UpdateFromAlphaFunc(
                            entry[0],
                            update_box,
                            rate_func=smooth,
                        ),
                        entry[1].animate(rate_func=smooth).move_to(destination),
                    )
                )
            prepared.append((left_entry, right_entry, center))

        self.play(*approach_animations, run_time=0.9)

        hulls = VGroup()
        seams = VGroup()
        for left_entry, right_entry, center in prepared:
            hull = self._horizontal_hull(left_entry, right_entry, center)
            hull.set_z_index(-1)
            seam = Line(
                center + np.array((0.0, -left_entry[0].get_height() / 2, 0.0)),
                center + np.array((0.0, left_entry[0].get_height() / 2, 0.0)),
                color=left_entry[0].get_stroke_color(),
                stroke_width=left_entry[0].get_stroke_width(),
                stroke_opacity=left_entry[0].get_stroke_opacity(),
            )
            self.add(hull, seam)
            left_entry[0].set_opacity(0)
            right_entry[0].set_opacity(0)
            hulls.add(hull)
            seams.add(seam)
        self.play(
            *(Uncreate(seam) for seam in seams),
            run_time=0.5,
        )

        results = VGroup()
        finish_animations = []
        for (left_entry, right_entry, center), hull in zip(prepared, hulls):
            common_entry = left_entry.copy()
            common_entry.set_opacity(1)
            common_entry.move_to(center)
            finish_animations.extend(
                (
                    Transform(hull, self._horizontal_common_box(common_entry)),
                    left_entry[1].animate(rate_func=smooth).move_to(center),
                    FadeOut(right_entry[1]),
                )
            )
            results.add(VGroup(hull, left_entry[1]))
        collapse_shift = cell_width / 2
        finish_animations.extend(
            node.animate(rate_func=smooth).shift(RIGHT * collapse_shift)
            for node in left_remaining_nodes
        )
        finish_animations.extend(
            node.animate(rate_func=smooth).shift(LEFT * collapse_shift)
            for node in right_remaining_nodes
        )
        finish_animations.extend(
            edge.animate(rate_func=smooth).shift(RIGHT * collapse_shift)
            for edge in left_remaining_edges
        )
        finish_animations.extend(
            edge.animate(rate_func=smooth).shift(LEFT * collapse_shift)
            for edge in right_remaining_edges
        )
        self.play(*finish_animations, run_time=0.78, rate_func=smooth)
        for left_entry, right_entry, _ in prepared:
            self.remove(left_entry, right_entry)
        self.add(results)
        return results


class SimplifiedNestTreeCancellationTest(NestTreeCancellationTest):
    """Cancel matching levels by placing identical cells on one another."""

    def _weld_common_level(
        self,
        left_nodes,
        right_nodes,
        preceding_edges,
        previous_results,
        left_remaining_nodes,
        right_remaining_nodes,
        left_remaining_edges,
        right_remaining_edges,
        extra_animations=(),
    ):
        if preceding_edges or previous_results:
            self.play(
                *(Uncreate(edge) for edge in preceding_edges),
                *(
                    (FadeOut(previous_results, scale=0.88),)
                    if previous_results
                    else ()
                ),
                run_time=0.42,
            )

        prepared = []
        overlap_animations = []
        cell_width = left_nodes[0][0].get_width()
        for left_entry, right_entry in zip(left_nodes, right_nodes):
            center = (
                left_entry.get_center() + right_entry.get_center()
            ) / 2
            initial_left_box = left_entry[0].copy()
            initial_right_box = right_entry[0].copy()
            initial_left_center = left_entry.get_center().copy()
            initial_right_center = right_entry.get_center().copy()
            initial_separation = (
                initial_right_center[0] - initial_left_center[0]
            )
            collision_alpha = 1.0 - cell_width / initial_separation
            corner_start_alpha = max(0.0, collision_alpha - 0.18)

            hull = self._horizontal_hull(left_entry, right_entry, center)
            hull.set_z_index(-1).set_opacity(0)
            self.add(hull)

            def update_collision(
                displayed_hull,
                alpha,
                left_entry=left_entry,
                right_entry=right_entry,
                initial_left_box=initial_left_box,
                initial_right_box=initial_right_box,
                initial_left_center=initial_left_center,
                initial_right_center=initial_right_center,
                center=center,
                collision_alpha=collision_alpha,
                corner_start_alpha=corner_start_alpha,
            ):
                left_position = initial_left_center + alpha * (
                    center - initial_left_center
                )
                right_position = initial_right_center + alpha * (
                    center - initial_right_center
                )
                if alpha < collision_alpha:
                    corner_alpha = np.clip(
                        (alpha - corner_start_alpha)
                        / (collision_alpha - corner_start_alpha),
                        0.0,
                        1.0,
                    )
                    left_entry[0].become(
                        self._horizontal_box_frame(
                            initial_left_box,
                            initial_left_center,
                            center,
                            True,
                            corner_alpha,
                            position_alpha=alpha,
                        )
                    )
                    right_entry[0].become(
                        self._horizontal_box_frame(
                            initial_right_box,
                            initial_right_center,
                            center,
                            False,
                            corner_alpha,
                            position_alpha=alpha,
                        )
                    )
                    displayed_hull.set_opacity(0)
                    return

                left_entry[0].set_opacity(0)
                right_entry[0].set_opacity(0)
                separation = right_position[0] - left_position[0]
                overlap_hull = selective_corner_box(
                    initial_left_box.get_height(),
                    cell_width + separation,
                    CELL_CORNER_RADIUS,
                    CELL_CORNER_RADIUS,
                    template=initial_left_box,
                )
                overlap_hull.rotate(-PI / 2).move_to(center)
                displayed_hull.become(overlap_hull)

            overlap_animations.extend(
                (
                    UpdateFromAlphaFunc(
                        hull,
                        update_collision,
                        rate_func=smooth,
                    ),
                    left_entry[1].animate(rate_func=smooth).move_to(center),
                    right_entry[1].animate(rate_func=smooth).move_to(center),
                )
            )
            prepared.append((left_entry, right_entry, hull))

        left_shift = (
            (left_nodes[0].get_center() + right_nodes[0].get_center()) / 2
            - left_nodes[0].get_center()
        )
        right_shift = (
            (left_nodes[0].get_center() + right_nodes[0].get_center()) / 2
            - right_nodes[0].get_center()
        )
        overlap_animations.extend(
            node.animate(rate_func=smooth).shift(left_shift)
            for node in left_remaining_nodes
        )
        overlap_animations.extend(
            node.animate(rate_func=smooth).shift(right_shift)
            for node in right_remaining_nodes
        )
        overlap_animations.extend(
            edge.animate(rate_func=smooth).shift(left_shift)
            for edge in left_remaining_edges
        )
        overlap_animations.extend(
            edge.animate(rate_func=smooth).shift(right_shift)
            for edge in right_remaining_edges
        )
        self.play(
            *overlap_animations,
            *extra_animations,
            run_time=0.9,
        )

        results = VGroup()
        for left_entry, right_entry, hull in prepared:
            self.remove(left_entry, right_entry)
            results.add(VGroup(hull, left_entry[1]))
        self.add(results)
        return results
