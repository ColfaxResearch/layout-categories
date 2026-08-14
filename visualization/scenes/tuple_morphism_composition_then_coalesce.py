"""Compose two large tuple morphisms, then coalesce their composite."""

from manim import (
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    ORIGIN,
    ReplacementTransform,
    Transform,
    Uncreate,
    UP,
    UpdateFromAlphaFunc,
    VGroup,
    config,
    smooth,
)
from tract import Tuple_morphism

from layout_categories_viz.coalesce import (
    cell_weld_box_frame,
    cell_weld_hull,
    cell_weld_product_box,
    cell_weld_seams,
    correspondence_line_frame,
    correspondence_lines,
    correspondence_pull_arrow,
    interpolate_mapsto_arrow,
)
from layout_categories_viz.style import BACKGROUND
from layout_categories_viz.tuple_morphism import TupleMorphismDiagram
from scenes.tuple_morphism_coalesce import _coalesce_partitions
from scenes.tuple_morphism_composition_curve import (
    ARROW_BEND_HANDLE,
    ARROW_ENDPOINT_INSET,
    ARROW_HORIZONTAL_RUN,
    TupleMorphismCurvedCompositionCollapse,
)


EXAMPLE = {
    "domain": (2, 3, 17, 5, 7, 29, 19, 23),
    "intermediate": (7, 13, 2, 19, 17, 29, 5, 23, 3, 11),
    "codomain": (13, 2, 3, 11, 5, 7, 17, 19, 23),
    "first_mapping": (3, 9, 5, 7, 1, 6, 4, 8),
    "second_mapping": (6, 1, 2, 8, 7, 0, 5, 9, 3, 4),
}

CORRESPONDENCE_LINE_COUNT = 11
CORRESPONDENCE_LINE_WIDTH = 1.25
CORRESPONDENCE_LINE_OPACITY = 0.42
CORRESPONDENCE_DRAW_RUN_TIME = 0.65
CORRESPONDENCE_DRAW_LAG_RATIO = 0.055
CORRESPONDENCE_PULL_RUN_TIME = 1.15
CORRESPONDENCE_ENDPOINT_MARGIN = 0.06


class TupleMorphismCompositionThenCoalesceTest(
    TupleMorphismCurvedCompositionCollapse
):
    """Animate ``f``, ``g``, ``g ∘ f``, and finally ``coalesce(g ∘ f)``."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND

        first = Tuple_morphism(
            EXAMPLE["domain"],
            EXAMPLE["intermediate"],
            EXAMPLE["first_mapping"],
        )
        second = Tuple_morphism(
            EXAMPLE["intermediate"],
            EXAMPLE["codomain"],
            EXAMPLE["second_mapping"],
        )
        composite = first.compose(second)
        expected_composite = Tuple_morphism(
            EXAMPLE["domain"],
            EXAMPLE["codomain"],
            (2, 3, 7, 5, 6, 0, 8, 9),
        )
        if composite.map != expected_composite.map:
            raise ValueError("The configured composition changed unexpectedly")
        expected_coalesced = (
            (6, 17, 35, 29, 437),
            (13, 6, 11, 35, 17, 437),
            (2, 5, 4, 0, 6),
        )
        coalesced = composite.coalesce()
        if (
            coalesced.domain,
            coalesced.codomain,
            coalesced.map,
        ) != expected_coalesced:
            raise ValueError("The configured coalescence changed unexpectedly")

        frame_width = config.frame_width
        frame_height = config.frame_height
        max_tuple_length = max(
            len(first.domain),
            len(first.codomain),
            len(second.codomain),
        )
        row_gap_ratio = 0.29
        cell_size = min(
            0.07 * frame_width,
            0.095 * frame_height,
            0.70
            * frame_height
            / (
                max_tuple_length
                + row_gap_ratio * (max_tuple_length - 1)
            ),
        )
        row_gap = row_gap_ratio * cell_size
        label_font_size = 28 * cell_size / (0.085 * frame_height)
        column_gap = 0.17 * frame_width

        state = self._play_example(
            **EXAMPLE,
            column_gap=column_gap,
            row_gap=row_gap,
            entry_width=cell_size,
            entry_height=cell_size,
            label_font_size=label_font_size,
            retain_composite=True,
        )
        self._coalesce_composite(composite, coalesced, state)

    def _coalesce_composite(self, morphism, coalesced, state) -> None:
        source_entries = state["source_entries"]
        target_entries = state["target_entries"]
        arrows = state["arrows"]
        entry_width = state["entry_width"]
        entry_height = state["entry_height"]
        row_gap = state["row_gap"]
        label_font_size = state["label_font_size"]

        domain_classes, codomain_classes = _coalesce_partitions(morphism)
        column_gap = (
            target_entries[0].get_left()[0]
            - source_entries[0].get_right()[0]
        )
        final_diagram = TupleMorphismDiagram(
            coalesced.domain,
            coalesced.codomain,
            coalesced.map,
            column_gap=column_gap,
            row_gap=row_gap,
            entry_width=entry_width,
            entry_height=entry_height,
            label_font_size=label_font_size,
            source_label="S",
            target_label="T",
            arrow_endpoint_inset=ARROW_ENDPOINT_INSET,
            arrow_horizontal_run=ARROW_HORIZONTAL_RUN,
            arrow_bend_handle=ARROW_BEND_HANDLE,
        )
        final_diagram.shift(
            source_entries[0].get_center()
            - final_diagram.source_entries[0].get_center()
        )
        for arrow in final_diagram.arrows:
            arrow.tail.set_opacity(0)

        interim_source_entries = []
        for class_index, class_ in enumerate(domain_classes):
            if len(class_) == 1:
                interim_source_entries.append(source_entries[class_[0]])
                continue
            center = sum(
                (source_entries[index].get_center() for index in class_),
                start=ORIGIN.copy(),
            ) / len(class_)
            destination = final_diagram.source_entries[class_index].copy()
            destination.move_to(center)
            interim_source_entries.append(destination)

        interim_target_entries = []
        for class_index, class_ in enumerate(codomain_classes):
            if len(class_) == 1:
                interim_target_entries.append(target_entries[class_[0]])
                continue
            center = sum(
                (target_entries[index].get_center() for index in class_),
                start=ORIGIN.copy(),
            ) / len(class_)
            destination = final_diagram.target_entries[class_index].copy()
            destination.move_to(center)
            interim_target_entries.append(destination)

        surviving_source_entries = [
            source_entries[class_[-1]] for class_ in domain_classes
        ]
        surviving_target_entries = [
            target_entries[class_[-1]] for class_ in codomain_classes
        ]
        arrows_by_source = dict(
            zip(
                (
                    index
                    for index, target in enumerate(morphism.map)
                    if target
                ),
                arrows,
            )
        )
        final_arrows_by_source = dict(
            zip(
                (
                    index
                    for index, target in enumerate(coalesced.map)
                    if target
                ),
                final_diagram.arrows,
            )
        )
        surviving_arrows = []
        for class_ in domain_classes:
            mapped_indices = [
                index for index in class_ if morphism.map[index]
            ]
            if mapped_indices:
                surviving_arrows.append(
                    arrows_by_source[mapped_indices[-1]]
                )

        collapse_batches = []
        for class_index, class_ in enumerate(domain_classes):
            if len(class_) == 1:
                continue

            source_destination = interim_source_entries[class_index]
            source_block_entries = [source_entries[index] for index in class_]
            source_touching_centers = [
                source_destination.get_center()
                + UP * (position - (len(class_) - 1) / 2) * entry_height
                for position in range(len(class_))
            ]
            animations = []
            for position, (entry, destination) in enumerate(
                zip(source_block_entries, source_touching_centers)
            ):
                initial_box = entry[0].copy()
                initial_center = entry.get_center().copy()

                def update_cell(
                    displayed,
                    alpha,
                    initial_box=initial_box,
                    initial_center=initial_center,
                    destination=destination,
                    position=position,
                    count=len(class_),
                ):
                    displayed.become(
                        cell_weld_box_frame(
                            initial_box,
                            initial_center,
                            destination,
                            position,
                            count,
                            alpha,
                        )
                    )

                animations.extend(
                    (
                        UpdateFromAlphaFunc(
                            entry[0],
                            update_cell,
                            run_time=CORRESPONDENCE_PULL_RUN_TIME,
                            rate_func=smooth,
                        ),
                        entry[1].animate(
                            run_time=CORRESPONDENCE_PULL_RUN_TIME,
                            rate_func=smooth,
                        ).move_to(destination),
                    )
                )

            target_class_index = coalesced.map[class_index] - 1
            target_class = codomain_classes[target_class_index]
            target_destination = interim_target_entries[target_class_index]
            target_block_entries = [
                target_entries[index] for index in target_class
            ]
            target_touching_centers = [
                target_destination.get_center()
                + UP
                * (position - (len(target_class) - 1) / 2)
                * entry_height
                for position in range(len(target_class))
            ]
            for position, (entry, destination) in enumerate(
                zip(target_block_entries, target_touching_centers)
            ):
                initial_box = entry[0].copy()
                initial_center = entry.get_center().copy()

                def update_cell(
                    displayed,
                    alpha,
                    initial_box=initial_box,
                    initial_center=initial_center,
                    destination=destination,
                    position=position,
                    count=len(target_class),
                ):
                    displayed.become(
                        cell_weld_box_frame(
                            initial_box,
                            initial_center,
                            destination,
                            position,
                            count,
                            alpha,
                        )
                    )

                animations.extend(
                    (
                        UpdateFromAlphaFunc(
                            entry[0],
                            update_cell,
                            run_time=CORRESPONDENCE_PULL_RUN_TIME,
                            rate_func=smooth,
                        ),
                        entry[1].animate(
                            run_time=CORRESPONDENCE_PULL_RUN_TIME,
                            rate_func=smooth,
                        ).move_to(destination),
                    )
                )

            mapped_indices = [
                index for index in class_ if morphism.map[index]
            ]
            block_arrows = tuple(
                arrows_by_source[index] for index in mapped_indices
            )
            lines = correspondence_lines(
                block_arrows,
                line_count=CORRESPONDENCE_LINE_COUNT,
                endpoint_margin=CORRESPONDENCE_ENDPOINT_MARGIN,
                stroke_width=CORRESPONDENCE_LINE_WIDTH,
                stroke_opacity=CORRESPONDENCE_LINE_OPACITY,
            )
            initial_lines = lines.copy()
            vertical_deltas = []
            shrink_arrow_specs = []
            target_positions = {
                target_index: position
                for position, target_index in enumerate(target_class)
            }
            for position, index in enumerate(mapped_indices):
                arrow = arrows_by_source[index]
                target_index = morphism.map[index] - 1
                source_touching = source_touching_centers[class_.index(index)]
                target_touching = target_touching_centers[
                    target_positions[target_index]
                ]
                source_delta = (
                    source_touching[1] - source_entries[index].get_y()
                )
                target_delta = (
                    target_touching[1] - target_entries[target_index].get_y()
                )
                if abs(source_delta - target_delta) > 1e-6:
                    raise ValueError(
                        "coalescing arrows must be vertical translations"
                    )
                vertical_delta = (source_delta + target_delta) / 2
                vertical_deltas.append(vertical_delta)
                shrink_arrow_specs.append(
                    (
                        arrow,
                        source_destination.get_y() - source_touching[1],
                        position < len(mapped_indices) - 1,
                    )
                )
                initial_arrow = arrow.copy()

                def update_arrow(
                    displayed,
                    alpha,
                    initial_arrow=initial_arrow,
                    vertical_delta=vertical_delta,
                ):
                    displayed.become(
                        correspondence_pull_arrow(
                            initial_arrow, vertical_delta, alpha
                        )
                    )

                animations.append(
                    UpdateFromAlphaFunc(
                        arrow,
                        update_arrow,
                        run_time=CORRESPONDENCE_PULL_RUN_TIME,
                        rate_func=smooth,
                    )
                )

            def update_lines(
                displayed,
                alpha,
                initial_lines=initial_lines,
                first_delta=vertical_deltas[0],
                last_delta=vertical_deltas[-1],
            ):
                displayed.become(
                    correspondence_line_frame(
                        initial_lines,
                        first_delta,
                        last_delta,
                        alpha,
                    )
                )

            animations.append(
                UpdateFromAlphaFunc(
                    lines,
                    update_lines,
                    run_time=CORRESPONDENCE_PULL_RUN_TIME,
                    rate_func=smooth,
                )
            )
            collapse_batches.append(
                (
                    lines,
                    animations,
                    tuple(
                        arrows_by_source[index]
                        for index in mapped_indices[:-1]
                    ),
                    source_block_entries,
                    source_destination,
                    target_block_entries,
                    target_destination,
                    shrink_arrow_specs,
                )
            )

        for (
            lines,
            animations,
            redundant_arrows,
            source_block_entries,
            source_destination,
            target_block_entries,
            target_destination,
            shrink_arrow_specs,
        ) in collapse_batches:
            self.play(
                LaggedStart(
                    *(GrowFromCenter(line) for line in lines),
                    lag_ratio=CORRESPONDENCE_DRAW_LAG_RATIO,
                ),
                run_time=CORRESPONDENCE_DRAW_RUN_TIME,
            )
            self.wait(0.25)
            self.play(*animations)

            source_hull = cell_weld_hull(
                source_block_entries, source_destination.get_center()
            )
            target_hull = cell_weld_hull(
                target_block_entries, target_destination.get_center()
            )
            source_seams = cell_weld_seams(
                source_block_entries, source_destination.get_center()
            )
            target_seams = cell_weld_seams(
                target_block_entries, target_destination.get_center()
            )
            source_hull.set_z_index(-1)
            target_hull.set_z_index(-1)
            self.add(source_hull, target_hull, source_seams, target_seams)
            for entry in (*source_block_entries, *target_block_entries):
                entry[0].set_opacity(0)
            self.remove(lines)
            self.play(
                *(
                    Uncreate(seam)
                    for seam in (*source_seams, *target_seams)
                ),
                run_time=0.55,
            )

            finish_animations = [
                Transform(
                    source_hull,
                    cell_weld_product_box(source_destination),
                ),
                ReplacementTransform(
                    VGroup(*(entry[1] for entry in source_block_entries)),
                    source_destination[1],
                ),
                Transform(
                    target_hull,
                    cell_weld_product_box(target_destination),
                ),
                ReplacementTransform(
                    VGroup(*(entry[1] for entry in target_block_entries)),
                    target_destination[1],
                ),
            ]
            for arrow, vertical_delta, redundant in shrink_arrow_specs:
                initial_arrow = arrow.copy()

                def finish_arrow(
                    displayed,
                    alpha,
                    initial_arrow=initial_arrow,
                    vertical_delta=vertical_delta,
                    redundant=redundant,
                ):
                    frame = correspondence_pull_arrow(
                        initial_arrow, vertical_delta, alpha
                    )
                    frame.tail.set_opacity(0)
                    if redundant:
                        frame.shaft.set_opacity(1.0 - alpha)
                        frame.tip.set_opacity(1.0 - alpha)
                    displayed.become(frame)

                finish_animations.append(
                    UpdateFromAlphaFunc(
                        arrow,
                        finish_arrow,
                        rate_func=smooth,
                    )
                )
            self.play(*finish_animations, run_time=0.8)

            source_survivor = source_block_entries[-1]
            target_survivor = target_block_entries[-1]
            self.remove(
                source_hull,
                target_hull,
                source_destination[1],
                target_destination[1],
                *source_block_entries,
                *target_block_entries,
            )
            source_survivor.become(source_destination)
            target_survivor.become(target_destination)
            self.add(source_survivor, target_survivor)
            for arrow in redundant_arrows:
                arrow.set_opacity(0)
            self.wait(0.18)
        self.wait(0.22)

        compaction_animations = [
            *(
                survivor.animate.move_to(
                    final_diagram.source_entries[
                        class_index
                    ].get_center()
                )
                for class_index, survivor in enumerate(
                    surviving_source_entries
                )
            ),
            *(
                survivor.animate.move_to(
                    final_diagram.target_entries[
                        class_index
                    ].get_center()
                )
                for class_index, survivor in enumerate(
                    surviving_target_entries
                )
            ),
        ]
        for class_index, survivor in zip(
            (
                index
                for index, target in enumerate(coalesced.map)
                if target
            ),
            surviving_arrows,
        ):
            target_class_index = coalesced.map[class_index] - 1
            initial_arrow = TupleMorphismDiagram._mapsto_arrow(
                surviving_source_entries[class_index].get_right(),
                surviving_target_entries[target_class_index].get_left(),
                endpoint_inset=ARROW_ENDPOINT_INSET,
                horizontal_run=ARROW_HORIZONTAL_RUN,
                bend_handle=ARROW_BEND_HANDLE,
            )
            initial_arrow.tail.set_opacity(0)
            final_arrow = final_arrows_by_source[class_index].copy()

            def update_arrow(
                displayed,
                alpha,
                initial_arrow=initial_arrow,
                final_arrow=final_arrow,
            ):
                displayed.become(
                    interpolate_mapsto_arrow(
                        initial_arrow,
                        final_arrow,
                        alpha,
                    )
                )

            compaction_animations.append(
                UpdateFromAlphaFunc(
                    survivor,
                    update_arrow,
                    rate_func=smooth,
                )
            )
        self.play(*compaction_animations, run_time=1.25)
        self.wait(0.45)

        self.wait(2.0)

        visible_result = VGroup(
            *surviving_source_entries,
            *surviving_target_entries,
            *surviving_arrows,
            state["first_label"],
            state["second_label"],
            state["composition_symbol"],
        )
        self.play(FadeOut(visible_result), run_time=0.85)
