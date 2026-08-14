"""Animate disjoint local coalesces of tuple-morphism entries."""

from manim import (
    DOWN,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    ORIGIN,
    ReplacementTransform,
    Scene,
    Text,
    Transform,
    Uncreate,
    UP,
    UpdateFromAlphaFunc,
    VGroup,
    config,
    smooth,
)
from tract import Tuple_morphism

from layout_categories_viz import TailToTipMapsto, TupleMorphismDiagram
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
from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK


EXAMPLES = (
    {
        "domain": (2, 3, 7, 5),
        "codomain": (2, 3, 5, 7, 11),
        "mapping": (1, 2, 4, 3),
    },
    {
        # Two mapped runs separated by the untouched 7 mode.
        "domain": (2, 3, 7, 11, 5),
        "codomain": (2, 3, 11, 5, 7),
        "mapping": (1, 2, 5, 3, 4),
    },
    {
        # Three mapped runs, visibly separated by untouched 7 and 13 modes.
        "domain": (2, 3, 7, 5, 11, 13, 17, 19),
        "codomain": (2, 3, 13, 5, 11, 7, 17, 19),
        "mapping": (1, 2, 6, 4, 5, 3, 7, 8),
    },
)

ARROW_HORIZONTAL_RUN = 0.2
ARROW_ENDPOINT_INSET = 0.08
ARROW_BEND_HANDLE = 0.8
CORRESPONDENCE_LINE_COUNT = 11
CORRESPONDENCE_LINE_WIDTH = 1.25
CORRESPONDENCE_LINE_OPACITY = 0.42
CORRESPONDENCE_DRAW_RUN_TIME = 0.65
CORRESPONDENCE_DRAW_LAG_RATIO = 0.055
CORRESPONDENCE_PULL_RUN_TIME = 1.15
CORRESPONDENCE_ENDPOINT_MARGIN = 0.06


def _coalesce_partitions(morphism: Tuple_morphism):
    """Return the zero-based domain and codomain classes used by coalesce."""
    mapping = morphism.map

    domain_classes = []
    current_class = [0]
    for index in range(1, len(morphism.domain)):
        previous_value = mapping[index - 1]
        current_value = mapping[index]
        if (previous_value == 0 and current_value == 0) or (
            previous_value != 0 and current_value == previous_value + 1
        ):
            current_class.append(index)
        else:
            domain_classes.append(current_class)
            current_class = [index]
    domain_classes.append(current_class)

    image = set(mapping)
    codomain_classes = []
    current_class = [0]
    for index in range(1, len(morphism.codomain)):
        previous_target = index
        if previous_target in image:
            source_index = mapping.index(previous_target)
            continues_mapped_run = (
                source_index + 1 < len(mapping)
                and mapping[source_index + 1] == index + 1
            )
        else:
            continues_mapped_run = False

        if continues_mapped_run:
            current_class.append(index)
        else:
            codomain_classes.append(current_class)
            current_class = [index]
    codomain_classes.append(current_class)

    return domain_classes, codomain_classes


class TupleMorphismCoalesce(Scene):
    """Collapse each consecutive order-preserving block to its product."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND

        for example in EXAMPLES:
            self._play_example(**example)

    @staticmethod
    def _place_labels(diagram, morphism_label) -> None:
        diagram.source_heading.scale(1.25)
        diagram.target_heading.scale(1.25)
        diagram.source_heading.next_to(diagram.source_entries[0], DOWN, buff=0.24)
        diagram.target_heading.next_to(diagram.target_entries[0], DOWN, buff=0.24)
        morphism_label.next_to(
            VGroup(diagram.source_heading, diagram.target_heading),
            DOWN,
            buff=0.22,
        )
        VGroup(diagram, morphism_label).move_to(ORIGIN)

    def _play_example(self, *, domain, codomain, mapping) -> None:
        morphism = Tuple_morphism(domain, codomain, mapping)
        if any(value == 1 for value in (*domain, *codomain)):
            raise ValueError("This scene handles coalesce examples without squeezing")

        coalesced = morphism.coalesce()
        domain_classes, codomain_classes = _coalesce_partitions(morphism)
        if sum(len(group) > 1 for group in domain_classes) < 1:
            raise ValueError("Each scene example must contain a coalescing run")

        frame_width = config.frame_width
        frame_height = config.frame_height
        max_tuple_length = max(len(morphism.domain), len(morphism.codomain))
        row_gap_ratio = 0.32
        cell_size = min(
            0.11 * frame_height,
            0.07 * frame_width,
            0.66
            * frame_height
            / (max_tuple_length + row_gap_ratio * (max_tuple_length - 1)),
        )
        row_gap = row_gap_ratio * cell_size
        entry_font_size = 28 * cell_size / (0.085 * frame_height)
        column_gap = 0.23 * frame_width
        layout_font_size = int(3.75 * frame_height)

        diagram = TupleMorphismDiagram(
            morphism.domain,
            morphism.codomain,
            morphism.map,
            column_gap=column_gap,
            row_gap=row_gap,
            entry_width=cell_size,
            entry_height=cell_size,
            label_font_size=entry_font_size,
            source_label="S",
            target_label="T",
            arrow_endpoint_inset=ARROW_ENDPOINT_INSET,
            arrow_horizontal_run=ARROW_HORIZONTAL_RUN,
            arrow_bend_handle=ARROW_BEND_HANDLE,
        )
        morphism_label = Text(
            "f", color=INK, font=CODE_FONT, font_size=layout_font_size
        )
        self._place_labels(diagram, morphism_label)

        final_diagram = TupleMorphismDiagram(
            coalesced.domain,
            coalesced.codomain,
            coalesced.map,
            column_gap=column_gap,
            row_gap=row_gap,
            entry_width=cell_size,
            entry_height=cell_size,
            label_font_size=entry_font_size,
            source_label="S",
            target_label="T",
            arrow_endpoint_inset=ARROW_ENDPOINT_INSET,
            arrow_horizontal_run=ARROW_HORIZONTAL_RUN,
            arrow_bend_handle=ARROW_BEND_HANDLE,
        )
        # Standard post-coalesce geometry is bottom-aligned with the original
        # tuple.  This makes the second stage a pure downward compaction and
        # leaves S, T, and f fixed throughout both stages.
        final_diagram.shift(
            diagram.source_entries[0].get_center()
            - final_diagram.source_entries[0].get_center()
        )

        for arrow in (*diagram.arrows, *final_diagram.arrows):
            arrow.tail.set_opacity(0)

        entries = tuple(diagram.source_entries) + tuple(diagram.target_entries)
        self.play(
            FadeIn(diagram.source_heading),
            FadeIn(diagram.target_heading),
            FadeIn(morphism_label),
            LaggedStart(
                *(FadeIn(entry, scale=0.9) for entry in entries),
                lag_ratio=0.08,
            ),
            run_time=1.1,
        )
        self.add(diagram.arrows)
        self.play(
            LaggedStart(
                *(TailToTipMapsto(arrow) for arrow in diagram.arrows),
                lag_ratio=0.14,
            ),
            run_time=1.35,
        )
        self.wait(1.0)

        # First collapse each nontrivial class at the vertical center of its
        # original span.  Singleton entries and their arrows do not move.
        interim_source_entries = []
        for class_index, class_ in enumerate(domain_classes):
            if len(class_) == 1:
                interim_source_entries.append(diagram.source_entries[class_[0]])
                continue
            center = sum(
                (diagram.source_entries[index].get_center() for index in class_),
                start=ORIGIN.copy(),
            ) / len(class_)
            destination = final_diagram.source_entries[class_index].copy()
            destination.move_to(center)
            interim_source_entries.append(destination)

        interim_target_entries = []
        for class_index, class_ in enumerate(codomain_classes):
            if len(class_) == 1:
                interim_target_entries.append(diagram.target_entries[class_[0]])
                continue
            center = sum(
                (diagram.target_entries[index].get_center() for index in class_),
                start=ORIGIN.copy(),
            ) / len(class_)
            destination = final_diagram.target_entries[class_index].copy()
            destination.move_to(center)
            interim_target_entries.append(destination)

        surviving_source_entries = [
            diagram.source_entries[class_[-1]] for class_ in domain_classes
        ]
        surviving_target_entries = [
            diagram.target_entries[class_[-1]] for class_ in codomain_classes
        ]

        arrows_by_source = dict(
            zip(
                (index for index, target in enumerate(morphism.map) if target),
                diagram.arrows,
            )
        )
        final_arrows_by_source = dict(
            zip(
                (index for index, target in enumerate(coalesced.map) if target),
                final_diagram.arrows,
            )
        )
        surviving_arrows = []
        for class_index, class_ in enumerate(domain_classes):
            mapped_indices = [index for index in class_ if morphism.map[index]]
            if not mapped_indices:
                continue
            survivor = arrows_by_source[mapped_indices[-1]]
            surviving_arrows.append(survivor)

        # Assemble one batch per nontrivial source class.  Its matching source
        # entries, target entries, and parallel arrows collapse together; the
        # batches themselves play sequentially from bottom to top.
        collapse_batches = []
        for class_index, class_ in enumerate(domain_classes):
            if len(class_) == 1:
                continue

            animations = []
            source_destination = interim_source_entries[class_index]
            source_block_entries = [
                diagram.source_entries[index] for index in class_
            ]
            source_touching_centers = [
                source_destination.get_center()
                + UP
                * (position - (len(class_) - 1) / 2)
                * cell_size
                for position in range(len(class_))
            ]
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

                animations.append(
                    UpdateFromAlphaFunc(
                        entry[0],
                        update_cell,
                        run_time=CORRESPONDENCE_PULL_RUN_TIME,
                        rate_func=smooth,
                    )
                )
                animations.append(
                    entry[1].animate(
                        run_time=CORRESPONDENCE_PULL_RUN_TIME,
                        rate_func=smooth,
                    ).move_to(destination)
                )

            target_class_index = coalesced.map[class_index] - 1
            target_class = codomain_classes[target_class_index]
            target_destination = interim_target_entries[target_class_index]
            target_block_entries = [
                diagram.target_entries[index] for index in target_class
            ]
            target_touching_centers = [
                target_destination.get_center()
                + UP
                * (position - (len(target_class) - 1) / 2)
                * cell_size
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

                animations.append(
                    UpdateFromAlphaFunc(
                        entry[0],
                        update_cell,
                        run_time=CORRESPONDENCE_PULL_RUN_TIME,
                        rate_func=smooth,
                    )
                )
                animations.append(
                    entry[1].animate(
                        run_time=CORRESPONDENCE_PULL_RUN_TIME,
                        rate_func=smooth,
                    ).move_to(destination)
                )

            mapped_indices = [index for index in class_ if morphism.map[index]]
            block_arrows = tuple(
                arrows_by_source[index] for index in mapped_indices
            )
            block_correspondence_lines = correspondence_lines(
                block_arrows,
                line_count=CORRESPONDENCE_LINE_COUNT,
                endpoint_margin=CORRESPONDENCE_ENDPOINT_MARGIN,
                stroke_width=CORRESPONDENCE_LINE_WIDTH,
                stroke_opacity=CORRESPONDENCE_LINE_OPACITY,
            )
            initial_correspondence_lines = block_correspondence_lines.copy()
            arrow_vertical_deltas = []
            shrink_arrow_specs = []
            target_positions = {
                target_index: position
                for position, target_index in enumerate(target_class)
            }
            for position, index in enumerate(mapped_indices):
                displayed_arrow = arrows_by_source[index]
                target_index = morphism.map[index] - 1
                initial_arrow = displayed_arrow.copy()
                source_touching = source_touching_centers[
                    class_.index(index)
                ]
                target_touching = target_touching_centers[
                    target_positions[target_index]
                ]
                source_delta = (
                    source_touching[1]
                    - diagram.source_entries[index].get_y()
                )
                target_delta = (
                    target_touching[1]
                    - diagram.target_entries[target_index].get_y()
                )
                if abs(source_delta - target_delta) > 1e-6:
                    raise ValueError(
                        "coalescing arrows must be vertical translations"
                    )
                vertical_delta = (source_delta + target_delta) / 2.0
                arrow_vertical_deltas.append(vertical_delta)
                shrink_delta = (
                    source_destination.get_y() - source_touching[1]
                )
                shrink_arrow_specs.append(
                    (
                        displayed_arrow,
                        shrink_delta,
                        position < len(mapped_indices) - 1,
                    )
                )

                def update(
                    displayed,
                    alpha,
                    initial_arrow=initial_arrow,
                    vertical_delta=vertical_delta,
                ):
                    displayed.become(
                        correspondence_pull_arrow(
                            initial_arrow,
                            vertical_delta,
                            alpha,
                        )
                    )

                animations.append(
                    UpdateFromAlphaFunc(
                        displayed_arrow,
                        update,
                        run_time=CORRESPONDENCE_PULL_RUN_TIME,
                        rate_func=smooth,
                    )
                )

            def update_lines(
                displayed,
                alpha,
                initial_lines=initial_correspondence_lines,
                first_delta=arrow_vertical_deltas[0],
                last_delta=arrow_vertical_deltas[-1],
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
                    block_correspondence_lines,
                    update_lines,
                    run_time=CORRESPONDENCE_PULL_RUN_TIME,
                    rate_func=smooth,
                )
            )
            redundant_arrows = tuple(
                arrows_by_source[index] for index in mapped_indices[:-1]
            )
            collapse_batches.append(
                (
                    block_correspondence_lines,
                    animations,
                    redundant_arrows,
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
                source_block_entries,
                source_destination.get_center(),
            )
            target_hull = cell_weld_hull(
                target_block_entries,
                target_destination.get_center(),
            )
            source_seams = cell_weld_seams(
                source_block_entries,
                source_destination.get_center(),
            )
            target_seams = cell_weld_seams(
                target_block_entries,
                target_destination.get_center(),
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
                        initial_arrow,
                        vertical_delta,
                        alpha,
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

        # Only after every block has collapsed in place do the surviving cells
        # and arrows close their gaps into ordinary tuple-morphism geometry.
        compaction_animations = []
        for class_index, survivor in enumerate(surviving_source_entries):
            compaction_animations.append(
                survivor.animate.move_to(
                    final_diagram.source_entries[class_index].get_center()
                )
            )
        for class_index, survivor in enumerate(surviving_target_entries):
            compaction_animations.append(
                survivor.animate.move_to(
                    final_diagram.target_entries[class_index].get_center()
                )
            )
        for class_index, survivor in zip(
            (index for index, target in enumerate(coalesced.map) if target),
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
            diagram.source_heading,
            diagram.target_heading,
            morphism_label,
        )
        self.play(FadeOut(visible_result), run_time=0.85)
        self.clear()
        self.wait(0.2)
