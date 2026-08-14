"""Reusable geometry for local tuple-morphism coalesce gestures."""

import numpy as np
from manim import (
    GrowFromCenter,
    LaggedStart,
    Line,
    ORIGIN,
    ReplacementTransform,
    Transform,
    Uncreate,
    UP,
    UpdateFromAlphaFunc,
    VGroup,
    VMobject,
    smooth,
)

from .paths import (
    ARROW_BEND_HANDLE,
    ARROW_ENDPOINT_INSET,
    ARROW_HORIZONTAL_RUN,
)
from .style import INK
from .tuple_morphism import TupleMorphismDiagram


CELL_CORNER_RADIUS = 0.08

CORRESPONDENCE_LINE_COUNT = 11
CORRESPONDENCE_LINE_WIDTH = 1.25
CORRESPONDENCE_LINE_OPACITY = 0.42
CORRESPONDENCE_DRAW_RUN_TIME = 0.65
CORRESPONDENCE_DRAW_LAG_RATIO = 0.055
CORRESPONDENCE_PULL_RUN_TIME = 1.15
CORRESPONDENCE_ENDPOINT_MARGIN = 0.06


def selective_corner_box(
    width,
    height,
    top_radius,
    bottom_radius,
    *,
    template,
):
    """Build a cell whose top and bottom corner radii vary independently."""
    left = -width / 2
    right = width / 2
    bottom = -height / 2
    top = height / 2
    kappa = 4 * (2**0.5 - 1) / 3

    box = VMobject()
    box.start_new_path(np.array((left + top_radius, top, 0.0)))
    box.add_line_to(np.array((right - top_radius, top, 0.0)))
    if top_radius:
        box.add_cubic_bezier_curve_to(
            np.array((right - top_radius + kappa * top_radius, top, 0.0)),
            np.array((right, top - top_radius + kappa * top_radius, 0.0)),
            np.array((right, top - top_radius, 0.0)),
        )
    box.add_line_to(np.array((right, bottom + bottom_radius, 0.0)))
    if bottom_radius:
        box.add_cubic_bezier_curve_to(
            np.array((right, bottom + bottom_radius - kappa * bottom_radius, 0.0)),
            np.array((right - bottom_radius + kappa * bottom_radius, bottom, 0.0)),
            np.array((right - bottom_radius, bottom, 0.0)),
        )
    box.add_line_to(np.array((left + bottom_radius, bottom, 0.0)))
    if bottom_radius:
        box.add_cubic_bezier_curve_to(
            np.array((left + bottom_radius - kappa * bottom_radius, bottom, 0.0)),
            np.array((left, bottom + bottom_radius - kappa * bottom_radius, 0.0)),
            np.array((left, bottom + bottom_radius, 0.0)),
        )
    box.add_line_to(np.array((left, top - top_radius, 0.0)))
    if top_radius:
        box.add_cubic_bezier_curve_to(
            np.array((left, top - top_radius + kappa * top_radius, 0.0)),
            np.array((left + top_radius - kappa * top_radius, top, 0.0)),
            np.array((left + top_radius, top, 0.0)),
        )
    box.close_path()
    box.match_style(template)
    return box


def cell_weld_box_frame(
    initial_box,
    initial_center,
    destination,
    index,
    count,
    alpha,
):
    """Return one approach frame with only future interior corners sharpening."""
    interior_radius = CELL_CORNER_RADIUS * (1.0 - alpha)
    top_radius = (
        CELL_CORNER_RADIUS if index == count - 1 else interior_radius
    )
    bottom_radius = CELL_CORNER_RADIUS if index == 0 else interior_radius
    frame = selective_corner_box(
        initial_box.get_width(),
        initial_box.get_height(),
        top_radius,
        bottom_radius,
        template=initial_box,
    )
    frame.move_to(initial_center + alpha * (destination - initial_center))
    return frame


def cell_weld_hull(entries, center):
    """Return the rounded outer hull of a touching stack of cells."""
    template = entries[0][0]
    return selective_corner_box(
        template.get_width(),
        sum(entry[0].get_height() for entry in entries),
        CELL_CORNER_RADIUS,
        CELL_CORNER_RADIUS,
        template=template,
    ).move_to(center)


def cell_weld_product_box(destination_entry):
    """Return a standard rounded product cell with weld-compatible topology."""
    template = destination_entry[0]
    return selective_corner_box(
        template.get_width(),
        template.get_height(),
        CELL_CORNER_RADIUS,
        CELL_CORNER_RADIUS,
        template=template,
    ).move_to(template)


def cell_weld_seams(entries, center):
    """Return the horizontal interior borders of a touching cell stack."""
    template = entries[0][0]
    count = len(entries)
    cell_width = template.get_width()
    cell_height = template.get_height()
    return VGroup(
        *(
            Line(
                center
                + np.array((-cell_width / 2, offset, 0.0)),
                center
                + np.array((cell_width / 2, offset, 0.0)),
                color=template.get_stroke_color(),
                stroke_width=template.get_stroke_width(),
                stroke_opacity=template.get_stroke_opacity(),
            )
            for offset in (
                (index - count / 2) * cell_height
                for index in range(1, count)
            )
        )
    )


def correspondence_lines(
    arrows,
    *,
    line_count,
    endpoint_margin,
    stroke_width,
    stroke_opacity,
):
    """Join matching path parameters on the outermost parallel shafts."""
    if len(arrows) < 2:
        raise ValueError("correspondence lines require at least two arrows")
    if line_count < 2:
        raise ValueError("line_count must be at least two")
    if not 0.0 <= endpoint_margin < 0.5:
        raise ValueError("endpoint_margin must lie in [0, 0.5)")

    first, last = arrows[0], arrows[-1]
    proportions = np.linspace(
        endpoint_margin,
        1.0 - endpoint_margin,
        line_count,
    )
    return VGroup(
        *(
            Line(
                first.shaft.point_from_proportion(proportion),
                last.shaft.point_from_proportion(proportion),
                color=INK,
                stroke_width=stroke_width,
                stroke_opacity=stroke_opacity,
            )
            for proportion in proportions
        )
    )


def correspondence_pull_arrow(initial_arrow, vertical_delta, alpha):
    """Rigidly pull one parallel arrow toward its common centerline."""
    return initial_arrow.copy().shift(UP * vertical_delta * alpha)


def correspondence_line_frame(
    initial_lines,
    first_vertical_delta,
    last_vertical_delta,
    alpha,
):
    """Return attached correspondence lines for one pull-animation frame."""
    progress = alpha
    remaining = 1.0 - progress
    rebuilt = VGroup()
    for initial_line in initial_lines:
        start = (
            initial_line.get_start()
            + UP * first_vertical_delta * progress
        )
        end = (
            initial_line.get_end()
            + UP * last_vertical_delta * progress
        )
        # Line normalizes its direction during construction, so retain an
        # imperceptible length on the fully transparent terminal frame.
        if remaining <= 1e-6:
            end = start + UP * 1e-6
        line = initial_line.copy()
        line.put_start_and_end_on(start, end)
        line.set_stroke(opacity=initial_line.get_stroke_opacity() * remaining)
        rebuilt.add(line)
    return rebuilt


def interpolate_mapsto_arrow(initial_arrow, final_arrow, alpha):
    """Interpolate equal-topology mapsto arrows without Manim realignment.

    Generic ``Transform`` may insert alignment points when an arrow has already
    passed through earlier transforms.  The coalesce compaction arrows are
    canonical and have identical topology, so direct control-point
    interpolation is both sufficient and the geometrically intended motion.
    """
    progress = alpha
    frame = initial_arrow.copy()
    initial_parts = (
        initial_arrow.tail,
        initial_arrow.shaft,
        *initial_arrow.tip,
    )
    final_parts = (
        final_arrow.tail,
        final_arrow.shaft,
        *final_arrow.tip,
    )
    frame_parts = (frame.tail, frame.shaft, *frame.tip)
    for displayed, initial, final in zip(
        frame_parts, initial_parts, final_parts
    ):
        if initial.points.shape != final.points.shape:
            raise ValueError(
                "mapsto-arrow interpolation requires identical topology"
            )
        displayed.points = (
            (1.0 - progress) * initial.points
            + progress * final.points
        )
    return frame


def coalesce_partitions(morphism):
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


def play_coalesce_collapse(
    scene,
    morphism,
    coalesced,
    *,
    source_entries,
    target_entries,
    arrows,
    column_gap,
    row_gap,
    entry_width,
    entry_height,
    label_font_size,
):
    """Play the coalesce ballet for ``morphism`` on an on-screen diagram.

    ``source_entries``, ``target_entries``, and ``arrows`` are the displayed
    stacks and arrows (arrows in source order, mapped entries only).  Each
    nontrivial coalesce class collapses in place -- correspondence lines
    draw, the block's cells weld at the vertical center of their span, and
    the parallel arrows merge -- with the batches playing sequentially; only
    then do the surviving cells and arrows close their gaps into ordinary
    tuple-morphism geometry, bottom-aligned with the original tuple so S, T,
    and the morphism label stay fixed throughout.

    Returns ``(surviving_source_entries, surviving_target_entries,
    surviving_arrows)``: the mobjects left on screen for the caller to
    dispose of.
    """
    domain_classes, codomain_classes = coalesce_partitions(morphism)

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
    # Standard post-coalesce geometry is bottom-aligned with the original
    # tuple.  This makes the second stage a pure downward compaction.
    final_diagram.shift(
        source_entries[0].get_center()
        - final_diagram.source_entries[0].get_center()
    )
    for arrow in final_diagram.arrows:
        arrow.tail.set_opacity(0)

    # First collapse each nontrivial class at the vertical center of its
    # original span.  Singleton entries and their arrows do not move.
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
            (index for index, target in enumerate(morphism.map) if target),
            arrows,
        )
    )
    final_arrows_by_source = dict(
        zip(
            (index for index, target in enumerate(coalesced.map) if target),
            final_diagram.arrows,
        )
    )
    surviving_arrows = []
    for class_ in domain_classes:
        mapped_indices = [index for index in class_ if morphism.map[index]]
        if mapped_indices:
            surviving_arrows.append(arrows_by_source[mapped_indices[-1]])

    # Assemble one batch per nontrivial source class.  Its matching source
    # entries, target entries, and parallel arrows collapse together; the
    # batches themselves play sequentially from bottom to top.
    collapse_batches = []
    for class_index, class_ in enumerate(domain_classes):
        if len(class_) == 1:
            continue

        animations = []
        source_destination = interim_source_entries[class_index]
        source_block_entries = [source_entries[index] for index in class_]
        source_touching_centers = [
            source_destination.get_center()
            + UP * (position - (len(class_) - 1) / 2) * entry_height
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

        mapped_indices = [index for index in class_ if morphism.map[index]]
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
        scene.play(
            LaggedStart(
                *(GrowFromCenter(line) for line in lines),
                lag_ratio=CORRESPONDENCE_DRAW_LAG_RATIO,
            ),
            run_time=CORRESPONDENCE_DRAW_RUN_TIME,
        )
        scene.wait(0.25)
        scene.play(*animations)

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
        scene.add(source_hull, target_hull, source_seams, target_seams)
        for entry in (*source_block_entries, *target_block_entries):
            entry[0].set_opacity(0)
        scene.remove(lines)
        scene.play(
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
        scene.play(*finish_animations, run_time=0.8)

        source_survivor = source_block_entries[-1]
        target_survivor = target_block_entries[-1]
        scene.remove(
            source_hull,
            target_hull,
            source_destination[1],
            target_destination[1],
            *source_block_entries,
            *target_block_entries,
        )
        source_survivor.become(source_destination)
        target_survivor.become(target_destination)
        scene.add(source_survivor, target_survivor)
        for arrow in redundant_arrows:
            arrow.set_opacity(0)
        scene.wait(0.18)
    scene.wait(0.22)

    # Only after every block has collapsed in place do the surviving cells
    # and arrows close their gaps into ordinary tuple-morphism geometry.
    compaction_animations = [
        *(
            survivor.animate.move_to(
                final_diagram.source_entries[class_index].get_center()
            )
            for class_index, survivor in enumerate(surviving_source_entries)
        ),
        *(
            survivor.animate.move_to(
                final_diagram.target_entries[class_index].get_center()
            )
            for class_index, survivor in enumerate(surviving_target_entries)
        ),
    ]
    for class_index, survivor in zip(
        (index for index, target in enumerate(coalesced.map) if target),
        surviving_arrows,
    ):
        target_class_index = coalesced.map[class_index] - 1
        initial_arrow = TupleMorphismDiagram.mapsto_arrow(
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
    scene.play(*compaction_animations, run_time=1.25)
    scene.wait(0.45)

    scene.wait(2.0)

    return (
        surviving_source_entries,
        surviving_target_entries,
        surviving_arrows,
    )
