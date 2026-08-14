"""Reusable geometry for local tuple-morphism coalesce gestures."""

import numpy as np
from manim import Line, UP, VGroup, VMobject

from .style import INK
from .tuple_morphism import TupleMorphismDiagram


CELL_CORNER_RADIUS = 0.08


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


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def zipper_progress(alpha, path_position, *, transition_width):
    """Return progress behind a soft source-to-target zipper front."""
    front = -transition_width + alpha * (1.0 + 2.0 * transition_width)
    return _smoothstep(
        (front - path_position + transition_width)
        / (2.0 * transition_width)
    )


def _sample_shaft_y(shaft, x_coordinates):
    """Sample a monotone-x Bézier shaft as y(x)."""
    samples = []
    for curve in shaft.points.reshape((-1, 4, 3)):
        start, first_handle, second_handle, end = curve
        for parameter in np.linspace(0.0, 1.0, 33, endpoint=False):
            complement = 1.0 - parameter
            samples.append(
                complement**3 * start
                + 3.0 * complement**2 * parameter * first_handle
                + 3.0 * complement * parameter**2 * second_handle
                + parameter**3 * end
            )
    samples.append(shaft.points[-1])
    samples = np.asarray(samples)
    order = np.argsort(samples[:, 0])
    return np.interp(x_coordinates, samples[order, 0], samples[order, 1])


def zipper_arrow(
    source_anchor,
    target_anchor,
    initial_source_y,
    initial_target_y,
    center_source_y,
    center_target_y,
    alpha,
    *,
    transition_width,
    sample_count,
    endpoint_inset,
    horizontal_run,
    bend_handle,
):
    """Return one frame of a continuous tail-to-tip arrow collapse.

    Parallel arrows differ from their product arrow by a vertical translation,
    including when the arrows are diagonal.  The zipper applies that translation
    behind a smoothly moving junction while retaining exact horizontal runs at
    the source and target cells.
    """
    start = np.asarray((source_anchor[0], initial_source_y, 0.0))
    end = np.asarray((target_anchor[0], initial_target_y, 0.0))
    arrow = TupleMorphismDiagram._mapsto_arrow(
        start,
        end,
        endpoint_inset=endpoint_inset,
        horizontal_run=horizontal_run,
        bend_handle=bend_handle,
    )
    arrow.tail.set_opacity(0)

    source_delta = center_source_y - initial_source_y
    target_delta = center_target_y - initial_target_y
    if not np.isclose(source_delta, target_delta):
        raise ValueError("zipper collapse requires vertically parallel arrows")
    vertical_delta = (source_delta + target_delta) / 2.0

    start_x = arrow.shaft.get_start()[0]
    end_x = arrow.shaft.get_end()[0]
    span = end_x - start_x
    flat_fraction = min(0.45, horizontal_run / abs(span))
    normalized = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, 1.0, sample_count + 1),
                np.asarray((flat_fraction, 1.0 - flat_fraction)),
            )
        )
    )
    interior = np.clip(
        (normalized - flat_fraction) / (1.0 - 2.0 * flat_fraction),
        0.0,
        1.0,
    )
    path_position = _smoothstep(interior)
    path_derivative = np.where(
        (normalized > flat_fraction)
        & (normalized < 1.0 - flat_fraction),
        6.0
        * interior
        * (1.0 - interior)
        / (1.0 - 2.0 * flat_fraction),
        0.0,
    )

    front = -transition_width + alpha * (1.0 + 2.0 * transition_width)
    transition = np.clip(
        (front - path_position + transition_width)
        / (2.0 * transition_width),
        0.0,
        1.0,
    )
    local_progress = _smoothstep(transition)
    transition_derivative = 6.0 * transition * (1.0 - transition)

    x_coordinates = start_x + normalized * span
    base_y = _sample_shaft_y(arrow.shaft, x_coordinates)
    base_slopes = np.gradient(base_y, x_coordinates)
    base_slopes[
        (normalized <= flat_fraction)
        | (normalized >= 1.0 - flat_fraction)
    ] = 0.0
    zipper_slopes = (
        vertical_delta
        * transition_derivative
        * (-1.0 / (2.0 * transition_width))
        * path_derivative
        / span
    )
    y_coordinates = base_y + local_progress * vertical_delta
    slopes = base_slopes + zipper_slopes

    shaft = VMobject(stroke_color=INK, stroke_width=3.6)
    shaft.start_new_path(
        np.asarray((x_coordinates[0], y_coordinates[0], 0.0))
    )
    for index in range(len(normalized) - 1):
        start_point = np.asarray(
            (x_coordinates[index], y_coordinates[index], 0.0)
        )
        end_point = np.asarray(
            (x_coordinates[index + 1], y_coordinates[index + 1], 0.0)
        )
        delta_x = end_point[0] - start_point[0]
        first_handle = start_point + np.asarray(
            (delta_x / 3.0, slopes[index] * delta_x / 3.0, 0.0)
        )
        second_handle = end_point - np.asarray(
            (delta_x / 3.0, slopes[index + 1] * delta_x / 3.0, 0.0)
        )
        shaft.add_cubic_bezier_curve_to(first_handle, second_handle, end_point)

    arrow.submobjects[1] = shaft
    arrow.shaft = shaft
    arrow.tail.shift(
        UP
        * vertical_delta
        * zipper_progress(alpha, 0.0, transition_width=transition_width)
    )
    arrow.tip.shift(
        UP
        * vertical_delta
        * zipper_progress(alpha, 1.0, transition_width=transition_width)
    )
    return arrow


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
