"""Shared bezier path helpers: route joining, matching, and repartitioning.

Composition scenes replace multi-stage routes by single composite arrows.
The helpers here glue existing shaft sections into one identical path, give
two paths exact point correspondence for a lump-free Transform, and split
cubics precisely, so every scene collapses routes the same way.
"""

import numpy as np
from manim import CubicBezier, VMobject

from .animations import _arc_length_parameterization
from .style import INK
from .stacks import STROKE_WIDTH

ARROW_HORIZONTAL_RUN = 0.2
ARROW_ENDPOINT_INSET = 0.08
ARROW_BEND_HANDLE = 0.8


def append_cubic_segments(route: VMobject, points: np.ndarray) -> None:
    """Append every four-point cubic segment in ``points`` to ``route``."""
    for offset in range(0, len(points), 4):
        route.add_cubic_bezier_curve_to(*points[offset + 1 : offset + 4])


def three_segment_route(first_shaft, connector, second_shaft) -> VMobject:
    """Join three existing shaft sections into one identical VMobject path."""
    route = VMobject(stroke_color=INK, stroke_width=3.6)
    first = first_shaft.points
    bridge_points = connector.points
    second = second_shaft.points
    route.start_new_path(first[0])
    append_cubic_segments(route, first)
    append_cubic_segments(route, bridge_points)
    append_cubic_segments(route, second)
    return route


def join_route(*paths) -> VMobject:
    """Join consecutive shaft sections into one identical path."""
    route = VMobject(stroke_color=INK, stroke_width=STROKE_WIDTH)
    route.start_new_path(paths[0].points[0])
    for path in paths:
        append_cubic_segments(route, path.points)
    return route


def bridge(start, end) -> CubicBezier:
    """A straight cubic crossing a ``W`` cell between two shaft ends."""
    start = np.array(start, dtype=float)
    span = np.array(end, dtype=float) - start
    return CubicBezier(
        start,
        start + span / 3,
        start + 2 * span / 3,
        start + span,
        color=INK,
        stroke_width=STROKE_WIDTH,
    )


def horizontal_matched_path_pair(
    initial: VMobject, final: VMobject, *, uniform_intervals: int = 64
) -> tuple[VMobject, VMobject]:
    """Match monotone shafts by horizontal position, not traveled distance.

    Arc-length matching lets a tall bend in one path correspond to a much
    later horizontal location in the other.  Interpolating those control
    points makes bends migrate and can introduce transient lumps.  Tuple-map
    shafts are monotone in x, so a shared normalized-x partition supplies the
    geometrically meaningful correspondence.
    """

    def normalized_x_boundaries(path: VMobject) -> np.ndarray:
        curves = path.points.reshape((-1, 4, 3))
        x_coordinates = np.concatenate(
            ((curves[0, 0, 0],), curves[:, -1, 0])
        )
        return (x_coordinates - x_coordinates[0]) / (
            x_coordinates[-1] - x_coordinates[0]
        )

    def sample_y_and_slope(path: VMobject, x_coordinates: np.ndarray):
        sampled_points = []
        sampled_slopes = []
        curves = path.points.reshape((-1, 4, 3))
        for start, first_handle, second_handle, end in curves:
            for parameter in np.linspace(0.0, 1.0, 65, endpoint=False):
                complement = 1.0 - parameter
                point = (
                    complement**3 * start
                    + 3 * complement**2 * parameter * first_handle
                    + 3 * complement * parameter**2 * second_handle
                    + parameter**3 * end
                )
                derivative = (
                    3 * complement**2 * (first_handle - start)
                    + 6
                    * complement
                    * parameter
                    * (second_handle - first_handle)
                    + 3 * parameter**2 * (end - second_handle)
                )
                sampled_points.append(point)
                sampled_slopes.append(
                    derivative[1] / derivative[0]
                    if abs(derivative[0]) > 1e-9
                    else 0.0
                )
        sampled_points.append(curves[-1, -1])
        final_derivative = 3 * (curves[-1, -1] - curves[-1, -2])
        sampled_slopes.append(
            final_derivative[1] / final_derivative[0]
            if abs(final_derivative[0]) > 1e-9
            else 0.0
        )

        sampled_points = np.asarray(sampled_points)
        sampled_slopes = np.asarray(sampled_slopes)
        order = np.argsort(sampled_points[:, 0], kind="stable")
        sampled_x = sampled_points[order, 0]
        unique_x, unique_indices = np.unique(sampled_x, return_index=True)
        ordered_points = sampled_points[order][unique_indices]
        ordered_slopes = sampled_slopes[order][unique_indices]
        return (
            np.interp(x_coordinates, unique_x, ordered_points[:, 1]),
            np.interp(x_coordinates, unique_x, ordered_slopes),
        )

    def hermite_reconstruction(
        path: VMobject, normalized_x: np.ndarray
    ) -> VMobject:
        start_x = path.get_start()[0]
        end_x = path.get_end()[0]
        x_coordinates = start_x + normalized_x * (end_x - start_x)
        y_coordinates, slopes = sample_y_and_slope(path, x_coordinates)
        result = VMobject(stroke_color=INK, stroke_width=3.6)
        result.start_new_path(
            np.asarray((x_coordinates[0], y_coordinates[0], 0.0))
        )
        for index in range(len(normalized_x) - 1):
            start = np.asarray(
                (x_coordinates[index], y_coordinates[index], 0.0)
            )
            end = np.asarray(
                (x_coordinates[index + 1], y_coordinates[index + 1], 0.0)
            )
            delta_x = end[0] - start[0]
            first_handle = start + np.asarray(
                (delta_x / 3, slopes[index] * delta_x / 3, 0.0)
            )
            second_handle = end - np.asarray(
                (delta_x / 3, slopes[index + 1] * delta_x / 3, 0.0)
            )
            result.add_cubic_bezier_curve_to(
                first_handle, second_handle, end
            )
        return result

    normalized_x = np.unique(
        np.round(
            np.concatenate(
                (
                    np.linspace(0.0, 1.0, uniform_intervals + 1),
                    normalized_x_boundaries(initial),
                    normalized_x_boundaries(final),
                )
            ),
            decimals=12,
        )
    )
    return (
        hermite_reconstruction(initial, normalized_x),
        hermite_reconstruction(final, normalized_x),
    )


def split_cubic(
    curve: np.ndarray, parameter: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split a four-control-point cubic exactly with de Casteljau's algorithm."""
    start, first_handle, second_handle, end = curve
    first = (1 - parameter) * start + parameter * first_handle
    second = (1 - parameter) * first_handle + parameter * second_handle
    third = (1 - parameter) * second_handle + parameter * end
    fourth = (1 - parameter) * first + parameter * second
    fifth = (1 - parameter) * second + parameter * third
    midpoint = (1 - parameter) * fourth + parameter * fifth
    return (
        np.asarray((start, first, fourth, midpoint)),
        np.asarray((midpoint, fifth, third, end)),
    )


def _cubic_subcurve(
    curve: np.ndarray, start_parameter: float, end_parameter: float
) -> np.ndarray:
    """Return the exact cubic restriction to the requested interval."""
    if np.isclose(start_parameter, 0) and np.isclose(end_parameter, 1):
        return curve.copy()
    left, _ = split_cubic(curve, end_parameter)
    if np.isclose(start_parameter, 0):
        return left
    _, subcurve = split_cubic(left, start_parameter / end_parameter)
    return subcurve


def _curve_location(
    global_parameter: float, curve_count: int, *, interval_end: bool
) -> tuple[int, float]:
    """Locate a global path parameter on one side of a curve boundary."""
    scaled = float(np.clip(global_parameter, 0.0, 1.0)) * curve_count
    if interval_end:
        curve_index = min(
            max(int(np.ceil(scaled - 1e-9)) - 1, 0),
            curve_count - 1,
        )
    else:
        curve_index = min(
            int(np.floor(scaled + 1e-9)), curve_count - 1
        )
    return curve_index, float(
        np.clip(scaled - curve_index, 0.0, 1.0)
    )


def _arc_boundaries(path: VMobject) -> np.ndarray:
    """Return normalized arc lengths of existing cubic boundaries."""
    arc_lengths, parameters = _arc_length_parameterization(path)
    curve_count = len(path.points) // 4
    boundaries = np.linspace(0.0, 1.0, curve_count + 1)
    return np.interp(boundaries, parameters, arc_lengths)


def repartition_path(path: VMobject, arc_breaks: np.ndarray) -> VMobject:
    """Exactly repartition a path at normalized arc lengths."""
    arc_lengths, parameters = _arc_length_parameterization(path)
    global_parameters = np.interp(arc_breaks, arc_lengths, parameters)
    curves = path.points.reshape((-1, 4, 3))
    subcurves = []
    for start, end in zip(global_parameters[:-1], global_parameters[1:]):
        start_index, local_start = _curve_location(
            start, len(curves), interval_end=False
        )
        end_index, local_end = _curve_location(
            end, len(curves), interval_end=True
        )
        if start_index != end_index:
            raise ValueError(
                "arc-length partition crossed an original curve boundary"
            )
        subcurves.append(
            _cubic_subcurve(curves[start_index], local_start, local_end)
        )

    result = VMobject(stroke_color=INK, stroke_width=3.6)
    result.start_new_path(subcurves[0][0])
    for subcurve in subcurves:
        result.add_cubic_bezier_curve_to(*subcurve[1:])
    return result


def matched_path_pair(
    initial: VMobject, final: VMobject, *, uniform_intervals: int = 24
) -> tuple[VMobject, VMobject]:
    """Give two paths exact, arc-length-matched point correspondence."""
    arc_breaks = np.unique(
        np.round(
            np.concatenate(
                (
                    np.linspace(0.0, 1.0, uniform_intervals + 1),
                    _arc_boundaries(initial),
                    _arc_boundaries(final),
                )
            ),
            decimals=12,
        )
    )
    return (
        repartition_path(initial, arc_breaks),
        repartition_path(final, arc_breaks),
    )
