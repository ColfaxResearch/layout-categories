"""Animate curved two-stage composition collapsing into a compact composite."""

import numpy as np
from manim import (
    CubicBezier,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    LaggedStart,
    ORIGIN,
    Scene,
    ShrinkToCenter,
    Text,
    Transform,
    VMobject,
    VGroup,
    Write,
)

from layout_categories_viz import (
    TailToTipMapsto,
    TailToTipUnmapsto,
    TipToTailUnmapsto,
    TupleMorphismCompositionDiagram,
    UncreateMapstoTip,
)
from layout_categories_viz.animations import _arc_length_parameterization
from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK
from layout_categories_viz.tuple_morphism import TupleMorphismDiagram


ARROW_DRAW_DURATION = 1.2
ARROW_HORIZONTAL_RUN = 0.2
ARROW_ENDPOINT_INSET = 0.08
ARROW_BEND_HANDLE = 0.8


def _append_cubic_segments(route: VMobject, points: np.ndarray) -> None:
    """Append every four-point cubic segment in ``points`` to ``route``."""
    for offset in range(0, len(points), 4):
        route.add_cubic_bezier_curve_to(*points[offset + 1 : offset + 4])


def _three_segment_route(first_shaft, connector, second_shaft) -> VMobject:
    """Join three existing shaft sections into one identical VMobject path."""
    route = VMobject(stroke_color=INK, stroke_width=3.6)
    first = first_shaft.points
    bridge = connector.points
    second = second_shaft.points
    route.start_new_path(first[0])
    _append_cubic_segments(route, first)
    _append_cubic_segments(route, bridge)
    _append_cubic_segments(route, second)
    return route


def _horizontal_matched_path_pair(
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


def _split_cubic(
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
    left, _ = _split_cubic(curve, end_parameter)
    if np.isclose(start_parameter, 0):
        return left
    _, subcurve = _split_cubic(left, start_parameter / end_parameter)
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


def _repartition_path(path: VMobject, arc_breaks: np.ndarray) -> VMobject:
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


def _matched_path_pair(
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
        _repartition_path(initial, arc_breaks),
        _repartition_path(final, arc_breaks),
    )


class TupleMorphismCurvedCompositionCollapse(Scene):
    """Collapse a curved two-stage route into a compact straight composite."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND

        examples = (
            {
                "domain": (2, 3, 4, 5, 6, 7),
                "intermediate": (7, 2, 4, 6, 3, 5),
                "codomain": (5, 3, 7, 2, 6),
                "first_mapping": (2, 5, 3, 6, 4, 0),
                "second_mapping": (3, 4, 0, 5, 2, 1),
            },
            {
                "domain": (2, 3, 4, 5, 6, 7),
                "intermediate": (3, 6, 2, 7, 5, 4),
                "codomain": (6, 4, 2, 5, 3),
                "first_mapping": (3, 1, 0, 5, 2, 4),
                "second_mapping": (5, 1, 3, 0, 4, 2),
            },
            {
                "domain": (2, 3, 4, 5, 6, 7),
                "intermediate": (5, 2, 7, 4, 6, 3),
                "codomain": (7, 5, 3, 6, 2),
                "first_mapping": (2, 6, 4, 0, 5, 3),
                "second_mapping": (2, 5, 1, 0, 4, 3),
            },
        )
        for example in examples:
            self._play_example(**example)

    def _play_example(
        self,
        *,
        domain,
        intermediate,
        codomain,
        first_mapping,
        second_mapping,
        column_gap=2.65,
        row_gap=0.22,
        entry_width=0.62,
        entry_height=0.62,
        label_font_size=28,
        retain_composite=False,
    ):
        diagram = TupleMorphismCompositionDiagram(
            domain=domain,
            intermediate=intermediate,
            codomain=codomain,
            first_mapping=first_mapping,
            second_mapping=second_mapping,
            column_gap=column_gap,
            row_gap=row_gap,
            entry_width=entry_width,
            entry_height=entry_height,
            label_font_size=label_font_size,
            arrow_endpoint_inset=ARROW_ENDPOINT_INSET,
            arrow_horizontal_run=ARROW_HORIZONTAL_RUN,
            arrow_bend_handle=ARROW_BEND_HANDLE,
        )
        diagram.first_label.next_to(
            VGroup(diagram.source_entries, diagram.intermediate_entries),
            DOWN,
            buff=0.36,
        )
        diagram.second_label.next_to(
            VGroup(diagram.intermediate_entries, diagram.target_entries),
            DOWN,
            buff=0.36,
        )
        composition_symbol = Text("∘", color=INK, font=CODE_FONT, font_size=30)
        # A normally typeset label supplies the exact glyph destinations. Its
        # glyph centers preserve the font's baseline and kerning, unlike
        # arranging standalone Text objects by their bounds.
        composition_reference = Text("g ∘ f", color=INK, font=CODE_FONT, font_size=30)
        composition_reference.next_to(
            VGroup(diagram.source_entries, diagram.target_entries), DOWN, buff=0.36
        )

        # Center the visible composition layout from its actual tuple geometry.
        # This keeps the scene balanced when f and g have differently sized
        # source, intermediate, or target tuples.
        visible_layout = VGroup(
            diagram.source_entries,
            diagram.intermediate_entries,
            diagram.target_entries,
            diagram.first_label,
            diagram.second_label,
            composition_reference,
        )
        layout_shift = ORIGIN - visible_layout.get_center()
        diagram.shift(layout_shift)
        composition_reference.shift(layout_shift)
        g_glyph, _, composition_glyph, _, f_glyph = tuple(composition_reference)
        composition_symbol.move_to(composition_glyph)

        first_by_source = dict(
            zip(
                (index for index, target in enumerate(diagram.first_mapping) if target),
                diagram.first_arrows,
            )
        )
        second_by_source = dict(
            zip(
                (index for index, target in enumerate(diagram.second_mapping) if target),
                diagram.second_arrows,
            )
        )
        routes = [
            (
                first_by_source[source_index],
                second_by_source[intermediate_index - 1],
                source_index,
                diagram.second_mapping[intermediate_index - 1] - 1,
            )
            for source_index, intermediate_index in enumerate(diagram.first_mapping)
            if intermediate_index
            and diagram.second_mapping[intermediate_index - 1]
        ]
        projected_first_arrows = [
            first_by_source[source_index]
            for source_index, intermediate_index in enumerate(diagram.first_mapping)
            if intermediate_index
            and not diagram.second_mapping[intermediate_index - 1]
        ]
        hit_intermediate_indices = {
            intermediate_index - 1
            for intermediate_index in diagram.first_mapping
            if intermediate_index
        }
        unhit_second_arrows = [
            arrow
            for intermediate_index, arrow in second_by_source.items()
            if intermediate_index not in hit_intermediate_indices
        ]
        stage_arrows = [*diagram.first_arrows, *diagram.second_arrows]
        for arrow in stage_arrows:
            arrow.tail.set_opacity(0)

        self.play(
            FadeIn(diagram.source_entries),
            FadeIn(diagram.intermediate_entries),
            FadeIn(diagram.target_entries),
            FadeIn(diagram.first_label),
            FadeIn(diagram.second_label),
        )
        self.add(diagram.first_arrows)
        self.play(
            LaggedStart(
                *(
                    TailToTipMapsto(arrow, run_time=ARROW_DRAW_DURATION)
                    for arrow in diagram.first_arrows
                ),
                lag_ratio=0.15,
            )
        )
        self.add(diagram.second_arrows)
        self.play(
            LaggedStart(
                *(
                    TailToTipMapsto(arrow, run_time=ARROW_DRAW_DURATION)
                    for arrow in diagram.second_arrows
                ),
                lag_ratio=0.15,
            )
        )
        self.wait(0.35)

        connectors = []
        for first_arrow, second_arrow, _, _ in routes:
            first_end = first_arrow.shaft.get_end()
            second_start = second_arrow.shaft.get_start()
            connector_span = second_start - first_end
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
            *(ShrinkToCenter(entry) for entry in diagram.intermediate_entries),
            *(UncreateMapstoTip(first_arrow) for first_arrow, _, _, _ in routes),
            *(TipToTailUnmapsto(arrow) for arrow in projected_first_arrows),
            *(TailToTipUnmapsto(arrow) for arrow in unhit_second_arrows),
            run_time=0.9,
        )
        self.play(
            *(FadeOut(second_arrow.tail) for _, second_arrow, _, _ in routes),
            *(Create(connector) for connector in connectors),
            diagram.first_label.animate.move_to(f_glyph),
            diagram.second_label.animate.move_to(g_glyph),
            Write(composition_symbol),
            run_time=1.35,
        )
        self.wait(0.25)

        # Every source mode deforms into exactly its composite target mode.
        source_center = diagram.source_entries.get_center()
        intermediate_center = diagram.intermediate_entries.get_center()
        target_center = diagram.target_entries.get_center()
        stage_span = (
            intermediate_center[0] - source_center[0]
            + target_center[0]
            - intermediate_center[0]
        ) / 2
        composite_center_x = (source_center[0] + target_center[0]) / 2
        source_shift = np.array(
            [composite_center_x - stage_span / 2 - source_center[0], 0.0, 0.0]
        )
        target_shift = np.array(
            [composite_center_x + stage_span / 2 - target_center[0], 0.0, 0.0]
        )

        curved_arrows = VGroup()
        contracted_arrows = VGroup()
        for (first_arrow, second_arrow, source_index, target_index), connector in zip(
            routes, connectors
        ):
            canonical = TupleMorphismDiagram._mapsto_arrow(
                diagram.source_entries[source_index].get_right(),
                diagram.target_entries[target_index].get_left(),
                endpoint_inset=ARROW_ENDPOINT_INSET,
                horizontal_run=ARROW_HORIZONTAL_RUN,
                bend_handle=ARROW_BEND_HANDLE,
            )
            canonical.tail.set_opacity(0)
            route_shaft = _three_segment_route(
                first_arrow.shaft, connector, second_arrow.shaft
            )
            contracted = TupleMorphismDiagram._mapsto_arrow(
                diagram.source_entries[source_index].get_right() + source_shift,
                diagram.target_entries[target_index].get_left() + target_shift,
                endpoint_inset=ARROW_ENDPOINT_INSET,
                horizontal_run=ARROW_HORIZONTAL_RUN,
                bend_handle=ARROW_BEND_HANDLE,
            )
            contracted.tail.set_opacity(0)
            initial_shaft, final_shaft = _matched_path_pair(
                route_shaft, contracted.shaft
            )
            curved_arrows.add(
                canonical.from_parts(
                    first_arrow.tail.copy(),
                    initial_shaft,
                    second_arrow.tip.copy(),
                )
            )
            contracted_arrows.add(
                contracted.from_parts(
                    contracted.tail,
                    final_shaft,
                    contracted.tip,
                )
            )

        # The replacement routes are disconnected copies, so the complete
        # two-stage diagram can vanish without leaving diagonal remnants.
        self.remove(
            *stage_arrows,
            *(arrow.tail for arrow in stage_arrows),
            *(arrow.shaft for arrow in stage_arrows),
            *(arrow.tip for arrow in stage_arrows),
            *connectors,
        )
        self.add(curved_arrows)
        self.wait(1 / 15)

        # All compatible route parts deform into g ∘ f while the outer tuple
        # columns contract to the width of either original morphism.
        self.play(
            Transform(curved_arrows, contracted_arrows),
            diagram.source_entries.animate.shift(source_shift),
            diagram.target_entries.animate.shift(target_shift),
            run_time=1.35,
        )

        # The final repartitioned shafts have exactly the canonical geometry.
        self.remove(curved_arrows)
        self.add(contracted_arrows)
        self.wait(1.0)
        if retain_composite:
            return {
                "source_entries": diagram.source_entries,
                "target_entries": diagram.target_entries,
                "arrows": contracted_arrows,
                "first_label": diagram.first_label,
                "second_label": diagram.second_label,
                "composition_symbol": composition_symbol,
                "entry_width": entry_width,
                "entry_height": entry_height,
                "row_gap": row_gap,
                "label_font_size": label_font_size,
            }
        self.play(
            FadeOut(diagram.source_entries),
            FadeOut(diagram.target_entries),
            FadeOut(diagram.first_label),
            FadeOut(diagram.second_label),
            FadeOut(composition_symbol),
            FadeOut(contracted_arrows),
            run_time=0.55,
        )
        return None
