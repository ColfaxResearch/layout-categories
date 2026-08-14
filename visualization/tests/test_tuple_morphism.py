import numpy as np
import pytest

from manim import CubicBezier
from tract import TupleMorphism, compute_flat_layout_components

from layout_categories_viz import MapstoArrow, TupleMorphismDiagram, compose_tuple_maps
from layout_categories_viz.animations import _arc_length_parameterization
from layout_categories_viz.coalesce import interpolate_mapsto_arrow
from layout_categories_viz.paths import (
    matched_path_pair,
    three_segment_route,
)


def test_diagram_draws_one_arrow_per_non_basepoint_map_value():
    diagram = TupleMorphismDiagram(
        domain=(2, 3, 4, 2),
        codomain=(4, 2, 3, 2, 5),
        mapping=(2, 3, 1, 0),
    )

    assert len(diagram.arrows) == 3
    assert all(len(arrow) == 3 for arrow in diagram.arrows)  # tail + shaft + tip
    assert all(isinstance(arrow.shaft, CubicBezier) for arrow in diagram.arrows)
    assert diagram.source_entry(4) is diagram.source_entries[3]
    assert diagram.target_entry(2) is diagram.target_entries[1]
    assert diagram.source_entries[0].get_y() < diagram.source_entries[-1].get_y()
    assert diagram.target_entries[0].get_y() < diagram.target_entries[-1].get_y()


def test_composing_tuple_maps_propagates_basepoints():
    assert compose_tuple_maps((2, 1, 0), (2, 0, 1)) == (0, 2, 0)


def test_mapsto_arrow_can_be_regrouped_without_changing_its_parts():
    arrow = MapstoArrow((0, 0, 0), (2, 0, 0))
    regrouped = MapstoArrow.from_parts(arrow.tail, arrow.shaft, arrow.tip)

    assert len(regrouped) == 3
    assert regrouped.shaft is arrow.shaft


def test_mapsto_arrow_has_tunable_horizontal_endpoint_runs():
    arrow = MapstoArrow(
        (0, 0, 0),
        (4, 2, 0),
        endpoint_inset=0.1,
        horizontal_run=0.6,
    )
    first_run, middle_curve, final_run = np.split(arrow.shaft.points, 3)

    assert np.allclose(first_run[-1] - first_run[0], (0.6, 0, 0))
    assert np.allclose(final_run[-1] - final_run[0], (0.6, 0, 0))
    assert np.allclose(first_run[:, 1], first_run[0, 1])
    assert np.allclose(final_run[:, 1], final_run[-1, 1])
    assert np.allclose(first_run[-1], middle_curve[0])
    assert np.allclose(middle_curve[-1], final_run[0])


def test_mapsto_arrow_is_exactly_horizontal_beside_the_tip():
    arrow = MapstoArrow((0, 0, 0), (4, 2, 0), horizontal_run=0.8)
    final_run = arrow.shaft.points[-4:]

    assert np.allclose(final_run[:, 1], arrow.shaft.get_end()[1])
    assert np.allclose(final_run[:, 2], 0)


def test_horizontal_mapsto_arrow_remains_straight():
    arrow = MapstoArrow((0, 1, 0), (4, 1, 0))

    assert np.allclose(arrow.shaft.points[:, 1], 1)


def test_arrow_reveal_parameterization_tracks_physical_length():
    arrow = MapstoArrow((0, 0, 0), (4, 2, 0), horizontal_run=0.2)
    arc_lengths, parameters = _arc_length_parameterization(arrow.shaft)

    assert np.all(np.diff(arc_lengths) >= 0)
    assert np.all(np.diff(parameters) > 0)
    assert np.isclose(arc_lengths[0], 0)
    assert np.isclose(arc_lengths[-1], 1)
    assert np.isclose(parameters[0], 0)
    assert np.isclose(parameters[-1], 1)


def test_deformation_paths_have_arc_length_matched_point_correspondence():
    first = MapstoArrow((0, 0, 0), (3, 2, 0), horizontal_run=0.2)
    second = MapstoArrow((4, 2, 0), (7, -1, 0), horizontal_run=0.2)
    connector = CubicBezier(
        first.shaft.get_end(),
        (10 / 3, 2, 0),
        (11 / 3, 2, 0),
        second.shaft.get_start(),
    )
    initial = three_segment_route(first.shaft, connector, second.shaft)
    final = MapstoArrow((0, 0, 0), (7, -1, 0), horizontal_run=0.2).shaft

    matched_initial, matched_final = matched_path_pair(initial, final)

    assert matched_initial.points.shape == matched_final.points.shape
    initial_lengths = np.asarray(
        [
            CubicBezier(*curve).get_arc_length()
            for curve in matched_initial.points.reshape(-1, 4, 3)
        ]
    )
    final_lengths = np.asarray(
        [
            CubicBezier(*curve).get_arc_length()
            for curve in matched_final.points.reshape(-1, 4, 3)
        ]
    )
    assert np.allclose(
        initial_lengths / initial_lengths.sum(),
        final_lengths / final_lengths.sum(),
        atol=2e-3,
    )


def test_coalesce_compaction_interpolates_arrow_control_points_directly():
    initial = MapstoArrow((0, 0, 0), (4, 2, 0), horizontal_run=0.2)
    final = MapstoArrow((0, 1, 0), (4, 3.5, 0), horizontal_run=0.2)

    midpoint = interpolate_mapsto_arrow(initial, final, 0.5)

    for displayed, start, end in zip(
        (midpoint.tail, midpoint.shaft, *midpoint.tip),
        (initial.tail, initial.shaft, *initial.tip),
        (final.tail, final.shaft, *final.tip),
    ):
        assert displayed.points.shape == start.points.shape == end.points.shape
        assert np.allclose(displayed.points, (start.points + end.points) / 2)


def test_flat_layout_components_match_the_displayed_prefix_products():
    morphism = TupleMorphism(
        domain=(4, 2, 6, 5, 3, 7),
        codomain=(2, 3, 4, 5, 6),
        map=(3, 1, 5, 4, 2, 0),
    )

    assert compute_flat_layout_components(morphism) == (
        (4, 2, 6, 5, 3, 7),
        (6, 1, 120, 24, 2, 0),
    )


@pytest.mark.parametrize(
    ("domain", "codomain", "mapping"),
    [
        ((2,), (2,), ()),  # map must cover every source mode
        ((2, 2), (2,), (1, 1)),  # non-basepoint map values are injective
        ((2,), (3,), (1,)),  # mapped values must agree
        ((2,), (2,), (2,)),  # one-based target index is out of bounds
    ],
)
def test_diagram_rejects_invalid_tuple_morphisms(domain, codomain, mapping):
    with pytest.raises(ValueError):
        TupleMorphismDiagram(domain, codomain, mapping)
