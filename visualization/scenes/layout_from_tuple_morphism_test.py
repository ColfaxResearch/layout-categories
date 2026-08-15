"""Extract the layout of a tuple morphism by collapsing its diagram.

The layout L(f) = S : D of a tuple morphism f : S -> T is read off in three
beats.  First the morphism itself: source and target stacks in the shape
fill, mapsto arrows carrying the underlying map.  Second the exclusive
prefix products of T, written as a stride-filled cell behind a colon beside
each target entry — the column-major layout of T, whose entry at position j
is the stride a mode sent to j picks up.  Third the collapse: the target
entries and their colons fall away; then, staggered but near simultaneous
like the arrow draw-ins, every arrow extends outward, its caret opening
into a grabber that seats flush on its stride entry's left edge; once all
grabs are complete, the pulls play out with the same stagger, each shaft
undrawing from its extended tip all the way back to its fixed start on
the source cell while its horizontal run out of the tail grows to end
exactly on the stride slot, where the stride is dropped; a basepoint
mode has no arrow to bring anything home,
so its stride appears as 0 only after every pulled stride is seated,
together with the colons.  What remains is exactly the
layout depiction of the previous scene — shape column, colons, stride
column — which slides to center.

The same extraction reads the layout of a span U ←b— X —f→ T of either
type — Span(Tuple, Fact) or Span(Tuple, Ref): the backward leg b tags
along inverted on the left of the apex stack — root product cells and
mirrored ref trees (pure fans for a Fact leg), exactly the layout
depiction's picture — untouched by the collapse, so what remains is the
nested layout whose shape is b's nested tuple and whose strides are
L(f)'s.

Colors follow the layout depiction: shape (and target-tuple) cells in
SHAPE_FILL, every stride-carrying cell in STRIDE_FILL.  The library
validates every example: the strides assembled on screen are read off
flat_layout_components, never restated.
"""

from layout_categories_viz.scene_base import LayoutScene

from dataclasses import dataclass

import numpy as np
from manim import (
    Animation,
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    RIGHT,
    VGroup,
)
from tract import RefMorphism, RefSpanMorphism, SpanMorphism, TupleMorphism
from tract.backends.base import flat_layout_components

from layout_categories_viz import TailToTipMapsto
from layout_categories_viz.animations import _arc_length_parameterization
from layout_categories_viz.layouts import fitted_cell, layout_colon
from layout_categories_viz.ref_trees import ref_tree_segments
from layout_categories_viz.stacks import (
    CELL_H,
    MAX_STACK_HEIGHT,
    SLOT_STEP,
    make_place,
    segment_arrow,
)
from layout_categories_viz.style import SHAPE_FILL, STRIDE_FILL
from scenes.tuple_morphism_to_flat_layout import prefix_products

SOURCE_X = -3.0
TARGET_X = 1.8
# A Ref span's backward leg needs room on the left, so its diagram uses its
# own columns: the apex where the source was, the root column further left.
SPAN_ROOT_X = -5.9
SPAN_SOURCE_X = -3.3
SPAN_TARGET_X = 1.6
# The stride column lands beside the source at the layout depiction's
# shape-to-stride gap, and the prefix products sit beside the target at the
# same gap: the target column reads as the layout T : prefix products.
PAIR_GAP = 1.6


@dataclass(frozen=True)
class MorphismExample:
    """A tuple morphism, in the repository's one-based map convention —
    or, when ``nest`` is given, the Ref span it is the forward leg of,
    with backward leg the Ref morphism of ``nest`` (whose flattening must
    be the domain: the span's apex)."""

    domain: tuple
    codomain: tuple
    mapping: tuple
    nest: tuple = None
    fact: bool = False  # validate as Span(Tuple, Fact) instead of Ref

    def morphism(self) -> TupleMorphism:
        return TupleMorphism(self.domain, self.codomain, self.mapping)

    def backward_leg(self) -> RefMorphism:
        """The span's backward leg, validated against the forward leg.

        A Fact-legged span validates as a SpanMorphism; either way the leg
        is returned as its Ref morphism, which is what the drawing reads —
        a Fact morphism is exactly a flat-modes Ref morphism, so the two
        span types share one picture.
        """
        leg = RefMorphism(self.nest)
        if self.fact:
            SpanMorphism(leg.to_fact_morphism(), self.morphism())
        else:
            RefSpanMorphism(leg, self.morphism())
        return leg


EXAMPLES = (
    # Every mode mapped: each stride travels an arrow home.
    MorphismExample(domain=(4, 2, 5), codomain=(5, 4, 2), mapping=(2, 3, 1)),
    # One basepoint mode: its stride appears as 0 with no arrow to follow.
    MorphismExample(
        domain=(4, 2, 5, 3), codomain=(2, 3, 4), mapping=(3, 1, 0, 2)
    ),
    # Larger and generic: five modes onto four, one projected away.
    MorphismExample(
        domain=(2, 6, 4, 3, 5),
        codomain=(4, 3, 2, 5),
        mapping=(3, 0, 1, 2, 4),
    ),
    # Six modes onto five, thoroughly shuffled, with a repeated entry.
    MorphismExample(
        domain=(7, 2, 3, 4, 2, 6),
        codomain=(2, 4, 7, 6, 3),
        mapping=(3, 1, 5, 2, 0, 4),
    ),
    # A Fact span — Span(Tuple, Fact): the backward leg's modes are flat,
    # so its trees are pure fans; the layout is ((4,2),(5,3)):((10,1),(2,0)).
    MorphismExample(
        domain=(4, 2, 5, 3),
        codomain=(2, 5, 4),
        mapping=(3, 1, 2, 0),
        nest=((4, 2), (5, 3)),
        fact=True,
    ),
    # A Ref span: the same extraction, with the backward Ref morphism of
    # ((2,3),4) standing inverted on the left, so the result is the nested
    # layout ((2,3),4) : ((12,1),3).
    MorphismExample(
        domain=(2, 3, 4),
        codomain=(3, 4, 2),
        mapping=(3, 1, 2),
        nest=((2, 3), 4),
    ),
    # A deeper Ref span with a projected-away apex entry: the layout
    # (2,(2,(3,2)),5) : (3,(30,(1,0)),6).
    MorphismExample(
        domain=(2, 2, 3, 2, 5),
        codomain=(3, 2, 5, 2),
        mapping=(2, 4, 1, 0, 3),
        nest=(2, (2, (3, 2)), 5),
    ),
)


def edge_jaws(cell) -> list:
    """The claw's two jaw polylines, relative to its anchor.

    The claw is exactly the left edge of the cell it grabs — a run up or
    down the edge and around the cell's own rounded corner, ending where
    the corner meets the top or bottom edge — traced flush, so the grip
    connects to the cell with no standoff and no overhanging lip.
    """
    width, height = cell.width, cell.height
    # The cell is a CELL_H x CELL_H RoundedRectangle of radius 0.08.
    radius = width * 0.08 / 0.7
    rise = height / 2 - radius
    run = np.linspace((0.0, 0.0, 0.0), (0.0, rise, 0.0), 5)
    angles = np.linspace(0.0, np.pi / 2, 12)[1:]
    corner = np.stack(
        [
            radius - radius * np.cos(angles),
            rise + radius * np.sin(angles),
            np.zeros_like(angles),
        ],
        axis=1,
    )
    jaw = np.concatenate([run, corner])
    return [jaw * np.array([1.0, side, 1.0]) for side in (1.0, -1.0)]


def ease(alpha: float, phase: tuple) -> float:
    return TailToTipMapsto._ease(alpha, *phase)


class GrabberReach(Animation):
    """Extend an arrow out to the stride entry and take hold of it.

    The shaft extends from its tip toward the cell where it stands, then
    the caret opens into the cell-edge claw on arrival, so the reach ends
    with the claw seated flush on the cell's left edge.  Nothing moves but
    the arrow: the grab must be complete before any pull begins.  The
    phase windows are TailToTipMapsto's shaft and caret windows, so a
    reach paces exactly like the draw-in it continues.
    """

    EXTEND = (0.07, 0.83)
    OPEN = (0.83, 1.00)

    def __init__(self, mapsto_arrow, carried, **kwargs) -> None:
        self.arrow = mapsto_arrow
        self.carried = carried
        super().__init__(mapsto_arrow, **kwargs)

    def begin(self) -> None:
        shaft = self.arrow.shaft
        grip = self.carried.width / 2
        start_anchor = self.carried.get_center() - RIGHT * grip
        self._template = shaft.copy()
        self._template.add_line_to(start_anchor)
        self._arc_lengths, self._parameters = _arc_length_parameterization(
            self._template
        )
        self._tip_fraction = (
            shaft.get_arc_length() / self._template.get_arc_length()
        )
        self._claw_jaws = edge_jaws(self.carried)
        self._caret_jaws = [
            np.linspace(
                np.zeros(3), stroke.get_end() - stroke.get_start(), len(jaw)
            )
            for stroke, jaw in zip(self.arrow.tip, self._claw_jaws)
        ]
        super().begin()

    def interpolate_mobject(self, alpha: float) -> None:
        extend = ease(alpha, self.EXTEND)
        opening = ease(alpha, self.OPEN)
        visible = self._tip_fraction + (1.0 - self._tip_fraction) * extend
        parameter = float(
            np.interp(visible, self._arc_lengths, self._parameters)
        )
        self.arrow.shaft.pointwise_become_partial(
            self._template, 0, parameter
        )
        anchor = self.arrow.shaft.get_end()
        for stroke, caret_jaw, claw_jaw in zip(
            self.arrow.tip, self._caret_jaws, self._claw_jaws
        ):
            jaw = caret_jaw * (1.0 - opening) + claw_jaw * opening
            stroke.set_points_as_corners(anchor + jaw)


def resample(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a polyline to ``count`` points uniform in arc length."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.stack(
        [np.interp(targets, cumulative, points[:, k]) for k in range(3)],
        axis=1,
    )


class GrabberPull(Animation):
    """Undraw the extended arrow to its start, dropping the stride en route.

    The shaft undraws from its extended tip all the way back to its fixed
    starting point on the source cell — the tail never moves.  To seat the
    stride exactly, the shaft's horizontal run at the tail grows as the
    undraw proceeds: each frame the curve morphs toward a run-and-bend
    shape — a straight run out of the tail along the source row, then a
    bend up to the retracting end — with the run growing to end exactly at
    the claw's final anchor, one grip left of the stride slot.  Undraw and
    growth finish together, so by the time the stride nears its slot the
    target sits at the end of the horizontal run: the claw and cell come
    off the bend straight onto the anchor.  There the claw releases and
    stays put, shrinking into the seated cell's edge, while the remaining
    horizontal run undraws into the tail along the row.
    """

    SAMPLES = 96
    RIDE = (0.00, 0.60)  # tip to the run's end, run growing to full
    RELEASE = (0.60, 0.74)
    CLEAR = (0.64, 1.00)  # the run itself, back into the tail

    def __init__(self, mapsto_arrow, carried, destination, **kwargs) -> None:
        self.arrow = mapsto_arrow
        self.carried = carried
        self.destination = np.array(destination, dtype=float)
        # The animation's mobject spans the arrow AND the cell it fetches:
        # the renderer decides its static/moving partition from animation
        # mobjects, and a cell moved as a side effect would be baked into
        # the static background and stand still until the play ends.
        super().__init__(VGroup(mapsto_arrow, carried), **kwargs)

    def begin(self) -> None:
        # The shaft was left fully extended by GrabberReach, its end flush
        # on the cell's edge: the pull retracts along the whole of it.
        template = self.arrow.shaft.copy()
        self._origin = np.array(
            [
                template.point_from_proportion(t)
                for t in np.linspace(0.0, 1.0, self.SAMPLES)
            ]
        )
        self._tail = self._origin[0]
        self._tip = self._origin[-1]
        self._grip = self.carried.width / 2
        self._final_anchor = self.destination - RIGHT * self._grip
        self._claw_jaws = edge_jaws(self.carried)
        # The full-grown run reaches from the tail to the final anchor,
        # which shares the tail's row; past it, only the run is left.
        self._run = max(self._final_anchor[0] - self._tail[0], 0.1)
        final_path = self._run_path(self._run)
        lengths = np.linalg.norm(np.diff(final_path, axis=0), axis=1)
        self._drop = self._run / lengths.sum()
        super().begin()

    def _run_path(self, run: float) -> np.ndarray:
        """The run-and-bend shape: a horizontal run out of the tail, then
        a bend to the tip with horizontal tangents at both ends."""
        corner = self._tail + RIGHT * run
        handle = 0.42 * (self._tip[0] - corner[0])
        ts = np.linspace(0.0, 1.0, 64)[:, None]
        bend = (
            (1 - ts) ** 3 * corner
            + 3 * (1 - ts) ** 2 * ts * (corner + RIGHT * handle)
            + 3 * (1 - ts) * ts**2 * (self._tip - RIGHT * handle)
            + ts**3 * self._tip
        )
        points = np.concatenate(
            [np.linspace(self._tail, corner, 16)[:-1], bend]
        )
        return resample(points, self.SAMPLES)

    def interpolate_mobject(self, alpha: float) -> None:
        ride = ease(alpha, self.RIDE)
        release = ease(alpha, self.RELEASE)
        clear = ease(alpha, self.CLEAR)

        curve = (
            self._origin * (1.0 - ride)
            + self._run_path(self._run * ride) * ride
        )
        visible = 1.0 - (1.0 - self._drop) * ride - self._drop * clear
        # Cut the polyline at the visible arc length; the cut point is the
        # retracting end the claw and cell ride.
        lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        cut = visible * cumulative[-1]
        kept = curve[cumulative <= cut]
        anchor = np.array(
            [np.interp(cut, cumulative, curve[:, k]) for k in range(3)]
        )
        if len(kept) < 2:
            kept = np.array([self._tail])
        self.arrow.shaft.set_points_as_corners(
            np.concatenate([kept, [anchor]])
        )

        # Once the ride ends the claw has let go: it stays at the anchor it
        # released on while the run goes on undrawing beneath it.
        riding = alpha < self.RIDE[1]
        claw_anchor = anchor if riding else self._final_anchor
        hold = max(1.0 - release, 1e-3)
        for stroke, claw_jaw in zip(self.arrow.tip, self._claw_jaws):
            stroke.set_points_as_corners(claw_anchor + claw_jaw * hold)
        if riding:
            self.carried.move_to(anchor + RIGHT * self._grip)
        else:
            self.carried.move_to(self.destination)


class LayoutFromTupleMorphismTest(LayoutScene):
    """Collapse a tuple morphism's diagram into its layout."""

    def construct(self) -> None:
        for index, example in enumerate(EXAMPLES):
            self._play_example(example)
            self.clear_scene(last=index == len(EXAMPLES) - 1)

    def _play_example(self, example: MorphismExample) -> None:
        f = example.morphism()
        _, strides = flat_layout_components(f)
        products = prefix_products(example.codomain)
        ref = example.backward_leg() if example.nest is not None else None
        source_x = SPAN_SOURCE_X if ref else SOURCE_X
        target_x = SPAN_TARGET_X if ref else TARGET_X

        slots = max(len(example.domain), len(example.codomain))
        usable_height = MAX_STACK_HEIGHT - 0.9
        scale = min(1.0, usable_height / ((slots - 1) * SLOT_STEP + CELL_H))
        step = SLOT_STEP * scale
        place = make_place(-((slots - 1) * step) / 2 - 0.05, step)

        source_cells = [
            fitted_cell(
                value, place(source_x, index), fill=SHAPE_FILL, scale=scale
            )
            for index, value in enumerate(example.domain)
        ]
        target_cells = [
            fitted_cell(
                value, place(target_x, index), fill=SHAPE_FILL, scale=scale
            )
            for index, value in enumerate(example.codomain)
        ]
        arrows = {
            index: segment_arrow(source_cells[index], target_cells[target - 1])
            for index, target in enumerate(example.mapping)
            if target != 0
        }

        # A Ref span's backward leg stands inverted on the left, exactly as
        # in the layout depiction: root product cells, mirrored ref trees
        # fanning out into the apex stack.
        root_cells = []
        trees = []
        if ref:
            root_cells = [
                fitted_cell(
                    value,
                    place(SPAN_ROOT_X, index),
                    fill=SHAPE_FILL,
                    scale=scale,
                )
                for index, value in enumerate(ref.codomain)
            ]
            global_depth = max(
                (mode.depth() for mode in ref.modes), default=1
            )
            start = 0
            for index, mode in enumerate(ref.modes):
                leaf_anchors = [
                    source_cells[start + offset].get_left()
                    for offset in range(mode.length())
                ]
                tree = ref_tree_segments(
                    mode,
                    leaf_anchors,
                    root_cells[index].get_right(),
                    global_depth,
                )
                # Strands are built child-to-parent; the backward leg reads
                # root-to-leaves, so they draw left to right, root first.
                for strand in tree:
                    strand.reverse_points()
                trees.append(VGroup(*reversed(list(tree))))
                start += mode.length()

        # Beat 1: the span — apex and target stacks, then the backward
        # leg's trees, then the forward leg's arrows.
        self.play(
            FadeIn(VGroup(*source_cells)),
            FadeIn(VGroup(*target_cells)),
            *((FadeIn(VGroup(*root_cells)),) if ref else ()),
        )
        if ref:
            self.play(
                LaggedStart(
                    *(
                        Create(strand)
                        for tree in trees
                        for strand in tree
                    ),
                    lag_ratio=0.12,
                    run_time=1.2,
                )
            )
        for arrow in arrows.values():
            self.add(arrow)
        self.play(
            LaggedStart(
                *(
                    TailToTipMapsto(arrow, run_time=0.85)
                    for arrow in arrows.values()
                ),
                lag_ratio=0.12,
            )
        )
        self.wait(0.25)

        # Beat 2: the prefix products beside the target, behind colons —
        # the column-major layout of T, built bottom-up as the products
        # accumulate.
        product_cells = [
            fitted_cell(
                value,
                place(target_x + PAIR_GAP, index),
                fill=STRIDE_FILL,
                scale=scale,
            )
            for index, value in enumerate(products)
        ]
        target_colons = [
            layout_colon(place(target_x + PAIR_GAP / 2, index), scale=scale)
            for index in range(len(products))
        ]
        self.play(
            LaggedStart(
                *(
                    FadeIn(VGroup(colon, cell), shift=RIGHT * 0.12)
                    for colon, cell in zip(target_colons, product_cells)
                ),
                lag_ratio=0.25,
            ),
            run_time=1.2,
        )
        self.wait(0.25)

        # Beat 3: the collapse.  The target entries and colons fall away
        # (with any prefix product no mode picks up); then, near
        # simultaneously with a slight stagger like the draw-ins, every
        # arrow extends out into a grabber, grips its stride where it
        # stands, and pulls it home; a basepoint stride appears as 0 from
        # nowhere in its turn.  The colons follow once everything is seated.
        used = set(example.mapping) - {0}
        unused_products = [
            product_cells[j]
            for j in range(len(products))
            if j + 1 not in used
        ]
        self.play(
            FadeOut(VGroup(*target_cells)),
            FadeOut(VGroup(*target_colons)),
            *(FadeOut(cell) for cell in unused_products),
            run_time=0.6,
        )

        stride_cells = []
        source_colons = []
        reaches = {}
        pulls = {}
        basepoint_strides = []
        for index, target in enumerate(example.mapping):
            destination = place(source_x + PAIR_GAP, index)
            source_colons.append(
                layout_colon(place(source_x + PAIR_GAP / 2, index), scale=scale)
            )
            if target == 0:
                stride_cell = fitted_cell(
                    strides[index], destination, fill=STRIDE_FILL, scale=scale
                )
                basepoint_strides.append(stride_cell)
            else:
                stride_cell = product_cells[target - 1]
                # The same run_time as a draw-in arrow, so the reaches
                # continue the draw-ins' stagger and pace.  Keyed by the
                # target slot: the stagger runs in target tuple order.
                reaches[target] = GrabberReach(
                    arrows[index], stride_cell, run_time=0.85
                )
                pulls[target] = GrabberPull(
                    arrows[index], stride_cell, destination
                )
            stride_cells.append(stride_cell)
        # Every grab completes before any stride moves: the reaches play
        # out staggered first, then the pulls, staggered the same way,
        # both in target tuple order — the order the strides stand in.
        # Basepoint strides have nothing to be pulled by: their 0s appear
        # only once every pulled stride is seated, with the colons.
        self.play(
            LaggedStart(
                *(reaches[t] for t in sorted(reaches)), lag_ratio=0.12
            )
        )
        self.play(
            LaggedStart(*(pulls[t] for t in sorted(pulls)), lag_ratio=0.12),
            run_time=1.8,
        )
        for arrow in arrows.values():
            self.remove(arrow)
        self.play(
            LaggedStart(
                *(FadeIn(colon) for colon in source_colons), lag_ratio=0.1
            ),
            *(FadeIn(cell) for cell in basepoint_strides),
            run_time=0.6,
        )

        # What remains is the layout depiction — nested, backward Ref leg
        # and all, for a span — which slides to center.
        remaining = VGroup(*source_cells, *stride_cells, *source_colons)
        left_x = source_x
        if ref:
            remaining.add(*root_cells, *trees)
            left_x = SPAN_ROOT_X
        self.play(
            remaining.animate.shift(
                RIGHT * -(left_x + source_x + PAIR_GAP) / 2
            )
        )
        self.wait(1.0)
