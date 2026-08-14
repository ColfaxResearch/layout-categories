"""Purpose-built animations for the layout-categories drawing primitives."""

import numpy as np
from manim import Animation

from .tuple_morphism import MapstoArrow


def _arc_length_parameterization(
    path, *, samples_per_curve: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Map normalized arc length to Manim's piecewise-cubic parameter."""
    curves = path.points.reshape((-1, 4, 3))
    curve_count = len(curves)
    parameters = [0.0]
    positions = [curves[0, 0]]

    sample_parameters = np.linspace(0.0, 1.0, samples_per_curve + 1)[1:]
    for curve_index, (start, first_handle, second_handle, end) in enumerate(curves):
        for curve_parameter in sample_parameters:
            complement = 1.0 - curve_parameter
            position = (
                complement**3 * start
                + 3 * complement**2 * curve_parameter * first_handle
                + 3 * complement * curve_parameter**2 * second_handle
                + curve_parameter**3 * end
            )
            parameters.append(
                (curve_index + curve_parameter) / curve_count
            )
            positions.append(position)

    distances = np.linalg.norm(np.diff(np.asarray(positions), axis=0), axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(distances)))
    total_length = cumulative_lengths[-1]
    if total_length == 0:
        return np.asarray((0.0, 1.0)), np.asarray((0.0, 1.0))
    return cumulative_lengths / total_length, np.asarray(parameters)


def handoff_mapsto_arrow(scene, displayed: MapstoArrow, canonical: MapstoArrow) -> None:
    """Swap geometrically identical arrow groups without producing a frame of motion.

    Call only after the displayed arrow's three parts have been transformed to
    exactly match ``canonical``.  The next rendered frame has the same pixels,
    but uses the canonical three-part group for later animations.
    """
    scene.remove(displayed)
    scene.add(canonical)


class TailToTipMapsto(Animation):
    """Draw a ``|→`` map arrow as one tail-to-tip gesture.

    The tail bar starts first, the shaft starts before that bar has finished,
    and the open caret grows from the shaft end only once the trace reaches it.
    ``TupleMorphismDiagram._mapsto_arrow`` defines the expected group shape.
    """

    def __init__(self, mapsto_arrow: MapstoArrow, **kwargs) -> None:
        self.bar = mapsto_arrow.tail
        self.shaft = mapsto_arrow.shaft
        self.caret = mapsto_arrow.tip
        super().__init__(mapsto_arrow, **kwargs)

    def begin(self) -> None:
        self._bar_template = self.bar.copy()
        self._shaft_template = self.shaft.copy()
        self._caret_templates = [stroke.copy() for stroke in self.caret]
        (
            self._shaft_arc_lengths,
            self._shaft_parameters,
        ) = _arc_length_parameterization(self._shaft_template)
        super().begin()

    @staticmethod
    def _ease(alpha: float, start: float, end: float) -> float:
        progress = float(np.clip((alpha - start) / (end - start), 0.0, 1.0))
        return progress * progress * (3.0 - 2.0 * progress)

    @staticmethod
    def _partial(target, template, progress: float) -> None:
        target.pointwise_become_partial(template, 0, progress)

    def interpolate_mobject(self, alpha: float) -> None:
        self._partial(self.bar, self._bar_template, self._ease(alpha, 0.0, 0.16))
        self._partial(
            self.shaft,
            self._shaft_template,
            float(
                np.interp(
                    self._ease(alpha, 0.07, 0.83),
                    self._shaft_arc_lengths,
                    self._shaft_parameters,
                )
            ),
        )
        caret_progress = self._ease(alpha, 0.83, 1.0)
        for stroke, template in zip(self.caret, self._caret_templates):
            self._partial(stroke, template, caret_progress)


class TailToTipUnmapsto(TailToTipMapsto):
    """Erase a ``|→`` map arrow from its tail toward its tip."""

    @staticmethod
    def _partial(target, template, progress: float) -> None:
        target.pointwise_become_partial(template, progress, 1)


class TipToTailUnmapsto(TailToTipMapsto):
    """Erase a ``|→`` map arrow from its tip back toward its tail."""

    def interpolate_mobject(self, alpha: float) -> None:
        super().interpolate_mobject(1.0 - alpha)


class UncreateMapstoTip(Animation):
    """Retract an open map-arrow tip back into its point."""

    def __init__(self, mapsto_arrow: MapstoArrow, **kwargs) -> None:
        self.caret = mapsto_arrow.tip
        super().__init__(mapsto_arrow, **kwargs)

    def begin(self) -> None:
        self._caret_templates = [stroke.copy() for stroke in self.caret]
        super().begin()

    def interpolate_mobject(self, alpha: float) -> None:
        progress = TailToTipMapsto._ease(alpha, 0.0, 1.0)
        for stroke, template in zip(self.caret, self._caret_templates):
            stroke.pointwise_become_partial(template, 0, 1.0 - progress)


class DrawMapstoTip(Animation):
    """Grow an open map-arrow tip out from its point.

    Both caret strokes are driven by one shared progress, so they extend from
    the shared vertex symmetrically and simultaneously.  ``tip`` is the caret
    itself (a group of stroke mobjects), not the whole arrow.
    """

    def __init__(self, tip, **kwargs) -> None:
        self.caret = tip
        super().__init__(tip, **kwargs)

    def begin(self) -> None:
        self._caret_templates = [stroke.copy() for stroke in self.caret]
        super().begin()

    @staticmethod
    def _progress(alpha: float) -> float:
        return TailToTipMapsto._ease(alpha, 0.0, 1.0)

    def interpolate_mobject(self, alpha: float) -> None:
        progress = self._progress(alpha)
        for stroke, template in zip(self.caret, self._caret_templates):
            stroke.pointwise_become_partial(template, 0, progress)


class UndrawMapstoTip(DrawMapstoTip):
    """Retract an open map-arrow tip back into its point (both strokes together)."""

    def interpolate_mobject(self, alpha: float) -> None:
        progress = self._progress(alpha)
        for stroke, template in zip(self.caret, self._caret_templates):
            stroke.pointwise_become_partial(template, 0, 1.0 - progress)
