"""Prototype for drawing arrows and segments in graphite.

Every connector in these scenes is one clean bezier of uniform width.  A pencil
does three things that ink does not, and this prototype does all three to an
existing connector without changing where it goes:

1. it wavers -- the path is displaced sideways by a little smooth noise, so it
   is never quite the curve it is aiming at;
2. it varies in pressure -- the stroke is laid down as a chain of short pieces,
   each with its own width and darkness, and lightens towards either end where
   the hand lifts;
3. it goes over itself -- two passes, each with its own waver, plus a wide faint
   one underneath for the bloom graphite leaves on paper.  Where the passes
   cross, the line darkens by itself.

``graphite`` turns any ``VMobject`` path into that, and ``draw_graphite``
animates it as the pencil travelling: the pieces appear in order, and the second
pass follows the first as a retrace.  The waver is seeded per connector, so a
line is always drawn the same way twice.

The scene shows the same three connectors in ink and in graphite, then a whole
morphism in graphite, then the tuning: light, standard, and bold.
"""

import numpy as np
from manim import (
    Create,
    FadeIn,
    FadeOut,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    VMobject,
    rate_functions,
)

from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK
from scenes.tuple_pullback_test import (
    SLOT_STEP,
    STROKE_WIDTH,
    TuplePullbackTest,
)


SPACING = 0.02  # how far apart the path is resampled, in scene units
PIECE = 0.09  # and how long each piece of the stroke is
SAMPLES = (24, 260)  # however long the path, this many samples
WAVER = 0.022  # how far the pencil strays from the curve it is aiming at
TOOTH = 0.004  # and how much the grain of the paper adds on top
PRESSURE = (0.82, 1.28)  # the range of widths pressure varies over
LIFT = 0.12  # the fraction of either end the hand lifts over
PASSES = 2
PASS_OPACITY = 0.62
BLOOM_OPACITY = 0.1  # the wide, faint pass underneath
BLOOM_WIDTH = 2.6


def _normals(points: np.ndarray) -> np.ndarray:
    """A unit normal at each sample, from the direction of travel."""
    steps = np.gradient(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1, keepdims=True)
    steps = steps / np.where(lengths == 0, 1.0, lengths)
    return np.stack((-steps[:, 1], steps[:, 0], np.zeros(len(steps))), axis=1)


def _waver(count: int, rng, amplitude: float) -> np.ndarray:
    """Smooth sideways drift, plus a little grain, along the stroke."""
    t = np.linspace(0.0, 1.0, count)
    drift = np.zeros(count)
    for _ in range(3):
        cycles = rng.uniform(0.6, 2.4)
        drift += rng.uniform(-1.0, 1.0) * np.sin(
            2 * np.pi * cycles * t + rng.uniform(0.0, 2 * np.pi)
        )
    drift *= amplitude / 3
    return drift + rng.normal(0.0, TOOTH, count)


def _pressure(count: int, rng) -> np.ndarray:
    """How hard the pencil is bearing down, along the stroke."""
    t = np.linspace(0.0, 1.0, count)
    low, high = PRESSURE
    varying = 0.5 + 0.5 * np.sin(
        2 * np.pi * rng.uniform(0.7, 1.6) * t + rng.uniform(0.0, 2 * np.pi)
    )
    # The hand lifts over the first and last stretch of the stroke.
    lift = np.clip(np.minimum(t, 1.0 - t) / LIFT, 0.0, 1.0)
    return (low + (high - low) * varying) * (0.55 + 0.45 * lift)


def graphite(
    path: VMobject,
    *,
    seed: int = 0,
    width: float = STROKE_WIDTH,
    color=INK,
    waver: float = WAVER,
    passes: int = PASSES,
) -> VGroup:
    """Redraw one path as a pencil would: wavering, pressed, gone over twice."""
    rng = np.random.default_rng(seed)
    # Sample by length, not by a fixed count: a caret arm is a twentieth of a
    # connector, and giving it as many pieces would pile them into a blot.
    coarse = np.array(
        [path.point_from_proportion(t) for t in np.linspace(0.0, 1.0, 64)]
    )
    length = float(np.linalg.norm(np.diff(coarse, axis=0), axis=1).sum())
    count = int(np.clip(length / SPACING, *SAMPLES))
    chunk = max(2, round(PIECE / SPACING))
    points = np.array(
        [path.point_from_proportion(t) for t in np.linspace(0.0, 1.0, count)]
    )
    normals = _normals(points)

    strokes = VGroup()
    for index in range(passes + 1):
        stroke = VGroup()
        bloom = index == passes  # the last one is the faint wide pass
        drifted = points + normals * _waver(
            count, rng, waver * (1.6 if bloom else 1.0)
        )[:, None]
        pressure = _pressure(count, rng)
        for start in range(0, count - 1, chunk):
            piece_points = drifted[start : start + chunk + 1]
            if len(piece_points) < 2:
                continue
            piece = VMobject()
            piece.set_points_as_corners(piece_points)
            piece.set_stroke(
                color=color,
                width=width
                * (BLOOM_WIDTH if bloom else 1.0)
                * float(pressure[start : start + chunk + 1].mean()),
                opacity=BLOOM_OPACITY if bloom else PASS_OPACITY,
            )
            stroke.add(piece)
        strokes.add(stroke)
    return strokes


def draw_graphite(strokes: VGroup, *, run_time: float = 1.0):
    """Draw the strokes as the pencil travelling: start to end, pass by pass.

    The reveal is a position along the stroke rather than an animation per
    piece: each piece owns a stretch of the path and shows exactly the fraction
    of itself the pencil has reached, so the line grows from one end at a steady
    speed instead of every piece coming up at once.  Pass two retraces from the
    start, and the bloom follows it.
    """
    travel = ValueTracker(0.0)
    passes = len(strokes)
    for index, stroke in enumerate(strokes):
        pieces = len(stroke)
        for position, piece in enumerate(stroke):
            template = piece.copy()
            piece.set_stroke(opacity=0)

            def reveal(
                mobject,
                template=template,
                start=(index + position / pieces) / passes,
                end=(index + (position + 1) / pieces) / passes,
            ):
                drawn = (travel.get_value() - start) / (end - start)
                if drawn <= 0.0:
                    mobject.set_stroke(opacity=0)
                    return
                mobject.become(
                    template.copy().pointwise_become_partial(
                        template, 0.0, min(1.0, drawn)
                    )
                )

            piece.add_updater(reveal)
    return travel.animate(
        run_time=run_time, rate_func=rate_functions.linear
    ).set_value(1.0)


def settle_graphite(strokes: VGroup) -> None:
    """Let the strokes be, once the pencil has gone over them."""
    for stroke in strokes:
        for piece in stroke:
            piece.clear_updaters()


class GraphiteLineTest(Scene):
    """The same connectors in ink and in graphite, and the tuning."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        self._compare()
        self._clear_scene()
        self._in_context()
        self._clear_scene()
        self._tuning()
        self._clear_scene(last=True)

    # ------------------------------------------------------------------ beats
    def _compare(self) -> None:
        """One of each connector, in ink above and in graphite below."""
        for row, (offset, label, pencil) in enumerate(
            ((1.5, "ink", False), (-1.2, "graphite", True))
        ):
            connectors = self._connectors(offset)
            drawn = (
                VGroup(
                    *(
                        graphite(connector, seed=index + 7 * row)
                        for index, connector in enumerate(connectors)
                    )
                )
                if pencil
                else connectors
            )
            if pencil:
                self.add(*drawn)
                self.play(
                    *(draw_graphite(strokes, run_time=1.6) for strokes in drawn)
                )
                for strokes in drawn:
                    settle_graphite(strokes)
            else:
                self.play(
                    *(Create(connector, run_time=1.6) for connector in drawn)
                )
            self.add(self._note(label, offset))
        self.wait(2.0)

    def _in_context(self) -> None:
        """A morphism with every connector drawn in graphite."""
        cell = TuplePullbackTest._cell
        left = [
            cell(value, np.array((-3.0, y, 0.0)))
            for value, y in ((6, -SLOT_STEP), (12, 0.0), (4, SLOT_STEP))
        ]
        right = [
            cell(value, np.array((3.0, y, 0.0)))
            for value, y in ((12, -SLOT_STEP), (4, 0.0), (6, SLOT_STEP))
        ]
        arrows = [
            TuplePullbackTest._segment_arrow(left[source], right[target])
            for source, target in ((0, 2), (1, 0), (2, 1))
        ]
        self.play(*(FadeIn(box) for box in (*left, *right)), run_time=0.6)
        strokes = VGroup(
            *(
                VGroup(
                    graphite(arrow.shaft, seed=index),
                    graphite(arrow.tip[0], seed=index + 31, waver=0.012),
                    graphite(arrow.tip[1], seed=index + 61, waver=0.012),
                )
                for index, arrow in enumerate(arrows)
            )
        )
        self.add(*strokes)
        self.play(
            *(draw_graphite(shaft, run_time=1.2) for shaft, _, _ in strokes)
        )
        self.play(
            *(
                draw_graphite(part, run_time=0.3)
                for _, upper, lower in strokes
                for part in (upper, lower)
            )
        )
        for parts in strokes:
            for part in parts:
                settle_graphite(part)
        self.add(self._note("a morphism, drawn in graphite", -1.9))
        self.wait(2.0)

    def _tuning(self) -> None:
        """The same connector at three weights."""
        settings = (
            ("light", dict(width=2.6, waver=0.03, passes=1)),
            ("standard", dict()),
            ("bold", dict(width=4.6, waver=0.016, passes=3)),
        )
        for index, (label, options) in enumerate(settings):
            offset = 1.7 - 1.7 * index
            connector = TuplePullbackTest._tree_segment(
                np.array((-3.4, offset - 0.35, 0.0)),
                np.array((3.4, offset + 0.35, 0.0)),
            )
            strokes = graphite(connector, seed=100 + index, **options)
            self.add(strokes)
            self.play(draw_graphite(strokes, run_time=1.2))
            settle_graphite(strokes)
            self.add(self._note(label, offset - 0.75))
        self.wait(2.2)

    # ------------------------------------------------------------------ parts
    @staticmethod
    def _connectors(offset: float):
        """A fan segment, a steep one, and a map arrow's shaft."""
        segment = TuplePullbackTest._tree_segment
        return VGroup(
            segment(
                np.array((-4.4, offset, 0.0)), np.array((-1.2, offset, 0.0))
            ),
            segment(
                np.array((-0.8, offset - 0.5, 0.0)),
                np.array((1.6, offset + 0.7, 0.0)),
            ),
            TuplePullbackTest._segment_arrow(
                TuplePullbackTest._cell(2, np.array((2.4, offset, 0.0))),
                TuplePullbackTest._cell(2, np.array((5.2, offset + 0.4, 0.0))),
            ).shaft,
        )

    @staticmethod
    def _note(text: str, y: float) -> Text:
        return Text(text, color=INK, font=CODE_FONT, font_size=20).move_to(
            np.array((-5.6, y, 0.0))
        )

    def _clear_scene(self, *, last: bool = False) -> None:
        self.play(*(FadeOut(m) for m in self.mobjects), run_time=0.6)
        self.clear()
        if not last:
            self.wait(0.2)
