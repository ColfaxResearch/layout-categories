"""Isolated playground for tuple-entry coalesce gestures."""

from math import prod

import numpy as np
from manim import (
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    ManimColor,
    ORIGIN,
    ReplacementTransform,
    Scene,
    Text,
    Transform,
    Uncreate,
    UP,
    UpdateFromAlphaFunc,
    VGroup,
    VMobject,
    config,
    smooth,
)

from layout_categories_viz.style import BACKGROUND, CODE_FONT, INK


EXAMPLES = (
    (2, 3),
    (2, 3, 5),
)
HIGHLIGHT = ManimColor("#F3D77A")
HIGHLIGHT_OPACITY = 0.42
CORNER_RADIUS = 0.08

ENTRY_HOLD = 0.7
RESULT_HOLD = 1.7


class TupleEntryCoalesceTest(Scene):
    """Weld cells by sharpening only the corners that become interior."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND

        for values in EXAMPLES:
            self._play_example(values)
            self.play(
                *(FadeOut(mobject) for mobject in tuple(self.mobjects)),
                run_time=0.5,
            )
            self.clear()
            self.wait(0.15)

    def _play_example(self, values) -> None:
        frame_width = config.frame_width
        frame_height = config.frame_height
        cell_size = min(0.12 * frame_height, 0.075 * frame_width)
        row_gap = 0.32 * cell_size
        font_size = 28 * cell_size / (0.085 * frame_height)

        entries = self._make_entries(
            values,
            cell_size,
            cell_size,
            font_size,
        )
        # Tuple coordinates follow the project convention: bottom to top.
        entries.arrange(UP, buff=row_gap)
        entries.move_to(ORIGIN)
        center = sum(
            (entry.get_center() for entry in entries),
            start=ORIGIN.copy(),
        ) / len(entries)
        product_entry = self._entry(
            prod(values), center, cell_size, font_size
        )
        title = Text(
            f"cell weld   {values} → {prod(values)}",
            color=INK,
            font=CODE_FONT,
            font_size=22,
        )
        title.to_edge(UP, buff=0.4)

        self.play(
            FadeIn(title),
            LaggedStart(
                *(FadeIn(entry, scale=0.9) for entry in entries),
                lag_ratio=0.12,
            ),
            run_time=0.8,
        )
        self.wait(ENTRY_HOLD)

        self._cell_weld(
            entries,
            product_entry,
            center,
            cell_size,
        )

        self.wait(RESULT_HOLD)

    @staticmethod
    def _box(width, height, top_radius, bottom_radius):
        """Return a box with independently controlled top/bottom corners."""
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
        box.set_stroke(INK, width=1.8)
        box.set_fill(HIGHLIGHT, opacity=HIGHLIGHT_OPACITY)
        return box

    @staticmethod
    def _make_entries(values, width, height, font_size):
        entries = VGroup()
        for value in values:
            box = TupleEntryCoalesceTest._box(
                width,
                height,
                CORNER_RADIUS,
                CORNER_RADIUS,
            )
            label = Text(
                str(value),
                color=INK,
                font=CODE_FONT,
                font_size=font_size,
            ).move_to(box)
            entries.add(VGroup(box, label))
        return entries

    @staticmethod
    def _entry(value, center, cell_size, font_size):
        entry = TupleEntryCoalesceTest._make_entries(
            (value,), cell_size, cell_size, font_size
        )[0]
        entry.move_to(center)
        return entry

    def _cell_weld(
        self,
        entries,
        product_entry,
        center,
        cell_size,
    ) -> None:
        count = len(entries)
        touching_centers = [
            center
            + UP * (index - (count - 1) / 2) * cell_size
            for index in range(count)
        ]
        seams = VGroup(
            *(
                Line(
                    center
                    + np.array((-cell_size / 2, offset, 0.0)),
                    center
                    + np.array((cell_size / 2, offset, 0.0)),
                    color=INK,
                    stroke_width=1.8,
                )
                for offset in (
                    (index - count / 2) * cell_size
                    for index in range(1, count)
                )
            )
        )
        approach_animations = []
        for index, (entry, destination) in enumerate(
            zip(entries, touching_centers)
        ):
            initial_center = entry.get_center().copy()

            def update_box(
                displayed,
                alpha,
                index=index,
                initial_center=initial_center,
                destination=destination,
            ):
                interior_radius = CORNER_RADIUS * (1.0 - alpha)
                top_radius = (
                    CORNER_RADIUS
                    if index == count - 1
                    else interior_radius
                )
                bottom_radius = (
                    CORNER_RADIUS if index == 0 else interior_radius
                )
                frame = self._box(
                    cell_size,
                    cell_size,
                    top_radius,
                    bottom_radius,
                )
                frame.move_to(
                    initial_center
                    + alpha * (destination - initial_center)
                )
                displayed.become(frame)

            approach_animations.extend(
                (
                    UpdateFromAlphaFunc(
                        entry[0],
                        update_box,
                        rate_func=smooth,
                    ),
                    entry[1].animate.move_to(destination),
                )
            )
        self.play(*approach_animations, run_time=0.65)

        hull = self._box(
            cell_size,
            count * cell_size,
            CORNER_RADIUS,
            CORNER_RADIUS,
        ).move_to(center)
        hull.set_z_index(-1)
        self.add(hull, seams)
        for entry in entries:
            entry[0].set_opacity(0)

        hull.set_z_index(-1)

        self.play(
            LaggedStart(
                *(Uncreate(seam) for seam in seams),
                lag_ratio=0.12,
            ),
            run_time=0.55,
        )

        labels = VGroup(*(entry[1] for entry in entries))
        self.play(
            Transform(hull, product_entry[0]),
            ReplacementTransform(labels, product_entry[1]),
            run_time=0.8,
            rate_func=smooth,
        )
        self.remove(*entries)
