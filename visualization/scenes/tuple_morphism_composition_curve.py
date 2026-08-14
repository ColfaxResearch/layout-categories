"""Animate curved two-stage composition collapsing into a compact composite."""

from layout_categories_viz.scene_base import LayoutScene

import numpy as np
from manim import (
    CubicBezier,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    LaggedStart,
    ORIGIN,
    ShrinkToCenter,
    Text,
    Transform,
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
from layout_categories_viz.paths import (
    ARROW_BEND_HANDLE,
    ARROW_ENDPOINT_INSET,
    ARROW_HORIZONTAL_RUN,
    append_cubic_segments,
    horizontal_matched_path_pair,
    matched_path_pair,
    repartition_path,
    split_cubic,
    three_segment_route,
)
from layout_categories_viz.style import CODE_FONT, INK
from layout_categories_viz.tuple_morphism import TupleMorphismDiagram


ARROW_DRAW_DURATION = 1.2

# Backward-compatible aliases for the path helpers' old private names; the
# helpers themselves live in layout_categories_viz.paths.
_append_cubic_segments = append_cubic_segments
_horizontal_matched_path_pair = horizontal_matched_path_pair
_matched_path_pair = matched_path_pair
_repartition_path = repartition_path
_split_cubic = split_cubic
_three_segment_route = three_segment_route


class TupleMorphismCurvedCompositionCollapse(LayoutScene):
    """Collapse a curved two-stage route into a compact straight composite."""

    def construct(self) -> None:

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
            canonical = TupleMorphismDiagram.mapsto_arrow(
                diagram.source_entries[source_index].get_right(),
                diagram.target_entries[target_index].get_left(),
                endpoint_inset=ARROW_ENDPOINT_INSET,
                horizontal_run=ARROW_HORIZONTAL_RUN,
                bend_handle=ARROW_BEND_HANDLE,
            )
            canonical.tail.set_opacity(0)
            route_shaft = three_segment_route(
                first_arrow.shaft, connector, second_arrow.shaft
            )
            contracted = TupleMorphismDiagram.mapsto_arrow(
                diagram.source_entries[source_index].get_right() + source_shift,
                diagram.target_entries[target_index].get_left() + target_shift,
                endpoint_inset=ARROW_ENDPOINT_INSET,
                horizontal_run=ARROW_HORIZONTAL_RUN,
                bend_handle=ARROW_BEND_HANDLE,
            )
            contracted.tail.set_opacity(0)
            initial_shaft, final_shaft = matched_path_pair(
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
