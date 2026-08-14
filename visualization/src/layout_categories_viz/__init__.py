"""Reusable Manim components for layout-categories explanations."""

from .style import BACKGROUND, CODE_FONT, INK, MODE_COLORS
from .animations import (
    DrawMapstoTip,
    TailToTipMapsto,
    TailToTipUnmapsto,
    TipToTailUnmapsto,
    UncreateMapstoTip,
    UndrawMapstoTip,
    handoff_mapsto_arrow,
)
from .composition import TupleMorphismCompositionDiagram, compose_tuple_maps
from .nested_tuple import NestedTupleTree
from .nest_morphism import NestMorphismDiagram
from .tuple_morphism import MapstoArrow, TupleMorphismDiagram

__all__ = [
    "BACKGROUND",
    "CODE_FONT",
    "INK",
    "MODE_COLORS",
    "MapstoArrow",
    "NestedTupleTree",
    "NestMorphismDiagram",
    "DrawMapstoTip",
    "TailToTipMapsto",
    "TailToTipUnmapsto",
    "TipToTailUnmapsto",
    "UncreateMapstoTip",
    "UndrawMapstoTip",
    "handoff_mapsto_arrow",
    "TupleMorphismCompositionDiagram",
    "TupleMorphismDiagram",
    "compose_tuple_maps",
]
