"""The composition hidden inside tiling: flash-attention's gQ partition.

``flat_divide`` never says "compose", but that is all it does:
``layout.flat_divide(tiler)`` is ``tiler.concat(tiler.complement())`` composed
with the layout's morphism — the tract form of CuTe's ``logical_divide``, the
engine under ``zipped_divide`` and ``local_tile``.  This scene animates that
hidden composite on the first partition FlashAttention-2 performs:
``gQ = local_tile(mQ, (kBlockM, kHeadDim), coord)`` with kBlockM = 64 and
head dimension 64.

``f_Q`` is the standard morphism of the row-major Q matrix for a sequence
length of 128, with the row mode pre-factored as (64, 2) — the split
``logical_divide``'s refinement stage would perform, taken here so the
composition is on the nose and the direct choreography applies.  The first
stage is the tiler concatenated with its complement (the block-index mode the
tiler leaves out); composing gives the morphism of the layout
``((64, 64), 2) : ((64, 1), 4096)`` — the stack of (64, 64) Q tiles the kernel
indexes by block, one per iteration of its outer loop.

Only the example and the validation prologue differ from
``tuple_morphism_composition_curve``; the choreography is that scene's.  The
composite's map has no mergeable runs, so unlike the coalesce variant nothing
follows the collapse.
"""

from manim import FadeIn, Text, UP
from tract import TupleMorphism
from tract.backends.base import flat_layout_components, standard_tuple_morphism

from layout_categories_viz.stacks import LABEL_FONT_SIZE
from layout_categories_viz.style import CODE_FONT, INK
from scenes.tuple_morphism_composition_curve import (
    TupleMorphismCurvedCompositionCollapse,
)


TITLE = "FlashAttention-2  gQ = local_tile(mQ, (64, 64))"

# first = tiler ++ complement (block row, column, block index),
# second = f_Q, the morphism of the Q matrix (64, 2, 64) : (64, 4096, 1).
EXAMPLE = {
    "domain": (64, 64, 2),
    "intermediate": (64, 2, 64),
    "codomain": (64, 64, 2),
    "first_mapping": (1, 3, 2),
    "second_mapping": (2, 3, 1),
}


class FlashAttentionDivideTest(TupleMorphismCurvedCompositionCollapse):
    """Animate the composition inside ``flat_divide`` on the gQ partition."""

    def construct(self) -> None:
        f_q = standard_tuple_morphism((64, 2, 64), (64, 4096, 1))
        tiler = TupleMorphism((64, 64), (64, 2, 64), (1, 3))
        first = tiler.concat(tiler.complement())
        if (first.domain, first.codomain, first.map) != (
            EXAMPLE["domain"],
            EXAMPLE["intermediate"],
            EXAMPLE["first_mapping"],
        ):
            raise ValueError("The configured tiler stage changed unexpectedly")
        if (f_q.domain, f_q.codomain, f_q.map) != (
            EXAMPLE["intermediate"],
            EXAMPLE["codomain"],
            EXAMPLE["second_mapping"],
        ):
            raise ValueError("The configured Q morphism changed unexpectedly")
        divided = f_q.flat_divide(tiler)
        if flat_layout_components(divided) != ((64, 64, 2), (64, 1, 4096)):
            raise ValueError("The configured division changed unexpectedly")

        title = Text(
            TITLE,
            color=INK,
            font=CODE_FONT,
            font_size=LABEL_FONT_SIZE,
        ).to_edge(UP, buff=0.2)
        self.play(FadeIn(title), run_time=0.4)
        self._play_example(**EXAMPLE)
