"""Weak composition on layouts from real tensor-core kernels.

Each example is the thread-to-address computation of an MMA atom: ``f`` is the
standard tuple morphism of the atom's TV layout (the map from (thread, value)
coordinates into the tile, ``standard_tuple_morphism`` applied to the CUTLASS
``CLayout``/``ALayout`` strides) and ``g`` is the morphism of a row-major tile
in global memory.  Composing them — what ``composition(tile, tv)`` does inside
CuTe's partitioning — answers where each thread's fragment values live in
memory.  The two are never composable on the nose: the TV codomain factors the
tile size along the atom's thread and value modes while the row-major tile
factors it by columns and rows, so the composite only exists weakly, through
the mutual refinement.  These are the compositions a warp performs in every
flash-attention epilogue and operand load.

The trivial contrast is column-major: against ``(16, 8):(1, 16)`` the tile
morphism is the identity and nothing needs refining, which is exactly why
CuTe's accumulator convention keeps C column-major inside the kernel.  The
row-major examples animate the price of leaving that convention.

Only the example list and the on-screen titles differ from
``weak_composition_test``; the choreography is that scene's.  Each example
records the flat (shape, stride) pairs it came from and the morphisms they
standardize to, and refuses to animate if the bridge stops agreeing.
"""

from dataclasses import dataclass

from manim import FadeIn, Text, UP
from tract.backends.base import standard_tuple_morphism

from layout_categories_viz.stacks import LABEL_FONT_SIZE
from layout_categories_viz.style import CODE_FONT, INK
from scenes.weak_composition_test import (
    WeakCompositionExample,
    WeakCompositionTest,
)


@dataclass(frozen=True)
class MmaCompositionExample:
    """A TV layout and a tile layout, with the morphisms they standardize to."""

    title: str
    tv_shape: tuple
    tv_stride: tuple
    tile_shape: tuple
    tile_stride: tuple
    expected: WeakCompositionExample

    def weak_example(self) -> WeakCompositionExample:
        f = standard_tuple_morphism(self.tv_shape, self.tv_stride)
        g = standard_tuple_morphism(self.tile_shape, self.tile_stride)
        derived = WeakCompositionExample(
            domain=f.domain,
            first_codomain=f.codomain,
            first_map=f.map,
            second_domain=g.domain,
            codomain=g.codomain,
            second_map=g.map,
        )
        if derived != self.expected:
            raise ValueError(
                "standard_tuple_morphism disagrees with the configured example"
            )
        return derived


EXAMPLES = (
    # mma.sync.m16n8k16 accumulator: 32 threads, 4 values each, over a 16x8 C
    # tile.  TV layout ((4,8),(2,2)):((32,1),(16,8)) from CUTLASS
    # mma_traits_sm80.hpp; the row-major tile splits the codomain 128 as
    # (8, 16) where the TV codomain factors it as (8, 2, 2, 4), so
    # W = (8, 2, 2, 4) refines the tile's 16-mode in three.
    MmaCompositionExample(
        title="SM80 mma.sync m16n8k16 C-tile, row-major C",
        tv_shape=(4, 8, 2, 2),
        tv_stride=(32, 1, 16, 8),
        tile_shape=(16, 8),
        tile_stride=(8, 1),
        expected=WeakCompositionExample(
            domain=(4, 8, 2, 2),
            first_codomain=(8, 2, 2, 4),
            first_map=(4, 1, 3, 2),
            second_domain=(16, 8),
            codomain=(8, 16),
            second_map=(2, 1),
        ),
    ),
    # wgmma.mma_async m64n8 accumulator: a 128-thread warpgroup, 4 values
    # each, over a 64x8 C tile.  CLayout_64x8 from CUTLASS
    # mma_traits_sm90_gmma.hpp, ((4,8,4),(2,2)):((128,1,16),(64,8)) with the
    # trivial N/8 = 1 mode dropped; W = (8, 2, 4, 2, 4) splits the tile's
    # 64-mode in three.
    MmaCompositionExample(
        title="SM90 wgmma m64n8 C accumulator, row-major C",
        tv_shape=(4, 8, 4, 2, 2),
        tv_stride=(128, 1, 16, 64, 8),
        tile_shape=(64, 8),
        tile_stride=(8, 1),
        expected=WeakCompositionExample(
            domain=(4, 8, 4, 2, 2),
            first_codomain=(8, 2, 4, 2, 4),
            first_map=(5, 1, 3, 4, 2),
            second_domain=(64, 8),
            codomain=(8, 64),
            second_map=(2, 1),
        ),
    ),
    # The A operand of the same sm80 atom: 8 values per thread over the 16x16
    # A tile, ((4,8),(2,2,2)):((32,1),(16,8,128)) from CUTLASS
    # mma_traits_sm80.hpp.  Row-major is A's natural (TN) layout, so unlike
    # the accumulators this composite is the one the kernel actually takes.
    MmaCompositionExample(
        title="SM80 m16n8k16 A-operand, row-major A",
        tv_shape=(4, 8, 2, 2, 2),
        tv_stride=(32, 1, 16, 8, 128),
        tile_shape=(16, 16),
        tile_stride=(16, 1),
        expected=WeakCompositionExample(
            domain=(4, 8, 2, 2, 2),
            first_codomain=(8, 2, 2, 4, 2),
            first_map=(4, 1, 3, 2, 5),
            second_domain=(16, 16),
            codomain=(16, 16),
            second_map=(2, 1),
        ),
    ),
)


class MmaWeakCompositionTest(WeakCompositionTest):
    """Weakly compose MMA thread-value layouts with row-major tiles."""

    def construct(self) -> None:
        for index, example in enumerate(EXAMPLES):
            title = Text(
                example.title,
                color=INK,
                font=CODE_FONT,
                font_size=LABEL_FONT_SIZE,
            ).to_edge(UP, buff=0.2)
            self.play(FadeIn(title), run_time=0.4)
            self._show_weak_composite(example.weak_example())
            self.clear_scene(last=index == len(EXAMPLES) - 1)
