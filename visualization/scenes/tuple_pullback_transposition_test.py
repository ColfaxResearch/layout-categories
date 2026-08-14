"""Isolated pullback examples for a morphism that reorders modes.

Only the example list differs from ``tuple_pullback_test``: the choreography is
that scene's, so the two cannot drift apart.  Kept for iterating on how the
pullback looks when ``f`` is not order preserving.

    S = (8, 8) --f--> T = (8, 8)      f over (2, 1)
    T' = ((4, 2), (2, 4))             the mutual refinement of T with (4, 4, 4)
    S' = ((2, 4), (4, 2))             f' over (3, 4, 1, 2)
"""

from scenes.tuple_pullback_test import PullbackExample, TuplePullbackTest


EXAMPLES = (
    # S = T = (8, 8) with f the transposition: the weak-composition example.
    PullbackExample(((4, 2), (2, 4)), (2, 1)),
    # S = (6, 6, 4) -> T = (6, 4, 6) over the three-cycle (3, 1, 2).
    PullbackExample(((2, 3), (4,), (3, 2)), (3, 1, 2), domain=(6, 6, 4)),
)


class TuplePullbackTranspositionTest(TuplePullbackTest):
    """Pull a mode-reordering morphism back along a refinement."""

    def construct(self) -> None:
        for index, example in enumerate(EXAMPLES):
            self._show_pullback(example)
            self.clear_scene(last=index == len(EXAMPLES) - 1)
