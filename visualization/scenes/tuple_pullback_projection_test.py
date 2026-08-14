"""Isolated pullback examples for domain modes that are projected away.

``f`` sends such a mode to the basepoint, so it has no cell in ``T``, nothing in
the refinement of ``T`` divides it, and it draws no arrow: in the opening frame
it sits unconnected.  The pullback keeps it regardless —
``f.pullback_along(T')`` leaves that mode of ``S'`` exactly as it stands in
``S``, unrefined, with ``f'`` sending it to the basepoint too — so a cell for it
has to reach the middle stack somehow.

It travels there out of its own ``S`` cell, during the same reordering that
permutes the blocks into the order of ``S'``: that reordering leaves its slot
free, and it arrives as its fan grows behind it.  A mapped mode enters ``S'`` by
peeling upward out of a ``T`` cell, so the two ways a cell can arrive read
differently, and neither invents one at an empty slot.

Only the example list differs from ``tuple_pullback_test``; the choreography is
that scene's.  The examples put the projected-away mode at the bottom, in the
middle and at the top of ``S``, alone and in pairs, with and without a reordering
of the modes that survive.  Every codomain mode is hit, so nothing else is going
on.
"""

from scenes.tuple_pullback_test import PullbackExample, TuplePullbackTest


EXAMPLES = (
    # S = (6, 5) -> T = (6,) over (1, 0): the top mode is projected away.
    PullbackExample(((2, 3),), (1, 0), domain=(6, 5)),
    # S = (5, 6) -> T = (6,) over (0, 1): the bottom mode is projected away, so
    # the block that survives has to make room above it.
    PullbackExample(((2, 3),), (0, 1), domain=(5, 6)),
    # S = (6, 5, 4) -> T = (4, 6) over (2, 0, 1): a projected-away mode between
    # two that are reordered.
    PullbackExample(((2, 2), (2, 3)), (2, 0, 1), domain=(6, 5, 4)),
    # S = (7, 6, 5) -> T = (6,) over (0, 1, 0): two projected-away modes, one
    # either side of the only mode f maps.
    PullbackExample(((2, 3),), (0, 1, 0), domain=(7, 6, 5)),
    # S = (6, 5, 12, 7) -> T = (12, 6) over (2, 0, 1, 0): two projected-away
    # modes, two blocks of different sizes, and a reordering.
    PullbackExample(((2, 2, 3), (2, 3)), (2, 0, 1, 0), domain=(6, 5, 12, 7)),
)


class TuplePullbackProjectionTest(TuplePullbackTest):
    """Pull back a morphism that projects some domain modes away."""

    def construct(self) -> None:
        for index, example in enumerate(EXAMPLES):
            self._show_pullback(example)
            self.clear_scene(last=index == len(EXAMPLES) - 1)
