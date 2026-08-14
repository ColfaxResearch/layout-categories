"""Prototype animation for composition in the category Span, single-beat.

An experimental variant of span_composition_test.py: the pullback plays
as ONE beat instead of two.  Every middle cell splits and its pieces
travel straight to their X' slots, so the split and the reordering are
a single motion; blocks with no place in X' fade unsplit, and the
basepoint arrivals join the same beat.  Everything else is unchanged --
the scene is the main one with ``single_beat`` set.
"""

from scenes.span_composition_test import SpanMorphismCompositionTest


class SpanMorphismCompositionSingleBeatTest(SpanMorphismCompositionTest):
    """The pullback in one beat: split and reorder as a single motion."""

    single_beat = True
