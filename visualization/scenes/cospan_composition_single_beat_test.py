"""Prototype animation for composition in the category CoSpan, single-beat.

An experimental variant of cospan_composition_test.py: the pushforward
plays as ONE beat instead of two.  Every middle cell splits and its
pieces travel straight to their Y' slots, so the split and the
reordering are a single motion; blocks with no place in Y' fade
unsplit, and the arrivals join the same beat.  Everything else is
unchanged -- the scene is the main one with ``single_beat`` set.
"""

from scenes.cospan_composition_test import CoSpanMorphismCompositionTest


class CoSpanMorphismCompositionSingleBeatTest(CoSpanMorphismCompositionTest):
    """The pushforward in one beat: split and reorder as a single motion."""

    single_beat = True
