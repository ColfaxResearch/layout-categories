"""Shared pytest configuration for the tract test suite."""

import random

import numpy as np


def seed_rngs(seed: int) -> None:
    """Seed both the stdlib and numpy RNGs so failures are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
