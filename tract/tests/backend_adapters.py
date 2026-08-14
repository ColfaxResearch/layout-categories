"""
Backend adapters for the cross-validation suite: a uniform interface over
the CuTe DSL and pycute layout algebras, so the agreement predicates in
test_cross_validation.py are written once.
"""

from tract.backends import base


def _flat_modes(layout) -> tuple:
    """
    The flat tuple of (shape, stride) modes of a layout, with the stride of
    every size-1 mode nullified and the empty layout normalized to ((1, 0),).
    Works for both cute and pycute layouts.
    """
    shape = base.flatten_nested(tuple(layout.shape) if not isinstance(layout.shape, int) else (layout.shape,))
    stride = base.flatten_nested(tuple(layout.stride) if not isinstance(layout.stride, int) else (layout.stride,))
    modes = tuple((s, d if s != 1 else 0) for s, d in zip(shape, stride))
    return modes if modes else ((1, 0),)


def layouts_agree(layout1, layout2) -> bool:
    """Check whether two layouts agree mode-by-mode after flattening."""
    return _flat_modes(layout1) == _flat_modes(layout2)


class CuteBackend:
    """Adapter over the cutlass CuTe DSL."""

    name = "cute"

    def __init__(self):
        import cutlass.cute as cute
        from tract.backends import cute_dsl

        self.cute = cute
        self.compute_flat_layout = cute_dsl.compute_flat_layout
        self.compute_layout = cute_dsl.compute_layout
        self.compute_Tuple_morphism = cute_dsl.compute_Tuple_morphism
        self.compute_Nest_morphism = cute_dsl.compute_Nest_morphism
        self.flatten_layout = cute_dsl.flatten_layout
        self.flat_concatenate = cute_dsl.flat_concatenate
        self.concatenate = cute_dsl.concatenate
        self.is_tractable = cute_dsl.is_tractable
        self.layouts_agree = layouts_agree

        import cutlass

        @cute.jit
        def _run1(
            pred: cutlass.Constexpr, bk: cutlass.Constexpr, x: cutlass.Constexpr
        ) -> bool:
            return pred(bk, x)

        @cute.jit
        def _run2(
            pred: cutlass.Constexpr,
            bk: cutlass.Constexpr,
            x: cutlass.Constexpr,
            y: cutlass.Constexpr,
        ) -> bool:
            return pred(bk, x, y)

        self._run1 = _run1
        self._run2 = _run2

    def check(self, pred, *args) -> bool:
        """Run an agreement predicate inside a cute.jit trace context,
        where the DSL's layout operations are available."""
        if len(args) == 1:
            return self._run1(pred, self, args[0])
        return self._run2(pred, self, args[0], args[1])

    def coalesce(self, layout):
        return self.cute.coalesce(layout)

    def coalesce_to_profile(self, layout, profile):
        return self.cute.coalesce(layout, target_profile=profile)

    def composition(self, B, A):
        return self.cute.composition(B, A)

    def complement(self, layout, N):
        return self.cute.complement(layout, N)

    def logical_divide(self, A, B):
        return self.cute.logical_divide(A, B)

    def logical_product(self, A, B):
        return self.cute.logical_product(A, B)

    def size(self, layout):
        return self.cute.size(layout)


class PycuteBackend:
    """Adapter over NVIDIA's pycute reference implementation."""

    name = "pycute"

    def __init__(self):
        import pycute

        from tract.backends import pycute as pb

        self._pycute = pycute
        self._pb = pb
        self.compute_flat_layout = pb.compute_flat_layout
        self.compute_layout = pb.compute_layout
        self.compute_Tuple_morphism = pb.compute_Tuple_morphism
        self.compute_Nest_morphism = pb.compute_Nest_morphism
        self.flatten_layout = pb.flatten_layout
        self.flat_concatenate = pb.flat_concatenate
        self.concatenate = pb.concatenate
        self.is_tractable = pb.is_tractable
        self.layouts_agree = pb.layouts_agree

    def check(self, pred, *args) -> bool:
        """Run an agreement predicate directly."""
        return pred(self, *args)

    def coalesce(self, layout):
        return self._pb.coalesce_layout(layout)

    def coalesce_to_profile(self, layout, profile):
        return self._pb.coalesce_layout(layout, profile)

    def composition(self, B, A):
        return self._pb.compose_layouts(B, A)

    def complement(self, layout, N):
        return self._pycute.complement(layout, N)

    def logical_divide(self, A, B):
        return self._pycute.logical_divide(A, B)

    def logical_product(self, A, B):
        return self._pb.logical_product_layouts(A, B)

    def size(self, layout):
        return self._pycute.size(layout)
