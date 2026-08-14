"""
Spans and cospans of Tuple morphisms over a category of refinements.

All four categories share one construction. Objects are flat tuples of
positive integers. A span morphism U → V is

    U ←b— X —f→ V

with backward leg b: X ↠ U in the refinement category (Fact or Ref) and
forward leg f: X → V a Tuple morphism, sharing the apex X. A cospan
morphism is the mirror image

    U —f→ X ←b— V

with the backward leg out of the nadir X. Composition pulls the forward
leg back along the next backward leg (spans), or pushes the next forward
leg forward along the backward leg (cospans); either way the chosen
(co)limit refines each apex/nadir entry, making composition strictly
associative and unital.

Concrete categories:

- :class:`SpanMorphism`      — Span(Tuple, Fact)
- :class:`CoSpanMorphism`    — CoSpan(Tuple, Fact)
- :class:`RefSpanMorphism`   — Span(Tuple, Ref)
- :class:`RefCoSpanMorphism` — CoSpan(Tuple, Ref)

A Fact morphism is a Ref morphism with flat top-level modes; flattening
the backward legs gives functors Span(Tuple, Ref) → Span and
CoSpan(Tuple, Ref) → CoSpan (``to_span_morphism`` / ``to_cospan_morphism``).
"""

from .fact_morphism import FactMorphism
from .ref_morphism import RefMorphism
from .tuple_morphism import TupleMorphism


class _LegwiseBase:
    """Shared scaffolding for span/cospan morphisms: two legs and a name."""

    #: Refinement category of the backward leg (FactMorphism or RefMorphism).
    backward_cls: type = None
    #: Attribute of the backward leg shown in __str__ ("modes" or "nest").
    backward_repr_attr: str = "modes"

    def __init__(self, left, right, name: str = ""):
        self.left = left
        self.right = right
        self.name = name
        self._validate_inputs()

    def __repr__(self):
        return f"{type(self).__name__}(left={self.left!r}, right={self.right!r})"

    def __eq__(self, other):
        """Structural equality on both legs; names are ignored."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash((self.left, self.right))

    def are_composable(self, g) -> bool:
        """Check whether ``g`` can follow ``self``."""
        return self.codomain == g.domain

    def sum(self, g):
        """Compute the sum f ⊕ g, taken legwise."""
        return type(self)(self.left.sum(g.left), self.right.sum(g.right))


class _SpanBase(_LegwiseBase):
    """
    A span U ←b— X —f→ V: backward (left) leg b: X ↠ U in the refinement
    category, forward (right) leg f: X → V a Tuple morphism, apex
    X = b.domain = f.domain.

    Composition of U ←b₁— X —f₁→ V and V ←b₂— Y —f₂→ W pulls f₁ back along
    b₂ to a square with corner X′, then composes the legs:

        X′ --f₁′--> Y --f₂--> W
        ↓r          ↓b₂
        X --f₁----> V
        ↓b₁
        U

    giving the span U ←(b₁ ∘ r)— X′ —(f₂ ∘ f₁′)→ W.
    """

    def __init__(self, left, right: TupleMorphism, name: str = ""):
        super().__init__(left, right, name)
        self.apex = left.domain
        self.domain = left.codomain
        self.codomain = right.codomain

    def _validate_inputs(self) -> None:
        """:raises ValueError: If the legs do not form a valid span."""
        if not isinstance(self.left, self.backward_cls):
            raise ValueError(
                f"Left leg must be a {self.backward_cls.__name__}, "
                f"got {type(self.left).__name__}"
            )
        if not isinstance(self.right, TupleMorphism):
            raise ValueError(
                f"Right leg must be a TupleMorphism, got {type(self.right).__name__}"
            )
        if self.left.domain != self.right.domain:
            raise ValueError(
                f"Legs must share an apex: left leg has domain "
                f"{self.left.domain}, right leg has domain {self.right.domain}"
            )

    def __str__(self):
        backward = getattr(self.left, self.backward_repr_attr)
        return (
            f"{self.domain} <--{backward}-- {self.apex} "
            f"--{self.right.map}--> {self.codomain}"
        )

    @classmethod
    def identity(cls, obj, name: str = ""):
        """Identity span U ←id— U —id→ U on a flat tuple U."""
        return cls(cls.backward_cls.identity(obj), TupleMorphism.identity(obj), name)

    def is_identity(self) -> bool:
        """Check whether both legs are identities."""
        return self.left.is_identity() and self.right.is_identity()

    def compose(self, g):
        """
        Compute the composition g ∘ f, where f = self, by pulling f's forward
        leg back along g's backward leg and composing legs around the
        resulting square.

        :raises ValueError: If not composable
        """
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")
        refinement, pulled = g.left.pullback_with_refinement(self.right)
        return type(self)(refinement.compose(self.left), pulled.compose(g.right))


class _CoSpanBase(_LegwiseBase):
    """
    A cospan U —f→ X ←b— V: forward (left) leg f: U → X a Tuple morphism,
    backward (right) leg b: X ↠ V in the refinement category (pointing
    backward, from the nadir to V), nadir X = f.codomain = b.domain.

    Composition of U —f₁→ X ←b₁— V and V —f₂→ Y ←b₂— W pushes f₂ forward
    along b₁ to a square with corner Y′, then composes the legs:

        U --f₁--> X --f₂′--> Y′
                  ↓b₁        ↓r
                  V --f₂---> Y
                             ↑b₂
                             W

    giving the cospan U —(f₂′ ∘ f₁)→ Y′ ←(b₂ ∘ r)— W (composites written in
    application order).
    """

    def __init__(self, left: TupleMorphism, right, name: str = ""):
        super().__init__(left, right, name)
        self.nadir = left.codomain
        self.domain = left.domain
        self.codomain = right.codomain

    def _validate_inputs(self) -> None:
        """:raises ValueError: If the legs do not form a valid cospan."""
        if not isinstance(self.left, TupleMorphism):
            raise ValueError(
                f"Left leg must be a TupleMorphism, got {type(self.left).__name__}"
            )
        if not isinstance(self.right, self.backward_cls):
            raise ValueError(
                f"Right leg must be a {self.backward_cls.__name__}, "
                f"got {type(self.right).__name__}"
            )
        if self.left.codomain != self.right.domain:
            raise ValueError(
                f"Legs must share a nadir: left leg has codomain "
                f"{self.left.codomain}, right leg has domain {self.right.domain}"
            )

    def __str__(self):
        backward = getattr(self.right, self.backward_repr_attr)
        return (
            f"{self.domain} --{self.left.map}--> {self.nadir} "
            f"<--{backward}-- {self.codomain}"
        )

    @classmethod
    def identity(cls, obj, name: str = ""):
        """Identity cospan U —id→ U ←id— U on a flat tuple U."""
        return cls(TupleMorphism.identity(obj), cls.backward_cls.identity(obj), name)

    def is_identity(self) -> bool:
        """Check whether both legs are identities."""
        return self.left.is_identity() and self.right.is_identity()

    def compose(self, g):
        """
        Compute the composition g ∘ f, where f = self, by pushing g's forward
        leg forward along f's backward leg and composing legs around the
        resulting square.

        :raises ValueError: If not composable
        """
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")
        refinement, pushed = self.right.pushforward_with_refinement(g.left)
        return type(self)(self.left.compose(pushed), refinement.compose(g.right))


class SpanMorphism(_SpanBase):
    """Morphisms in Span(Tuple, Fact): U ←b— X —f→ V with b a Fact morphism."""

    backward_cls = FactMorphism
    backward_repr_attr = "modes"


class CoSpanMorphism(_CoSpanBase):
    """Morphisms in CoSpan(Tuple, Fact): U —f→ X ←b— V with b a Fact morphism."""

    backward_cls = FactMorphism
    backward_repr_attr = "modes"


class RefSpanMorphism(_SpanBase):
    """
    Morphisms in Span(Tuple, Ref): U ←b— X —f→ V with b a Ref morphism.

    The backward leg is presented by a nested tuple N whose flattening is
    the apex X and whose depth-1 reduction is U, so the apex carries an
    arbitrary nested factorization of the domain.
    """

    backward_cls = RefMorphism
    backward_repr_attr = "nest"

    def to_span_morphism(self) -> SpanMorphism:
        """
        The underlying Span morphism, obtained by flattening the backward
        leg's top-level modes to a Fact morphism. This is a functor
        Span(Tuple, Ref) → Span: it preserves identities, composition, and
        sums, and is the identity on objects.
        """
        return SpanMorphism(self.left.to_fact_morphism(), self.right)

    @classmethod
    def from_span_morphism(cls, f: SpanMorphism, name: str = "") -> "RefSpanMorphism":
        """
        The RefSpan morphism presented by a Span morphism, with backward
        leg the depth ≤ 2 Ref morphism of the Fact leg's modes. Satisfies
        from_span_morphism(f).to_span_morphism() == f.
        """
        return cls(RefMorphism.from_fact_morphism(f.left), f.right, name)

    # Deprecated aliases
    to_Span_morphism = to_span_morphism
    from_Span_morphism = from_span_morphism


class RefCoSpanMorphism(_CoSpanBase):
    """
    Morphisms in CoSpan(Tuple, Ref): U —f→ X ←b— V with b a Ref morphism.

    The backward leg is presented by a nested tuple N whose flattening is
    the nadir X and whose depth-1 reduction is V, so the nadir carries an
    arbitrary nested factorization of the codomain.
    """

    backward_cls = RefMorphism
    backward_repr_attr = "nest"

    def to_cospan_morphism(self) -> CoSpanMorphism:
        """
        The underlying CoSpan morphism, obtained by flattening the backward
        leg's top-level modes to a Fact morphism. This is a functor
        CoSpan(Tuple, Ref) → CoSpan: it preserves identities, composition,
        and sums, and is the identity on objects.
        """
        return CoSpanMorphism(self.left, self.right.to_fact_morphism())

    @classmethod
    def from_cospan_morphism(
        cls, f: CoSpanMorphism, name: str = ""
    ) -> "RefCoSpanMorphism":
        """
        The RefCoSpan morphism presented by a CoSpan morphism, with backward
        leg the depth ≤ 2 Ref morphism of the Fact leg's modes. Satisfies
        from_cospan_morphism(f).to_cospan_morphism() == f.
        """
        return cls(f.left, RefMorphism.from_fact_morphism(f.right), name)

    # Deprecated aliases
    to_CoSpan_morphism = to_cospan_morphism
    from_CoSpan_morphism = from_cospan_morphism
