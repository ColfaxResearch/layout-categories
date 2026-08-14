"""
The category Fact for Tract library.

This module implements morphisms in the category Fact, whose objects are
flat tuples of positive integers and whose morphisms are entrywise
factorizations. A morphism (u₁,...,u_p) → (t₁,...,t_n) is a tuple
(F₁,...,F_n) such that each Fᵢ is a flat tuple of positive integers whose
entries multiply to tᵢ, and the concatenation of F₁,...,F_n equals
(u₁,...,u_p). Such a morphism is precisely a flat refinement
(u₁,...,u_p) ↠ (t₁,...,t_n) recorded by its relative modes.

Kept separate from categories.py for now while the theory is developed.
"""

from typing import Tuple

from .categories import NestedTuple, Tuple_morphism


# *************************************************************************
# THE CATEGORY Fact
# *************************************************************************


class Fact_morphism:
    """
    Morphisms in the category Fact.

    A morphism F: (u₁,...,u_p) → (t₁,...,t_n) is encoded by its modes
    (F₁,...,F_n), where each Fᵢ is a flat tuple of positive integers with
    prod(Fᵢ) = tᵢ, and flat(F₁,...,F_n) = (u₁,...,u_p). Equivalently, the
    domain is partitioned into consecutive blocks, one per codomain entry,
    with each block's product equal to that entry.

    The identity on (t₁,...,tₙ) is ((t₁),...,(tₙ)), and composition
    concatenates blocks of blocks.

    :param domain: Domain tuple
    :type domain: Tuple[int]
    :param codomain: Codomain tuple
    :type codomain: Tuple[int]
    :param modes: Tuple (F₁,...,F_n) of factorizations, one per codomain entry
    :type modes: Tuple[Tuple[int]]
    :param name: Optional name
    :type name: str
    """

    def __init__(
        self,
        domain: Tuple[int],
        codomain: Tuple[int],
        modes: Tuple[Tuple[int]],
        name: str = "",
    ):
        self.domain = domain
        self.codomain = codomain
        self.modes = modes
        self.name = name
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Verify that the input data defines a valid morphism in the Fact category.

        :raises ValueError: If morphism is invalid
        """
        for entry in self.domain:
            if not isinstance(entry, int) or entry < 1:
                raise ValueError(
                    f"Domain entries must be positive integers, got {entry}"
                )

        for entry in self.codomain:
            if not isinstance(entry, int) or entry < 1:
                raise ValueError(
                    f"Codomain entries must be positive integers, got {entry}"
                )

        for mode in self.modes:
            if not isinstance(mode, tuple):
                raise ValueError(f"Each mode must be a tuple, got {mode}")
            for entry in mode:
                if not isinstance(entry, int) or entry < 1:
                    raise ValueError(
                        f"Mode entries must be positive integers, got {entry}"
                    )

        if len(self.modes) != len(self.codomain):
            raise ValueError(
                f"Number of modes ({len(self.modes)}) must match codomain length "
                f"({len(self.codomain)})"
            )

        for i, mode in enumerate(self.modes):
            product = 1
            for entry in mode:
                product *= entry
            if product != self.codomain[i]:
                raise ValueError(
                    f"Must satisfy prod(F_i) = t_i for all i: mode {i + 1} has "
                    f"product {product}, but codomain entry is {self.codomain[i]}"
                )

        flattening = tuple(entry for mode in self.modes for entry in mode)
        if flattening != self.domain:
            raise ValueError(
                f"Flattening of modes ({flattening}) must equal domain "
                f"({self.domain})"
            )

    def __repr__(self):
        return f"Fact_morphism(domain={self.domain}, codomain={self.codomain}, modes={self.modes})"

    def __str__(self):
        return f"{self.domain} --{self.modes}--> {self.codomain}"

    def __eq__(self, other):
        """
        Structural equality on (domain, codomain, modes); names are ignored.

        :param other: Object to compare against
        :return: True if other is a Fact_morphism with the same data
        :rtype: bool
        """
        if not isinstance(other, Fact_morphism):
            return NotImplemented
        return (
            self.domain == other.domain
            and self.codomain == other.codomain
            and self.modes == other.modes
        )

    def __hash__(self):
        return hash((self.domain, self.codomain, self.modes))

    def size(self) -> int:
        """
        Product of domain entries. Morphisms in Fact preserve size, so
        size() == cosize() always.

        :return: Size of domain
        :rtype: int
        """
        size = 1
        for entry in self.domain:
            size *= entry
        return size

    def cosize(self) -> int:
        """
        Product of codomain entries. Morphisms in Fact preserve size, so
        size() == cosize() always.

        :return: Size of codomain
        :rtype: int
        """
        cosize = 1
        for entry in self.codomain:
            cosize *= entry
        return cosize

    @classmethod
    def identity(cls, codomain: Tuple[int], name: str = "") -> "Fact_morphism":
        """
        Identity morphism on (t₁,...,tₙ), with modes ((t₁),...,(tₙ)).

        :param codomain: Object to take the identity of
        :type codomain: Tuple[int]
        :param name: Optional name
        :type name: str
        :return: Identity morphism
        :rtype: Fact_morphism
        """
        return cls(codomain, codomain, tuple((t,) for t in codomain), name)

    def is_identity(self) -> bool:
        """
        Check if the morphism is an identity, i.e. every mode is a singleton.

        :return: True if identity
        :rtype: bool
        """
        return all(len(mode) == 1 for mode in self.modes)

    def are_composable(self, g: "Fact_morphism") -> bool:
        """
        Check if morphisms are composable.

        :param g: Second morphism
        :type g: Fact_morphism
        :return: True if composable
        :rtype: bool
        """
        return self.codomain == g.domain

    def compose(self, g: "Fact_morphism") -> "Fact_morphism":
        """
        Compute composition g ∘ f, where f = self.

        If f: U → T has modes (F₁,...,F_p) and g: T → S has modes
        (G₁,...,G_n), the composite U → S has j-th mode the concatenation of
        the Fᵢ over the len(G_j) consecutive entries of T covered by G_j.

        :param g: Second morphism (must have domain = self.codomain)
        :type g: Fact_morphism
        :return: The composition g ∘ f
        :rtype: Fact_morphism
        :raises ValueError: If not composable
        """
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")

        composite_modes = []
        index = 0
        for mode in g.modes:
            block = ()
            for _ in range(len(mode)):
                block += self.modes[index]
                index += 1
            composite_modes.append(block)

        return Fact_morphism(self.domain, g.codomain, tuple(composite_modes))

    def sum(self, g: "Fact_morphism") -> "Fact_morphism":
        """
        Compute sum f ⊕ g.

        :param g: Second morphism
        :type g: Fact_morphism
        :return: Sum of morphisms
        :rtype: Fact_morphism
        """
        return Fact_morphism(
            self.domain + g.domain,
            self.codomain + g.codomain,
            self.modes + g.modes,
        )

    def _sublengths(self) -> Tuple[int]:
        """
        Prefix sums of mode lengths: entry j is the number of domain entries
        lying over codomain entries 1,...,j.

        :return: Tuple of prefix sums, of length len(codomain) + 1
        :rtype: Tuple[int]
        """
        sublengths = [0]
        for mode in self.modes:
            sublengths.append(sublengths[-1] + len(mode))
        return tuple(sublengths)

    def pullback(self, f: Tuple_morphism) -> Tuple_morphism:
        """
        Pull back a Tuple morphism f: S → T along self: T′ ↠ T, viewing self
        as the flat refinement of T = codomain by T′ = domain.

        The result f′: S′ → T′ replaces each domain entry sᵢ of f with
        α(i) = j ≠ * by the mode F_j (of which sᵢ = t_j is the product),
        mapped to the corresponding consecutive positions of T′; entries with
        α(i) = * are unchanged. This is the flat special case of
        Nest_morphism.pullback_along.

        :param f: Tuple morphism with codomain equal to self.codomain
        :type f: Tuple_morphism
        :return: The pullback of f along self
        :rtype: Tuple_morphism
        :raises ValueError: If f's codomain does not match
        """
        if f.codomain != self.codomain:
            raise ValueError(
                f"Codomain of f ({f.codomain}) must equal codomain of the "
                f"Fact morphism ({self.codomain})."
            )

        sublengths = self._sublengths()
        domain = []
        map_ = []
        for i, j in enumerate(f.map):
            if j != 0:
                mode = self.modes[j - 1]
                domain.extend(mode)
                map_.extend(sublengths[j - 1] + k + 1 for k in range(len(mode)))
            else:
                domain.append(f.domain[i])
                map_.append(0)

        return Tuple_morphism(tuple(domain), self.domain, tuple(map_))

    def pullback_with_refinement(self, f: Tuple_morphism):
        """
        Pull back a Tuple morphism f: S → T along self: T′ ↠ T, returning
        both the induced refinement of the domain and the pulled-back
        morphism.

        The pullback square is

            S′ --f′--> T′
            ↓r         ↓self
            S ---f---> T

        where r: S′ ↠ S is the Fact morphism refining each domain entry sᵢ
        with α(i) = j ≠ * by the mode F_j, and leaving entries with
        α(i) = * unrefined.

        :param f: Tuple morphism with codomain equal to self.codomain
        :type f: Tuple_morphism
        :return: Pair (r, f′) of the induced Fact morphism r: S′ ↠ S and the
            pullback f′: S′ → T′
        :rtype: tuple[Fact_morphism, Tuple_morphism]
        :raises ValueError: If f's codomain does not match
        """
        pullback = self.pullback(f)
        modes = tuple(
            self.modes[j - 1] if j != 0 else (f.domain[i],)
            for i, j in enumerate(f.map)
        )
        refinement = Fact_morphism(pullback.domain, f.domain, modes)
        return refinement, pullback

    def pushforward(self, f: Tuple_morphism) -> Tuple_morphism:
        """
        Push forward a Tuple morphism f: U → V along self: U′ ↠ U, viewing
        self as the flat refinement of U = codomain by U′ = domain.

        The result f′: U′ → V′ replaces each codomain entry v_j of f in the
        image of α (say v_j = u_i with α(i) = j) by the mode Fᵢ, and maps the
        entries of U′ over uᵢ to the corresponding consecutive positions of
        V′; codomain entries outside the image are unchanged. This is the
        flat special case of Nest_morphism.pushforward_along.

        :param f: Tuple morphism with domain equal to self.codomain
        :type f: Tuple_morphism
        :return: The pushforward of f along self
        :rtype: Tuple_morphism
        :raises ValueError: If f's domain does not match
        """
        if f.domain != self.codomain:
            raise ValueError(
                f"Domain of f ({f.domain}) must equal codomain of the "
                f"Fact morphism ({self.codomain})."
            )

        # Each codomain entry of f contributes a block to the new codomain:
        # its refining mode if it is hit by the map, itself otherwise.
        blocks = []
        for j in range(1, len(f.codomain) + 1):
            if j in f.map:
                blocks.append(self.modes[f.map.index(j)])
            else:
                blocks.append((f.codomain[j - 1],))
        codomain = tuple(entry for block in blocks for entry in block)

        sublengths = [0]
        for block in blocks:
            sublengths.append(sublengths[-1] + len(block))

        map_ = []
        for i, j in enumerate(f.map):
            if j == 0:
                map_.extend(0 for _ in range(len(self.modes[i])))
            else:
                map_.extend(
                    sublengths[j - 1] + k + 1 for k in range(len(blocks[j - 1]))
                )

        return Tuple_morphism(self.domain, codomain, tuple(map_))

    def pushforward_with_refinement(self, f: Tuple_morphism):
        """
        Push forward a Tuple morphism f: U → V along self: U′ ↠ U, returning
        both the induced refinement of the codomain and the pushed-forward
        morphism.

        The pushforward square is

            U′ --f′--> V′
            ↓self      ↓r
            U ---f---> V

        where r: V′ ↠ V is the Fact morphism refining each codomain entry
        v_j in the image of α (say v_j = uᵢ with α(i) = j) by the mode Fᵢ,
        and leaving entries outside the image unrefined.

        :param f: Tuple morphism with domain equal to self.codomain
        :type f: Tuple_morphism
        :return: Pair (r, f′) of the induced Fact morphism r: V′ ↠ V and the
            pushforward f′: U′ → V′
        :rtype: tuple[Fact_morphism, Tuple_morphism]
        :raises ValueError: If f's domain does not match
        """
        pushforward = self.pushforward(f)
        blocks = tuple(
            self.modes[f.map.index(j)] if j in f.map else (f.codomain[j - 1],)
            for j in range(1, len(f.codomain) + 1)
        )
        refinement = Fact_morphism(pushforward.codomain, f.codomain, blocks)
        return refinement, pushforward

    def refined_codomain(self) -> NestedTuple:
        """
        The nested tuple (F₁,...,F_n) of modes, which refines the codomain:
        refined_codomain().refines(NestedTuple(codomain)) always holds.

        :return: Modes as a nested tuple
        :rtype: NestedTuple
        """
        return NestedTuple(self.modes)

    @classmethod
    def from_refinement(
        cls, refined: NestedTuple, coarse: NestedTuple, name: str = ""
    ) -> "Fact_morphism":
        """
        Build the Fact morphism flat(refined) → flat(coarse) from a flat
        refinement, i.e. a refinement refined ↠ coarse in which coarse and
        each relative mode of refined are flat.

        :param refined: Refining nested tuple
        :type refined: NestedTuple
        :param coarse: Refined nested tuple
        :type coarse: NestedTuple
        :param name: Optional name
        :type name: str
        :return: Corresponding Fact morphism
        :rtype: Fact_morphism
        :raises ValueError: If refined does not refine coarse, coarse is not
            flat, or some relative mode is not flat
        """
        if coarse.depth() > 1:
            raise ValueError(f"Coarse nested tuple {coarse} must be flat.")
        if not refined.refines(coarse):
            raise ValueError(f"{refined} does not refine {coarse}.")

        modes = []
        for i in range(1, coarse.length() + 1):
            relative_mode = refined.relative_mode(i, coarse)
            if relative_mode.depth() > 1:
                raise ValueError(
                    f"Relative mode {relative_mode} of {refined} relative to "
                    f"{coarse} is not flat."
                )
            modes.append(relative_mode.flatten())

        return cls(refined.flatten(), coarse.flatten(), tuple(modes), name)
