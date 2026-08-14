"""Nested tuples: the objects of the categories Nest and Ref."""
# *************************************************************************
# NESTED TUPLES
# *************************************************************************


class NestedTuple:
    """
    Nested tuple structure for hierarchical data representation.

    Wraps an integer or an arbitrarily nested tuple of integers, with
    operations for flattening, substitution, and refinement checking.
    """
    
    def __init__(self, data):
        if not self._validate(data):
            raise ValueError("Only integers or nested tuples of integers are allowed.")
        self.data = data

    def _validate(self, obj):
        """Recursively validate the nested structure."""
        if isinstance(obj, int):
            return True
        elif isinstance(obj, tuple):
            return all(self._validate(item) for item in obj)
        return False

    def _custom_repr(self, obj):
        """Custom representation without trailing commas for single elements."""
        if isinstance(obj, int):
            return str(obj)
        elif isinstance(obj, tuple):
            if not obj:
                return "()"
            elif len(obj) == 1:
                return f"({self._custom_repr(obj[0])})"
            else:
                inner = ",".join(self._custom_repr(item) for item in obj)
                return f"({inner})"

    def __repr__(self):
        return self._custom_repr(self.data)

    def __str__(self):
        return repr(self)

    def _flatten(self, obj):
        """Generator for flattening nested structure."""
        if isinstance(obj, int):
            yield obj
        elif isinstance(obj, tuple):
            for item in obj:
                yield from self._flatten(item)

    def __eq__(self, other):
        """Structural equality on the underlying nested data."""
        if not isinstance(other, NestedTuple):
            return NotImplemented
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def flatten(self) -> tuple:
        """Return flattened tuple of all integers."""
        return tuple(self._flatten(self.data))

    def __iter__(self):
        return iter(self.flatten())

    def __getitem__(self, index):
        return self.flatten()[index]

    def length(self) -> int:
        """Number of integers in flattened representation."""
        return len(self.flatten())

    def rank(self) -> int:
        """Number of top-level modes."""
        if isinstance(self.data, int):
            return 1
        return len(self.data)

    def size(self) -> int:
        """Product of all integers in the nested tuple."""
        size = 1
        for entry in self.flatten():
            size *= entry
        return size

    def entry(self, i: int):
        """Get i-th entry (1-indexed)."""
        if i < 1 or i > self.length():
            raise IndexError("Index out of range")
        return self[i - 1]

    def mode(self, i: int) -> "NestedTuple":
        """Get i-th mode as a NestedTuple (1-indexed)."""
        if not isinstance(i, int) or i < 1:
            raise IndexError("Mode index must be a positive integer.")

        if isinstance(self.data, int):
            if i == 1:
                return NestedTuple(self.data)
            else:
                raise IndexError("An integer NestedTuple S has only one mode.")

        if i > self.rank():
            raise IndexError(f"Mode index {i} out of range.")

        return NestedTuple(self.data[i - 1])
    
    def depth(self) -> int:
        """
        Maximum depth of nesting in the nested tuple.

        An integer has depth 1, a flat tuple has depth 1,
        a nested tuple has depth 1 + max depth of its modes.
        """
        if isinstance(self.data, int):
            return 0
        
        max_depth = 0
        for i in range(1, self.rank() + 1):
            mode_depth = self.mode(i).depth()
            max_depth = max(max_depth, mode_depth)
        
        return max_depth + 1

    def sub(self, values: tuple) -> "NestedTuple":
        """Substitute values into the nested structure; values must have the same length as the flattening."""
        if len(values) != self.length():
            raise ValueError("Replacement tuple must have same length as NestedTuple")

        it = iter(values)

        def _replace(obj):
            if isinstance(obj, int):
                return next(it)
            elif isinstance(obj, tuple):
                return tuple(_replace(item) for item in obj)

        new_data = _replace(self.data)
        return NestedTuple(new_data)

    def profile(self) -> "NestedTuple":
        """Return profile (structure with all zeros)."""
        return self.sub(tuple([0] * self.length()))

    def is_congruent_to(self, other: "NestedTuple") -> bool:
        """Check if two NestedTuples have the same profile."""
        return self.profile().data == other.profile().data

    def replace_empty_tuples_with_one(self) -> "NestedTuple":
        """Replace all empty tuples with 1."""
        def _replace(obj):
            if obj == ():
                return 1
            elif isinstance(obj, int):
                return obj
            elif isinstance(obj, tuple):
                return tuple(_replace(item) for item in obj)
            else:
                raise ValueError("Invalid element type.")

        new_data = _replace(self.data)
        return NestedTuple(new_data)

    def replace_empty_tuples_with_zero(self) -> "NestedTuple":
        """Replace all empty tuples with 0."""
        def _replace(obj):
            if obj == ():
                return 0
            elif isinstance(obj, int):
                return obj
            elif isinstance(obj, tuple):
                return tuple(_replace(item) for item in obj)
            else:
                raise ValueError("Invalid element type.")

        new_data = _replace(self.data)
        return NestedTuple(new_data)

    def refines(self, other: "NestedTuple") -> bool:
        """
        Check if self refines other.

        S refines T if:
        1. S = T, or
        2. T = size(S), or
        3. rank(S) = rank(T) and mode_i(S) refines mode_i(T) for all i
        """
        # Case 1: S = T
        if self.data == other.data:
            return True
        # Case 2: T = size(S)
        if isinstance(other.data, int) and other.data == self.size():
            return True
        # Case 3: rank(S) = rank(T) and mode_i(S) refines mode_i(T) for all i
        if self.rank() == other.rank():
            for i in range(1, self.rank() + 1):
                if not self.mode(i).refines(other.mode(i)):
                    return False
            return True
        return False

    def is_refined_by(self, other: "NestedTuple") -> bool:
        """Check if self is refined by other."""
        return other.refines(self)

    def relative_mode(self, i: int, other: "NestedTuple") -> "NestedTuple":
        """Get i-th relative mode (1-indexed) with respect to another NestedTuple that self refines."""
        assert self.refines(other), "Self must refine other"
        assert 1 <= i <= other.length()
        
        if i == 1 and other.data == self.size():
            return self
        else:
            l = 0
            N = 0
            while N + other.mode(l + 1).length() < i:
                l += 1
                N += other.mode(l).length()
            l += 1
            return self.mode(l).relative_mode(i - N, other.mode(l))

    def relative_flattening(self, other: "NestedTuple") -> "NestedTuple":
        """Compute relative flattening with respect to another NestedTuple that self refines."""
        assert self.refines(other)
        result_list = []
        for i in range(1, other.length() + 1):
            result_list.append(self.relative_mode(i, other).data)
        return NestedTuple(tuple(result_list))

    def underlying_map(self, other: "NestedTuple") -> tuple:
        """Compute underlying map for the refinement of other by self."""
        assert self.refines(other)
        map_ = []
        for i in range(1, other.length() + 1):
            for _ in range(self.relative_mode(i, other).length()):
                map_.append(i)
        return tuple(map_)

    def sublength(self, i: int, refined: "NestedTuple") -> int:
        """Compute sublength up to position i (1-indexed)."""
        assert 1 <= i <= self.length()
        result = 0
        for j in range(1, i):
            result += self.relative_mode(j, refined).length()
        return result

