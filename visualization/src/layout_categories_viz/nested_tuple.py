"""Tree diagrams for nested tuples."""

from dataclasses import dataclass
from math import prod

import numpy as np
from manim import CubicBezier, LEFT, RIGHT, VGroup

from .style import INK
from .tuple_morphism import TupleMorphismDiagram


@dataclass
class _TreeNode:
    value: int
    children: tuple["_TreeNode", ...]
    depth: int
    leaf_index: int | None = None
    y: float = 0.0
    mobject: VGroup | None = None


class NestedTupleTree(VGroup):
    """A nested tuple drawn as a horizontal forest with its root omitted."""

    def __init__(
        self,
        nested_tuple,
        *,
        direction=RIGHT,
        cell_size=0.66,
        level_gap=1.28,
        leaf_gap=0.92,
        label_font_size=28,
        edge_stroke_width=3.6,
    ) -> None:
        super().__init__()
        if direction[0] == 0:
            raise ValueError("nested-tuple trees require a horizontal direction")
        if not isinstance(nested_tuple, tuple) or not nested_tuple:
            raise ValueError("the omitted root must be a nonempty tuple")

        self.nested_tuple = nested_tuple
        self.direction = RIGHT if direction[0] > 0 else LEFT
        self.cell_size = cell_size
        self.level_gap = level_gap
        self.leaf_gap = leaf_gap

        leaf_counter = [0]
        self.roots = tuple(
            self._parse(component, 0, leaf_counter)
            for component in nested_tuple
        )
        self._position_nodes(self.roots)

        all_nodes = tuple(self._walk(self.roots))
        max_depth = max(node.depth for node in all_nodes)
        for node in all_nodes:
            node_font_size = label_font_size * min(
                1.0,
                2.1 / len(str(node.value)),
            )
            node.mobject = TupleMorphismDiagram._make_entries(
                (node.value,),
                cell_size,
                cell_size,
                node_font_size,
            )[0]
            target_level = max_depth - self._height(node)
            node.mobject.move_to(
                np.array(
                    (
                        self.direction[0] * target_level * level_gap,
                        node.y,
                        0.0,
                    )
                )
            )

        # Rebuild the depth groups now that every node has its mobject.
        self.nodes_by_depth = tuple(
            VGroup(
                *(
                    node.mobject
                    for node in all_nodes
                    if node.depth == depth
                )
            )
            for depth in range(max_depth + 1)
        )
        edges_by_depth = [VGroup() for _ in range(max_depth + 1)]
        for parent in all_nodes:
            for child in parent.children:
                if self.direction[0] > 0:
                    start = parent.mobject.get_right()
                    end = child.mobject.get_left()
                else:
                    # Store mirrored edges from leaf-side to root-side so
                    # ``Create`` still draws every tree edge left to right.
                    start = child.mobject.get_right()
                    end = parent.mobject.get_left()
                handle = RIGHT * 0.42 * abs(end[0] - start[0])
                edge = CubicBezier(
                    start,
                    start + handle,
                    end - handle,
                    end,
                    color=INK,
                    stroke_width=edge_stroke_width,
                )
                edges_by_depth[child.depth].add(edge)
        self.edges_by_depth = tuple(edges_by_depth)
        self.edges = VGroup(*(edge for group in self.edges_by_depth for edge in group))
        self.nodes = VGroup(*(node.mobject for node in all_nodes))
        self.leaf_entries = VGroup(
            *(
                node.mobject
                for node in sorted(
                    (node for node in all_nodes if node.leaf_index is not None),
                    key=lambda node: node.leaf_index,
                )
            )
        )

        # Edges are added first so node cells remain crisp at attachment points.
        self.add(self.edges, self.nodes)
        self.move_to(np.zeros(3))

    @classmethod
    def _parse(cls, value, depth, leaf_counter):
        if isinstance(value, int):
            node = _TreeNode(value, (), depth, leaf_index=leaf_counter[0])
            leaf_counter[0] += 1
            return node
        if not isinstance(value, tuple) or not value:
            raise ValueError("nested tuple components must be integers or nonempty tuples")
        children = tuple(
            cls._parse(child, depth + 1, leaf_counter) for child in value
        )
        return _TreeNode(prod(child.value for child in children), children, depth)

    def _position_nodes(self, roots):
        def position(node):
            if not node.children:
                node.y = node.leaf_index * self.leaf_gap
            else:
                for child in node.children:
                    position(child)
                node.y = sum(child.y for child in node.children) / len(node.children)

        for root in roots:
            position(root)

    @classmethod
    def _walk(cls, roots):
        for node in roots:
            yield node
            yield from cls._walk(node.children)

    @classmethod
    def _height(cls, node):
        if not node.children:
            return 0
        return 1 + max(cls._height(child) for child in node.children)
