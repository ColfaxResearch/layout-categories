from manim import LEFT, RIGHT

from layout_categories_viz import NestMorphismDiagram, NestedTupleTree


def _labels(group):
    return [node[1].text for node in group]


def test_nested_tuple_tree_omits_root_and_labels_internal_products():
    tree = NestedTupleTree(((2, 3), (5, 7)), direction=RIGHT)

    assert _labels(tree.nodes_by_depth[0]) == ["6", "35"]
    assert _labels(tree.nodes_by_depth[1]) == ["2", "3", "5", "7"]
    assert len(tree.nodes) == 6
    assert len(tree.edges) == 4


def test_nested_tuple_tree_mirrors_horizontal_orientation():
    left_to_right = NestedTupleTree((2, (3, (5, 7)), 11), direction=RIGHT)
    right_to_left = NestedTupleTree((2, (3, (5, 7)), 11), direction=LEFT)

    assert left_to_right.nodes_by_depth[1].get_x() > left_to_right.nodes_by_depth[0].get_x()
    assert right_to_left.nodes_by_depth[1].get_x() < right_to_left.nodes_by_depth[0].get_x()
    assert _labels(left_to_right.nodes_by_depth[0]) == ["2", "105", "11"]
    assert _labels(left_to_right.nodes_by_depth[1]) == ["3", "35"]
    assert _labels(left_to_right.nodes_by_depth[2]) == ["5", "7"]

    left_leaf_x = {
        round(node.get_x(), 6)
        for node in left_to_right.nodes
        if node[1].text in {"2", "3", "5", "7", "11"}
    }
    right_leaf_x = {
        round(node.get_x(), 6)
        for node in right_to_left.nodes
        if node[1].text in {"2", "3", "5", "7", "11"}
    }
    assert len(left_leaf_x) == 1
    assert len(right_leaf_x) == 1
    assert all(edge.get_start()[0] < edge.get_end()[0] for edge in left_to_right.edges)
    assert all(edge.get_start()[0] < edge.get_end()[0] for edge in right_to_left.edges)


def test_nest_morphism_uses_tree_leaves_as_flat_tuple_columns():
    diagram = NestMorphismDiagram(
        (2, (3, (5, 7)), 11),
        ((5, 7), (2, 11), 3),
        (3, 5, 1, 2, 4),
    )

    assert _labels(diagram.source_tree.leaf_entries) == ["2", "3", "5", "7", "11"]
    assert _labels(diagram.target_tree.leaf_entries) == ["5", "7", "2", "11", "3"]
    assert len(diagram.arrows) == 5
    assert len({round(node.get_x(), 6) for node in diagram.source_tree.leaf_entries}) == 1
    assert len({round(node.get_x(), 6) for node in diagram.target_tree.leaf_entries}) == 1
    assert diagram.source_heading.get_x() == diagram.source_tree.get_x()
    assert diagram.target_heading.get_x() == diagram.target_tree.get_x()
