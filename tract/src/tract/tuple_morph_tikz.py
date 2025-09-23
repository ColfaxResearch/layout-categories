from tract import Tuple_morphism, NestedTuple, Nest_morphism, make_morphism

PREAMBLE = r"""
\documentclass{standalone}
\usepackage{tikz}
\usepackage{amsmath}
\usetikzlibrary{arrows.meta, positioning}
\newcommand{\mapArrow}[2]{\draw[maparrow] (#1.east) -- (#2.west);}
\begin{document}
"""
TIKZ_START = r"""
\begin{tikzpicture}[
    entry/.style={minimum width=5mm, minimum height=7mm, inner sep = 2pt},
    maparrow/.style={|->}
]
\def\colspacing{3}
\def\rowspacing{0.8}
"""
TIKZ_END = r"""
\end{tikzpicture}
"""
EPILOGUE = r"""
\end{document}
"""


def calculate_junction_layers(mode) -> int:
    if isinstance(mode.data, int):
        return 0
    mx = 0
    for i in range(1, mode.rank() + 1):
        sm = mode.mode(i)
        if not isinstance(sm.data, int):
            mx = max(mx, calculate_junction_layers(sm))
    return 1 + mx


def get_leaf_indices(mode, base_offset: int) -> list:
    """Return flat leaf indices covered by `mode` (advancing by .length())."""
    if not hasattr(mode, "rank"):
        return [base_offset]
    if isinstance(mode.data, int):
        return [base_offset]
    idxs = []
    off = base_offset
    for i in range(1, mode.rank() + 1):
        sm = mode.mode(i)
        if isinstance(sm.data, int):
            idxs.append(off)
            off += 1
        else:
            idxs.extend(get_leaf_indices(sm, off))
            off += sm.length()  # number of leaves
    return idxs


def process_mode(
    mode,
    base_offset: int,
    leaf_x: float,
    junctions: list,
    root_x: float,
    junction_counter: list,
    depth: int,
    max_junction_layers: int,
    row_spacing: float,
) -> list:
    """Return connectors upward; junction y = weighted avg of descendant leaves."""
    if isinstance(mode.data, int):
        return [f"s{base_offset + 1}.west"]  # branches go left on the tree half

    jid = junction_counter[0]
    junction_counter[0] += 1

    spacing = leaf_x - root_x
    denom = max_junction_layers + 1
    t = depth / denom
    jx = root_x + spacing * t

    conns = []
    sub_off = base_offset
    all_leaf_idxs = []

    for i in range(1, mode.rank() + 1):
        sm = mode.mode(i)
        if isinstance(sm.data, int):
            conns.append(f"s{sub_off + 1}.west")
            all_leaf_idxs.append(sub_off)
            sub_off += 1
        else:
            sub_juncs = []
            sub_conns = process_mode(
                sm,
                sub_off,
                leaf_x,
                sub_juncs,
                root_x,
                junction_counter,
                depth + 1,
                max_junction_layers,
                row_spacing,
            )
            junctions.extend(sub_juncs)
            conns.extend(sub_conns)
            for j in range(sm.length()):
                all_leaf_idxs.append(sub_off + j)
            sub_off += sm.length()

    jy = (
        (sum(i * row_spacing for i in all_leaf_idxs) / len(all_leaf_idxs))
        if all_leaf_idxs
        else (base_offset * row_spacing)
    )
    junctions.append({"id": jid, "x": jx, "y": jy, "connections": conns})
    return [f"j{jid}"]


def nested_tuple_morphism_to_tikz(
    morphism: Nest_morphism,
    *,
    row_spacing: float = 0.8,
    tree_width: float = 5.0,
    map_width: float = 4.0,
    root_y_offset: float = 0.0,
    label: str | None = None,
    full_doc: bool = False,
) -> str:
    """
    Build a single picture with one set of (s{i}).
    - If domain.depth() <= 1: skip tree half entirely.
    - Mode roots m<i>:
        * if root_y_offset == -1: align y to midpoint of the mode's leaf span
        * else: y = (i-1)*row_spacing + root_y_offset (stacked by mode order)
    - Intermediate junctions get dynamic y (weighted centers).
    - Targets at x_tgt = x_leaf + map_width.
    - Label (if provided) is horizontally centered over what is actually drawn.
    """
    domain_nt: NestedTuple = morphism.domain
    flat_tm: Tuple_morphism = morphism.flatten()
    ret = []

    has_tree = domain_nt.depth() > 1

    x_root = 0.0
    x_leaf = x_root + (tree_width if has_tree else 0.0)
    x_tgt = x_leaf + map_width

    flat_vals = domain_nt.flatten()
    for j, val in enumerate(flat_vals):
        y = j * row_spacing
        ret.append(f"\\node[entry] (s{j + 1}) at ({x_leaf:.2f}, {y:.2f}) {{{val}}};")
    ret.append("")

    if has_tree:
        off = 0
        for mode_idx in range(1, domain_nt.rank() + 1):
            mode = domain_nt.mode(mode_idx)
            if root_y_offset == -1:
                span = get_leaf_indices(mode, off)
                y_root = (sum(span) / len(span)) * row_spacing if span else off * row_spacing
            else:
                y_root = (mode_idx - 1) * row_spacing + root_y_offset
            ret.append(
                f"\\node[entry] (m{mode_idx}) at ({x_root:.2f}, {y_root:.2f}) {{{mode.size()}}};"
            )
            off += mode.length()
        ret.append("\n% Trees")

        jcounter = [0]
        global_off = 0
        for mode_idx in range(1, domain_nt.rank() + 1):
            mode = domain_nt.mode(mode_idx)
            max_layers = calculate_junction_layers(mode)
            junctions = []
            conns = process_mode(
                mode=mode,
                base_offset=global_off,
                leaf_x=x_leaf,
                junctions=junctions,
                root_x=x_root,
                junction_counter=jcounter,
                depth=1,
                max_junction_layers=max_layers,
                row_spacing=row_spacing,
            )
            for junc in junctions:
                ret.append(
                    f"\\coordinate (j{junc['id']}) at ({junc['x']:.2f}, {junc['y']:.2f});"
                )
            root_name = f"m{mode_idx}"
            for c in conns:
                ret.append(f"\\draw ({c}) -- ({root_name}.east);")
            for junc in junctions:
                for c in junc["connections"]:
                    ret.append(f"\\draw ({c}) -- (j{junc['id']});")
            global_off += mode.length()
        ret.append("")

    codomain = flat_tm.codomain
    for j, c in enumerate(codomain):
        y = j * row_spacing
        ret.append(f"\\node[entry] (t{j + 1}) at ({x_tgt:.2f}, {y:.2f}) {{{c}}};")
    ret.append("")
    for i, j in enumerate(flat_tm.map):
        if j != 0:
            ret.append(f"\\mapArrow{{s{i + 1}}}{{t{j}}};")

    if label:
        x_left = x_root if has_tree else x_leaf
        x_center = 0.5 * (x_left + x_tgt)
        ret.append(f"\\node at ({x_center:.2f}, -0.8) {{{label}}};")

    out = TIKZ_START + "\n" + "\n".join(ret) + "\n" + TIKZ_END
    return PREAMBLE + "\n" + out + "\n" + EPILOGUE if full_doc else out


# ============================================================
# Two-parenthesization trees over integer leaves (supports duplicates)
# ============================================================

from collections import defaultdict, deque

def _is_leaf_tuple(x):
    return not isinstance(x, (tuple, list))


def _calc_layers_tuple(tree):
    if _is_leaf_tuple(tree):
        return 0
    mx = 0
    for t in tree:
        if not _is_leaf_tuple(t):
            mx = max(mx, _calc_layers_tuple(t))
    return 1 + mx


def _map_tree_to_indices(tree, value_pools):
    """
    Return a new tree with leaves replaced by their GLOBAL leaf indices.
    Each time a value v is encountered, pop the leftmost index from value_pools[v].
    This is side-local (call separately for P and Q) and supports duplicates correctly.
    """
    if _is_leaf_tuple(tree):
        if tree not in value_pools or not value_pools[tree]:
            raise ValueError(f"No unused leaf index available for value {tree}")
        return value_pools[tree].popleft()
    return tuple(_map_tree_to_indices(ch, value_pools) for ch in tree)


def _gather_indices_idx(tree):
    """On an index-mapped tree, collect the (global) leaf indices under 'tree'."""
    if _is_leaf_tuple(tree):  # now leaf is an int index
        return [tree]
    out = []
    for ch in tree:
        out.extend(_gather_indices_idx(ch))
    return out


def _process_tuple_tree_idx(
    tree_idx,
    *,
    side,
    leaf_x,
    root_x,
    row_spacing,
    jcounter,
    nodes_out,
    edges_out,
    depth,
    max_layers_root,
):
    """
    Build junctions for an INDEXED tree (leaves are global indices).
    Junction x uses uniform subdivision depth/(max_layers_root+1) from root_x → leaf_x.
    """
    if _is_leaf_tuple(tree_idx):
        ix = tree_idx
        return [f"s{ix + 1}.{'west' if side == 'L' else 'east'}"]

    jid = jcounter[0]; jcounter[0] += 1

    denom = max_layers_root + 1 if max_layers_root >= 0 else 1
    t = depth / denom
    jx = root_x + (leaf_x - root_x) * t

    conns = []
    all_ix = []
    for ch in tree_idx:
        ch_conns = _process_tuple_tree_idx(
            ch,
            side=side,
            leaf_x=leaf_x,
            root_x=root_x,
            row_spacing=row_spacing,
            jcounter=jcounter,
            nodes_out=nodes_out,
            edges_out=edges_out,
            depth=depth + 1,
            max_layers_root=max_layers_root,
        )
        conns.extend(ch_conns)
        all_ix.extend(_gather_indices_idx(ch))

    jy = (sum(all_ix) / len(all_ix) * row_spacing) if all_ix else 0.0
    nodes_out.append(f"\\coordinate ({side}j{jid}) at ({jx:.3f},{jy:.3f});")
    for c in conns:
        edges_out.append(f"\\draw ({c}) -- ({side}j{jid});")
    return [f"{side}j{jid}"]


def _place_roots_and_connect_tuple_idx(
    tree_idx,
    *,
    side,
    root_x,
    leaf_x,
    row_spacing,
    values_by_index,
    root_y_offset,
    jcounter,
    nodes_out,
    edges_out,
):
    """
    Create a visible root node for EVERY top-level item in the INDEXED tree:
      • Leaf item: label is the leaf value (via values_by_index); draw a direct edge.
      • Internal: label is the product of its leaves; build junctions and connect.

    y-position:
      • root_y_offset == -1 → midpoint of the item's leaf rows.
      • else                → stacked by top-level order: y = k*row_spacing + root_y_offset.
    """
    def midpoint_y_from_indices(idxs):
        return (sum(idxs) / len(idxs)) * row_spacing if idxs else 0.0

    rendered_idx = 0
    m_id = 1

    for item in tree_idx:
        idxs = _gather_indices_idx(item)
        # product label
        prod = 1
        for ix in idxs:
            prod *= values_by_index[ix]

        # y
        if root_y_offset == -1:
            y = midpoint_y_from_indices(idxs)
        else:
            y = rendered_idx * row_spacing + root_y_offset

        # root node
        nodes_out.append(f"\\node[entry] ({side}m{m_id}) at ({root_x:.3f},{y:.3f}) {{{prod}}};")

        if _is_leaf_tuple(item):
            ix = item
            edges_out.append(
                f"\\draw (s{ix+1}.{'west' if side=='L' else 'east'}) -- "
                f"({side}m{m_id}.{'east' if side=='L' else 'west'});"
            )
        else:
            L = _calc_layers_tuple(item)
            conns = _process_tuple_tree_idx(
                item,
                side=side,
                leaf_x=leaf_x,
                root_x=root_x,
                row_spacing=row_spacing,
                jcounter=jcounter,
                nodes_out=nodes_out,
                edges_out=edges_out,
                depth=1,
                max_layers_root=L,
            )
            for c in conns:
                edges_out.append(
                    f"\\draw ({c}) -- ({side}m{m_id}.{'west' if side == 'R' else 'east'});"
                )

        rendered_idx += 1
        m_id += 1


def two_parenthesizations_to_tikz_values(
    values,
    P,
    Q,
    *,
    row_spacing: float = 0.8,
    left_width: float = 5.0,
    right_width: float = 5.0,
    root_y_offset: float = -1.0,
    center_label: str | None = None,
    full_doc: bool = False,
) -> str:
    """
    Draw two tree halves sharing a single integer leaf stack (duplicates supported).
    Left = P, Right = Q. Each side is resolved to GLOBAL leaf indices first.
    """
    assert all(isinstance(v, int) for v in values), "values must be integers"

    # Shared geometry
    x_leaf = 0.0
    x_left_root = -left_width
    x_right_root = right_width

    # Values by index (for labels & products)
    values_by_index = {i: v for i, v in enumerate(values)}

    # Build global value→deque(indices) pools once
    pools_template = defaultdict(deque)
    for i, v in enumerate(values):
        pools_template[v].append(i)

    # Resolve P and Q to index trees (independently)
    def clone_pools():
        return {k: deque(v) for k, v in pools_template.items()}

    P_idx = _map_tree_to_indices(P, clone_pools())
    Q_idx = _map_tree_to_indices(Q, clone_pools())

    body: list[str] = []

    # Shared leaves s{i}
    for i, val in enumerate(values):
        y = i * row_spacing
        body.append(f"\\node[entry] (s{i + 1}) at ({x_leaf:.3f},{y:.3f}) {{{val}}};")
    body.append("")

    # Left = P
    if not _is_leaf_tuple(P_idx):
        body.append("% Left-side roots and tree for P")
        Lj = [0]
        _place_roots_and_connect_tuple_idx(
            P_idx,
            side="L",
            root_x=x_left_root,
            leaf_x=x_leaf,
            row_spacing=row_spacing,
            values_by_index=values_by_index,
            root_y_offset=root_y_offset,
            jcounter=Lj,
            nodes_out=body,
            edges_out=body,
        )
        body.append("")

    # Right = Q
    if not _is_leaf_tuple(Q_idx):
        body.append("% Right-side roots and tree for Q")
        Rj = [0]
        _place_roots_and_connect_tuple_idx(
            Q_idx,
            side="R",
            root_x=x_right_root,
            leaf_x=x_leaf,
            row_spacing=row_spacing,
            values_by_index=values_by_index,
            root_y_offset=root_y_offset,
            jcounter=Rj,
            nodes_out=body,
            edges_out=body,
        )
        body.append("")

    if center_label:
        x_center = 0.5 * (x_left_root + x_right_root)
        body.append(f"\\node at ({x_center:.3f}, -0.8) {{{center_label}}};")

    out = TIKZ_START + "\n" + "\n".join(body) + "\n" + TIKZ_END
    return PREAMBLE + "\n" + out + "\n" + EPILOGUE if full_doc else out


# ============================
# Examples
# ============================
if __name__ == "__main__":
    # Existing morphism example (unchanged)
    m = Nest_morphism(
        domain=(16, ((4, 4), (4, 4)), (2, 2)),
        codomain=(16, 4, 4),
        map=(1, 2, 0, 3, 0, 0, 0),
    )
    print(
        nested_tuple_morphism_to_tikz(
            m,
            row_spacing=0.8,
            tree_width=2.5,
            map_width=3.0,
            root_y_offset=0,  # -1 aligns roots to midpoints of their leaf spans
            label="$L$",
            full_doc=False,
        )
    )

    # New two-parenthesization examples over integer leaves (duplicates supported)
    vals = (2, 3, 5, 7, 11, 13, 17)
    P = (2, (3, (5, 7)), 11)                # initial segment
    Q = ((2, 3, 5), (7, 11, 13), 17)        # full tuple
    print(
        two_parenthesizations_to_tikz_values(
            vals, P, Q,
            row_spacing=0.8,
            left_width=2.5, right_width=2.5,
            root_y_offset=-1.0,
            center_label="$P$ (left) vs. $Q$ (right)",
            full_doc=False,
        )
    )

    vals2 = (6, 2, 3, 6)
    P2 = (6, (2, 3))                         # initial segment
    Q2 = ((6, 2), 3, 6)
    print(
        two_parenthesizations_to_tikz_values(
            vals2, P2, Q2,
            row_spacing=0.8,
            left_width=2.5, right_width=2.5,
            root_y_offset=-1.0,
            center_label="$P$ (left) vs. $Q$ (right)",
            full_doc=False,
        )
    )
