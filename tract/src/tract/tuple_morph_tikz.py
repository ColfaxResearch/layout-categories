from torch import nested
from tract import Tuple_morphism, NestedTuple, Nest_morphism

def single_mode_nested_tuple_to_tree(tup: NestedTuple) -> str:
    breakpoint()
    if tup.rank() != 1:
        raise ValueError("Input NestedTuple must have rank 1")
    flat = tup.flatten()
    ret = ""
    for i in range(len(flat)):
        ret += f"\\node[entry] (s{i+1}) at (0, {(i * 0.8):.2f}) {{{flat[i]}}};\n"
    ret += "\n"
    for j in range(tup.mode(1).size()):
        ret += f"\\node[entry] (m{j+1}) at (3, {(j * 0.8):.2f}) {{{tup.mode(1).size()}}};\n"
    ret += "\n"
    for k in range(len(flat)):
        ret += f"\\mapArrow{{s{k+1}}}{{m{(k % tup.mode(1).size()) + 1}}};\n"
    return ret

def nested_tuple_to_tikz(tup: NestedTuple, start_coord = 0) -> str:
    rank = tup.rank()
    flat = tup.flatten()
    ret = ""
    
    # Create mode nodes (left column)
    for i in range(rank):
        ret += f"\\node[entry] (m{i+1}) at ({start_coord}, {(i * 0.8):.2f}) {{{tup.mode(i+1).size()}}};\n"
    ret += "\n"
    
    # Create flat nodes (middle column)  
    for j in range(len(flat)):
        ret += f"\\node[entry] (s{j+1}) at ({start_coord + 3}, {(j * 0.8):.2f}) {{{flat[j]}}};\n"
    
    # Add trees
    ret += "\n% Trees\n"
    ret += generate_trees(tup, start_coord)
    
    return ret

def generate_trees(tup: NestedTuple, start_coord: float) -> str:
    """Generate tree connections from modes to flattened entries"""
    ret = ""
    junction_counter = [0]  # Shared counter across all modes
    
    # Process each mode
    offset = 0
    for mode_idx in range(1, tup.rank() + 1):
        mode = tup.mode(mode_idx)
        junctions = []
        
        # Get max depth for this mode to calculate spacing
        max_depth = mode.depth()
        
        mode_nodes = process_mode(mode, offset, start_coord + 3, junctions, start_coord, junction_counter, 1, max_depth)
        
        # Output junction coordinates for this mode
        for junction in junctions:
            ret += f"\\coordinate (j{junction['id']}) at ({junction['x']:.1f}, {junction['y']:.2f});\n"
        
        # Connect final nodes to mode root
        root_name = f"m{mode_idx}"
        for node in mode_nodes:
            ret += f"\\draw ({node}) -- ({root_name}.east);\n"
        
        # Output junction connections
        for junction in junctions:
            for conn in junction['connections']:
                ret += f"\\draw ({conn}) -- (j{junction['id']});\n"
        
        offset += mode.length()
    
    return ret

def process_mode(mode, base_offset: int, leaf_x: float, junctions: list, root_x: float, 
                junction_counter: list, depth: int, max_depth: int) -> list:
    """Recursively process a mode and return connection points"""
    
    # If mode is a single integer
    if isinstance(mode.data, int):
        return [f"s{base_offset + 1}.west"]
    
    # Create junction for this level
    junction_id = junction_counter[0]
    junction_counter[0] += 1
    
    # Calculate x position with proportional spacing
    # For n levels, distribute junctions evenly between columns
    # Column 1 is at x=0, column 2 is at x=3
    # We want junctions at 3/(n+1), 2*3/(n+1), etc.
    spacing = 3.0  # Distance between columns
    junction_x = spacing * depth / (max_depth + 2)
    
    # Process sub-modes to get connections and leaf indices
    connections = []
    sub_offset = base_offset
    all_leaf_indices = []
    
    for i in range(1, mode.rank() + 1):
        sub_mode = mode.mode(i)
        if isinstance(sub_mode.data, int):
            # Direct leaf
            connections.append(f"s{sub_offset + 1}.west")
            all_leaf_indices.append(sub_offset)
            sub_offset += 1
        else:
            # Nested structure - recurse with incremented depth
            sub_junctions = []
            sub_connections = process_mode(sub_mode, sub_offset, leaf_x, sub_junctions, 
                                         root_x, junction_counter, depth + 1, max_depth)
            
            # Add sub-junctions to our list
            junctions.extend(sub_junctions)
            
            # Connect to the sub-junction(s)
            connections.extend(sub_connections)
            
            # Track leaf indices
            for j in range(sub_mode.length()):
                all_leaf_indices.append(sub_offset + j)
            sub_offset += sub_mode.length()
    
    # Calculate y-position as center of all involved leaves
    if all_leaf_indices:
        y_center = sum(idx * 0.8 for idx in all_leaf_indices) / len(all_leaf_indices)
    else:
        y_center = base_offset * 0.8
    
    # Add this junction
    junctions.append({
        'id': junction_id,
        'x': junction_x,
        'y': y_center,
        'connections': connections
    })
    
    return [f"j{junction_id}"]

def get_leaf_indices(mode, base_offset: int) -> list:
    """Get all leaf indices for a mode"""
    # Check if mode is just an integer
    if not hasattr(mode, 'rank'):
        return [base_offset]
    
    # Check if it's a NestedTuple containing a single integer
    if isinstance(mode.data, int):
        return [base_offset]
    
    indices = []
    offset = base_offset
    for i in range(1, mode.rank() + 1):
        sub_mode = mode.mode(i)
        if isinstance(sub_mode.data, int):
            indices.append(offset)
            offset += 1
        else:
            indices.extend(get_leaf_indices(sub_mode, offset))
            offset += sub_mode.size()
    
    return indices

def nested_tuple_morphism_to_tikz(morphism: Tuple_morphism, profile: NestedTuple) -> str:
    assert morphism.domain == profile.flatten()
    ret = ""
    ret += nested_tuple_to_tikz(profile, start_coord=0)
    ret += "\n"
    ret += tuple_morphism_to_tikz(morphism, start_coord=3)
    return ret

def tuple_morphism_to_tikz(morphism: Tuple_morphism, start_coord = 0)-> str:
    domain = morphism.domain
    codomain = morphism.codomain
    mapping = morphism.map
    
    ret = ""
    for i, d in enumerate(domain):
        ret += f"\\node[entry] (s{i+1}) at ({start_coord}, {(i * 0.8):.2f}) {{{d}}};\n"
    ret += "\n"
    for j, c in enumerate(codomain):
        ret += f"\\node[entry] (t{j+1}) at ({start_coord + 3}, {(j * 0.8):.2f}) {{{c}}};\n"
    ret += "\n"
    for i, j in enumerate(mapping):
        if j != 0:
            ret += f"\\mapArrow{{s{i+1}}}{{t{j}}};\n"

    return ret

if __name__ == "__main__":
    m = Tuple_morphism((6, 2, 3, 6, 3, 2), (6, 2, 6, 3, 2, 3), (1, 2, 4, 3, 6, 5))
    n = NestedTuple(((6, 2, 3), (6, 2)))
    profile_m = NestedTuple((6, (2, (3, (6, (3, 2))))))
    k = NestedTuple((((6, 2))))
    # print(tuple_morphism_to_tikz(m))
    # print(nested_tuple_to_tikz(n))
    print(nested_tuple_morphism_to_tikz(m, profile_m))
    # print(single_mode_nested_tuple_to_tree(k))
