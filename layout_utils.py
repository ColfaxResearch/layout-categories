from unittest import result
import cutlass
import cutlass.cute as cute

from categories import *

def nullify_trivial_strides(flat_layout: cute.Layout) -> cute.Layout:
    """
    Description: 
    Given a flat layout L = (s_1,...,s_m):(d_1,...,d_m),
    sets d_i = 0 if s_i = 1.
    """
    cute.is_int_tuple
    shape  = flat_layout.shape
    stride = flat_layout.stride
    new_stride = []
    for i in range(len(shape)):
        if shape[i] != 1:
            new_stride.append(stride[i])
        else:
            new_stride.append(0)
    result = cute.make_layout(shape,stride=tuple(new_stride))
    return result 

def nullify_zero_strides(layout: cute.Layout) -> cute.Layout:
    """
    Description: 
    Given a flat layout L = (s_1,...,s_m):(d_1,...,d_m),
    sets d_i = 0 if s_i = 1.
    """
    cute.is_int_tuple
    flat_layout = nullify_trivial_strides(flatten_layout(layout))
    shape  = NestedTuple(layout.shape).sub(flat_layout.shape).data
    stride = NestedTuple(layout.stride).sub(flat_layout.stride).data
    result = cute.make_layout(shape,stride=stride)
    return result 


def flatten_layout(layout):
    """
    Description: 
    Computes the flattening flat(L) of a given flat layout L.
    """
    flat_layout = cute.make_layout(cute.flatten_to_tuple(layout.shape),stride = cute.flatten_to_tuple(layout.stride))
    return flat_layout

def sort_flat_layout(flat_layout):
    """
    Description: 
    Computes the sorting sort(L) of a given flat layout L.
    """
    if len(flat_layout.shape) == 0:
        return flat_layout
    indexed = list(zip(flat_layout.shape,flat_layout.stride))
    sorted_pairs = sorted(indexed, key=lambda x: (x[1], x[0]))
    sorted_shape, sorted_stride = zip(*sorted_pairs)
    sorted_layout = cute.make_layout(sorted_shape, stride = sorted_stride)
    return sorted_layout

def sort_flat_layout_with_perm(flat_layout):
    """
    Description: 
    Computes the sorting sort(L) of a given flat layout L, and also returns the permutation used in the sorting.
    """
    if len(flat_layout.shape) == 0:
        return flat_layout, []
    indexed = list(enumerate(zip(flat_layout.shape, flat_layout.stride)))
    sorted_indexed = sorted(indexed, key=lambda x: (x[1][1], x[1][0]))
    permutation = [index+1 for index, _ in sorted_indexed]
    sorted_shape, sorted_stride = zip(*[item for _, item in sorted_indexed])
    sorted_layout = cute.make_layout(sorted_shape, stride=sorted_stride)
    return sorted_layout, permutation

def is_tractable(layout):
    """
    Description: 
    Checks if a given flat layout L is tractable.
    """
    flat_layout        = flatten_layout(layout)
    sorted_flat_layout = sort_flat_layout(flat_layout)
    shape              = sorted_flat_layout.shape
    stride             = sorted_flat_layout.stride
    tractable          = cute.Boolean(True)
    for i in cutlass.range_constexpr(len(shape)-1):
        if tractable and stride[i] != 0:
            if stride[i+1] % (shape[i]*stride[i]) != 0:
                tractable = False
    return tractable

@cute.jit
def compute_Tuple_morphism(flat_layout):
    """
    Description:
    Given a tractable flat layout L, produces a tuple morphism f with L_f = L.
    """
    if cutlass.const_expr(not is_tractable(flat_layout)):
        raise ValueError("The given layout is not tractable.")
    
    domain = tuple(flat_layout.shape)
    sorted_flat_layout, permutation = sort_flat_layout_with_perm(flat_layout)
    shape = tuple(sorted_flat_layout.shape)
    stride = tuple(sorted_flat_layout.stride)
    m = len(shape)

    # Find the largest integer k such that stride(L)_k = stride[k-1] = 0
    k = 0
    seen_nonzero = False
    for i in cutlass.range_constexpr(len(stride)):
        if cutlass.const_expr(stride[i] == 0 and not seen_nonzero):
            k += 1
        else:
            seen_nonzero = True

    # build codomain
    codomain = tuple()
    if cutlass.const_expr(k < m):
        cod = [stride[k], shape[k]]
        for j in cutlass.range_constexpr(k + 1, m):
            denom = shape[j - 1] * stride[j - 1]
            # mimic original: integer factorization step; use // for exact integer division
            factor = (stride[j] // denom) if denom != 0 else 0
            cod.append(int(factor))
            cod.append(shape[j])
        codomain = tuple(cod)
    else:
        codomain = tuple()

    # construct the map alpha'
    alpha_prime = [0] * m
    for j in cutlass.range_constexpr(k, m):
        alpha_prime[j] = 2 * (j - k + 1)

    # construct the inverse permutation
    inverse_permutation = [0] * m
    for i in cutlass.range_constexpr(m):
        inverse_permutation[permutation[i] - 1] = i + 1

    # alpha = alpha'[σ^{-1}(i)]
    alpha = tuple(alpha_prime[inverse_permutation[i] - 1] for i in range(m))

    return Tuple_morphism(domain, codomain, alpha)

# @cute.jit
# def compute_Tuple_morphism(flat_layout):
#     """
#     Description: 
#     Given a tractable flat layout L, produces a tuple morphism f with L_f = L.
#     """
#     if cutlass.const_expr(not is_tractable(flat_layout)):
#         raise ValueError("The given layout is not tractable.")
    
#     domain  = flat_layout.shape
#     sorted_flat_layout, permutation = sort_flat_layout_with_perm(flat_layout)
#     shape   = sorted_flat_layout.shape
#     stride  = sorted_flat_layout.stride
#     m = len(shape)

#     #Find the largest integer k such that stride(L)_k = stride[k-1] = 0
#     k=0
#     while k<m and stride[k]==0:
#         k+=1

#     codomain = []
#     if k < m:
#         codomain.append(stride[k])
#         codomain.append(shape[k])
#         for j in range(k+2,m+1):
#             codomain.append(int(stride[j-1]/(shape[j-2]*stride[j-2])))
#             codomain.append(shape[j-1])
#     codomain = tuple(codomain)

#     #Construct the map alpha'
#     alpha_prime = [] 
#     for _ in range(k):
#         alpha_prime.append(0)
#     for j in range(k+1,m+1):
#         alpha_prime.append(2*(j-k))

#     #Construct the inverse permutation sigma^{-1}
#     inverse_permutation = [0]*m
#     for i in range(m):
#         inverse_permutation[permutation[i]-1] = i+1
    
#     #Construct alpha = alpha' o sigma^{-1}
#     alpha = tuple([alpha_prime[inverse_permutation[i]-1] for i in range(m)])

#     return Tuple_morphism(domain,codomain,alpha)

def compute_flat_layout(morphism: Tuple_morphism):
    """
    Description: 
    Computes the layout L_f associated to a tuple morphism f.
    """

    domain   = morphism.domain
    codomain = morphism.codomain
    alpha    = morphism.map

    m = len(domain)
    stride_list = [0]*m
    for i in range(m):
        if alpha[i] != 0:
            t = 1
            for j in range(alpha[i]-1):
                t*=codomain[j]
            stride_list[i]=t

    shape_tuple  = tuple(domain)
    stride_tuple = tuple(stride_list)
    layout       = cute.make_layout(shape_tuple,stride=stride_tuple)
    return layout

def flat_concatenate(base: cute.Layout, stack: cute.Layout) -> cute.Layout:
    """
    Description: 
    Computes the flat concatenation conat(L_1,L_2) of flat layouts L_1 and L_2.
    """
    # Convert shapes to tuples for proper concatenation
    def intuple_to_tuple(shape):
        if isinstance(shape, int):
            return (shape,)
        else:
            return shape
    
    base_shape_tuple = intuple_to_tuple(base.shape)
    stack_shape_tuple = intuple_to_tuple(stack.shape)
    base_stride_tuple = intuple_to_tuple(base.stride)
    stack_stride_tuple = intuple_to_tuple(stack.stride)
    
    concat_shape = base_shape_tuple + stack_shape_tuple
    concat_stride = base_stride_tuple + stack_stride_tuple 
    concat_layout = cute.make_layout(concat_shape, stride=concat_stride)
    return concat_layout

def concatenate(base: cute.Layout, stack: cute.Layout) -> cute.Layout:
    """
    Description: 
    Computes the (nested) concatenation (L_1,L_2) of layouts L_1 and L_2.
    """
    return cute.make_layout(
        (base.shape, stack.shape),
        stride=(base.stride, stack.stride)
    )

def compute_layout(morphism: Nest_morphism) -> cute.Layout:
    flat_layout = compute_flat_layout(morphism.flatten())
    shape = morphism.domain.data
    stride = morphism.domain.sub(flat_layout.stride).data
    layout = cute.make_layout(shape, stride = stride)
    return layout

@cute.jit
def compute_Nest_morphism(layout):
    """
    Description:
    Given a tractable layout L, produces a nested tuple morphism f with L_f = L.
    """
    if cutlass.const_expr(not is_tractable(layout)):
        raise ValueError("The given layout is not tractable.")

    flat_layout = flatten_layout(layout)
    flat_morphism = compute_Tuple_morphism(flat_layout)
    domain = NestedTuple(layout.shape)
    codomain = NestedTuple(flat_morphism.codomain)
    map = flat_morphism.map
    morphism = Nest_morphism(domain, codomain, map)

    return morphism


# @cute.jit
# def compute_Nest_morphism(layout):
#     """
#     Description: 
#     Given a tractable layout L, produces a nested tuple morphism f with L_f = L.
#     """
#     if not is_tractable(layout):
#         raise ValueError("The given layout is not tractable.")
    
#     flat_layout   = flatten_layout(layout)
#     flat_morphism = compute_Tuple_morphism(flat_layout)
#     domain        = NestedTuple(layout.shape)
#     codomain      = NestedTuple(flat_morphism.codomain)
#     map           = flat_morphism.map
#     morphism      = Nest_morphism(domain,codomain,map)

#     return morphism

def flat_complement(flat_layout, N):
    reduced_layout = sort_flat_layout(flat_layout)
    S = reduced_layout.shape
    D = reduced_layout.stride
    m = len(S)
    shape = [D[0]]
    for i in range(1, m):
        shape.append(D[i] // (S[i-1] * D[i-1]))
    shape.append(N // (S[-1] * D[-1]))
    stride = [1]
    for i in range(m):
        stride.append(S[i] * D[i])
    return cute.make_layout(tuple(shape), stride=tuple(stride))

def mutual_refinement(nestedtuple1,nestedtuple2):
    """
    Given nested tuples nestedtuple1 = T and nestedtuple2 = U, computes nested tuples 
    T' and U' such that 
     1. T' refines T
     2. U' refines U, and 
     3. T' divides U'
    For example 
    T = (6,6)
    U = (2,6,6)
    ->
    T' = ((2,3),(2,3))
    U' = (2,(3,2),(3,2))
    """
    tuple1 = nestedtuple1.flatten()
    tuple2 = nestedtuple2.flatten()
    list1  = list(tuple1)
    list2  = list(tuple2)
    i = 0
    j = 0
    result1   = []
    cur_mode1 = []
    result2   = []
    cur_mode2 = []
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            cur_mode1.append(list1[i])
            result1.append(cur_mode1[0] if len(cur_mode1) == 1 else tuple(cur_mode1))
            cur_mode1 = []
            cur_mode2.append(list2[j])
            result2.append(cur_mode2[0] if len(cur_mode2) == 1 else tuple(cur_mode2))
            cur_mode2 = []
            i += 1
            j += 1
        elif list1[i] < list2[j] and list2[j]%list1[i] == 0:
            cur_mode1.append(list1[i])
            result1.append(cur_mode1[0] if len(cur_mode1) == 1 else tuple(cur_mode1))
            cur_mode1 = []
            cur_mode2.append(list1[i])
            list2[j] //= list1[i]
            i += 1
        elif list2[j] < list1[i] and list1[i]%list2[j] == 0:
            cur_mode1.append(list2[j])
            cur_mode2.append(list2[j])
            result2.append(cur_mode2[0] if len(cur_mode2) == 1 else tuple(cur_mode2))
            cur_mode2 = []
            list1[i] //= list2[j]
            j += 1
        else:
            raise ValueError("The given nested tuples are not mutually refinable.")
    if i < len(list1):
        raise ValueError("The given nested tuples are not mutually refinable.")
    if cur_mode2 != []:
        cur_mode2.append(list2[j])
        result2.append(tuple(cur_mode2))
        j+=1 

    while j < len(list2):
        result2.append(list2[j])
        j += 1
    
    result1 = nestedtuple1.sub(tuple(result1))
    result2 = nestedtuple2.sub(tuple(result2))
    return result1, result2

def weak_composite(f:'Nest_morphism',g: 'Nest_morphism'):
    S = f.domain
    T = f.codomain
    U = g.domain
    V = g.codomain
    Tprime,Uprime = mutual_refinement(T,U)
    assert Tprime.refines(T) and Uprime.refines(U)
    inclusion = Nest_morphism(Tprime,Uprime,tuple(range(1,Tprime.length()+1)))
    fprime = f.pullback_along(Tprime)
    gprime = g.pushforward_along(Uprime)
    weak_composite = fprime.compose(inclusion).compose(gprime)
    return weak_composite

def main():  
    pass
    
if __name__ == "__main__":
    main()