import cutlass
import cutlass.cute as cute

from categories import *

#*************************************************************************
# CUTE FUNCTIONS
#*************************************************************************

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

    #Flatten layout
    flat_layout = flatten_layout(layout)

    #Sort layout
    sorted_flat_layout = sort_flat_layout(flat_layout)

    #Check divisibility condition
    shape = sorted_flat_layout.shape
    stride = sorted_flat_layout.stride
    tractable = cute.Boolean(True)
    for i in cutlass.range_constexpr(len(shape)-1):
        if tractable and stride[i] != 0:
            if stride[i+1] % (shape[i]*stride[i]) != 0:
                tractable = False
    return tractable

def compute_Tuple_morphism(flat_layout):
    """
    Description: 
    Given a tractable flat layout L, produces a tuple morphism f with L_f = L.
    """
    if not is_tractable(flat_layout):
        raise ValueError("The given layout is not tractable.")
    
    sorted_flat_layout, permutation = sort_flat_layout_with_perm(flat_layout)

    shape = sorted_flat_layout.shape
    stride = sorted_flat_layout.stride
    domain = shape
    m = len(domain)

    #Find the largest integer k such that stride(L)_k = stride[k-1] = 0
    k=0
    while k<m and stride[k]==0:
        k+=1

    #Construct codomain(alpha)
    codomain=[]
    if k < m:
        codomain.append(stride[k])
        codomain.append(shape[k])
        for j in range(k+2,m+1):
            codomain.append(int(stride[j-1]/(shape[j-2]*stride[j-2])))
            codomain.append(shape[j-1])
    codomain=tuple(codomain)

    #Construct the map alpha'
    alpha_prime = [] 
    for _ in range(k):
        alpha_prime.append(0)
    for j in range(k+1,m+1):
        alpha_prime.append(2*(j-k))

    #Construct the inverse permutation sigma^{-1}
    inverse_permutation = [0]*m
    for i in range(m):
        inverse_permutation[permutation[i]-1] = i+1
    
    #Construct alpha = alpha' o sigma^{-1}
    alpha = [alpha_prime[inverse_permutation[i]-1] for i in range(m)]
    

    return Tuple_morphism(domain,codomain,alpha)

def compute_flat_layout(morphism: Tuple_morphism):
    """
    Description: 
    Computes the layout L_f associated to a tuple morphism f.
    """

    domain      = morphism.domain
    codomain    = morphism.codomain
    alpha       = morphism.map

    m = len(domain)
    stride_list = [0]*m
    for i in range(m):
        if alpha[i] != 0:
            t = 1
            for j in range(alpha[i]-1):
                t*=codomain[j]
            stride_list[i]=t

    shape_tuple = tuple([int(x) for x in domain])
    stride_tuple = tuple([int(x) for x in stride_list])
    
    layout = cute.make_layout(shape_tuple,stride=stride_tuple)
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




def main():  
    pass
    
if __name__ == "__main__":
    main()