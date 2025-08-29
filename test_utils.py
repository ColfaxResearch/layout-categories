import torch 
import numpy as np
import cutlass
import cutlass.cute as cute 

from categories import *



#*************************************************************************
# RANDOM GENERATION FUNCTIONS
#*************************************************************************

def random_ordered_subtuple(m: int, k: int) -> tuple:
    """
    Generate a tuple of k distinct integers sampled uniformly at random from {1,...,m}.
    
    Parameters:
    - m (int): The upper bound of the set (inclusive).
    - k (int): The number of distinct elements to sample, 1 <= k <= m.
    
    Returns:
    - tuple of k distinct integers from {1,...,m}.
    """
    if not (0 <= k <= m):
        raise ValueError("k must satisfy 0 <= k <= m")

    sampled = random.sample(range(1, m+1), k)
    return tuple(sampled)

def random_Fin_morphism(min_length = 0, max_length = 9):
    """
    Description:
    Randomly generate a morphism in Fin_*
    """
    domain   = np.random.randint(1,max_length)
    codomain = np.random.randint(1,max_length)
    map_     = [0 for _ in range(domain)]

    max_size      = min(domain, codomain)
    u             = np.random.rand()
    skewed        = 1 - u**3 # skew toward larger values
    size_of_image = int(np.round(skewed * max_size))
    size_of_image = max(0, min(size_of_image, max_size))

    image             = random_ordered_subtuple(codomain, size_of_image)
    surviving_indices = sorted(random_ordered_subtuple(domain, size_of_image))
    for i, index in enumerate(surviving_indices):
        map_[index - 1] = int(image[i])
    map_ = tuple(map_)

    return Fin_morphism(domain, codomain, map_)
    
def random_Tuple_morphism(min_length = 0, max_length = 9, max_value = 10):
    """
    Description:
    Randomly generates a tuple morphism f.
    """
    domain_length   = np.random.randint(min_length,max_length)
    codomain_length = np.random.randint(min_length,max_length)
    map_            = [0 for _ in range(domain_length)]

    max_size      = min(domain_length, codomain_length)
    u             = np.random.rand()
    skewed        = 1 - u**3 # skew toward larger values
    size_of_image = int(np.round(skewed * max_size))
    size_of_image = max(0, min(size_of_image, max_size))

    image             = random_ordered_subtuple(codomain_length, size_of_image)
    surviving_indices = sorted(random_ordered_subtuple(domain_length, size_of_image))
    for i, index in enumerate(surviving_indices):
        map_[index - 1] = int(image[i])
    map_ = tuple(map_)

    domain = []
    for i in range(domain_length):
        domain.append(int(np.random.randint(1,max_value)))
    domain = tuple(domain)
    
    codomain = []
    for j in range(codomain_length):
        codomain.append(int(np.random.randint(1,max_value)))
    
    for i,value in enumerate(map_):
        if value>0:
            codomain[value-1] = domain[i]
    codomain = tuple(codomain) 

    return Tuple_morphism(domain,codomain, map_)

def random_Tuple_composable_morphisms(min_length = 0, max_length = 9, max_value = 10):
    """
    Description:
    Randomly generate a pair of composable tuple morphisms f and g.
    """
    f = random_Tuple_morphism(min_length = min_length, max_length = max_length, max_value = max_value)

    domain        = f.codomain
    domain_length = len(domain)

    codomain_length = np.random.randint(min_length,max_length+1)
    codomain        = []
    for k in range(codomain_length):
        codomain.append(np.random.randint(1,max_value+1))

    map_ = [0 for _ in range(domain_length)]

    max_size      = min(domain_length, codomain_length)
    u             = np.random.rand()
    skewed        = 1 - u**3 # skew toward larger values
    size_of_image = int(np.round(skewed * max_size))
    size_of_image = max(0, min(size_of_image, max_size))

    image             = random_ordered_subtuple(codomain_length,size_of_image)
    surviving_indices = sorted(random_ordered_subtuple(domain_length,size_of_image))
    for i,index in enumerate(surviving_indices):
        map_[index-1] = image[i]
    map_ = tuple(map_)

    for i,value in enumerate(map_):
        if value>0:
            codomain[value-1] = domain[i]
    codomain = tuple(codomain)

    g = Tuple_morphism(domain,codomain,map_)

    return f, g 

def random_Tuple_morphisms_with_disjoint_images(min_length = 0, max_length = 9, max_value = 10):
    """
    Description:
    Randomly generates a pair of tuple morphisms f, g with the same codomain and disjoint images.
    """
    f               = random_Tuple_morphism()
    codomain        = f.codomain
    codomain_length = len(codomain)

    domain_length   = np.random.randint(min_length,max_length+1)
    domain          = []
    for i in range(domain_length):
        domain.append(np.random.randint(1,max_value+1))

    #Compute possible values for the map beta underlying g.
    possible_values = []
    for j in range(1,codomain_length+1):
        if j not in f.map:
            possible_values.append(j)

    #Select a random subset of those possible values
    max_size      = min(domain_length, len(possible_values))
    u             = np.random.rand()
    skewed        = 1 - u**3 # skew toward larger values
    size_of_image = int(np.round(skewed * max_size))
    size_of_image = max(0, min(size_of_image, max_size))

    image             = [int(x) for x in np.random.choice(possible_values,size_of_image,replace=False)]
    map_              = [0]*domain_length
    surviving_indices = sorted(random_ordered_subtuple(domain_length,size_of_image))
    for i,index in enumerate(surviving_indices):
        map_[index-1] = image[i]
    map_ = tuple(map_)

    for i in range(len(domain)):
        if map_[i] > 0:
            domain[i] = codomain[map_[i]-1]
    domain = tuple(domain)

    g = Tuple_morphism(domain,codomain,map_)
    return f,g

def random_Tuple_complementable_morphism(min_length = 2, max_length = 9, max_value = 10):
    """
    Description:
    Randomly generates a complementable tuple morphism f 
    with length(codomain)>1 and 0<length(domain)<length(codomain). Still need
    to cover the case that length(domain) in {0,length(codomain)}.
    """
    codomain_length = np.random.randint(min_length,max_length+1)
    codomain = []
    for _ in range(codomain_length):
        codomain.append(np.random.randint(1,max_value+1))
    codomain = tuple(codomain)

    size_of_image = np.random.randint(1,len(codomain))
    map_          = random_ordered_subtuple(len(codomain),size_of_image)

    domain = []
    for _, value in enumerate(map_):
        domain.append(codomain[value - 1])
    domain = tuple(domain)
    
    return Tuple_morphism(domain,codomain,map_)

def random_Tuple_divisible_morphisms(min_length = 2, max_length = 9, max_value = 10):
    """
    Description:
    Randomly generates a pair of tuple morphisms f, g such that
    g is injective, and codomain(g) = domain(f).
    """
    f        = random_Tuple_morphism(min_length = min_length, max_length = max_length, max_value = max_value)
    codomain = f.domain

    domain_length = np.random.randint(1,len(codomain)+1)
    domain        = [0]*domain_length

    #Generate map
    map_ = random_ordered_subtuple(len(codomain),len(domain))

    for i,value in enumerate(map_):
        if value>0:
            domain[i] = codomain[value-1]
    domain = tuple(domain)
    
    g = Tuple_morphism(domain,codomain,map_)

    return f, g 

def random_Tuple_productable_morphisms(min_length = 2, max_length = 9, max_value = 10):
    f        = random_Tuple_complementable_morphism(min_length = min_length, max_length = max_length, max_value = max_value)
    codomain = f.complement().domain

    domain        = [] 
    size_of_image = np.random.randint(0,len(codomain)+1)
    m             = size_of_image + np.random.randint(0,5)
    for _ in range(m):
        domain.append(np.random.randint(1,max_value+1))
    image   = random_ordered_subtuple(len(codomain),size_of_image)
    map_    = [0 for _ in range(len(domain))]
    surviving_indices = sorted(random_ordered_subtuple(len(domain),size_of_image))
    for i,index in enumerate(surviving_indices):
        map_[index-1] = image[i]
    
    for i,value in enumerate(map_):
        if value>0:
            domain[i] = codomain[value-1]
    domain = tuple(domain)
    map_ = tuple(map_)
    
    g = Tuple_morphism(domain,codomain,map_)

    return f, g

def random_NestedTuple(length=5, max_depth=4, max_width=5, int_range=(1, 10)):
    """
    Generate a NestedTuple containing exactly `length` integers.
    """
    def generate(depth, remaining):
        if remaining <= 0:
            return ()

        if depth >= max_depth:
            return tuple(random.randint(*int_range) for _ in range(remaining))
        
        if remaining == 1:
            return random.randint(*int_range)

        valid_width = min(max_width, remaining)
        if valid_width < 1:
            return ()

        width = random.randint(1, valid_width)

        if width == 1:
            counts = [remaining]
        else:
            cuts   = sorted(random.sample(range(1, remaining), width - 1))
            counts = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + [remaining - cuts[-1]]

        return tuple(generate(depth + 1, count) for count in counts)

    return NestedTuple(generate(0, length))

def random_profile(length = 5, max_depth = 4, max_width = 5):
    return random_NestedTuple(length = length, max_depth = max_depth, max_width = max_width, int_range = (0,0))

def random_Nest_morphism():
    flat_morphism = random_Tuple_morphism()
    flat_domain   = flat_morphism.domain
    flat_codomain = flat_morphism.codomain
    domain        = random_profile(length = len(flat_domain)).sub(flat_domain)
    codomain      = random_profile(length = len(flat_codomain)).sub(flat_codomain)
    return Nest_morphism(domain,codomain,flat_morphism.map)

def random_Nest_composable_morphisms():
    flat_f, flat_g = random_Tuple_composable_morphisms()
    domain_f   = random_profile(length = len(flat_f.domain)).sub(flat_f.domain)
    codomain_f = random_profile(length = len(flat_f.codomain)).sub(flat_f.codomain)
    domain_g   = codomain_f
    codomain_g = random_profile(length = len(flat_g.codomain)).sub(flat_g.codomain)
    f = Nest_morphism(domain_f,codomain_f,flat_f.map)
    g = Nest_morphism(domain_g,codomain_g,flat_g.map)
    return f, g

def random_Nest_morphisms_with_disjoint_images():
    flat_f, flat_g = random_Tuple_morphisms_with_disjoint_images()
    domain_f   = random_profile(length = len(flat_f.domain)).sub(flat_f.domain)
    codomain_f = random_profile(length = len(flat_f.codomain)).sub(flat_f.codomain)
    domain_g   = random_profile(length = len(flat_g.domain)).sub(flat_g.domain)
    codomain_g = codomain_f
    f = Nest_morphism(domain_f,codomain_f,flat_f.map)
    g = Nest_morphism(domain_g,codomain_g,flat_g.map)
    return f,g

def random_Nest_complementable_morphism():
    flat_f = random_Tuple_complementable_morphism()
    domain_f   = random_NestedTuple(length = len(flat_f.domain)).sub(flat_f.domain)
    codomain_f = random_NestedTuple(length = len(flat_f.codomain)).sub(flat_f.codomain)
    map_f      = flat_f.map
    f = Nest_morphism(domain_f,codomain_f,map_f)
    return f

def random_Nest_divisible_morphisms(min_length = 2, max_length = 9, max_value = 10):
    flat_f, flat_g = random_Tuple_divisible_morphisms(min_length = min_length, max_length = max_length, max_value = max_value)
    domain_f    = random_profile(length = len(flat_f.domain)).sub(flat_f.domain)
    codomain_f  = random_profile(length = len(flat_f.codomain)).sub(flat_f.codomain)
    map_f       = flat_f.map
    domain_g    = random_profile(length = len(flat_g.domain)).sub(flat_g.domain)
    codomain_g  = NestedTuple(domain_f.data)
    map_g       = flat_g.map
    f = Nest_morphism(domain_f,codomain_f,map_f)
    g = Nest_morphism(domain_g,codomain_g,map_g)
    return f,g

def random_Nest_productable_morphisms(min_length = 2, max_length = 9, max_value = 10):
    flat_f,flat_g = random_Tuple_productable_morphisms(min_length = min_length, max_length = max_length, max_value = max_value)
    domain_f    = random_profile(length = len(flat_f.domain)).sub(flat_f.domain)
    codomain_f  = random_profile(length = len(flat_f.codomain)).sub(flat_f.codomain)
    map_f       = flat_f.map
    domain_g    = random_profile(length = len(flat_g.domain)).sub(flat_g.domain)
    codomain_g  = random_profile(length = len(flat_g.codomain)).sub(flat_g.codomain)
    map_g       = flat_g.map
    f = Nest_morphism(domain_f,codomain_f,map_f)
    g = Nest_morphism(domain_g,codomain_g,map_g)
    return f,g
