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

def random_Fin_morphism():
    """
    Description:
    Randomly generate a morphism alpha in Fin_*
    """
    domain = np.random.randint(1,9)
    codomain = np.random.randint(1,9)
    map_ = [0 for _ in range(domain)]

    max_size = min(domain, codomain)
    u = np.random.rand()
    skewed = 1 - u**3 # skew toward larger values
    size_of_image = int(np.round(skewed * max_size))
    size_of_image = max(0, min(size_of_image, max_size))

    image = random_ordered_subtuple(codomain, size_of_image)
    surviving_indices = sorted(random_ordered_subtuple(domain, size_of_image))
    for i, index in enumerate(surviving_indices):
        map_[index - 1] = int(image[i])
    map_ = tuple(map_)

    return Fin_morphism(domain, codomain, map_)
    
def random_Tuple_morphism():
    """
    Description:
    Randomly generates a tuple morphism f.
    """
    domain_length = np.random.randint(1,9)
    codomain_length = np.random.randint(1,9)
    map_ = [0 for _ in range(domain_length)]

    max_size = min(domain_length, codomain_length)
    u = np.random.rand()
    skewed = 1 - u**3 # skew toward larger values
    size_of_image = int(np.round(skewed * max_size))
    size_of_image = max(0, min(size_of_image, max_size))

    image = random_ordered_subtuple(codomain_length, size_of_image)
    surviving_indices = sorted(random_ordered_subtuple(domain_length, size_of_image))
    for i, index in enumerate(surviving_indices):
        map_[index - 1] = int(image[i])
    map_ = tuple(map_)

    domain = []
    for i in range(domain_length):
        domain.append(int(np.random.randint(1,10)))
    domain = tuple(domain)
    

    codomain = []
    for j in range(codomain_length):
        codomain.append(int(np.random.randint(1,10)))
    

    for i,value in enumerate(map_):
        if value>0:
            codomain[value-1] = domain[i]
    codomain = tuple(codomain) 

    return Tuple_morphism(domain,codomain, map_)

def random_Tuple_composable_morphisms():
    """
    Description:
    Randomly generate a pair of composable tuple morphisms f and g.
    """
    f = random_Tuple_morphism()

    domain = f.codomain
    domain_length = len(domain)

    codomain_length = np.random.randint(1,9)
    codomain = []
    for k in range(codomain_length):
        codomain.append(np.random.randint(1,10))

    #Generate map
    map_ = [0 for _ in range(domain_length)]

    max_size = min(domain_length, codomain_length)
    u = np.random.rand()
    skewed = 1 - u**3 # skew toward larger values
    size_of_image = int(np.round(skewed * max_size))
    size_of_image = max(0, min(size_of_image, max_size))
    image = random_ordered_subtuple(codomain_length,size_of_image)
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

def random_Tuple_morphisms_with_disjoint_images():
    """
    Description:
    Randomly generates a pair of tuple morphisms f, g with the same codomain and disjoint images.
    """
    f = random_Tuple_morphism()
    codomain = f.codomain
    codomain_length = len(codomain)

    domain_length = np.random.randint(1,9)
    domain = []
    for i in range(domain_length):
        domain.append(np.random.randint(1,10))

    #Compute possible values for the map beta underlying g.
    possible_values = []
    for j in range(1,codomain_length+1):
        if j not in f.map:
            possible_values.append(j)

    #Select a random subset of those possible values
    max_size = min(domain_length, len(possible_values))
    u = np.random.rand()
    skewed = 1 - u**3 # skew toward larger values
    size_of_image = int(np.round(skewed * max_size))
    size_of_image = max(0, min(size_of_image, max_size))

    image = [int(x) for x in np.random.choice(possible_values,size_of_image,replace=False)]
    map_ = [0]*domain_length
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

def random_Tuple_divisible_morphisms():
    """
    Description:
    Randomly generates a pair of tuple morphisms f, g such that
    g is injective, and codomain(g) = domain(f).
    """
    f = random_Tuple_morphism()
    codomain = f.domain

    domain_length = np.random.randint(1,len(codomain)+1)
    domain = [0]*domain_length

    #Generate map
    map_ = random_ordered_subtuple(len(codomain),len(domain))

    for i,value in enumerate(map_):
        if value>0:
            domain[i] = codomain[value-1]
    domain = tuple(domain)
    
    g = Tuple_morphism(domain,codomain,map_)

    return f, g 

def random_Tuple_complementable_morphism():
    """
    Description:
    Randomly generates a complementable tuple morphism f 
    with length(codomain)>1 and 0<length(domain)<length(codomain). Still need
    to cover the case that length(domain) \in {0,length(codomain)}.
    """
    codomain = []
    for _ in range(np.random.randint(2,10)):
        codomain.append(np.random.randint(1,10))
    codomain = tuple(codomain)

    size_of_image = np.random.randint(1,len(codomain))
    map_ = random_ordered_subtuple(len(codomain),size_of_image)

    domain = []
    for _, value in enumerate(map_):
        domain.append(codomain[value - 1])
    domain = tuple(domain)
    
    return Tuple_morphism(domain,codomain,map_)

def random_Tuple_productable_morphisms():
    f = random_Tuple_complementable_morphism()
    codomain = f.complement().domain

    domain = [] 
    size_of_image = np.random.randint(0,len(codomain)+1)
    m = size_of_image + np.random.randint(0,5)
    for _ in range(m):
        domain.append(np.random.randint(1,9))
    image = random_ordered_subtuple(len(codomain),size_of_image)
    map_ = [0 for _ in range(len(domain))]
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


    

    
