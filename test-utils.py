import torch 
import numpy as np
import cutlass
import cutlass.cute as cute 

from categories import *



#*************************************************************************
# RANDOM GENERATION FUNCTIONS
#*************************************************************************

def random_subset(m: int, k: int):  
    """
    Description:
    Randomly generate an ordered subset of [1,...,m] of size k. 
    """
    if m < 0:
        raise ValueError("m must be a non-negative integer.")
    if not (0 <= k <= m):
        raise ValueError("k must satisfy 0 ≤ k ≤ m.")

    if m == 0:
        subset = []
    else:
        full_set = list(range(1, m + 1))
        subset = [int(x) for x in list(np.random.choice(full_set, size=k, replace=False))]
    return subset

def random_Fin_morphism():
    """
    Description:
    Randomly generate a morphism alpha in Fin_*
    """
    #Generate domain = m and codomain = n
    domain = int(np.random.randint(1,5))
    codomain = int(np.random.randint(1,5))

    #Generate map
    map_ = [0 for _ in range(domain)]
    size_of_image = min(int(np.random.randint(0,15)),min(domain,codomain))
    image = random_subset(codomain,size_of_image)
    surviving_indices = sorted(random_subset(domain,size_of_image))
    for i,index in enumerate(surviving_indices):
        map_[index-1] = int(image[i])

    return Fin_morphism(domain,codomain,map_)
    
def random_Tuple_morphism():
    """
    Description:
    Randomly generates a tuple morphism f.
    """
    underlying_map = random_Fin_morphism()

    domain = []
    for i in range(underlying_map.domain):
        domain.append(int(np.random.randint(1,5)))
    
    codomain = []
    for j in range(underlying_map.codomain):
        codomain.append(int(np.random.randint(1,5)))
    
    for i,value in enumerate(underlying_map.map):
        if value>0:
            codomain[value-1] = domain[i]
    return Tuple_morphism(domain,codomain,underlying_map.map)

def random_Tuple_composable_morphisms():
    """
    Description:
    Randomly generate a pair of composable tuple morphisms f and g.
    """
    f = random_Tuple_morphism()

    domain = f.codomain

    codomain = []
    for k in range(int(np.random.randint(0,10))):
        codomain.append(int(np.random.randint(1,10)))

    #Generate map
    map_ = [0 for _ in range(len(domain))]
    size_of_image = min(int(np.random.randint(0,15)),min(len(domain),len(codomain)))
    image = random_subset(len(codomain),size_of_image)
    surviving_indices = sorted(random_subset(len(domain),size_of_image))
    for i,index in enumerate(surviving_indices):
        map_[index-1] = image[i]

    for i,value in enumerate(map_):
        if value>0:
            codomain[value-1] = domain[i]
    
    g = Tuple_morphism(domain,codomain,map_)

    return f, g 

def random_Tuple_morphisms_with_disjoint_images():
    """
    Description:
    Randomly generates a pair of tuple morphisms f, g with the same codomain and disjoint images.
    """
    f = random_Tuple_morphism()
    domain = []
    for i in range(int(np.random.randint(0,20))):
        domain.append(int(np.random.randint(1,20)))
    codomain = f.codomain

    #Compute possible values for beta
    possible_values = []
    for j in range(1,len(codomain)):
        if j not in f.map:
            possible_values.append(j)

    #Select a random subset of those possible values
    size_of_image = int(np.random.randint(0,min(len(domain),len(possible_values))+1))
    image = [int(x) for x in list(np.random.choice(possible_values,size_of_image,replace=False))]
    map_ = [0]*len(domain)
    surviving_indices = sorted(random_subset(len(domain),size_of_image))
    for i,index in enumerate(surviving_indices):
        map_[index-1] = image[i]

    for i in range(len(domain)):
        if map_[i] > 0:
            domain[i] = codomain[map_[i]-1]
    
    g = Tuple_morphism(domain,codomain,map_)
    return f,g