import random
import numpy as np
import cutlass
import cutlass.cute as cute

#*************************************************************************
# THE CATEGORY E_0 = E_0^otimes
#*************************************************************************

class Fin_morphism: 
    """
    Objects of the class Fin_morphism encode morphisms in E_0 = E_0^otimes. 
    If alpha:<m>_* -> <n>_* is a morphism in E_0, we encode alpha by
    alpha.domain       = m
    alpha.codomain     = n
    alpha.map          = the tuple of length m whose ith entry is = 0          if alpha(i) = *, and 
                                                                  = alpha(i)   otherwise.
    """
    def __init__(self, domain: int, codomain: int, map_: tuple, name: str = ''):
        self.domain     = domain
        self.codomain   = codomain
        self.map        = map_
        self.name       = name
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Checks that the input data define a valid morphism in E_0.
        
        :return: True if the input data define a valid morphism in E_0, False otherwise.
        :rtype: bool
        """
        if not all([value >= 0 and value <= self.codomain for value in self.map]):
            raise ValueError(f"All values in the map must be between 0 and {self.codomain}")
        
        nonzero_vals = [x for x in self.map if x > 0]
        
        if len(set(nonzero_vals)) < len(nonzero_vals):
            raise ValueError(f"The map ({self.map}) must contain no duplicate values")
 
    # def __str__(self):
    #     # Description:
    #     # Print the given morphism in Fin_*.
    #     line1 = f"{'Domain':<10} = <{self.domain}>_*"
    #     line2 = f"{'Codomain':<10} = <{self.codomain}>_*"
    #     line3 = f"{'Map':<10} = {self.map}"
    #     max_length = max(len(line1), len(line2), len(line3))
        
    #     out = '\n'
    #     out += '=' * max_length + '\n'
    #     out += f'Fin_morphism {self.name}:\n'
    #     out += '-' * max_length + '\n'
    #     out += line1 + '\n'
    #     out += line2 + '\n'
    #     out += line3 + '\n'
    #     out += '-' * max_length + '\n'
    #     return out

    def __str__(self):
        """
        Description:
        Print the given morphism in Fin_*.
        """
        domain_string   = str(self.domain)
        codomain_string = str(self.codomain)
        map_string      = str(self.map)
        output          = domain_string + ":" + codomain_string + ":" + map_string
        return output
    
    def __repr__(self):
        return f"Fin_morphism(domain={self.domain}, codomain={self.codomain}, map={self.map})"

    def compose(self, beta):
        """
        Compute the composition 
              beta o alpha:<m>_* -> <p>_*
        of morphisms 
                     alpha:<m>_* -> <n>_*, and 
                      beta:<n>_* -> <p>_* 
        in Fin_*.

        :param beta: second morphism to compose
        :type beta: Fin_morphism
        
        :return beta o alpha:
        :rtype: Fin_morphism
        """
        if self.codomain != beta.domain:
            raise ValueError("The given morphisms are not composable.")
        
        composite = []
        for value in self.map:
            if value>0:
                composite.append(beta.map[value-1])
            else:
                composite.append(0)
        map_   = tuple(composite)
        result = Fin_morphism(self.domain,beta.codomain,map_)
        return result

    def sum(self, beta):
        """
        Compute the sum 
        alpha oplus beta: <m+p>_* -> <n+q>_*
        of morphisms 
        alpha:<m>_* -> <n>_*, and 
        beta:<p>_* -> <q>_*
        in E_0
        
        :param beta: second morphism in sum
        :type beta: Fin_morphism
        
        :return: alpha oplus beta
        :rtype: Fin_morphism
        """
        shifted = []
        for value in beta.map:
            if value == 0:
                shifted.append(0)
            else:
                shifted.append(value+self.codomain)
        return Fin_morphism(self.domain+beta.domain, self.codomain+beta.codomain, self.map+tuple(shifted))

    def images_are_disjoint(self, beta):
        """
        Description: 
        Checks if morphisms alpha and beta in E_0 have disjoint images.
        """

        if self.codomain != beta.codomain:
            raise ValueError("The given maps do not have the same codomain.")

        # Construct the image of alpha
        seen_values = set()
        for _,value in enumerate(self.map):
            if value > 0:
                seen_values.add(value)
        
        # Check that no value of beta is in the image of alpha
        for _,value in enumerate(beta.map):
            if (value > 0) and value in seen_values:
                return False
            
        return True

    def wedge(self, beta):
        """
        Compute the wedge sum 
        alpha v beta : <m+p>_* -> <n>_*
        of two morphisms 
        alpha:<m>_* -> <n>_*, and 
        beta:<p>_* -> <n>_*
        in E_0 with the same codomain and disjoint images.

        :param beta: second morphism in wedge sum
        :type beta: Fin_morphism        
        
        :return: alpha v beta
        :rtype: Fin_morphism
        """
        if not self.codomain == beta.codomain:
            raise ValueError("The given morphisms do not have the same codomain.")
        if not self.images_are_disjoint(beta):
            raise ValueError("The given morphisms do not have disjoint images.")
        if self.images_are_disjoint(beta):
            alpha_wedge_beta = Fin_morphism(self.domain+beta.domain,self.codomain,self.map+beta.map)
            return alpha_wedge_beta
        else:
            return None

#*************************************************************************
# NESTED TUPLES
#*************************************************************************

class NestedTuple:
    def __init__(self, data):
        if not self._validate(data):
            raise ValueError("Only integers or nested tuples of integers are allowed.")
        self.data = data

    def _validate(self, obj):
        if isinstance(obj, int):
            return True
        elif isinstance(obj, tuple):
            return all(self._validate(item) for item in obj)
        return False

    def _custom_repr(self, obj):
        if isinstance(obj, int):
            return str(obj)
        elif isinstance(obj, tuple):
            if not obj:
                return "()"  # empty tuple
            elif len(obj) == 1:
                # Single-element tuple → no trailing comma
                return f"({self._custom_repr(obj[0])})"
            else:
                # Multiple elements → comma-separated
                inner = ",".join(self._custom_repr(item) for item in obj)
                return f"({inner})"

    def __repr__(self):
        return self._custom_repr(self.data)

    def _flatten(self, obj):
        if isinstance(obj, int):
            yield obj
        elif isinstance(obj, tuple):
            for item in obj:
                yield from self._flatten(item)
    
    def flatten(self):
        return tuple(self._flatten(self.data))

    def __iter__(self):
        return iter(self.flatten())
    
    def __getitem__(self, index):
        return self.flatten()[index]
    
    def length(self):
        return len(self.flatten())
    
    def rank(self):
        if isinstance(self.data,int):
            return 1
        return len(self.data)
    
    def size(self):
        size = 1
        for entry in self.flatten():
            size*=entry
        return size

    def entry(self, i):
        if i < 1 or i > self.length():
            raise IndexError("Index out of range")
        return self[i-1]

    def mode(self, i):
        if not isinstance(i, int) or i < 1:
            raise IndexError("Mode index must be a positive integer.")
        
        if isinstance(self.data, int):
            if i == 1:
                return NestedTuple(self.data)
            else:
                raise IndexError("An integer NestedTuple S has only one mode: mode_1(S).")
        
        if i > self.length():
            raise IndexError(f"Mode index {i} out of range.")
        
        return NestedTuple(self.data[i - 1])
    
    def sub(self, values):
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
    
    def profile(self):
        return self.sub(tuple([0]*self.length()))
    
    def is_congruent_to(self,other: 'NestedTuple'):
        return self.profile().data == other.profile().data
    

    
#*************************************************************************
# THE CATEGORY Tuple
#*************************************************************************

class Tuple_morphism:
    """Objects of the class Tuple_morphism encode morphisms in the category Tuple.
    If f:(s_1,..s_m) -> (t_1,...,t_n) is a tuple morphism lying over 
    alpha: <m>_* -> <n>_*, we encode f by
    f.domain              = (s_1,...,s_m)
    f.codomain            = (t_1,...,t_n)
    f.underlying_map      = alpha
    
    :param domain: domain of the morphism
    :type domain: list
    :param codomain: codomain of the morphism
    :type codomain: list
    :param map: map of the morphism
    :type map: list
    
    :return: Tuple_morphism
    :rtype: Tuple_morphism
    """
    def __init__(self, domain: tuple, codomain: tuple, map: tuple, name: str = ''):
        self.domain         = domain
        self.codomain       = codomain
        self.map            = map
        self.name           = name
        self.underlying_map = Fin_morphism(len(self.domain),len(self.codomain),self.map, self.name)
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """
        Verifies that the input data define a valid morphism in C.
        """
        if len(self.domain) != self.underlying_map.domain:
            raise ValueError(f"Domain of the morphism ({self.domain}) must match the domain of the underlying map ({self.underlying_map.domain})")
        
        if len(self.codomain) != self.underlying_map.codomain:
            raise ValueError(f"Codomain of the morphism ({self.codomain}) must match the codomain of the underlying map ({self.underlying_map.codomain})")
        
        for i, value in enumerate(self.underlying_map.map):
            if value != 0:
                if self.domain[i] != self.codomain[value - 1]:
                    raise ValueError(f"The map ({self.underlying_map.map}) must satisfy s_i = t_alpha(i) for all i in <{self.underlying_map.domain}>_*")

    def __repr__(self):
        return f"Tuple_morphism(domain={self.domain}, codomain={self.codomain}, map={self.map})"
    
    # def __str__(self):
    #     line1 = f"{'Domain':<10} = {self.domain}"
    #     line2 = f"{'Codomain':<10} = {self.codomain}"
    #     line3 = f"{'Map':<10} = {self.map}"
    #     max_length = max(len(line1), len(line2), len(line3))
        
    #     out = '\n'
    #     out += '=' * max_length + '\n'
    #     out += f'Tuple_morphism {self.name}:\n'
    #     out += '-' * max_length + '\n'
    #     out += line1 + '\n'
    #     out += line2 + '\n'
    #     out += line3 + '\n'
    #     out += '=' * max_length + '\n'
    #     return out

    def __str__(self):
        """
        Description:
        Print the given tuple morphism.
        """
        domain_string   = str(self.domain)
        codomain_string = str(self.codomain)
        map_string      = str(self.map)
        output          = domain_string + ":" + codomain_string + ":" + map_string
        return output
    
    def size(self):
        size = 1
        for entry in self.domain:
            size*=entry
        return size

    def cosize(self):
        cosize = 1
        for entry in self.codomain:
            cosize*=entry
        return cosize

    def is_sorted(self):
        """
        Checks if the given morphism in C is sorted.
        
        :return: True if the given morphism is sorted, False otherwise.
        :rtype: bool
        """
        m    = len(self.domain)
        n    = len(self.codomain)
        map_ = self.map
        for i, value in enumerate(map_):
            if (value == 0) and (i>0) and ((map_[i-1]!=0) or ((map_[i-1]==0) and self.domain[i-1]>self.domain[i])):
                return False
            if (i<m-1) and (value != 0) and (map_[i+1]!=0) and (value > map_[i+1]):
                for j in range(map_[i+1]-1,value):
                    if self.codomain[j]!=1:
                        return False  
        return True
    
    def is_coalesced(self):
        """
        Checks if the given morphism in C is coalesced.
        
        :return: True if the given morphism is coalesced, False otherwise.
        :rtype: bool
        """
        m = len(self.domain)
        map_ = self.map
        for i in range(m):
            if self.domain[i] == 1:
                return False
            if (i<m-1) and (map_[i]>0) and (map_[i]<map_[i+1]):
                result = False
                for j in range(map_[i]+1,map_[i+1]):
                    if self.codomain[j-1]>1:
                        result = True
                if not result:
                    return False
        return True

    def are_composable(self, g: 'Tuple_morphism') -> bool:
        """
        Checks if the given morphisms in C are composable.
        
        :param g: second morphism in composition
        :type g: Tuple_morphism 
        
        :return: True if the given morphisms are composable, False otherwise.
        :rtype: bool
        """
        if self.codomain != g.domain:
            return False
        return True

    def compose(self, g: 'Tuple_morphism') -> 'Tuple_morphism':
        """
        Computes the composition g o f of two morphisms f and g in C.
        
        :param g: second morphism in composition
        :type g: Tuple_morphism 
        
        :return: g o f
        :rtype: Tuple_morphism
        """
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")
        
        composite = Tuple_morphism(self.domain,g.codomain,self.underlying_map.compose(g.underlying_map).map)
        return composite
        
    def sum(self, g: 'Tuple_morphism') -> 'Tuple_morphism':
        """
        Computes the sum f oplus g of two morphisms f and g in C.
        
        :param g: second morphism in sum
        :type g: Tuple_morphism
        
        :return: f oplus g
        :rtype: Tuple_morphism
        """
        sum = Tuple_morphism(self.domain+g.domain,self.codomain+g.codomain,self.underlying_map.sum(g.underlying_map).map)
        return sum

    def restrict(self, subtuple: tuple):
        """
        Restricts a morphism f in C to some subtuple of its domain.
        
        :param subtuple: subtuple of the domain of f
        :type subtuple: tuple
        
        :return: restricted morphism
        :rtype: Tuple_morphism
        """
        if not all(1 <= index <= len(self.domain) for index in subtuple):
            raise ValueError("Invalid subtuple indices.")
        
        if not all(subtuple[i] < subtuple[i+1] for i in range(len(subtuple)-1)):
            raise ValueError("Subtuple must be strictly increasing.")
        
        restricted_domain = tuple([self.domain[index-1] for index in subtuple])
        restricted_map    = tuple([self.map[index-1] for index in subtuple])
        
        return Tuple_morphism(
            restricted_domain,
            self.codomain,
            restricted_map
        )

    def factorize(self, subtuple: tuple):
        """ 
        Factorizes a morphism f:(s_1,...,s_m) -> (t_1,...,t_n)
        through some subtuple (t_{j_1},...,t_{j_r}) -> (t_1,...,t_n).
        
        :param subtuple: subtuple of the codomain of f
        :type subtuple: tuple
        
        :return: factorized morphism
        :rtype: Tuple_morphism
        """

        domain   = self.domain
        codomain = tuple([self.codomain[j-1] for j in subtuple])
        map_     = []
        for value in self.map:
            if value == 0:
                map_.append(value)
            else:
                missing_count = sum(1 for i in range(1,value) if i not in subtuple)
                map_.append(value - missing_count)
        return Tuple_morphism(domain,codomain,tuple(map_))

    def sort(self):
        """
        Sorts the given morphism f in C.
        
        :return: sorted morphism
        :rtype: Tuple_morphism
        """
        alpha = self.map

        # Extract   P = The collection of indices i in <m> with alpha(i) = *,
        #           Q = The collection of indices i in <m> with alpha(i) != *.
        P = []
        Q = []
        for i,value in enumerate(alpha):
            if value == 0:
                P.append(i+1)
            else:
                Q.append(i+1)

        #Reorder P so that i_1 < i_2 implies 
        #               s_{i_1} < s_{i_2}, or 
        #               s_{i_1} = s_{i_2} and i_1 < i_2
        P_sorted = sorted(P, key=lambda i: (self.domain[i - 1], i))

        # Reorder Q so that j_1 < j_2 implies alpha(j_1)<alpha(j_2)
        Q_sorted = sorted(Q, key=lambda j: alpha[j - 1])

        permutation = P_sorted+Q_sorted
        domain_of_g = []
        for entry in permutation:
            domain_of_g.append(self.domain[entry-1])
        
        g = Tuple_morphism(tuple(domain_of_g),self.domain,tuple(permutation))
        sorted_morphism = g.compose(self)
        return sorted_morphism

    def images_are_disjoint(self, g):
        """
        Checks if morphisms f and g in C with the same codomain have disjoint images.
        
        :param g: second morphism in comparison
        :type g: Tuple_morphism
        
        :return: True if the given morphisms have disjoint images, False otherwise.
        :rtype: bool
        """
        return self.underlying_map.images_are_disjoint(g.underlying_map)

    def concat(self, g):    
        """
        Computes the concatenation concat(f,g) of morphisms f and g in C
        with the same codomain and disjoint images.
        
        :param g: second morphism in concatenation
        :type g: Tuple_morphism
        
        :return: concatenation of f and g
        :rtype: Tuple_morphism
        """
        if not self.images_are_disjoint(g):
            raise ValueError("The given morphisms do not have the same codomain and disjoint images.")
        
        wedge_result = self.underlying_map.wedge(g.underlying_map)
        if wedge_result is None:
            raise ValueError("The given morphisms do not have disjoint images.")
        concat = Tuple_morphism(self.domain+g.domain,self.codomain,wedge_result.map)
        return concat

    def squeeze(self):
        """
        Removes all ones from domain and codomain of f.
        
        :return: morphism with ones removed
        :rtype: Tuple_morphism
        """
        domain_subtuple     = tuple([i+1 for i in range(len(self.domain)) if self.domain[i] != 1])
        restricted_morphism = self.restrict(domain_subtuple)
        codomain_subtuple   = tuple([j+1 for j in range(len(restricted_morphism.codomain)) if restricted_morphism.codomain[j] != 1])

        return restricted_morphism.factorize(codomain_subtuple)

    def coalesce(self):
        """Computes the coalescence of a morphism in C. 
        
        :return: A coalesced morphism in C
        :rtype: Tuple_morphism
        """
        morphism = self.squeeze()
        m = len(morphism.domain)
        n = len(morphism.codomain)

        # Form equivalence classes for domain
        if m>0:
            domain_equivalence_classes = []
            current_equivalence_class = [1]
            for i in range(1, m):
                previous_value  = morphism.map[i - 1]
                current_value   = morphism.map[i]
                if (previous_value == 0 and current_value == 0) or ((previous_value != 0) and current_value == previous_value + 1):
                    current_equivalence_class.append(i + 1)
                else:
                    domain_equivalence_classes.append(current_equivalence_class)
                    current_equivalence_class = [i + 1]
            domain_equivalence_classes.append(current_equivalence_class) 
        else: 
            domain_equivalence_classes = []

        # Form equivalence classes for codomain
        image_of_map = set(morphism.map)
        if n>0:
            codomain_equivalence_classes = []
            current_equivalence_class = [1]
            for j in range(2, n+1):
                previous_index  = j-1
                current_index   = j
                if (j-1 not in image_of_map and j not in image_of_map):
                    current_equivalence_class.append(j)
                elif (j-1 in image_of_map):
                    i=0
                    while morphism.map[i] != j-1:
                        i+=1
                    if (i+1 < m) and (morphism.map[i+1] == j):
                        current_equivalence_class.append(j)
                    else:
                        codomain_equivalence_classes.append(current_equivalence_class)
                        current_equivalence_class = [j]
                else:
                    codomain_equivalence_classes.append(current_equivalence_class)
                    current_equivalence_class = [j]
            codomain_equivalence_classes.append(current_equivalence_class) 
        else: codomain_equivalence_classes = [] 
        coalesced_domain   = []
        for i in range(len(domain_equivalence_classes)):
            equivalence_class = domain_equivalence_classes[i]
            product = 1
            for index in equivalence_class:
                product *= morphism.domain[index-1]
            coalesced_domain.append(product)
        coalesced_codomain = []
        for j in range(len(codomain_equivalence_classes)):
            equivalence_class = codomain_equivalence_classes[j]
            product = 1
            for index in equivalence_class:
                product *= morphism.codomain[index-1]
            coalesced_codomain.append(product)
        coalesced_map = []
        for i in range(len(coalesced_domain)):
            domain_representative = domain_equivalence_classes[i][0]
            codomain_representative = morphism.map[domain_representative-1]
            if morphism.map[domain_representative-1] == 0: 
                coalesced_map.append(0)
            else: 
                index = 0
                while (index < len(coalesced_codomain)) and (codomain_representative not in codomain_equivalence_classes[index]):
                    index+=1
                coalesced_map.append(index+1)

        return Tuple_morphism(tuple(coalesced_domain),tuple(coalesced_codomain),tuple(coalesced_map))

        
    def is_complementable(self):
        """
        Checks if morphism is complementable.

        :return: True if morphism is complementable, False otherwise.
        :rtype: bool
        """
        if 0 in set(self.map):
            return False
        else:
            return True
    
    def complement(self):
        """
        Computes the complement of f.

        :return: Complement of given morphism in C.
        :rtype: Tuple_morphism
        """

        if not self.is_complementable():
            raise ValueError("The given morphism is not admissible for complementation.")
        
        codomain = self.codomain
        image_indices = set(self.map)
        domain = tuple(codomain[i] for i in range(len(codomain)) if i+1 not in image_indices)
        map = tuple(i+1 for i in range(len(codomain)) if i+1 not in image_indices)
        return Tuple_morphism(domain,codomain,map)
    
    def is_isomorphism(self):
        """
        Checks if f is an isomorphism.
        
        :return: True if the given morphisms is an isomorphism, False otherwise.
        :rtype: bool
        """
        m = len(self.domain)
        n = len(self.codomain)
        if (m == n) and set(self.map) == set(range(1,m+1)):
            return True
        else:
            return False
            
    def is_complementary_to(self,other):
        """
        Checks if morphism is complementary to other morphism.
        
        :param g: other morphism
        :type g: Tuple_morphism 
        
        :return: True if morphism is complementary to other morphism, False otherwise.
        :rtype: bool
        """
        if self.codomain != other.codomain:
            raise ValueError("The given morphisms do not have the same codomain.")
        concat = self.concat(other)
        return concat.is_isomorphism()

#*************************************************************************
# THE CATEGORY NestTuple (work in progress)
#*************************************************************************

class Nest_morphism:
    """Objects of the class Nest_morphism encode morphisms in the category NestTuple.
    If f:S -> T is a nested tuple morphism lying over 
    alpha: <m>_* -> <n>_*, we encode f by
    f.domain              = S
    f.codomain            = T
    f.underlying_map      = alpha
    
    :param domain: domain of the morphism
    :type domain: NestedTuple
    :param codomain: codomain of the morphism
    :type codomain: NestedTuple
    :param map: map of the morphism
    :type map: list
    
    :return: Nest_morphism
    :rtype: Nest_morphism
    """
    def __init__(self, domain: NestedTuple, codomain: NestedTuple, map: tuple, name: str = ''):
        self.domain         = domain
        self.codomain       = codomain
        self.map            = map
        self.name           = name
        self.underlying_map = Fin_morphism(len(self.domain.flatten()),len(self.codomain.flatten()),self.map, self.name)
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """
        Verifies that the input data define a valid morphism in C.
        """
        if len(self.domain.flatten()) != self.underlying_map.domain:
            raise ValueError(f"Domain of the morphism ({self.domain}) must match the domain of the underlying map ({self.underlying_map.domain})")
        
        if len(self.codomain.flatten()) != self.underlying_map.codomain:
            raise ValueError(f"Codomain of the morphism ({self.codomain}) must match the codomain of the underlying map ({self.underlying_map.codomain})")
        
        for i, value in enumerate(self.underlying_map.map):
            if value != 0:
                if self.domain.flatten()[i] != self.codomain.flatten()[value - 1]:
                    raise ValueError(f"The map ({self.underlying_map.map}) must satisfy s_i = t_alpha(i) for all i in <{self.underlying_map.domain}>_*")

    def __repr__(self):
        return f"Nest_morphism(domain={self.domain}, codomain={self.codomain}, map={self.map})"
    
    def __str__(self):
        line1 = f"{'Domain':<10} = {self.domain}"
        line2 = f"{'Codomain':<10} = {self.codomain}"
        line3 = f"{'Map':<10} = {self.map}"
        max_length = max(len(line1), len(line2), len(line3))
        
        out = '\n'
        out += '=' * max_length + '\n'
        out += f'Nest_morphism {self.name}:\n'
        out += '-' * max_length + '\n'
        out += line1 + '\n'
        out += line2 + '\n'
        out += line3 + '\n'
        out += '=' * max_length + '\n'
        return out
    
    def flatten(self):
        domain = self.domain.flatten()
        codomain = self.codomain.flatten()
        return Tuple_morphism(domain,codomain,self.map)

    def size(self):
        size = 1
        for entry in self.domain.flatten():
            size*=entry
        return size

    def cosize(self):
        cosize = 1
        for entry in self.codomain.flatten():
            cosize*=entry
        return cosize

    def is_sorted(self):
        """
        Checks if the given nested tuple morphism is sorted.
        
        :return: True if the given morphism is sorted, False otherwise.
        :rtype: bool
        """
        return self.flatten().is_sorted()
    
    # def is_coalesced(self):
    #     """
    #     Checks if the given morphism in C is coalesced.
        
    #     :return: True if the given morphism is coalesced, False otherwise.
    #     :rtype: bool
    #     """
    #     m = len(self.domain)
    #     map_ = self.map
    #     for i in range(m):
    #         # If any entry in domain(f) is = 1, then f is not coalesced.
    #         if self.domain[i] == 1:
    #             return False
    #         # If there is any instance of alpha(i)<alpha(i+1), and there does not exist 
    #         # alpha(i)<j<alpha(i+1) with t_j = 1, then f is not coalesced. 
    #         if (i<m-1) and (map_[i]>0) and (map_[i]<map_[i+1]):
    #             result = False
    #             for j in range(map_[i]+1,map_[i+1]):
    #                 if self.codomain[j-1]>1:
    #                     result = True
    #             if not result:
    #                 return False
    #     return True

    def are_composable(self, g: 'Nest_morphism') -> bool:
        """
        Checks if the given morphisms in C are composable.
        
        :param g: second morphism in composition
        :type g: Tuple_morphism 
        
        :return: True if the given morphisms are composable, False otherwise.
        :rtype: bool
        """
        if self.codomain.data != g.domain.data:
            return False
        return True

    def compose(self, g: 'Nest_morphism') -> 'Nest_morphism':
        """
        Computes the composition g o f of two nested tuple morphisms f and g.
        
        :param g: second morphism in composition
        :type g: Nest_morphism 
        
        :return: g o f
        :rtype: Nest_morphism
        """
        if self.codomain.data != g.domain.data:
            raise ValueError("The given morphisms are not composable.")
        
        composite = Nest_morphism(self.domain,g.codomain,self.underlying_map.compose(g.underlying_map).map)
        return composite
        
    # def sum(self, g: 'Tuple_morphism') -> 'Tuple_morphism':
    #     """
    #     Computes the sum f oplus g of two morphisms f and g in C.
        
    #     :param g: second morphism in sum
    #     :type g: Tuple_morphism
        
    #     :return: f oplus g
    #     :rtype: Tuple_morphism
    #     """
    #     sum = Tuple_morphism(self.domain+g.domain,self.codomain+g.codomain,self.underlying_map.sum(g.underlying_map).map)
    #     return sum

    # def restrict(self, subtuple: list):
    #     """
    #     Restricts a morphism f in C to some subtuple of its domain.
        
    #     :param subtuple: subtuple of the domain of f
    #     :type subtuple: list
        
    #     :return: restricted morphism
    #     :rtype: Tuple_morphism
    #     """
    #     if not all(1 <= index <= len(self.domain) for index in subtuple):
    #         raise ValueError("Invalid subtuple indices.")
        
    #     if not all(subtuple[i] < subtuple[i+1] for i in range(len(subtuple)-1)):
    #         raise ValueError("Subtuple must be strictly increasing.")
        
    #     restricted_domain = [self.domain[index-1] for index in subtuple]
    #     restricted_map = [self.map[index-1] for index in subtuple]
        
    #     return Tuple_morphism(
    #         restricted_domain,
    #         self.codomain,
    #         restricted_map
    #     )

    # def factorize(self, subtuple: list):
    #     """ 
    #     Factorizes a morphism f:(s_1,...,s_m) -> (t_1,...,t_n)
    #     through some subtuple (t_{j_1},...,t_{j_r}) -> (t_1,...,t_n).
        
    #     :param subtuple: subtuple of the codomain of f
    #     :type subtuple: list
        
    #     :return: factorized morphism
    #     :rtype: Tuple_morphism
    #     """
    #     domain = self.domain
    #     codomain = [self.codomain[j-1] for j in subtuple]

    #     domain = self.domain
    #     codomain = [self.codomain[j-1] for j in subtuple]

    #     map_ = []

    #     for value in self.map:
    #         if value == 0:
    #             map_.append(value)
    #         else:
    #             missing_count = sum(1 for i in range(1,value) if i not in subtuple)
    #             map_.append(value - missing_count)

    #     return Tuple_morphism(domain,codomain,map_)

    # def sort(self):
    #     """
    #     Sorts the given morphism f in C.
        
    #     :return: sorted morphism
    #     :rtype: Tuple_morphism
    #     """
    #     alpha = self.map

    #     # Extract   P = The collection of indices i in <m> with alpha(i) = *,
    #     #           Q = The collection of indices i in <m> with alpha(i) != *.
    #     P = []
    #     Q = []
    #     for i,value in enumerate(alpha):
    #         if value == 0:
    #             P.append(i+1)
    #         else:
    #             Q.append(i+1)

    #     #Reorder P so that i_1 < i_2 implies 
    #     #               s_{i_1} < s_{i_2}, or 
    #     #               s_{i_1} = s_{i_2} and i_1 < i_2
    #     P_sorted = sorted(P, key=lambda i: (self.domain[i - 1], i))

    #     # Reorder Q so that j_1 < j_2 implies alpha(j_1)<alpha(j_2)
    #     Q_sorted = sorted(Q, key=lambda j: alpha[j - 1])

    #     permutation = P_sorted+Q_sorted
    #     domain_of_g = []
    #     for entry in permutation:
    #         domain_of_g.append(self.domain[entry-1])
        
    #     g = Tuple_morphism(domain_of_g,self.domain,permutation)
    #     sorted_morphism = g.compose(self)
    #     return sorted_morphism

    def images_are_disjoint(self, g):
        """
        Checks if morphisms f and g in C with the same codomain have disjoint images.
        
        :param g: second morphism in comparison
        :type g: Tuple_morphism
        
        :return: True if the given morphisms have disjoint images, False otherwise.
        :rtype: bool
        """
        if self.codomain.data != g.codomain.data:
            raise ValueError("Morphisms do not have the same codomain.")
        return self.flatten().images_are_disjoint(g.flatten())

    def concat(self, g):    
        """
        Computes the concatenation (f,g) of nested tuple morphisms f and g
        with the same codomain and disjoint images.
        
        :param g: second morphism in concatenation
        :type g: Nest_morphism
        
        :return: concatenation of f and g
        :rtype: Nest_morphism
        """
        if not self.images_are_disjoint(g):
            raise ValueError("The given morphisms do not have the same codomain and disjoint images.")
        concat = Nest_morphism(NestedTuple((self.domain.data,g.domain.data)),
                                self.codomain,
                                self.underlying_map.wedge(g.underlying_map).map)
        return concat

    # def squeeze(self):
    #     """
    #     Removes all ones from domain and codomain of f.
        
    #     :return: morphism with ones removed
    #     :rtype: Tuple_morphism
    #     """
    #     domain_subtuple     = [i+1 for i in range(len(self.domain)) if self.domain[i] != 1]
    #     restricted_morphism = self.restrict(domain_subtuple)
    #     codomain_subtuple   = [j+1 for j in range(len(restricted_morphism.codomain)) if restricted_morphism.codomain[j] != 1]

    #     return restricted_morphism.factorize(codomain_subtuple)

    # def coalesce(self):
    #     """Computes the coalescence of a morphism in C. 
        
    #     :return: A coalesced morphism in C
    #     :rtype: Tuple_morphism
    #     """
    #     morphism = self.squeeze()
    #     m = len(morphism.domain)
    #     n = len(morphism.codomain)

    #     # Form equivalence classes for domain
    #     if m>0:
    #         domain_equivalence_classes = []
    #         current_equivalence_class = [1]

    #         for i in range(1, m):
    #             previous_value  = morphism.map[i - 1]
    #             current_value   = morphism.map[i]

    #             if (previous_value == 0 and current_value == 0) or ((previous_value != 0) and current_value == previous_value + 1):
    #                 current_equivalence_class.append(i + 1)
    #             else:
    #                 domain_equivalence_classes.append(current_equivalence_class)
    #                 current_equivalence_class = [i + 1]

    #         domain_equivalence_classes.append(current_equivalence_class) 
    #     else: 
    #         domain_equivalence_classes = []

    #     # Form equivalence classes for codomain
    #     image_of_map = set(morphism.map)
    #     if n>0:
    #         codomain_equivalence_classes = []
    #         current_equivalence_class = [1]

    #         for j in range(2, n+1):
    #             previous_index  = j-1
    #             current_index   = j

    #             if (j-1 not in image_of_map and j not in image_of_map):
    #                 current_equivalence_class.append(j)
    #             elif (j-1 in image_of_map):
    #                 i=0
    #                 while morphism.map[i] != j-1:
    #                     i+=1
    #                 if (i+1 < m) and (morphism.map[i+1] == j):
    #                     current_equivalence_class.append(j)
    #                 else:
    #                     codomain_equivalence_classes.append(current_equivalence_class)
    #                     current_equivalence_class = [j]
    #             else:
    #                 codomain_equivalence_classes.append(current_equivalence_class)
    #                 current_equivalence_class = [j]

    #         codomain_equivalence_classes.append(current_equivalence_class) 
    #     else: codomain_equivalence_classes = [] 

    #     coalesced_domain   = []
    #     for i in range(len(domain_equivalence_classes)):
    #         equivalence_class = domain_equivalence_classes[i]
    #         product = 1
    #         for index in equivalence_class:
    #             product *= morphism.domain[index-1]
    #         coalesced_domain.append(product)
        
    #     coalesced_codomain = []
    #     for j in range(len(codomain_equivalence_classes)):
    #         equivalence_class = codomain_equivalence_classes[j]
    #         product = 1
    #         for index in equivalence_class:
    #             product *= morphism.codomain[index-1]
    #         coalesced_codomain.append(product)

    #     coalesced_map = []
    #     for i in range(len(coalesced_domain)):
    #         domain_representative = domain_equivalence_classes[i][0]
    #         codomain_representative = morphism.map[domain_representative-1]
    #         if morphism.map[domain_representative-1] == 0: 
    #             coalesced_map.append(0)
    #         else: 
    #             index = 0
    #             while (index < len(coalesced_codomain)) and (codomain_representative not in codomain_equivalence_classes[index]):
    #                 index+=1
    #             coalesced_map.append(index+1)

    #     return Tuple_morphism(coalesced_domain,coalesced_codomain,coalesced_map)
    
    #     def is_admissible_for_complementation(self):
    #         if 0 in set(self.map):
    #             return False
    #         else:
    #             return True 
        
    def is_complementable(self):
        """
        Checks if nested tuple morphism is complementable.

        :return: True if morphism is complementable, False otherwise.
        :rtype: bool
        """
        if 0 in set(self.map):
            return False
        else:
            return True
    
    def complement(self):
        """
        Computes the complement of f.

        :return: Complement of nested tuple morphism.
        :rtype: Nest_morphism
        """

        if not self.is_complementable():
            raise ValueError("The given morphism is not complementable.")
        
        codomain = self.codomain
        image_indices = set(self.map)
        domain = [codomain[i] for i in range(codomain.length()) if i+1 not in image_indices]
        domain = NestedTuple(tuple(domain))
        map = tuple([i+1 for i in range(codomain.length()) if i+1 not in image_indices])
        return Nest_morphism(domain,codomain,map)
    
    def is_isomorphism(self):
        """
        Checks if f is an isomorphism.
        
        :return: True if the given nested tuple morphism is an isomorphism, False otherwise.
        :rtype: bool
        """
        return self.flatten().is_isomorphism()
            
    def is_complementary_to(self,other):
        """
        Checks if morphism is complementary to other morphism.
        
        :param g: other morphism
        :type g: Nest_morphism 
        
        :return: True if morphism is complementary to other morphism, False otherwise.
        :rtype: bool
        """
        if self.codomain.data != other.codomain.data:
            raise ValueError("The given morphisms do not have the same codomain.")
        
        concat = self.concat(other)

        if concat.is_isomorphism():
            return True
        else:
            return False


