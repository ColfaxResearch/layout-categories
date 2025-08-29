import numpy as np
import cutlass
import cutlass.cute as cute 

from categories import *
from test_utils import *
from layout_utils import *
from tests import *

@cute.jit
def example():
    f = Tuple_morphism((6,6),(6,6),(2,1))
    g = Tuple_morphism((2,6,3),(6,10,2,10,3),(3,1,5))
    A = compute_flat_layout(f)
    B = compute_flat_layout(g)
    print(cute.composition(B,A))

X = NestedTuple( ())
Y = NestedTuple( ( (4,4,4,4),(4,(4,4)),(4,4,(64,64))))
Xprime, Yprime = composability_algorithm(X,Y)
print(X)
print(Xprime)
print(Y)
print(Yprime)
print(Xprime.refines(X))
print(Yprime.refines(Y))
