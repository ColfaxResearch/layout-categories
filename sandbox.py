import numpy as np
import cutlass
import cutlass.cute as cute 

from categories import *
from test_utils import *
from layout_utils import *
from tests import *
import torch
import torch.utils.dlpack as dlpack

@cute.jit
def example():
    f=Nest_morphism(NestedTuple((2,2,2,2)), NestedTuple((2,5,2,5,2,2)), (1,3,5,6))
    g=Nest_morphism(NestedTuple((2,2)), NestedTuple((2,2,2,2)), (3,1))
    L_f = compute_layout(f)
    L_g = compute_layout(g)
    quotient = cute.logical_divide(L_f,L_g)
    print(f)
    print(g)
    print("L_f:", L_f)
    print("L_g:", L_g)
    print("quotient:", quotient)
example()


