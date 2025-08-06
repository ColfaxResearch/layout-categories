import numpy as np
import cutlass
import cutlass.cute as cute 

from categories import *
from test_utils import *
from layout_utils import *
from tests import *


@cute.jit
def example():
    A = cute.make_layout(((3,3),(()),(4,)), stride = ((2,12),(()),(36,)))
    f = compute_Nest_morphism(A)
    print(f)
example()
