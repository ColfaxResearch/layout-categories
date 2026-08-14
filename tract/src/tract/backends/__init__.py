"""
Layout backends: bridges between tract morphisms and concrete CuTe layout
implementations.

- :mod:`tract.backends.cute_dsl` — NVIDIA's cutlass CuTe DSL (GPU toolchain)
- :mod:`tract.backends.pycute` — NVIDIA's pycute reference implementation
  (pure Python, no GPU)

Neither backend is imported at package-import time; import the one whose
dependency you have installed. The shared, backend-independent algorithms
live in :mod:`tract.backends.base`.
"""
