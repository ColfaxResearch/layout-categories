# layout-categories

This respository contains the companion software for the paper "Categorical Foundations for CuTe layouts", by Colfax Research. The paper and our accompanying blog post can be found [on our website](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/). 

In the paper, we develop a robust algebra of morphisms in the categories $\text{Tuple}$ and $\text{Nest}$, which encode flat layous and layouts, respectively. Here, we implement this algebra and demonstrate empirically that the operations align with their counterparts in CuTe, using CuTe DSL. 

Project structure:
```
tract/
├── pyproject.toml
├── README.md
├── examples/
|   ├── images/
|   |   └── nest_morphism_to_tikz_example.png
│   └── example_notebook.ipynb
├── src/
│   └── tract/
│       ├── __init__.py
│       ├── categories.py
│       ├── layout_utils.py
│       ├── test_utils.py
│       └── tuple_morph_tikz.py
└── tests/
    ├── __init__.py
    └── morphism_tests.py
```

To install `tract`, run the following:
```
cd tract
pip install .
```

Run the tests using pytest:
```
pytest tests/morphism_tests.py
```

