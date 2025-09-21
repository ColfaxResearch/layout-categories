# layout-categories

This respository contains the companion software for the paper "Categorical Foundations for CuTe Layouts", by Colfax Research.

Project structure:
```
layout-categories
|> categories.py        # Core category-theoretic definitions
|> layout_utils.py      # CuTe layout manipulation logic
|> test_utils.py        # Morphism generators and helpers
|> tests.py             # Unit tests for correctness and agreement
|> notebook.ipynb       # Interactive notebook for demonstrations
```

Dependencies can be found in requirements.txt, and installed with 
```
pip install -r requirements.txt
```

Run the tests using pytest:
```
pytest tests.py
```
