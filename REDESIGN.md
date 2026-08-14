# Layout-Categories Redesign

Audit of what the repo currently supports, and a design for refactoring it into
canonical, human-readable code with concept notebooks. Prepared 2026-08-14.

> **Status: implemented 2026-08-14** (all four phases, working tree on
> `jcarlisle/ref-categories`, uncommitted). Decisions taken: renames **with**
> deprecated aliases; both `flat_*`/`logical_*` names kept; all uncalled
> methods kept and given tests; generators tests-only. Deviations from the
> plan below: module names ended up `fin/nested_tuple/tuple_morphism/
> nest_morphism/fact_morphism/ref_morphism/spans/refinement/backends/`
> (with `categories.py`, `layout_utils.py`, `pycute_utils.py` kept as compat
> shims); notebooks live in `tract/notebooks/01…05`; the span-family scenes
> share `scenes/span_common.py` plus the library rather than one parameterized
> scene. End state: `pytest` collects 10,884 tract tests (was 0 from a bare
> run), cross-validation runs the same predicates against both backends,
> `import tract` no longer needs cutlass, and the merged scenes re-render
> frame-identically. §1–§2 below describe the repo **before** the refactor.

---

## 1. What the repo supports today

**`tract`** (~5,700 source lines, ~3,200 test lines) — the categorical algebra:

| Layer | Where | Contents |
|---|---|---|
| Core categories | `categories.py` (1,640 lines) | `Fin_morphism`, `NestedTuple`, `Tuple_morphism`, `Nest_morphism` with compose / coalesce / complement / sort / concat / restrict / factorize / squeeze / logical divide & product / pullback & pushforward along refinements |
| Fact & Ref | `fact_morphism.py`, `ref_morphism.py` | Factorization morphisms and nested-refinement morphisms, with pullback/pushforward and the Ref→Fact flattening functor |
| Spans & cospans | `span.py`, `cospan.py`, `ref_span.py`, `ref_cospan.py` | Span(Tuple, Fact), CoSpan(Tuple, Fact), and their Ref variants, each with identity / compose / sum, plus bridge functors |
| Layout bridges | `layout_utils.py` (cutlass DSL), `pycute_utils.py` (pycute) | Layout ↔ morphism round trips, tractability, `mutual_refinement`, `weak_composite`, complements, concatenation — implemented twice, once per backend |
| TikZ export | `tuple_morph_tikz.py` | Morphism and mutual-refinement diagrams for the paper |
| Test scaffolding | `test_utils.py` (750 lines) | Random generators for property tests, currently exported as public API |
| Tests | 8 files | Cross-validation against cutlass DSL and pycute, plus category-law property tests for every variant |
| Notebooks | `examples/` (3) | Core operations with CuTe checks; Fact; Span/CoSpan. All current with the code, but nothing covers Ref, the span Ref-variants, or pycute |

**`visualization`** (~1,300 library lines, ~11,000 scene lines) — Manim animations:

- Library (`layout_categories_viz`): mapsto-arrow glyphs and animations, morphism stack diagrams, composition diagrams, nested-tuple trees, coalesce geometry, style constants.
- 29 scenes covering: tuple-morphism create/remove, composition, coalesce, layout read-off, nested trees, nest-morphism composition via tree cancellation, mutual refinement, pullback/pushforward, weak composition, Fact/Ref morphism create & compose, Span/CoSpan/RefSpan composition, plus look-and-feel prototypes (graphite lines, decorative variations).
- `ANIMATION_AUDIT.md` tracks done/to-do accurately; `COMPOSABILITY_NOTES.md` records the composability-vs-tractability investigation with verified counterexamples.

---

## 2. Problems the refactor should fix

### tract

1. **`categories.py` is four modules in one** — Fin, NestedTuple, Tuple, Nest, already separated by banner comments at lines 19 / 172 / 545 / 1238.
2. **The span construction is copy-pasted four times.** `span.py` / `cospan.py` / `ref_span.py` / `ref_cospan.py` differ only in leg type, direction, and docstrings (`compose` bodies are byte-identical modulo class name). ~840 lines that should be ~250.
3. **The two layout backends duplicate 14 identically-named functions** (`compute_Tuple_morphism`, `is_tractable`, `flat_complement`, …) with no shared protocol, and the split leaks: `mutual_refinement` and `weak_composite` are pure categorical code but live in `layout_utils.py`, which hard-imports `cutlass` — so the "no-GPU" pycute test path still requires the cutlass DSL.
4. **`pytest` collects zero tests.** Files are named `*_tests.py`; there is no `[tool.pytest.ini_options]` and no `conftest.py`. Only explicit paths (`pytest tests/span_tests.py`) work, so 7 of 8 test files are invisible to a bare `pytest` run.
5. **Tests are not reproducible.** Only `np.random` is seeded; five test files and `test_utils.random_ordered_subtuple` use stdlib `random`, which is never seeded.
6. **`test_utils` is shipped as public API** (18 names in `__all__`), and is the sole reason `numpy` is a runtime dependency.
7. **Circular import**: `tuple_morph_tikz.py` imports from the package root while `layout_utils.py` imports it; works only by import-order luck. The file also contains two independent TikZ generations back-to-back.
8. **Naming drift**: `Tuple_morphism`-style names (and a shadowing `typing.Tuple` import that makes one annotation wrong); `map` vs `map_` constructor args; `flat_divide`/`flat_product` vs `logical_divide`/`logical_product` for the same operation at two levels; the undocumented `compute_morphism` alias.
9. **API asymmetries**: `__eq__`/`__hash__` and `identity`/`is_identity` exist on the new Fact/Ref/Span classes but not on `Tuple_morphism` / `Nest_morphism` — the newer code documents the tuple-comparison workaround instead of fixing it.
10. **Dead code**: unused cutlass try/excepts in `categories.py` and `test_utils.py`, a no-op `main()`, unused imports; several methods with no caller anywhere (see §5, decision 3).
11. **Stale docs**: `tract/README.md` predates the entire Fact/Ref/Span/pycute layer; the root README documents a pytest invocation that only exercises one file.

### visualization

1. **The real drawing library lives inside scene files.** `tuple_pullback_test.py` owns cell/deck/split/arrow primitives imported by 9 other scenes; `tuple_morphism_composition_curve.py` owns the path-matching helpers (6 importers); scenes subclass other scenes' `Scene` classes across files. None of it is shippable from `src/`.
2. **Whole-file forks**: `span_composition_test` vs its `_single_beat` variant are 87% line-identical (real delta ≈ 30 lines in one beat); cospan pair 88%; span vs ref_span 77%; coalesce pair 67% *despite subclassing*; span vs cospan 53% (mirror images).
3. **Boilerplate everywhere**: `self.camera.background_color = BACKGROUND` in all 28 `construct`s (no shared base scene); a `place(column, index)` closure redefined in 10 files; arrow-geometry constants copied in 3 files shadowing `MapstoArrow` defaults; a per-file `@dataclass` + `EXAMPLES` idiom re-invented 14 times with inconsistent field names.
4. **Fragile imports**: `from scenes.X import Y` (37 occurrences) with no `scenes/__init__.py`, no `conftest.py`, no `manim.cfg` — works only when invoked from `visualization/`.
5. **Dead code** (~160 lines: `zipper_progress`, `zipper_arrow`, `_sample_shaft_y`, `handoff_mapsto_arrow`) and an `__init__.py` `__all__` that omits modules scenes actually use (`coalesce`, style constants) while private `_make_entries`/`_mapsto_arrow` are de-facto public across 10 files.
6. **Repo hygiene**: `visualization/TupleMorphismRefinementTest.mp4` (882 KB) and `papers/` are untracked and unignored; everything else is correctly gitignored.

---

## 3. Target design — `tract`

```
tract/src/tract/
├── __init__.py          # curated public API only
├── fin.py               # FinMorphism (pointed finite sets)
├── nested_tuple.py      # NestedTuple
├── tuple_morphism.py    # TupleMorphism   (category Tuple)
├── nest_morphism.py     # NestMorphism    (category Nest)
├── fact.py              # FactMorphism    (category Fact)
├── ref.py               # RefMorphism     (category Ref)
├── spans.py             # one generic construction → Span/CoSpan × Fact/Ref
├── refinement.py        # mutual_refinement, weak_composite (pure, no cutlass)
├── tikz.py              # single-generation TikZ export
└── backends/
    ├── base.py          # shared layout↔morphism algorithms over a small protocol
    ├── cute_dsl.py      # cutlass DSL backend (optional import)
    └── pycute.py        # pycute backend + empty-layout workarounds (optional)
tract/tests/
├── conftest.py          # seeds random + np.random; shared fixtures
├── generators.py        # what test_utils.py is today (numpy → dev dependency)
└── test_*.py            # pytest-collectable names
tract/notebooks/         # renamed from examples/, see §4
```

Key moves, in order of leverage:

**Split `categories.py` along its own banner comments.** Four files, no logic changes. `NestedTuple` is the shared data structure every other module needs, so it goes first.

**One generic span.** A single base parameterized by leg category and direction:

```python
class _SpanBase:
    backward_leg_cls: type       # FactMorphism | RefMorphism
    direction: Literal["span", "cospan"]
    # identity / is_identity / are_composable / compose / sum / __eq__ / __hash__
    # compose uses backward_leg_cls.pullback (span) or .pushforward (cospan)

class SpanMorphism(_SpanBase): ...      # Span(Tuple, Fact)
class CoSpanMorphism(_SpanBase): ...
class RefSpanMorphism(_SpanBase): ...   # + to/from_SpanMorphism functor
class RefCoSpanMorphism(_SpanBase): ...
```

The four Ref↔Fact bridge functors stay as thin methods on the Ref variants.

**Backends behind a protocol.** The 14 duplicated functions become one implementation in `backends/base.py` written against a ~6-method protocol (make layout, get shape/stride, coalesce, compose, …); `cute_dsl.py` and `pycute.py` implement the protocol and keep only genuinely backend-specific code (pycute's empty-layout bug wrappers, `@cute.jit` glue). `mutual_refinement`/`weak_composite` move to `refinement.py` with zero backend imports. Neither backend is imported at package-import time; `pycute` and `nvidia-cutlass-dsl` become optional extras (`tract[cute]`, `tract[pycute]`).

**Naming, one pass, with compat aliases.** PEP 8 class names (`TupleMorphism`, `NestMorphism`, `FactMorphism`, `RefMorphism`, `FinMorphism`); constructor arg `map` everywhere (drop `map_`); pick **`logical_divide` / `logical_product` at both levels** (they're the CuTe names) and delete `flat_divide`/`flat_product`; delete the `compute_morphism` alias. `__init__.py` keeps `Tuple_morphism = TupleMorphism` etc. as deprecated aliases for one release so paper-companion code keeps running.

**Symmetric core API.** Add `__eq__`/`__hash__` and `identity()`/`is_identity()` to `TupleMorphism`, `NestMorphism`, `NestedTuple`; delete the tuple-comparison workarounds in `ref_span.py` and the tests.

**Tests.** Rename to `test_*.py`, add `[tool.pytest.ini_options]` with `testpaths = ["tests"]`, seed both RNGs in a `conftest.py` fixture, hoist the shared generators and the cross-file test imports into `conftest.py`/`generators.py`. Unify `morphism_tests.py` and `pycute_morphism_tests.py`: the 13 `*_agree` predicates are written once against the backend protocol and parameterized over available backends (`pytest.importorskip` per backend).

**Docstring policy** (this is the "without tons of unnecessary documentation" pass): one-line summary per public function; keep a short mathematical statement where the operation has one (e.g. the divisibility condition in `coalesce`, the chain condition in `is_tractable`); delete `:param:`/`:type:`/`:rtype:` boilerplate that restates the signature — type hints carry that. Validation errors keep their precise messages.

---

## 4. Target design — `visualization`

```
visualization/src/layout_categories_viz/
├── style.py             # palette, fonts (unchanged)
├── scene_base.py        # LayoutScene(Scene): background, example-record protocol, run_examples()
├── mapsto.py            # MapstoArrow + its Animation subclasses (merge animations.py in)
├── stacks.py            # cells, decks, place() grid, split gesture  ← promoted from tuple_pullback_test.py
├── paths.py             # path matching/repartition helpers          ← promoted from tuple_morphism_composition_curve.py
├── trees.py             # NestedTupleTree, banded Ref trees          ← merge nested_tuple.py + ref_tree_with_paths
├── morphism_diagram.py  # TupleMorphismDiagram + composition diagram (merge composition.py in)
├── nest_morphism.py     # NestMorphismDiagram
└── coalesce.py          # pruned of the ~160 dead lines
visualization/scenes/
├── __init__.py          # make it a real package; add conftest.py at visualization/
└── ...                  # thin scenes: EXAMPLES + beat sequencing only
```

Principles:

- **A scene file contains only what is specific to that scene**: its examples and its beat sequencing. Anything imported by a second scene moves to `src/`. After this pass, no scene imports another scene except explicit gallery subclassing of *examples* (the projection/transposition benches).
- **`LayoutScene` base** sets the background, standardizes the `Example` dataclass (one definition, not 14), and provides `place()`. Every scene's `construct` shrinks by its current preamble.
- **Kill the forks.** Each `*_single_beat_test` becomes a class attribute or `Example` flag on the main scene (the real difference is ~30 lines in one beat). Span/cospan/ref-span composition become one parameterized scene family: leg category and mirror direction are exactly the parameters the library classes already take, so the scene should take them too. Expected effect: the five span/cospan scene files (≈3,300 lines) collapse to one module of ≈800.
- **Fix the public surface**: export `coalesce` and the style constants from `__init__.py`; rename `_make_entries`/`_mapsto_arrow` to public since ten files already treat them that way.
- **Keep the prototypes** (`graphite_line_test.py`, `decorative_variations_test.py`) in a `scenes/prototypes/` folder so the main listing is all real deliverables.
- **Hygiene**: add `*.mp4` (root-level strays) and decide `papers/` (recommend: commit the PDF — it's the companion paper — or add to `.gitignore` explicitly; don't leave it ambiguous).

---

## 5. Notebooks

Replace the three `examples/` notebooks with a numbered concept series in `tract/notebooks/`, each runnable top-to-bottom with only the pycute backend (no GPU required), using the cute-DSL backend in clearly-marked optional cells:

1. **`01_layouts_and_tuple_morphisms.ipynb`** — flat layouts, tractability, the standard-representation round trip layout ↔ morphism, reading a layout off a morphism via prefix products. (Salvages the first half of `example_notebook.ipynb`.)
2. **`02_operations.ipynb`** — composition, coalescence (weak vs strong), complements, concatenation/sums, logical divide and product — each paired with its CuTe cross-check. (Second half of `example_notebook.ipynb`.)
3. **`03_nested_tuples_and_nest.ipynb`** — `NestedTuple`, profiles/refinement, `NestMorphism`, flattening functors, TikZ export.
4. **`04_fact_ref_and_spans.ipynb`** — Fact and Ref morphisms, pullback/pushforward, the four span categories and the bridge functors. (Absorbs and extends `fact_category.ipynb` + `span_category.ipynb`; covers the currently-undocumented Ref layer.)
5. **`05_composability_vs_tractability.ipynb`** — `COMPOSABILITY_NOTES.md` turned into runnable form: the refuted conjecture with its counterexamples, the anatomy of pycute composition, the dictionary to mutual refinement, and the V1–V3 verification experiments (with fixed seeds and small N so the notebook runs in seconds). Keep the md file as the prose record; the notebook is the executable companion.

Notebooks import only the public `tract` API — no `from tract.test_utils import *`.

---

## 6. Migration plan

Ordered so every phase leaves the repo green and shippable.

**Phase 0 — hygiene (no API changes, ~1 session).** Pytest config + `test_*.py` renames; seed both RNGs; fix the circular import (`tuple_morph_tikz` imports from `.categories`); delete dead code (unused cutlass blocks, `main()`, unused imports, the ~160 dead lines in `coalesce.py`); `.gitignore` the stray mp4 pattern and settle `papers/`; fix the two READMEs' factual drift.

**Phase 1 — tract restructure (~2–3 sessions).** Split `categories.py`; generic span base; backend protocol + `refinement.py`; move `test_utils` → `tests/generators.py` (numpy → dev dep); renames with compat aliases; add `__eq__`/`identity` symmetry; unify the two cross-validation suites; docstring prune. The existing 800+ property tests are the safety net — run them against both backends after every step.

**Phase 2 — visualization restructure (~2 sessions).** Promote scene-owned primitives to `src/`; introduce `LayoutScene`; merge the single-beat forks; unify the span-family scenes; fix packaging/imports. Verify by re-rendering one scene per family at low quality and eyeballing against the existing renders in `media/`.

**Phase 3 — notebooks (~1–2 sessions).** Write the five-notebook series; delete the old three; update `ANIMATION_AUDIT.md` links to the new module paths.

### Decisions to confirm before Phase 1

1. **Class renames** — recommended (with one-release aliases), but if the paper's published companion-code links must match the paper's `Tuple_morphism` notation exactly, keep the old names and skip this item only.
2. **`flat_divide`/`flat_product` vs `logical_divide`/`logical_product`** — recommended: `logical_*` at both levels; reverse if the paper's terminology distinguishes them deliberately.
3. **Never-called methods** (`strong_coalesce`, `squeeze`, `restrict`, `factorize`, `coalesce_with_equiv`, `update_codomain`, `Fin_morphism.wedge`, `flatten_codomain`, …) — recommended: **keep** the ones that implement paper constructions or appear in `ANIMATION_AUDIT.md`'s to-do list, and give each a test; delete `coalesce_with_equiv` and `update_codomain` if no paper construction claims them.
4. **`tract.testing` as public API vs tests-only generators** — recommended: tests-only, unless the visualization package or external users need the random generators.
