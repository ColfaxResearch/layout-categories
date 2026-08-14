# Composability vs. Tractability: Notes on CuTe Composition and Mutual Refinement

*Notes from an empirical + theoretical investigation (2026-08-14) using `tract` and
NVIDIA's official pure-Python CuTe reference implementation
([NVlabs/CuTe](https://github.com/NVlabs/CuTe), the `pycute` package).
Experiments live in ad-hoc scripts against `tract.pycute_utils`; the
cross-validation suite is `tract/tests/pycute_morphism_tests.py`.*

## Conventions

- **B∘A** means "apply A first, then B": `pycute.composition(B, A)`, with A the
  *inner* layout and B the *outer* layout. This matches how
  `morphism_tests.py` and `pycute_morphism_tests.py` call composition for a
  morphism composite g∘f.
- A flat layout is **tractable** if, after sorting modes by stride, each
  stride divides the next stride-times-shape product
  (`layout_utils.is_tractable`); equivalently, it is the layout of a tuple
  morphism. For tractability *as a function*, strides of shape-1 modes should
  be nullified first (`nullify_trivial_strides`), else layouts like
  `(1,1):(22,23)` are spuriously flagged non-tractable.
- **In-bounds / bona fide composition**: cosize(A) ≤ size(B). pycute permits
  out-of-bounds compositions by silently extending through B's last mode
  ("extendability"); C++ CuTe would reject these. Unless stated otherwise,
  all claims below are restricted to in-bounds pairs.

## 1. The conjecture, and its refutation

**Conjecture.** If B∘A exists (CuTe composition), then A is tractable.

**Status: false**, though quantitatively "close to true" in-bounds.
Verified in-bounds counterexamples:

| B | A | B∘A | remarks |
|---|---|---|---|
| `12:1` | `(2,2):(1,3)` | `(2,2):(1,3)` | any B coalescing to rank 1 — which includes **every compact layout** — imposes no condition on A beyond bounds |
| `(4,6):(1,8)` | `(2,2):(8,12)` | `(2,2):(16,24)` | B tractable *and* non-coalescible; A injective, non-tractable (12 ∤ 2·8) |
| `8:1` | `(2,2):(1,1)` | `(2,2):(1,1)` | overlapping (non-injective) A composes with anything large enough |

Random-search statistics (500k random pairs, in-bounds only, tractability
normalized for shape-1 modes): among composable pairs, A is non-tractable in
**0.2%** of cases when coalesce(B) has rank 1, rising to **~3%** at rank 3–4.
Without the in-bounds restriction (pycute's permissive truncation/extension),
violation rates are 20–60%.

**Sharper negative result.** Even A and B *both tractable* with B∘A defined
does not make B∘A tractable:

- A = `4:3`, B = `(6,3):(21,7)` (both tractable) ⟹ B∘A = `(2,2):(63,7)`,
  which is **non-tractable** (63 ∤ 2·7·k chain fails).
- tract diagnoses why: f_A: (4) → (3,4), f_B: (6,3) → (7,3,6), and
  `mutual_refinement((3,4), (6,3))` **fails** (prefix products 12 and 18 are
  divisibility-incomparable).

So CuTe composability is strictly weaker than tract weak-composability, and
tractable layouts are *not* closed under raw CuTe composition; mutual
refinability of cod(f_A) with dom(f_B) is exactly the missing hypothesis.

## 2. Anatomy of pycute's composition algorithm

`Layout._composition` (pycute `layout.py`) reduces everything to one
primitive:

1. **Distributes over A's modes**: B∘(A₁,A₂,…) = (B∘A₁, B∘A₂, …), with **no
   cross-mode condition whatsoever**.
2. **Coalesces B** (`coalesce_z`), so only M := shape(coalesce_z(B)) =
   (M₁,…,Mₘ) matters. B's strides play no role in *existence* (in-bounds) —
   they only multiply into the result — except indirectly through which modes
   coalesce.
3. Handles the primitive **B∘(s:d)** in two stages, with exactly two `raise`
   sites.

Write Pⱼ = M₁·M₂···Mⱼ (P₀ = 1) for the prefix products of M — a divisibility
chain P₀ | P₁ | ⋯ | Pₘ marking the boundaries of B's mixed-radix coordinate
box. B∘(s:d) walks the arithmetic progression 0, d, 2d, …, (s−1)d through
that box, and the conditions say the walk must respect the box structure:

- **Stage 1 (shape condition, on the extent E = s·d).** The footprint must
  consume whole modes then stop: E = Pₖ·c with c ≤ Mₖ₊₁ for some k.
  Divisibility is demanded at every mode fully crossed, but the **final**
  covered mode is exempt: c may be any prefix of Mₖ₊₁, dividing or not
  (the quotient-0 truncation branch).
- **Stage 2 (stride condition, on the step d).** Within that footprint, d
  must factor through the boundaries it crosses: d = Pⱼ·d′ with d′ | Mⱼ₊₁.
  Again the last covered mode is exempt (the `for…else` branch floor-divides
  with no remainder check).

**Slogan: divisibility is required at every internal boundary of
coalesce(B) that the walk must cross; the walk may *end* ragged
(mid-mode) but may never *cross* a boundary misaligned.**

*Verification V1*: a verbatim transcription of the two loops as a predicate
matched `pycute.composition` success/failure on **85,874 random in-bounds
pairs with 0 mismatches**.

## 3. The dictionary to mutual refinement

**Step 1 — a single mode is the two-entry tuple (d, s).** The standard
morphism of the layout s:d is f: (s) → (d, s). Tract's greedy
`mutual_refinement((d, s), M)` splits d against M — forcing d to consume
whole modes and *divide* the one it lands in — then splits s across the rest
the same way. These are precisely Stages 2 and 1 above **with the
final-mode exemptions removed**.

*Verification V2*: `mutual_refinement((d,s), M)` succeeding implies pycute
composes, on **57,551 random in-bounds pairs with 0 exceptions**. The gap
(pycute yes, mutual refinement no) is exactly the ragged-final-edge cases —
e.g. refining (3, 4) against (6, 3): d = 3 divides M₁ = 6, but
s·d = 12 = 6·2 ends with c = 2 ∤ M₂ = 3. pycute takes the first two-thirds
of the mode; mutual refinement refuses; and this is exactly where the
tractable∘tractable → non-tractable example above lives.

**Step 2 — tractability glues the modes together.** For tractable A with
sorted standard form, T_A = (d₁, s₁, d₂/(s₁d₁), s₂, …) has prefix products
{1, d₁, s₁d₁, d₂, s₂d₂, …} — a divisibility chain, by tractability. Mutual
refinability of flat tuples amounts to their prefix-product sets merging
into one chain, and merging a chain with the chain {Pⱼ} only requires each
element to be comparable with each Pⱼ. Hence the joint condition decomposes
per-mode.

*Verification V3*: for random tractable A and arbitrary B (**18,150 in-bounds
cases, 0 mismatches**):
`mutual_refinement(T_A, M)` succeeds ⟺ `mutual_refinement((dᵢ, sᵢ), M)`
succeeds for every mode individually.

**Step 3 — the full correspondence.**

| condition | per mode | across A's modes | ragged final edge |
|---|---|---|---|
| pycute composability | boundary divisibility | nothing | allowed |
| tract weak-composability, i.e. `mutual_refinement(T_A, S_B)` | boundary divisibility | chain condition = **tractability of A** | forbidden |

So tract's mutual refinement = pycute's divisibility conditions strengthened
in exactly two independent ways:

1. each mode's walk must **end on a boundary** of the common refinement
   (no truncation), and
2. all modes' walks must be simultaneously compatible with **one**
   refinement of M — forcing the dᵢ, sᵢdᵢ to interleave into a single
   chain, which is precisely tractability of A.

The original conjecture fails on (2) alone — per-mode conditions cannot see
cross-mode structure. The tractable∘tractable → non-tractable counterexample
fails on (1) alone. Note also that a chain among the multiset
{dᵢ, sᵢdᵢ} is *not* sufficient for tractability: `(2,2):(1,1)` has chain
{1, 2} yet is non-tractable — tractability needs the intervals [dᵢ, sᵢdᵢ]
to be disjointly stacked (d₁ ≤ s₁d₁ | d₂ ≤ s₂d₂ | ⋯), not merely
comparable.

**Coalescing caveat.** Everything is stated against coalesce_z(B), not B.
Coalescing coarsens the boundary chain {Pⱼ} and strictly weakens the
conditions; on the tract side this corresponds to composing with the
*coalesced* standard morphism of B. Mutual refinement against B's raw
(uncoalesced) shape is conservative: e.g. T_A = (3,2) fails against (4,6)
but succeeds against the coalesced (24).

## 4. pycute implementation notes (relevant to testing)

- **Empty-layout (rank-0) bugs**: `make_layout` crashes on an empty iterable,
  breaking `composition(A, ():())`, `logical_product(A, ():())`, and
  `coalesce(A, profile=())`, all of which C++ CuTe handles (returning empty
  layouts). Workaround wrappers live in `tract.pycute_utils`:
  `compose_layouts`, `logical_product_layouts`, `coalesce_layout`.
  Candidates for an upstream PR; the local CuTe clone is kept pristine.
- **Permissiveness**: pycute never checks cosize(A) ≤ size(B); a rank-1
  outer layout composes with anything (both truncation and out-of-bounds
  extension happen silently via the last mode). Any theorem stated from
  pycute behavior should carry the in-bounds hypothesis explicitly.
- **Final-mode floor**: in Stage 2's `else` branch the last shape is
  floor-divided by the residual stride with no remainder check
  (`result_s[-1] //= strideB`), another quiet relaxation at the ragged edge.

## 5. Candidate theorem statements

1. **Characterization of composability.** For in-bounds B∘A, pycute
   composition exists iff for every mode s:d of A (with s > 1, d > 0):
   d = Pⱼ·d′ with d′ | Mⱼ₊₁ (or d lands in the final covered mode), and
   s·d = Pₖ·c with c ≤ Mₖ₊₁, where P are the prefix products of
   shape(coalesce_z(B)). *(Verified V1.)*
2. **Strict composability = per-mode mutual refinability.** The
   no-ragged-edge strengthening of (1) holds iff (d, s) and
   shape(coalesce_z(B)) are mutually refinable, for every mode. *(V2.)*
3. **Bridge.** If A is tractable, per-mode mutual refinability is equivalent
   to joint mutual refinability of T_A with shape(coalesce_z(B)), i.e. to
   tract weak-composability of the standard morphisms. *(V3.)*
4. **Closure.** Under (3)'s hypotheses the composite is tractable with
   f_{B∘A} the weak composite; without them, B∘A of tractable layouts can be
   non-tractable (§1).
