<h1 align="center">separatrix</h1>

<p align="center"><b>Your top-10 changed when the batch size changed. Get a proof it was never yours.</b></p>

<p align="center">
  Every ranking, <code>argmin</code> and threshold in your stack is a discrete decision read
  off a float computation. <code>separatrix</code> proves the decision came from your
  <b>data</b> and not from the <b>rounding of the kernel you happened to call</b> — or it
  <b>refuses</b>, names the exact pair it cannot separate, and tells you what to change.<br>
  One extra <code>O(nd)</code> pass. One run, not two. No probability anywhere in the output.
</p>

<p align="center">
  Invented by <b>Teerth Sharma</b> · <a href="mailto:teerths57@gmail.com">teerths57@gmail.com</a> · <a href="https://github.com/teerthsharma/separatrix">github.com/teerthsharma/separatrix</a>
</p>

<p align="center">
  <a href="https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md"><img src="https://img.shields.io/badge/tests-195%20passed-2ea043?style=flat-square" alt="195 tests passed"></a>
  <a href="https://github.com/teerthsharma/separatrix/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/deps-numpy%20%2B%20scipy-013243?style=flat-square" alt="numpy and scipy only"></a>
  <img src="https://img.shields.io/badge/CPU--only-no%20GPU%2C%20no%20cloud-0b7285?style=flat-square" alt="CPU only">
  <img src="https://img.shields.io/badge/tuning-zero%20fitted%20constants-8b5cf6?style=flat-square" alt="zero fitted constants">
  <a href="https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#8-not-earned"><img src="https://img.shields.io/badge/output-a%20proof%20or%20a%20refusal-b45309?style=flat-square" alt="a proof or a refusal"></a>
  <img src="https://img.shields.io/badge/never-a%20probability-9b2a2a?style=flat-square" alt="never a probability">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/teerthsharma/separatrix/main/assets/frame1.svg" width="820" alt="separatrix demo --frame cancellation: two points 1e-6 apart at magnitude 1e6, the Gram identity returns 0.0 in float64, and separatrix returns REFUSED (GRAM_CANCELLATION) naming the code change">
</p>

<p align="center"><i>Two distinct points, <code>1e-6</code> apart. The Gram identity your
library switched to returns a distance of <b>exactly 0.0</b> — in <b>float64</b>, so
upcasting is not the fix. <code>separatrix</code> refuses, and the refusal names the
formula, not the precision.</i></p>

```bash
pip install separatrix        # or:  uv pip install separatrix
separatrix demo --frame preview
```

> ### 📈 `0` of **1,116** certified top-10 sets moved — across **nine** different evaluations of one formula
>
> Same stored bytes, nine numerically distinct engines: numpy Gram fp32, the same with the
> reduction permuted, numpy direct, numpy fp64, `scipy.cdist`, `sklearn.euclidean_distances`,
> `torch.cdist` on both compute modes, and `torch.cdist` at batch 32. Out of 1,500 top-10
> decisions on five corpora, **every set that moved had been refused first**.
> → [RESULTS.md §2](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#2-the-headline)

```python
import separatrix

idx, v = separatrix.certified_topk(corpus, queries, k=10)   # replaces your argpartition
j,   v = separatrix.certified_argmin(corpus, queries)       # the k=1 case, axis squeezed
e      = separatrix.enclose_scores(corpus, queries)         # the scores and their radii
trit   = separatrix.certified_threshold(e.D, e.R, t)        # +1 above, -1 below, 0 undetermined

v.status        # CERTIFIED | CERTIFIED_UPCAST | NOT CERTIFIED | REFUSED
v.n_refused     # an exact integer. never a score, never a probability
v.frontiers     # per undetermined row: the pair, both intervals, the gap, the deficit
v.next_action   # what to change — one refusal names a code change, the rest a re-observation
```

**Zero training. No model. No dataset. No GPU. No fitted constant.** `numpy`, `scipy`, and
one page of Higham. It never raises for an arithmetic outcome and it has no `on_refuse`
knob — a refusal is a return value, because a library that throws on 26% of rows —
384 of the 1,500 measured here — is uninstalled the same afternoon.

---

## ⚙️ How it works

```mermaid
flowchart LR
  A["📦 X, Q<br/>float32 as stored"] --> B["🚧 preconditions<br/>range · finite · canary"]
  B --> C["📏 one extra O(nd) pass<br/>float64 norms → radius R"]
  C --> D["✂️ max-in + R<br/>vs min-out − R"]
  D --> E["✅ CERTIFIED<br/>🛑 REFUSED + the pair"]
```

1. 🚧 **Precondition, before a single score is read.** `‖x‖²+‖q‖²+2‖x‖‖q‖` overflows the working dtype → `RANGE_UNSAFE`, per row. A 4×4 canary catches arithmetic coarser than the declared unit roundoff — TF32, bfloat16 inputs, AMX — with no vendor flag taxonomy.
2. 📏 **Enclose.** Every score gets a rigorous a-priori radius `R` with `|D − s| ≤ R`, where `s` is the exact real value of *the named formula on the bytes you handed it*. One extra `O(nd)` pass, riding the BLAS call the scores already cost. **No sampling. No reruns. No posterior.**
3. ✂️ **Decide.** `max_{i∈T}(D_i+R_i) < min_{j∉T}(D_j−R_j)`. Disjoint intervals → `T` is the top-k of the exact score, **and of every vector in the box**.
4. ✅ **Certify.** Then any other evaluation of the same formula on the same bytes returns the same `T`. Batch size, BLAS backend, thread count, chunk size, reduction order, FMA, and `torch.cdist`'s 25-row switch **cannot move it**. Determinism across backends is a theorem here, not a measurement over reruns.
5. 🛑 **Or refuse.** Overlap → `REFUSED`, with the frontier pair, both intervals, the `gap`, the `width`, the `deficit`, a typed code, and a next action. Never a hedge, never a score.

<p align="center">
  <img src="https://raw.githubusercontent.com/teerthsharma/separatrix/main/assets/switch.svg" width="820" alt="torch.cdist switches formula above 25 rows: max |mm − direct| is 0.000e+00 at 24 and 25 rows and 9.766e-04 at 26 rows on one 40x8 float32 array">
</p>

**This is why the library exists.** `torch.cdist` switches to the cancellation-prone Gram
identity above 25 rows because it is faster. Below the switch the two compute modes are
**bit-identical**; above it, on the same stored bytes, they are not — measured on the
installed `torch 2.14.0+cpu`, not quoted from docs.
→ [RESULTS.md §5.1](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#51-the-25-row-switch-measured-on-the-installed-torch)

---

## 🧮 The math, in sixty seconds

**The enclosure.** `fl(x·y)` deviates from `x·y` by at most `γ_d · Σ|x_i||y_i|` — Higham,
*ASNA* 2nd ed., Theorem 3.1. The Gram identity `d² = ‖x‖² + ‖y‖² − 2⟨x,y⟩` is three such
reductions plus two additions, so its constant is `γ_{d+2}`, and Cauchy–Schwarz collapses
the whole thing into two norms you can get for free:

$$
\gamma_n = \frac{n u}{1 - n u}
\qquad
R = \gamma_{d+2}\left(\lVert x \rVert + \lVert y \rVert\right)^{2}
$$

with `u = eps/2`. **No extra matmul.** The norms are computed in float64 — reusing the
working-dtype `‖x‖²` already sitting in the identity is the tempting shortcut, and a norm
that rounds low yields a radius that is low, and a low radius is not a bound.

**The rule.** One inequality, over `(D, R)` alone, and it knows nothing about kernels:

$$
\max_{i \in T}\left(D_i + R_i\right)
\quad \lt \quad
\min_{j \notin T}\left(D_j - R_j\right)
\qquad \Longrightarrow \qquad
T = \mathrm{topk}(s)
$$

🔴 **The obvious version of this rule is a false theorem.** Comparing only the rank-k and
rank-(k+1) scores is unsound the moment the radii vary: scores `[0, 1, 2, 10]` with radii
`[12, 0, 0, 0]` and `k = 2` gives a disjoint boundary pair and certifies `{0, 1}`, while the
vector `(11, 1, 2, 10)` sits inside the box with top-2 `{1, 2}`. On **every** benchmark
corpus here the naive rule and the sound rule return identical refusal counts — 15/15,
300/300, 35/35, 32/32, 2/2 — so the naive rule ships green and only the counterexample test
catches it. [→ §1.4](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#14-the-false-theorem-and-why-no-benchmark-finds-it)

🔴 **And the relative-error model is false under subnormals.** `fl(ab) = ab(1+δ)` does not
hold when a product lands subnormal, so an unconditional additive `η` rides every radius.
Its control: at float32 with components ~1e-25, **4000/4000** trials escaped the radius
alone and **0/4000** escaped with `η` — while at components ~1, the ordinary regime,
**0/4000** escaped either way. [→ §1.3](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#13-the-underflow-term-with-its-control)

> ### 🧠 Explainable, minus the AI
> The output is an integer, the pair it came from, the interval it came from, and the
> theorem it came from. Nothing is learned, nothing is fitted, nothing is sampled. There is
> no `eps` to tune and no threshold to pick. Run it twice, get the same answer — and that
> is the *weakest* thing it promises.

---

## ✅🛑 What it certifies, and what it refuses

`CERTIFIED` means **rounding did not choose this ranking**. It says *nothing* about the
model, the quantiser or the sensor: a 384-dim embedding out of a float16 forward pass carries
~1e-3 relative error, three to four orders above the float32 rounding being certified. This
is a heading and not a footnote because it is the way the certificate will be misread.

| status | exit | what it means |
|---|---|---|
| ✅ **CERTIFIED** | `0` | **Proved.** Every row's set is the exact score's set, and every other evaluation of the formula on these bytes returns it |
| 🟨 **CERTIFIED_UPCAST** | `4` | Proved *at a wider dtype than production runs*. Never `0`, because CI must not read a pass for a computation your index does not run |
| 🟡 **NOT CERTIFIED** | `1` | An exact tie. The only outcome no precision removes |
| 🛑 **REFUSED** | `2` | No certificate was formed, with a cause and a next action |
| ⚫ usage error | `3` | An exception, not a verdict — a statement about your **code**, not your data |

Every refusal is typed, names a concrete object, and carries a next action:

| code | what it names | next action |
|---|---|---|
| `BOUNDARY_UNDETERMINED` | the frontier pair, both intervals, `gap`, `width`, `deficit` | re-observe, escalate, or widen `k` |
| `GRAM_CANCELLATION` | **the only one that names a code change** — the direct kernel separates this pair, the Gram identity does not | `kernel="direct"`, or `compute_mode="donot_use_mm_for_euclid_dist"` |
| `EXACT_TIE` | escalation proved the two scores equal | adopt a tie-break; no precision fixes this |
| `RANGE_UNSAFE` | the row whose norms overflow the working dtype, **before any score is read** | `upcast=True`, or normalise |
| `NONFINITE_INPUT` | the row and the column. separatrix will not impute | fix the data |
| `BOUND_VACUOUS` | `(d+2)·u > 1/2` — a stated domain limit, not a property of your data | use a wider dtype |
| `REDUCED_PRECISION_ARITHMETIC` | the 4×4 canary caught TF32 / bfloat16 / AMX | set `float32_matmul_precision="highest"` |
| `ESCALATION_BUDGET` | the frontier exceeded `max_escalations` | raise the budget |

🎯 **A refusal is not a finding.** It means no certificate was formed. Only `escalate=True`
decides which way a boundary actually falls, and on these corpora it decided **379 of 384
refusals were pessimism rather than damage**. That number is printed here rather than
discovered by a reader.

---

## 🎬 The demo — three frames, no network, no download

```bash
separatrix demo --frame cancellation   # two points 1e-6 apart at magnitude 1e6
separatrix demo --frame batch          # one set of bytes, two evaluations, two answers
separatrix demo --frame preview        # which boundaries were never separable, named first
separatrix probe                       # this machine's u, canary, gamma table, torch flags
```

**Frame 3 is the one no a-posteriori method can run.** A float64 diff, two batch sizes and
stochastic rounding all find what *moved*. None of them can say a boundary was never
determined on a row where the two runs happened to agree.

```
  corpus 400 x 64 float32, 60 queries, k = 5, 40 near-duplicate pairs at 3e-07
  named undetermined, from one evaluation:  9 of 60  [5, 13, 14, 22, 37, 39, 40, 54, 57]
  actually differed, over two evaluations:   3 of 60  [14, 22, 54]
  differed and NOT named:                    0  []
```

`differed and NOT named` is the only row that is evidence, and it is a soundness check — a
row two evaluations decided differently that was *not* named in advance is a counterexample
to the certificate, and the frame exits non-zero on one. **The control is in the corpus:** a
corpus of nothing but near-duplicates refuses every row and makes the claim true by
construction, so 51 of the 60 rows here are certified and had something to lose.
→ [RESULTS.md §5.3](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#53-frame-3--the-one-frame-no-a-posteriori-method-can-run)

---

## 🔧 The CLI and the build gate

```bash
separatrix check --corpus corpus.npy --queries queries.npy --k 10 \
                 [--kernel gram|direct] [--bound cheap|tight] [--per-pair] \
                 [--escalate] [--upcast] [--ordered] [--largest] \
                 [--max-refused 0.05] [--max-report 1] [--json]
```

Real output, 2000×384 float32 corpus, 300 queries, `k = 10`, `--max-report 1` — the same
iid arm as the table above, with the two-line recipe that regenerates the corpus from
scratch in
[RESULTS.md §3.1](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#31-what-one-of-those-refusals-looks-like):

```
==============================================================================
  REFUSED (BOUNDARY_UNDETERMINED)                         285/300 determined
==============================================================================
  detail      15 of 300 rows have a rank-10 boundary this enclosure does not
              decide; the direct kernel separates 7 of those 15 frontier
              pairs
  computed    kernel gram   bound cheap   per-row   k 10
  dtype       float32
  canary      numpy/float32 clean
  accumulator assumed storage dtype (P5 is not testable)
------------------------------------------------------------------------------
  boundary    row 12: in #1046 [1.721866e+00, 1.722050e+00]  out #1342
              [1.721932e+00, 1.722116e+00]  gap 6.616116e-05  width
              1.840634e-04  deficit -1.179022e-04
              14 further boundaries not shown (--max-report 1)
------------------------------------------------------------------------------
  next        The two enclosures at the rank-k boundary overlap. Pass
              escalate=True to decide it exactly, or bound='tight', or
              per_pair=True, or recompute in a wider dtype.
==============================================================================
```

Gate a build on it:

```python
with separatrix.gate(max_refused=0.05, fixture="scifact-d384-fp32"):
    run_your_evaluation()          # every certified_topk call reports into the open gate
```

`max_refused` is **required** — no default. The fraction is keyed to a config digest over
the nine fields that change what a refused fraction *means*, and on a digest mismatch the
gate **refuses to compare** rather than failing, because a gate that goes red on a numpy
upgrade is deleted in a week. It reads `.separatrix-gate.json` and never writes it: a gate
that records its own budget is vacuous.

---

## 📊 Benchmarks

Pasted from the commands beside them, one draw, seed 11, commit `4ce5e0b`, on one machine —
`WIN-16QAL06O9GB`, CPython 3.11.9, numpy 2.4.6, scipy 1.17.1, torch 2.14.0+cpu.
**[Every table, every control, and every arm that lost →](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md)**

### The headline — determinism across nine engines

<p align="center">
  <img src="https://raw.githubusercontent.com/teerthsharma/separatrix/main/assets/agreement.svg" width="820" alt="0 certified top-10 sets moved across nine evaluations; the only movement, 8 rows, is on the clustered corpus and every one was refused first">
</p>

| corpus | shape | refused | moved, CERTIFIED | moved, REFUSED first |
|---|---|---|---|---|
| iid normalised d=384 | 2000×384 f32 | 15/300 | **0** | 0 |
| clustered normalised d=384 | 2000×384 f32 | 300/300 | **0** | **8** |
| MNIST-shaped (generated) d=784 | 2000×784 f32 | 35/300 | **0** | 0 |
| MNIST (downloaded) | 5000×784 f32 | 32/300 | **0** | 0 |
| BEIR SciFact + all-MiniLM-L6-v2 (downloaded) | 4883×384 f32 | 2/300 | **0** | 0 |

**Four of five corpora returned 0 in the last column. Those four are arms where this package
had nothing to say**, and they are printed at the same size as the one that did. Only the
clustered arm is evidence the corpus was not too easy. Three of these five corpora are
generated and two are downloads of a few thousand rows; the nine-engine comparison is not
run at a million rows, and the real-data table below carries one external answer instead of
nine internal ones.

### Real data at scale — SIFT1M, scored against a ground truth this repository did not compute

`1,000,000 × 128` INRIA SIFT descriptors, 1,000 queries, k = 10, `chunk=100`, 48.0 s. The
ANN\_SIFT1M release ships the authors' exact top-100 list, so this is the one table on this
page whose answer key comes from outside this repository.

| what | measurement |
|---|---|
| refused | **52 of 1,000** (5.2%) |
| CERTIFIED sets agreeing with the published top-10 | **948 of 948** |
| rows where the published answer differs | **9** |
| how many of those 9 had been refused first | **9 of 9** |
| how many of those 9 exact integer arithmetic calls a tie | **9 of 9** |
| float32 Gram scores differing from exact int64 | **0 of 20,000,000** |
| float16 on the same bytes | **REFUSED (RANGE\_UNSAFE)**, before any score is read |

The descriptors are integers 0..255, so float32 computes these scores **exactly** — which
makes the flip count 0 by arithmetic and every non-tie refusal measurable pessimism: the
median frontier has gap 7.0 against an enclosure width of 16.12, and a true error of 0.
Three things broke getting here and are fixed in this build: frontiers reported row 0 after
`--escalate`, `chunk=` was validated and then ignored (1,000 queries × 1M rows is 8.00 GB of
scores in one allocation), and the float64 norm pass copied the whole corpus (1.02 GB).
→ [§10](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#10-real-data-at-scale--sift1m-and-a-third-party-holding-the-answer)

```bash
.venv/Scripts/python bench.py --sift        # downloads 516 MB once, caches under .donotcommit/
```

### The kernel switch — where the product actually is

| corpus | gram refused | direct refused | median gram width | median direct width |
|---|---|---|---|---|
| iid normalised d=384 | 15/300 | 8/300 | 1.841e-04 | 9.180e-05 |
| clustered normalised d=384 | **300/300** | **2/300** | 1.841e-04 | 9.167e-05 |
| MNIST-shaped (generated) d=784 | 35/300 | 11/300 | 1.414e+03 | 5.191e+02 |
| MNIST (downloaded) | 32/300 | 4/300 | 3.580e+03 | 6.290e+02 |
| SciFact (downloaded) | 2/300 | 1/300 | 1.841e-04 | 8.085e-05 |

**298 of the 300 refusals on the clustered index are attributable to the kernel alone** —
the one refusal in the catalogue that names a *code change* rather than a re-observation.
→ [§5](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#5-the-kernel-switch--where-the-product-is)

### Cost — best of 5, 5,000×784 float32, 300 queries, k = 10

| | seconds | vs float64 gram | vs float32 gram |
|---|---|---|---|
| float32 gram + argpartition — the status quo | 0.0229 | 0.73× | 1.00× |
| float64 gram + argpartition — **the honest control** | 0.0314 | 1.00× | 1.37× |
| scipy.cdist float64 + argpartition | 0.5357 | 17.1× | 23.4× |
| **separatrix rung 1**, per-row radii | **0.0529** | **1.68×** | 2.31× |
| separatrix rung 2, per-pair radii | 0.0730 | 2.33× | 3.18× |
| the float32-vs-float64 diff — the competing practice | 0.0498 | 1.59× | 2.17× |

**Parity with the diff it replaces (1.06×), and roughly 1.7× the float64 control.** Neither
is a selling point, and both are stated at their real size. Run-to-run spread over six runs
of the same file, same corpus: rung 1 at 0.0517–0.0555 s, **1.59×–1.89×** the control and
**0.96×–1.12×** the diff. Two earlier builds quoted 0.0505 s and 0.0650 s for this row;
neither absolute reading reproduces and both are struck. The seconds move between builds
and the ratios do not, so the ratios are the measurement and the seconds are the draw.
→ [§6](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#6-cost)

### The soundness gate — mandatory, an instance and not a budget

| what | measurement |
|---|---|
| CERTIFIED verdicts the exact lattice contradicts | **0** |
| enclosure escapes on the adversarial corpus | **0 of 656 pairs** — 82 per configuration, 8 configurations, 10 corpora |
| escalation contradicting a certificate | **0** over all 384 refused rows, 5,827 exact dot products |
| CERTIFIED on the recorded float16 range case | **0** |

One CERTIFIED decision that exact arithmetic contradicts withdraws the package.
→ [§1](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#1-the-soundness-gate--mandatory-no-threshold)

### Reproduce all of it

```bash
python -m venv .venv
.venv/Scripts/pip install -e .              # numpy and scipy are the only hard dependencies
.venv/Scripts/python -m pytest tests/ -q    # 200 passed; 1 skips without the SIFT cache
.venv/Scripts/python bench.py --out results.json --assets assets
.venv/Scripts/python bench.py --sift        # §10, downloads SIFT1M (516 MB) once
```

`bench.py --no-download` skips the two corpora that need a network and runs the rest. `torch`,
`scikit-learn`, `datasets` and `sentence-transformers` are optional, imported lazily, and
named in `bench.py`'s own output when absent rather than skipped silently. Every module owns
a runnable self-check: `python -m separatrix.enclose`.

---

## 📚 Prior art

**Nothing in the certified layer is new, and saying so first is what makes the rest
believable.** Three of these beat this design on their own axis.

| work | what they had first, or do better | what `separatrix` adds |
|---|---|---|
| **CGAL** `Filtered_predicate` / `Interval_nt` ([manual](https://doc.cgal.org/latest/Number_types/index.html)) | **This exact rule, shipping since ~2001**: interval enclosure, disjoint-from-zero certifies the sign, overlap escalates to exact | a different adversary — a rank-k *set* boundary in 384 dimensions in Python, where the escalation target is a set rather than a sign |
| **Melquiond & Pion**, static filter certification ([RAIRO Theor. Inform. Appl. 2007](https://doi.org/10.1051/ita:2007005)) | the same class of a-priori bound, **formally machine-checked** where these are only tested against an integer oracle | nothing on rigour. This is a gap, stated as one |
| **Ogita–Rump–Oishi `Dot2`** ([SIAM J. Sci. Comput. 2005](https://doi.org/10.1137/030601818)) | an *a-posteriori* bound at `u` where this is *a-priori* at `(d+1)u`. Benchmarked here: it certifies **64 of 64** boundaries this package refuses | throughput only — Dot2 is elementwise and forfeits the gemm, at **61× to 99×** the cost over six runs. **If you can afford roughly 100×, use Dot2** |
| **Higham**, *ASNA* 2nd ed. ch. 3 | the bound itself, one page, Theorem 3.1 and (3.1) | the plumbing to a top-k set and a typed refusal |
| **scikit-learn** [#9354](https://github.com/scikit-learn/scikit-learn/issues/9354) / [PR 13554](https://github.com/scikit-learn/scikit-learn/pull/13554) | fixed `euclidean_distances` fp32 instability by chunked upcasting, **on by default since it landed** | the diagnosis this is the mitigation *for*. It also shrinks the addressable population to `torch.cdist` above 25 rows, hand-rolled Gram code, and low-precision indexes |
| **CADNA / Discrete Stochastic Arithmetic** ([site](http://cadna.lip6.fr/)) | unstable branching as a first-class counter — **the same output shape**, and older | a proof over a box instead of a 95% confidence estimate from three random-rounding runs, and a pip channel | <!-- # sx: quote -->
| **`python-flint` (Arb)**, **`mpmath.iv`** | both pip-installable, both rigorous, both doing enclose-or-escalate **today** | it rides the gemm. Arb is scalar; this needs no arithmetic beyond the one BLAS call the scores already cost |
| **`torch.use_deterministic_algorithms`**, `CUBLAS_WORKSPACE_CONFIG`, ReproBLAS | same-machine run-to-run reproducibility, in-tree, free | they make two runs agree on a value that was **never determined**. This reports whether the answer was ever in doubt |
| **`torch.cdist(compute_mode="donot_use_mm_for_euclid_dist")`** | one keyword, already installed, and it *removes* the cancellation class rather than diagnosing it | knowing which of your decisions needed it |

"The mechanism is unknown to the Python ecosystem" is not a sustainable claim and is not
made anywhere in this repository.

The origin of the observation is the author's own, from an earlier clustering pipeline:
`torch.cdist` switching to the Gram identity above 25 rows, where it flipped single-linkage
merge order on clustered low-precision keys, and where points at offset `1e6` with `1e-6`
separation cancelled to exactly `0.0` so a persistence routine reported two distinct points
as one. That last case is frame 1, at the top of this page, and it is the only part of that
story with a number in [RESULTS.md](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#52-frame-1--changing-the-formula-is-the-fix-changing-the-precision-is-not).

---

## 💥 What we got wrong

Six things this build killed, and every one of them was found by running the modules against
each other rather than by reading them.

- 🪦 **A negative radius certified everything.** Precondition P3 admits `n·u == 1/2`, where `γ_n` is exactly 1.0 and rounds outward to `1.0000000000000002`. The Gram form multiplies by γ and stays sound; the direct kernel's *relative* form divides by `1 − γ` and produced a radius of **−9.224903e+14**, which inverts every interval so max-in falls below min-out and **the rule certifies every row**. Now `BOUND_VACUOUS`, pinned by two tests asserting `R ≥ 0` and `lo ≤ hi` over every kernel × bound × adversarial case. [→ §1.1](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#11-two-soundness-bugs-this-build-found-and-fixed)
- 🪦 **A 64-frontier cap made refused rows read as certified.** With 69 refused rows out of 200, the 5 past the cap were invisible to any consumer reconstructing the refused set from `v.frontiers`, and two evaluations then "disagreed on 5 CERTIFIED rows". Measured: **5 apparent disagreements before, 0 after**, with 41 among refused on the same corpus. A `Verdict` now caps nothing; the CLI caps what it *prints*.
- 🪦 **`0.14× the cost of the fp64 reference`.** The design's headline. It does not reproduce, and neither does its revision to `0.91×`. **Withdrawn.** The honest words are parity with the diff and ~1.7× the float64 control. [→ §6](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#6-cost)
- 🪦 **`summation="pairwise"` as a certificate.** Deleted. OpenBLAS accumulates with ~8 unrolled accumulators, giving reduction depth ≈ `d/8 + 3` ≈ 101 at d=784 against the pairwise model's `2·log₂(784)+2 = 21.2` — **4.8× too shallow, by counting, before any measurement**.
- 📉 **The pre-registered prediction of `flipped = 0` on every corpus did not hold** — and it failed in the direction that helps least. The clustered arm flips **5 of 300**, and **the plain float32-vs-float64 diff finds the same 5**. So that arm is not one where separatrix saw what the practice missed; it is one where both saw the same thing and only one needed a second run. [→ §3](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#3-the-refusal-triple)
- 📉 **The pessimism gate fired on the synthetic generator and the design's number was wrong in both directions.** The design predicted 107/300 on a clustered normalised d=384 index and generalised to real embeddings. Measured: **300/300 on the synthetic clustered generator, 2/300 on real BEIR SciFact**. The synthetic generator is far more adversarial than a real index, and **no number from it may be quoted as a statement about one**. [→ §4](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#4-the-pessimism-gate-and-how-it-landed)
- 📉 **A four-line tuned margin beats this on coverage.** `gap > eps·|score|` certifies **295 of 300** on the clustered arm where separatrix certifies **0**. What it does not have is any `eps` below `1e-3` that avoids issuing certificates a witness contradicts — and the `eps` that does costs **122 of 300** certificates on the iid arm and 108 on MNIST-shaped. The differentiator is **no tuning, and a proof**, which is smaller than the pitch. [→ §7.2](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#72-the-tuned-margin-baseline-draws-on-four-of-five-corpora)

---

## ⚠️ Limits

Collected once, here.

- **It certifies the rounding of one named formula on the bytes it was handed, and nothing else.** A 384-dim embedding out of a float16 forward pass carries ~1e-3 relative error, three to four orders above the float32 rounding certified. `CERTIFIED` means *rounding did not choose this ranking* — never that the ranking is right.
- **Not a detection.** The float32-vs-float64 diff was right on **1,495 of 1,500** rankings measured here. The status quo is mostly fine; this returns a proof instead of an empty diff, at the same cost.
- **Not faster than what it replaces.** 1.68× the float64 control, 1.06× the diff, on the printed draw. There is no speed story.
- **Pessimistic where it matters most.** 379 of 384 refusals here were pessimism, not damage. The exact lattice puts the pessimism factor at **≤256×**, and that is an upper bound on the gap rather than a measurement of it. [→ §7.5](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#75-the-exact-lattices-pessimism-factor)
- **The refusal actions that work offline do not ship at the 26% refusal rate measured here** — 384 of 1,500 rows. The action that works in production is free and needs no library: retrieve `k+5` and let the reranker absorb the boundary.
- **`ordered=True` compares k boundaries instead of one** and refuses correspondingly more often. Its column is never merged with the set column.
- **The rung-1 memory collapse costs refusals where norms vary**: 35 against 26 on MNIST-shaped, 32 against 15 on real MNIST. It costs 0 at d=384 with unit norms. [→ §7.3](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#73-the-shuffled-enclosure-control-is-a-no-op-where-the-design-said-it-would-be)
- **Never claimed anywhere in this repository:** any GPU number (no CUDA path is designed); any claim about an approximate index (IVF/PQ/HNSW search error is ~1e-2 relative, four orders above the float32 enclosure width, so certifying it would certify the wrong quantity while printing CERTIFIED — `faiss` is not a dependency, not a loader and not a refusal code); accumulator width (P5 is not testable by either probe tried and travels on every verdict as a declared assumption); a machine-checked bound; and **a probability of any kind** — Higham & Mary's `sqrt(n)·u` would cut the width ~19× at d=384 and is cited and refused for exactly that reason. [→ §8](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#8-not-earned)
- **One corpus carries the real-data-at-scale evidence, and it is an easy one for the arithmetic.** SIFT1M's integer components make its float32 scores exact, so its flip count is 0 *by arithmetic* and it can never produce evidence that rounding chose a ranking. `flipped > 0` stays a property of the generated clustered corpus; the nine-engine agreement table, the tuned-margin baseline, the Dot2 arm, the shuffled-enclosure control and the cost table are all generated-corpus only. [→ §10.7](https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md#107-what-this-section-does-not-carry-and-what-stays-synthetic)
- **Measured on CPython 3.11.9 only**, on one machine. Two of the five corpora need a network, and every row drawn from them is labelled `(downloaded)`: the three generated corpora reproduce offline and carry 550 of the 1,116 certified sets, so a reader without a network can check that much and no more.

---

## 🗺️ Roadmap

- 🧊 **Make the pessimism the product.** The tight rung and per-pair radii already exist; the missing piece is an escalation that runs only on the frontier and turns a refusal into a decision at a bounded cost, so the 379 pessimistic refusals become 379 answers.
- ⚡ **Dot2 as an opt-in rung.** It wins on coverage by a mile and loses on throughput by 61×–99×. That is a knob, not a verdict, and it should be exposed as one.
- 🔬 **A machine-checked radius.** Melquiond and Pion did this for the static filters. The `gamma`/`eta` derivation is one page and is the part of this repository whose bugs are unsound rather than merely wrong.
- 🧵 **A benchmark row for the argmin and threshold surfaces.** Both ship — `certified_argmin` is `certified_topk(k=1, largest=False)` and `certified_threshold` returns a trit array whose `0` is *undetermined* — and neither has a refusal count on any corpus above, so neither is quoted on this page.
- 🖥️ **A GPU path** — currently a host copy before the enclosure, which is why there is no GPU number anywhere on this page.

---

<p align="center"><sub>
<a href="https://github.com/teerthsharma/separatrix/blob/main/LICENSE">MIT</a> · python ≥ 3.10 · <a href="https://github.com/teerthsharma/separatrix/blob/main/RESULTS.md">RESULTS.md</a> ·
Invented by <b>Teerth Sharma</b> · teerths57@gmail.com ·
<a href="https://github.com/teerthsharma/separatrix">github.com/teerthsharma/separatrix</a><br>
<code>floating-point · rounding-error · interval-arithmetic · certificate · abstention ·
determinism · top-k · nearest-neighbors · numerical-reproducibility</code>
</sub></p>
