# separatrix — measured results

Every number here was produced on this machine by a command printed beside it. Nothing was
copied forward from the design document; where a design number did not reproduce, the
design number is struck and the measurement stands in its place. Arms that lost have their
own sections, at the same size as the arms that won.

```
provenance
  machine   WIN-16QAL06O9GB                 date 2026-09-05
  python    3.11.9 (CPython, MSC v.1938, 64-bit)
  numpy     2.4.6      scipy 1.17.1      torch 2.14.0+cpu
  extras    scikit-learn 1.9.0, datasets 5.0.1, sentence-transformers 6.0.1
  suite     195 tests, 195 passed, 0 skipped
  one draw  every table below comes from one bench.py run, seed 11
```

Reproduce everything:

```
python -m venv .venv
.venv/Scripts/pip install -e .
.venv/Scripts/python -m pytest tests/ -q
.venv/Scripts/python bench.py --out results.json --assets assets
```

`bench.py --no-download` skips the two corpora that need a network and runs the rest.

---

## 1. The soundness gate — mandatory, no threshold

One CERTIFIED decision that exact arithmetic contradicts withdraws the package. Not a
budget; an instance.

| what | measurement | the control it scored against |
|---|---|---|
| CERTIFIED verdicts the exact lattice contradicts | **0** | integer coordinates, so the top-4 is known before any float runs — the only arm whose truth is not an oracle's |
| enclosure escapes on the adversarial corpus | **0 of 656 pairs** — 82 pairs in each of 8 configurations (2 kernels × 2 bounds × 2 rungs), over 10 corpora | `exact.exact_sq`, scaled Python integers, no third-party arithmetic |
| enclosure escapes on the integer lattice | **0** | the known integer difference |
| escalation contradicting a certificate | **0** over all 384 refused rows, 5,827 exact dot products | `exact_sq` re-decision of every refused row |
| CERTIFIED on the recorded float16 range case | **0** | the float32 top-10 sets the float16 run was measured wrong against |

Command: `.venv/Scripts/python -m pytest tests/ -q`, then `.venv/Scripts/python bench.py`.

### 1.1 Two soundness bugs this build found and fixed

Both were found by running the modules against each other, not by reading them.

**(a) The frontier cap made refused rows read as certified.** `api.certified_topk` carried
at most 64 frontiers on a Verdict. On a 200-query corpus with 69 refused rows, the 5 rows
past the cap were invisible to any consumer that reconstructs the refused set from
`v.frontiers`, and two evaluations of one formula then appeared to disagree on 5 *certified*
rows. Measured before the fix: 5 apparent disagreements among certified. After: **0**, with
41 among refused, on the same corpus. The cap is gone; the CLI still caps what it *prints*
with `--max-report`. Pinned by `test_backends_agree_on_certified`, which now asserts
`len({f.row for f in v.frontiers}) == v.n_refused`.

**(b) A negative radius certified everything.** Precondition P3 admits `n·u == 1/2`, where
`gamma_n` is exactly 1.0 and is rounded outward to `1.0000000000000002`. The Gram form
multiplies by gamma and stays sound — an enormous, useless, *positive* radius. The direct
kernel's relative form divides by `1 - gamma` and produced a **negative** one, which inverts
every interval so that max-in falls below min-out and the rule certifies every row.

```
  corpus.adversarial("vacuous_f16_d1023")   float16, d = 1023, (d+1)u = 0.5 exactly
    gamma_1024(float16)                     1.0000000000000002
    direct radius                          -9.224903e+14
    certified_topk(..., kernel="direct")    CERTIFIED   <- over a bound that was not one
  after the guard in enclose.direct_radii
    both kernels                            REFUSED (BOUND_VACUOUS)
  d = 1022, the last legal width            REFUSED (BOUNDARY_UNDETERMINED), not vacuous
```

Pinned by `test_a_negative_radius_can_never_reach_the_rule` and by
`test_no_radius_anywhere_is_negative`, which asserts `R >= 0` and `lo <= hi` over every
kernel × bound × adversarial case.

### 1.2 The must-fix from the design, still holding

Precondition P2 fires **before any score is read**, so the refusal names the cause and not
the damage:

```
  corpus.adversarial("fp16_range_784")   raw 0..255 pixels, d = 784, float16
    max ||x||^2                          1.789e+07     against float16's 6.55e+04
    real MNIST, seed 0, 5,000 rows       max 1.489e+07, median 5.482e+06
    verdict                              REFUSED (RANGE_UNSAFE), 0 rankings returned
    reason is NOT                        NONFINITE_INPUT, which would name the damage
    upcast=True                          CERTIFIED_UPCAST, exit 4, sets equal to the
                                         float32 exact top-k on every query
```

The design's `||x||² ≈ 5.6e6` for MNIST is the **median**, not the maximum; corrected above
and in the docstrings.

### 1.3 The underflow term, with its control

`fl(ab) = ab(1+δ)` is false when a product lands subnormal. Command:
`.venv/Scripts/python -m pytest tests/test_enclose.py -q -k underflow`. d=8, 4,000 trials,
seed 21, exact scaled-integer ground truth.

| regime | radius alone | radius + η |
|---|---|---|
| float32, components ~1e-25 | **4000/4000** escaped | **0/4000** |
| float16, components ~3e-4 | **3078/4000** escaped | **0/4000** |
| float32, components ~1 — **the control** | **0/4000** | **0/4000** |

The control row is what makes this a measurement rather than a patch: the escapes are
specifically underflow, and η is not covering for a broken bound in the ordinary regime.

### 1.4 The false theorem, and why no benchmark finds it

The rank-k / rank-(k+1) boundary-pair rule is unsound whenever the radii vary. Verified:

```
  scores [0, 1, 2, 10]   radii [12, 0, 0, 0]   k = 2 smallest
    naive boundary pair   hi = 1.0 < lo = 2.0  ->  disjoint  ->  CERTIFIED {0, 1}
    the vector (11, 1, 2, 10) is inside the box and its top-2 is {1, 2}
    max-in / min-out      12.0 vs 2.0          ->  REFUSED
```

On every benchmark corpus the naive rule and the sound rule return **identical** refusal
counts — 15/15, 300/300, 35/35, 32/32, 2/2. The naive rule ships green. Only the
counterexample test catches it, which is why that test exists.

### 1.5 One conservatism, measured rather than assumed

`decide.topk_determined` uses the unclamped interval `[D-R, D+R]`. `Enclosure.interval`
clamps the lower end at zero, since a squared distance cannot be negative, and that clamp
would make the rule strictly stronger — but `decide.py` does not know the kernel, and a
negative value is legal for an inner-product score. Cost of the separation, measured across
all five corpora at 300 queries and k=10: **0 of 4,764,900 enclosure lower bounds fell
below zero**, so the clamp changes 0 refusals on every corpus here.

---

## 2. The headline

![agreement](assets/agreement.svg)

**Of 1,500 top-10 decisions across five corpora, 1,116 were certified and 0 of them moved
between nine numerically distinct evaluations of one formula on one set of stored bytes.
Every set that did move had been refused first.**

The nine evaluations, none of them separatrix's own arithmetic:

```
  numpy gram float32                     numpy gram float32, reduction order permuted
  numpy direct float32                   numpy gram float64
  scipy.spatial.distance.cdist float64   sklearn.metrics.pairwise.euclidean_distances
  torch.cdist (mm path)                  torch.cdist donot_use_mm_for_euclid_dist
  torch.cdist at batch 32
```

| corpus | shape | refused | sets that moved, CERTIFIED | sets that moved, REFUSED first |
|---|---|---|---|---|
| iid normalised d=384 | 2000×384 f32 | 15/300 | **0** | 0 |
| clustered normalised d=384 | 2000×384 f32 | 300/300 | **0** | **8** |
| MNIST-shaped (generated) d=784 | 2000×784 f32 | 35/300 | **0** | 0 |
| MNIST (downloaded) | 5000×784 f32 | 32/300 | **0** | 0 |
| BEIR SciFact + all-MiniLM-L6-v2 (downloaded) | 4883×384 f32 | 2/300 | **0** | 0 |

**Four of five corpora returned 0 in the last column. Those four are arms where this
package had nothing to say**, and they are printed here at the same size as the one that
did. Only the clustered arm carries evidence that the corpus was not too easy.

Command: `.venv/Scripts/python bench.py`, section *agreement among CERTIFIED sets*.

---

## 3. The refusal triple

A refusal is not a detection. `flipped` is the only column that is evidence this package
found anything, and `flipped = 0` was the prediction on record before the run.

| corpus | refused | flipped | exact tie | confirmed fine | float32-vs-float64 diff |
|---|---|---|---|---|---|
| iid normalised d=384 | 15/300 | 0 | 0 | 15 | 0/300 |
| clustered normalised d=384 | 300/300 | **5** | 0 | 295 | **5/300** |
| MNIST-shaped (generated) d=784 | 35/300 | 0 | 0 | 35 | 0/300 |
| MNIST (downloaded) | 32/300 | 0 | 0 | 32 | 0/300 |
| SciFact (downloaded) | 2/300 | 0 | 0 | 2 | 0/300 |

**The pre-registered prediction of `flipped = 0` on every corpus did not hold.** The
clustered arm flips 5 of 300 — rows 91, 96, 212, 236, 274, each verified against
`exact.exact_topk`. **The float32-vs-float64 diff finds the same 5**, so that arm is *not*
one where separatrix saw what the practice missed. It is one where both saw the same thing,
and only one of the two needed a second run to do it.

Everything in the `confirmed fine` column is the a-priori bound's pessimism, measured.

### 3.1 What one of those refusals looks like

The iid row of the table above, through the CLI rather than through `bench.py`. Reproduced
from scratch on this machine:

```
python -c "import numpy as np; r=np.random.default_rng(11); X=r.standard_normal((2000,384)).astype(np.float32); X/=np.linalg.norm(X,axis=1,keepdims=True); Q=r.standard_normal((300,384)).astype(np.float32); Q/=np.linalg.norm(Q,axis=1,keepdims=True); np.save('X.npy',X); np.save('Q.npy',Q)"
.venv/Scripts/python -m separatrix check --corpus X.npy --queries Q.npy --k 10 --max-report 1
```

`--max-report 1` caps what the CLI *prints*; the `Verdict` caps nothing, and the line
naming the boundaries it did not print is part of the output.

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

Exit code 2. The `15 of 300` matches the iid row of §3's table and of §5's `gram refused`
column, from an independent draw of the same generator, and `width 1.840634e-04` is §5's
median gram width at d=384.

The `7 of those 15` is a **pair**-level measurement and is why `GRAM_CANCELLATION` is a
reason code and never a certificate: certifying needs max-in/min-out over all n, not a
verdict on one pair. Re-running the same corpus with the direct kernel, which is the
row-level question:

```
.venv/Scripts/python -m separatrix check --corpus X.npy --queries Q.npy --k 10 --kernel direct --max-report 1

  REFUSED (BOUNDARY_UNDETERMINED)                         292/300 determined
  detail      8 of 300 rows have a rank-10 boundary this enclosure does not
              decide
  computed    kernel direct   bound relative   per-pair   k 10
```

(the header lines of that run; the boundary, dtype, canary and next blocks are the same
shape as above and are not repeated here)

`8 of 300` reproduces §5's `direct refused` column for this corpus, and 15 − 7 = 8 is
consistent with it — but the two are separate measurements and the agreement is reported,
not derived.

---

## 4. The pessimism gate, and how it landed

The design pre-committed: if the refused fraction on a normalised d=384 corpus exceeds ~5%
**and** escalation confirms every refusal was fine, the a-priori bound is too loose to ship
as a general top-k certifier.

```
  synthetic clustered normalised d=384    300/300 refused (100%)   295/300 confirmed fine
  real BEIR SciFact, all-MiniLM-L6-v2       2/300 refused (0.67%)    2/2   confirmed fine
  iid normalised d=384                     15/300 refused (5.0%)    15/15  confirmed fine
```

**The gate fires on the synthetic clustered generator and does not fire on the real
corpus.** The design predicted 107/300 on a clustered normalised d=384 index and
generalised from it to real embeddings; on the real corpus the measurement is 2/300. The
correction runs both ways: the synthetic clustered generator (20 points per cluster at
spread 0.02) is far more adversarial than BEIR SciFact, and no number from it may be quoted
as a statement about a real index.

So the narrowed claim is narrower than the design's, and in a different direction:

> On a real normalised float32 retrieval index, separatrix refuses **2 of 300** rank-10
> boundaries and every one of those 2 was pessimism rather than damage. The refusals that
> carry weight are on **un-normalised features, low-precision or out-of-range indexes, and
> the Gram-versus-direct kernel switch** — and the claim that stands on every corpus is
> determinism, not detection.

---

## 5. The kernel switch — where the product is

The Gram identity and the direct sum compute the same real number and have enclosures that
differ by orders of magnitude: absolute-and-cancellation-dominated against relative.

| corpus | gram refused | direct refused | median gram width | median direct width |
|---|---|---|---|---|
| iid normalised d=384 | 15/300 | 8/300 | 1.841e-04 | 9.180e-05 |
| clustered normalised d=384 | **300/300** | **2/300** | 1.841e-04 | 9.167e-05 |
| MNIST-shaped (generated) d=784 | 35/300 | 11/300 | 1.414e+03 | 5.191e+02 |
| MNIST (downloaded) | 32/300 | 4/300 | 3.580e+03 | 6.290e+02 |
| SciFact (downloaded) | 2/300 | 1/300 | 1.841e-04 | 8.085e-05 |

**298 of the 300 refusals on the clustered index are attributable to the kernel alone.**
That is the one refusal in the catalogue that names a *code change* rather than a
re-observation.

### 5.1 The 25-row switch, measured on the installed torch

![the 25-row switch](assets/switch.svg)

One 40×8 float32 array, `torch 2.14.0+cpu`, the same stored bytes at every row count:

| rows | max &#124;mm − direct&#124; | |
|---|---|---|
| 24 | **0.000e+00** | bit-identical |
| 25 | **0.000e+00** | bit-identical |
| 26 | **9.766e-04** | the Gram identity |
| 32 | 1.381e-03 | the Gram identity |
| 40 | 1.381e-03 | the Gram identity |

The keyword that turns it off was **verified against the installed library**, not quoted:
`compute_mode="donot_use_mm_for_euclid_dist"` accepts, and so do
`"use_mm_for_euclid_dist"` and `"use_mm_for_euclid_dist_if_necessary"`. This was marked NOT
EARNED in the design until checked.

### 5.2 Frame 1 — changing the formula is the fix; changing the precision is not

![frame 1](assets/frame1.svg)

```
  x = (1e6, 0)   y = (1e6 + 1e-6, 0)          all values float64
    gram identity  ||x||^2 + ||y||^2 - 2<x,y>   =  0.0                      exactly
    direct sum     sum_l (x_l - y_l)^2         =  1.0000152290447206e-12
    exact, scaled integers                     =  1.0000152290447206e-12
    torch.cdist use_mm_for_euclid_dist          =  0.0
    torch.cdist donot_use_mm_for_euclid_dist    =  1.00000761449337e-06     (a distance)
    R_cheap at d=2                              =  1.776357e-03
    verdict                                     =  REFUSED (GRAM_CANCELLATION)
```

At float32 the two points **are** the same stored vector — `np.float32(1e6 + 1e-6) ==
np.float32(1e6)` is `True` — so `0.0` is the correct squared distance for the data as it
stands and there is nothing to refuse. The frame runs in float64, where the inputs are
distinct and the Gram identity still returns exactly `0.0`. Command:
`.venv/Scripts/python -m separatrix demo --frame cancellation`.

### 5.3 Frame 3 — the one frame no a-posteriori method can run

A float64 diff, two batch sizes and stochastic rounding all find what *moved*. None of them
can say a boundary was never determined on a row where the two runs happened to agree.
Frame 3 names the undetermined rows from **one** evaluation, then runs a second one.
Command: `.venv/Scripts/python -m separatrix demo --frame preview`.

```
  corpus 400 x 64 float32, 60 queries, k = 5, 40 near-duplicate pairs at 3e-07
  named undetermined, from one evaluation:  9 of 60  [5, 13, 14, 22, 37, 39, 40, 54, 57]
  actually differed, over two evaluations:   3 of 60  [14, 22, 54]
  differed and NOT named:                    0  []
```

The second evaluation is `cli.permuted_evaluation` — the same formula on the same stored
bytes with the `d` columns permuted identically, which leaves every exact score unchanged
and changes every rounding. It needs nothing installed.

**`differed and NOT named` is the only row that is evidence.** It is a soundness check, not
a demonstration: a row that two evaluations decided differently and that was not named in
advance is a counterexample to the certificate, and `frame_preview` returns exit 2 on one.
Pinned by `test_frame_three_names_every_row_two_evaluations_decided_differently`.

**The control is in the corpus, not in the table.** A corpus of nothing but near-duplicates
refuses every row, and then "every row that differed was named" is true by construction and
measures nothing. `cli.near_duplicate_corpus` puts 40 near-duplicate pairs in 400 points, so
**51 of the 60 rows are certified and had something to lose** (60 minus the 9 named above).
`test_the_demo_corpus_is_seeded_and_leaves_the_frame_something_to_lose` asserts the
property — `0 < named < 60` — rather than the number. The 6 rows named
and not differing are the a-priori bound's pessimism, in the same table as the 3 that were
real. Seeded (`seed=3`, `seed=7`), deterministic across runs on this machine.

---

## 6. Cost

Best of 5, 5,000×784 float32 corpus, 300 queries, k = 10. Command:
`.venv/Scripts/python bench.py`, section *cost*.

| | seconds | vs float64 gram | vs float32 gram |
|---|---|---|---|
| float32 gram + argpartition — the status quo | 0.0233 | 0.66× | 1.00× |
| float64 gram + argpartition — **the honest control** | 0.0353 | 1.00× | 1.51× |
| scipy.cdist float64 + argpartition | 0.6112 | 17.32× | 26.19× |
| separatrix rung 1, per-row radii | **0.0650** | **1.84×** | 2.78× |
| separatrix rung 2, per-pair radii | 0.0861 | 2.44× | 3.69× |
| the float32-vs-float64 diff — the competing practice | 0.0625 | 1.77× | 2.68× |

**rung 1 against the diff it replaces: 1.04×.**

Run-to-run spread on this machine, six runs over the same corpus — the printed draw above
plus five repeats of `bench.cost_table` on the identical arrays: rung 1 sits at
0.0544–0.0650 s, **1.59×–1.84×** the float64 control and **0.94×–1.17×** the diff. The
printed draw is the slowest of the six and is printed anyway, because the run that produced
the rest of this file is the run whose cost belongs beside them.

**Struck:** an earlier build of this file quoted 0.0505 s, 1.66× the control and a
0.0393–0.0505 s spread. None of that reproduces here — every one of the six observations
above is slower than 0.0505 s — so it is removed rather than carried forward. The low end
of a spread is the one number a cost table must not quietly retain.

**Withdrawn:** the brief's *0.14× the cost of the fp64 reference* and the design's revised
*0.91× fp64 / 0.54× the diff* both fail to reproduce. The honest words are **parity with
the diff** and **roughly 1.8× the float64 control**, and neither is a selling point.

---

## 7. Arms that lost

### 7.1 Ogita–Rump–Oishi Dot2 beats this on coverage, decisively

The direct competitor to the whole engine: an *a-posteriori* bound at `u` where this is an
*a-priori* bound at `(d+1)u`. Implemented in float32 arithmetic — Dekker `TwoProduct`,
Knuth `TwoSum` — not simulated in float64, because a float64 simulation gives the same
coverage at a fictitious cost, which is the one way this arm could be made to look good
dishonestly.

```
  64 queries, clustered normalised d=384, one draw
    refused, a-priori gamma bound (this package)      64 / 64
    refused, Dot2 a-posteriori bound (the competitor)   0 / 64
    throughput cost of Dot2                           81x to 117x, over six runs
```

Dot2 certifies every boundary this package refuses, and it costs two orders of magnitude
more because it is elementwise and forfeits the gemm entirely. The a-priori choice is a
throughput decision, and it is a decision this measurement makes expensive to defend on
coverage. Anyone who can afford two orders of magnitude should use Dot2.

The spread on the cost ratio is the a-priori denominator: one 600×384 gemm reads 2.4–4.4 ms
across those six runs while Dot2 holds at 0.26–0.36 s, so the denominator's noise is most
of the ratio's range. The direction is not in doubt. An earlier build of this file quoted
48× to 78× over four runs; that range does not reproduce and is struck.

### 7.2 The tuned-margin baseline draws on four of five corpora

`gap > eps * |score|`, four lines and one fitted constant. Each cell is
`certified/queries` and the number of those certificates a witness contradicts; a witness is
two evaluations differing or exact arithmetic moving the set, so the wrong count is a
**lower bound**, not an audit.

| eps | iid d=384 | clustered d=384 | MNIST-shaped | MNIST | SciFact |
|---|---|---|---|---|---|
| 1e-08 | 300/300, 0 wrong | 295/300, **5 wrong** | 300/300, 0 | 300/300, 0 | 300/300, 0 |
| 1e-07 | 300/300, 0 | 295/300, **5** | 300/300, 0 | 300/300, 0 | 300/300, 0 |
| 1e-06 | 300/300, 0 | 295/300, **5** | 299/300, 0 | 300/300, 0 | 300/300, 0 |
| 1e-05 | 299/300, 0 | 295/300, **5** | 297/300, 0 | 300/300, 0 | 300/300, 0 |
| 1e-04 | 286/300, 0 | 293/300, **4** | 289/300, 0 | 296/300, 0 | 299/300, 0 |
| 1e-03 | 178/300, 0 | 268/300, 0 | 192/300, 0 | 280/300, 0 | 273/300, 0 |

**On coverage this package loses badly**: separatrix certifies 0 of 300 on the clustered
arm where the tuned margin certifies 295. What the tuned margin does not have is any eps
below 1e-3 that avoids issuing certificates a witness contradicts, and the eps that does —
1e-3 — costs 122 of 300 certificates on the iid arm and 108 on MNIST-shaped. So no single
constant holds across all five corpora, which is the differentiator the design said had to
be measured rather than argued. It is smaller than the pitch, and this is it stated at its
real size: **no tuning, and a proof.**

### 7.3 The shuffled-enclosure control is a no-op where the design said it would be

Permuting the per-pair radii within a query row, scored against rung 2 beside it. At rung 1
the radius is constant across a row and the shuffle is the identity, so this control can
only run against rung 2 — the design's version of it was a tautology and is corrected here.

| corpus | rung 1 | naive rule | rung 2 | shuffled |
|---|---|---|---|---|
| iid normalised d=384 | 15 | 15 | 15 | 15 |
| clustered normalised d=384 | 300 | 300 | 300 | 300 |
| MNIST-shaped (generated) d=784 | 35 | 35 | 26 | **28** |
| MNIST (downloaded) | 32 | 32 | 15 | **18** |
| SciFact (downloaded) | 2 | 2 | 2 | 2 |

A no-op on both normalised corpora, exactly as predicted before the run — with unit norms
the cheap per-pair radius is `gamma_386 · 4` for every pair. It bites only where the norms
vary. The rung-1 memory collapse costs 0 refusals at d=384 and **35 against 26** on
MNIST-shaped, **32 against 15** on real MNIST.

### 7.4 The float32-vs-float64 diff, the practice this replaces

0 of 300 disagreements on four of five corpora, and the same 5 as separatrix on the fifth.
**The status quo was right on 1,495 of 1,500 rankings measured here.** The diff needs two
runs and proves nothing when it comes back empty; separatrix costs 0.94×–1.17× the diff
over six runs — parity, inside the run-to-run spread — and returns a proof. That is the
whole of the difference, and it is smaller than a headline.

### 7.5 The exact lattice's pessimism factor

Integer coordinates make the top-4 known before any float runs. n=16, d=32, k=4, float32,
8 trials at seeds 1000.., `delta` the exact integer margin at the rank-k frontier.

```
  delta*        smallest margin this package certifies       1024   deterministic
  delta_wrong   largest margin the float decision got wrong      4   MAX over 8 trials
  pessimism factor                                            256x
```

`delta*` is a property of the bound and needs no sampling; `delta_wrong` is a maximum over
a sample and can only rise, so 256× is an upper bound on the gap and not a measurement of
it. `delta = 0` — an exact tie, where no correct answer exists — gives
`NOT CERTIFIED (EXACT_TIE)` at every dtype and is never certified.

---

## 8. NOT EARNED

Claims that appear nowhere in this repository because nothing here measured them:

- **Any GPU number.** No CUDA path is designed. A CUDA tensor forces a host copy before the
  enclosure. Every number above is CPU.
- **Any claim about an approximate index.** IVF/PQ/HNSW search error is ~1e-2 relative, four
  orders above the float32 enclosure width; certifying the rounding of an approximate score
  would be certifying the wrong quantity while printing CERTIFIED. faiss is not a dependency,
  not a loader, and not a refusal code.
- **Accumulator width.** Precondition P4 tests the multiplier with a 4×4 canary and catches
  TF32, bfloat16 inputs and AMX with no vendor flag taxonomy. P5, the accumulator *width*,
  is not testable by either probe tried: the ones-vector probe `dot(1,1)==n` is exact at
  n=32768 for float16 and float32 alike because every partial sum of ones is a power of two,
  and the eps-tail probe retains 1.000 of the tail at every dtype because it cannot separate
  accumulator width from summation order. It is a declared assumption and it travels on
  every Verdict as `accum_assumed`, where a consumer can see it.
- **Machine-checked bounds.** Melquiond and Pion's static a-priori filters are formally
  certified; these are tested against a scaled-integer oracle. That is a gap, stated as one.
- **`summation="pairwise"` as a certificate.** Deleted. OpenBLAS accumulates with ~8 unrolled
  accumulators, giving reduction depth ≈ d/8 + 3 ≈ 101 at d=784 against the pairwise model's
  2·log₂(784)+2 = 21.2 — the pairwise envelope is 4.8× too shallow, by counting, before any
  measurement. `gamma_{d+2}` is a provable envelope over every reduction order, because every
  reduction tree over d products performs d−1 additions and so has depth at most d−1.
- **A probability of any kind.** Higham & Mary's `sqrt(n)·u` would cut the width by ~19× at
  d=384 and would very plausibly close the pessimism gap. It is cited and refused: it is a
  probability, and no output of this package carries one.

---

## 9. Limits

The certificate is about the rounding of one named formula on the bytes it was handed. It
says nothing about how those bytes got there: a 384-dimensional embedding out of a float16
forward pass carries ~1e-3 relative error, three to four orders above the float32 rounding
certified here. CERTIFIED means *rounding did not choose this ranking*. A refusal exhibits
no flip; only escalation decides which way a boundary falls, and on these corpora it decided
that 379 of 384 refusals were pessimism. `ordered=True` compares k boundaries instead of one
and refuses correspondingly more often; its column is never merged with the set column. The
refusal actions that work offline — adopt a tie-break, escalate to exact — do not ship at
the 26% refusal rate measured here, 384 of 1,500 rows, and the action that works in
production is free and needs no library: retrieve k+5 and let the reranker absorb the
boundary. Two of the five corpora above need a network; every row drawn from them is
labelled `(downloaded)` in this file and in `README.md`, and the three generated corpora
reproduce offline and carry 550 of the 1,116 certified sets. Measured on CPython 3.11.9
only.
