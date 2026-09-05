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
  suite     200 tests, 200 passed with the SIFT1M cache present, 0 skipped
            (199 passed, 1 skipped without it: test_sift1m_when_cached)
  one draw  every table below comes from one bench.py run, seed 11
  re-run    on the build that adds section 10, `bench.py` was re-run and its results.json
            compared field by field against the previous one: 6 fields differ and all 6
            are wall-clock timings measured under load. Every refusal count, agreement
            count and soundness field reproduces exactly.
```

Reproduce everything:

```
python -m venv .venv
.venv/Scripts/pip install -e .
.venv/Scripts/python -m pytest tests/ -q
.venv/Scripts/python bench.py --out results.json --assets assets
.venv/Scripts/python bench.py --sift --out .donotcommit/sift.json   # section 10, 516 MB
```

`bench.py --no-download` skips the two corpora that need a network and runs the rest.
**Sections 1-9 are generated corpora plus two downloads of a few thousand rows. Section 10
is the only real-data-at-scale section**: one downloaded million-row corpus, scored against
a ground truth this repository did not compute, and it lists at its end every claim above it
that stays synthetic-only.

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
| float32 gram + argpartition — the status quo | 0.0229 | 0.73× | 1.00× |
| float64 gram + argpartition — **the honest control** | 0.0314 | 1.00× | 1.37× |
| scipy.cdist float64 + argpartition | 0.5357 | 17.06× | 23.36× |
| separatrix rung 1, per-row radii | **0.0529** | **1.68×** | 2.31× |
| separatrix rung 2, per-pair radii | 0.0730 | 2.33× | 3.18× |
| the float32-vs-float64 diff — the competing practice | 0.0498 | 1.59× | 2.17× |

**rung 1 against the diff it replaces: 1.06×.**

Run-to-run spread on this machine, six runs over the same corpus — the printed draw above
plus five repeats of `bench.cost_table` on the identical arrays: rung 1 sits at
0.0517–0.0555 s, **1.59×–1.89×** the float64 control and **0.96×–1.12×** the diff. The
printed draw is the run that produced the rest of this file, so its cost belongs beside
them; it is the second-slowest of the six and neither end of the spread is hidden by it.

**Struck:** two earlier builds of this file quoted 0.0650 s and 0.0505 s for the rung-1
row, with spreads of 0.0544–0.0650 s and 0.0393–0.0505 s. Neither absolute reading
reproduces here and both are removed rather than carried forward. What survives across all
three builds is the ratio band — rung 1 has never left 1.5×–1.9× the float64 control or
0.9×–1.2× the diff — so the ratios are the measurement and the seconds are the draw.

**Withdrawn:** the brief's *0.14× the cost of the fp64 reference* and the design's revised
*0.91× fp64 / 0.54× the diff* both fail to reproduce. The honest words are **parity with
the diff** and **roughly 1.7× the float64 control**, and neither is a selling point.

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
    throughput cost of Dot2                           64.1x on the printed draw,
                                                      61x to 99x over six runs
```

Dot2 certifies every boundary this package refuses, and it costs roughly 100× more because
it is elementwise and forfeits the gemm entirely. The a-priori choice is a throughput
decision, and it is a decision this measurement makes expensive to defend on coverage.
Anyone who can afford roughly 100× should use Dot2.

The spread on the cost ratio is the a-priori denominator: one 600×384 gemm reads 2.7–5.9 ms
across those six runs while Dot2 holds at 0.23–0.36 s, so the denominator's noise is most
of the ratio's range. **Struck:** two earlier builds of this file quoted 48×–78× over four
runs and 81×–117× over six; neither range reproduces here and both are removed. A ratio
whose denominator is a 3 ms gemm is a measurement of this machine's scheduler as much as of
either algorithm, and the union of everything three builds have observed is 48×–117×. Only
the direction is not in doubt.

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
runs and proves nothing when it comes back empty; separatrix costs 0.96×–1.12× the diff
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

---

## 10. Real data at scale — SIFT1M, and a third party holding the answer

Everything above section 10 is either generated on this machine or downloaded at a few
thousand rows. This section is one corpus, downloaded whole, at a million rows, and it is
the only place in this repository where the answer a certificate is scored against was
computed by **somebody else**: the ANN\_SIFT1M release ships the authors' exact top-100
neighbour list beside the vectors.

```
corpus    SIFT1M, 1,000,000 x 128 float32 base + 10,000 x 128 queries + top-100 truth
source    huggingface.co/datasets/qbo-odp/sift1m (apache-2.0), mirroring INRIA TEXMEX
size      516 MB base, 5 MB queries, 4 MB ground truth; cached under .donotcommit/
fetch     separatrix.corpus.sift1m() -- the loader is committed, the 516 MB is not
shape     d = 128, k = 10, gram/cheap float32 unless a row says otherwise
```

Command for every number in this section:

```
.venv/Scripts/python bench.py --sift --out .donotcommit/sift.json
```

### 10.1 Three things that broke on it, and what the break was

None of the three is a soundness bug. All three are the difference between a package that
runs on a 2,000-row corpus in a test and one that runs on the corpus a user has.

**(a) Every frontier reported row 0 after `--escalate`.** `exact.escalate_row` built its
`Frontier` with `row=0` hard-coded — it is called one row at a time and was never told
which row it had. On the 100-query SIFT block, escalation leaves two rows undecided, 82 and
93, and both printed as row 0: a consumer reconstructing the refused set from `v.frontiers`
read **one** refused row where there were **two**. This is the reporting half of the
soundness bug 1.1(a) — the same wrong answer to "which rows did you refuse", arrived at a
different way. `escalate_row` now takes `row=` and `api.certified_topk` passes it;
`test_every_frontier_names_its_own_row_after_escalation` asserts
`len({f.row for f in v.frontiers}) == v.n_refused` on the escalated path, which is where
the invariant was missing. The disputed-row measurement in 10.3 could not have been
written before this fix: it selects rows by `f.row`.

**(b) `chunk=` was validated and then ignored.** `certified_topk` accepted `chunk`, raised
on `chunk <= 0`, and never used it. The scores are an `(m, n)` float64 array: 1,000 queries
against this corpus is **8.00 GB** in one allocation, on a machine with 2.9 GB free. It now
blocks the query rows — 0.80 GB at `chunk=100` — and the 1,000-query run below is the first
one that completes here. The CLI takes `--chunk` for the same reason.

Chunking is **not** a no-op on the arithmetic, and calling it one would be the same mistake
this package exists to name: BLAS picks a different gemm path for a 1-row right-hand side
than for a 17-row one, so `chunk=1` is a tenth numerically distinct evaluation of the same
formula on the same stored bytes. Measured on the near-duplicate corpus: **1 row of 17
moves between `chunk=1` and the unchunked call, and it is REFUSED under both**;
0 certified rows move, at any of `chunk = 1, 4, 17, 64`. `test_chunk_is_a_tenth_engine`
asserts both halves and fails if the gemv path ever stops moving that row.

**(c) The float64 norm pass allocated a second copy of the corpus.** `enclose._norms64`
computed `A.astype(np.float64)` on the whole array before any score existed: **1.02 GB**
for this corpus's 512 MB of float32, at the moment the caller has least room. It is now
blocked at ~32 MB of float64 at a time. Blocking is over rows and each row's reduction is
unchanged, so the values are bitwise identical — `test_norms64_blocking_is_exact` asserts
equality, not closeness.

Peak working set of the whole arm after all three fixes: **2,565 MB**, against 8.00 GB for
the score array alone before (b).

### 10.2 Why this corpus is the real-data twin of the exact lattice

SIFT descriptors are integers 0..255 stored as float32. Every Gram intermediate is then an
integer below 2^24: `||x||^2 <= 2.61e5`, `2<x,q> <= 5.2e5` by Cauchy–Schwarz, and float32
carries every integer below 1.68e7 exactly. So the float32 score **is** the exact integer
squared distance, and it is measured rather than argued:

| what | measurement | the control it scored against |
|---|---|---|
| float32 Gram scores differing from the exact value | **0 of 20,000,000** (max abs difference 0) | int64 arithmetic over the same bytes, 20 queries × 1,000,000 rows |
| components that are integer-valued | 100,000 of 100,000 sampled rows | `x == floor(x)` |

The consequence is the strongest statement available about a refusal: on this corpus the
flip count is **0 by arithmetic**, not by an oracle's opinion. Every refusal that is not an
exact tie is pessimism, exactly, and the pessimism is now visible as a number: the median
frontier has **gap 7.0 against width 16.12**, while the true error is **0**.

### 10.3 The certificate against a third party's answer

1,000 queries against the full 1,000,000-row base, `chunk=100`, 48.0 s:

| what | measurement | the control it scored against |
|---|---|---|
| refused | **52 of 1,000** (5.2%) | — |
| CERTIFIED sets agreeing with the published top-10 | **948 of 948** | the ANN\_SIFT1M authors' ground-truth file, computed outside this repository |
| rows where the published answer differs | **9**, rows 93, 170, 460, 574, 614, 731, 760, 930, 934 | — |
| how many of those 9 had been refused first | **9 of 9** | — |
| how many of those 9 exact arithmetic calls a tie | **9 of 9** | `exact.escalate_row`, scaled-integer |
| how many of those 9 were a float set exact arithmetic moved | **0** | same |

**Every disagreement with the third party is an exact tie, and every one of them was
refused before it was found.** On a tie both answers are correct and no arithmetic decides
it; what a certificate has to do there is refuse, and it did. Row 93 is the one small enough to
print in full: `#196106` and `#274922` are both at squared distance **42,192** from query
93 in exact integers. The published list takes `#274922`, this float32 run takes `#196106`,
and the frontier reports `gap 0.000000e+00` before either is preferred.

On the 100-query block, the same claim with the counts small enough to print:

| n | refused, m = 100 | reason | seconds |
|---|---|---|---|
| 10,000 | 2 | GRAM_CANCELLATION | 0.03 |
| 100,000 | 2 | GRAM_CANCELLATION | 0.38 |
| 1,000,000 | 11 | BOUNDARY_UNDETERMINED | 5.05 |

Certified rows agreeing with the published truth at n = 1,000,000: **89 of 89**. All rows:
99 of 100 — the one exception is row 93, refused.

**The prediction on record before this ran was that the refused fraction grows with n**,
because a fixed-width enclosure has more chances to straddle the rank-k boundary as the
corpus fills in around it. It held, and it is a statement about the corpus and not about
the bound: 2%, 2%, 11% at 10k, 100k, 1M.

### 10.4 Escalation, and the two rows nothing decides

100 queries, full base, `escalate=True`, 9.9 s:

| what | measurement |
|---|---|
| refused before escalation | 11 of 100 |
| still refused after | **2** — rows 82 and 93 |
| verdict | NOT CERTIFIED (EXACT\_TIE), exit 1 |
| exact scaled-integer products spent | 22 |
| float sets exact arithmetic moved | **0** |

9 of the 11 refusals were pessimism the escalation rung cleared for 22 exact dot products.
The other 2 are ties, which is the one outcome no precision removes.

### 10.5 The refusal's advice, run rather than printed

At n = 10,000 and n = 100,000 the refusal is GRAM\_CANCELLATION, whose `next_action` says
to run the direct kernel. So it was run, on the same queries at the same n:

| kernel | refused, m = 100 | verdict | seconds |
|---|---|---|---|
| gram, cheap | 2 of 100 | REFUSED (GRAM\_CANCELLATION) | 0.38 |
| direct | **0 of 100** | **CERTIFIED** | 3.3 |

The advice works and it costs **8.7×**. That ratio is the product: a code change that
certifies, at a price named before it is paid, against a re-run at higher precision that
certifies nothing.

Through the CLI, on the same arrays written to `.npy`:

```
.venv/Scripts/python -m separatrix check --corpus sift_X.npy --queries sift_Q.npy \
    --k 10 --chunk 25
  REFUSED (GRAM_CANCELLATION)                              98/100 determined
  boundary    row 30: in #54361 [6.928394e+04, 6.930006e+04]  out #31025
              [6.929094e+04, 6.930706e+04]  gap 7.000000e+00  width 1.611169e+01
  next        The direct kernel separates this pair; the Gram identity does not.
exit 2

.venv/Scripts/python -m separatrix check --corpus sift_X.npy --queries sift_Q.npy \
    --k 10 --chunk 25 --kernel direct
  CERTIFIED                                               100/100 determined
exit 0
```

### 10.6 The float16 must-fix, on downloaded bytes

`||x||^2` has median 2.587e5 and max 2.612e5 on this corpus, against float16's 6.55e4. The
Gram intermediate `(||q|| + ||x||)^2` reaches 1.04e6.

| what | measurement |
|---|---|
| float16 verdict on 100 real queries | **REFUSED (RANGE\_UNSAFE)**, exit 2, before any score is read |
| query rows the precondition names | 100 of 100 |

MNIST (section 2) was the first real corpus to show this and it was downloaded at 5,000
rows; SIFT1M shows it again at a million, on different bytes from a different decade.

### 10.7 What this section does not carry, and what stays synthetic

Claims that remain measured only on generated corpora, listed so `README.md` can label
them:

- **The nine-engine agreement table (section 2).** SIFT1M is not in it. Nine evaluations of
  a 1,000 × 1,000,000 score array is 72 GB and was not attempted; the real-data agreement
  evidence here is against the published ground truth instead, which is one external
  answer rather than nine internal ones. The tenth engine — `chunk` — is measured on the
  generated near-duplicate corpus only.
- **`flipped > 0` anywhere.** It stays a property of the clustered generated corpus
  (section 3). On SIFT1M the flip count is 0 *by arithmetic*, so this corpus can never
  produce one, and it is not evidence either way about corpora that can.
- **The tuned-margin baseline (7.2), the shuffled-enclosure control (7.3), the Dot2 arm
  (7.1), the cost table (6) and the pessimism factor (7.5).** All generated, all unchanged
  by this section.
- **Every GPU claim, every approximate-index claim.** Still not earned; see section 8.
