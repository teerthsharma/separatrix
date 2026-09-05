"""The rule, over (D, R) alone.

Pure.  No dtypes, no matmuls, no corpus, no kernel ever reaches this module -- which is
the point: the claim this layer makes is "given a correct enclosure the verdict is exact",
and **nothing in this file is evidence that the enclosure is correct**.

The rule
--------

    T is the k smallest (or largest) of D.  T is DETERMINED when

        max_{i in T} (D_i + R_i)   <   min_{j not in T} (D_j - R_j)         [strict]

    and then T is the top-k set of every vector in the box prod_i [D_i-R_i, D_i+R_i], in
    particular of the exact scores.

The false theorem this replaces
-------------------------------

Comparing only the rank-k and rank-(k+1) enclosures is UNSOUND whenever the radii vary
across rows -- and they do: R scales as (||q||+||x||)^2, which is monotone in ||x|| and has
nothing to do with the order of D.  Counterexample, verified and pinned as a test:

    scores [0, 1, 2, 10]   radii [12, 0, 0, 0]   k = 2 smallest
      naive rank-k/rank-(k+1) pair:  hi = 1.0 < lo = 2.0  ->  disjoint  ->  CERTIFIED {0,1}
      the vector (11, 1, 2, 10) lies inside the box and its 2 smallest are {1, 2}
      this rule:  max-in 12.0  vs  min-out 2.0  ->  not determined

The two rules coincide exactly when R is constant, so nothing is given up.  **A benchmark
cannot find this**: on all three prototype corpora the naive rule and this one returned
identical refusal counts (19/19, 107/107, 20/20).  Only the counterexample test and a
heteroscedastic-norm corpus catch it.

`largest=True` is negate-and-reuse, so there is exactly one rule and exactly one place to
be wrong.

Order is not claimed unless asked.  `ordered=True` additionally requires the k-1 adjacent
enclosures to be disjoint; adjacent suffices because "entirely to the left of" is
transitive.  The SET claim has no such chain across the interior it skips, which is exactly
why the set rule needs max/min.  Measured cost: k boundaries instead of one, so a 6% set
refusal rate becomes roughly 40% at k=10.

Self-check:  python -m separatrix.decide
"""

from __future__ import annotations

import numpy as np

from .verdict import Frontier


def topk_set(D: np.ndarray, k: int, largest: bool = False) -> np.ndarray:
    """The k smallest (or largest) indices of D, ascending in score.  Not a certificate."""
    n = D.shape[-1]
    if not (0 < k < n):
        raise ValueError(f"k must satisfy 0 < k < n; got k={k}, n={n}")
    s = -D if largest else D
    part = np.argpartition(s, k - 1)[:k]
    return part[np.argsort(s[part], kind="stable")]


def topk_determined(
    D: np.ndarray,
    R: np.ndarray | float,
    k: int,
    largest: bool = False,
    *,
    ordered: bool = False,
    row: int = 0,
) -> Frontier | None:
    """None when the top-k set is determined; the blocking Frontier when it is not.

    One-directional: disjoint enclosures certify, overlapping enclosures certify nothing in
    either direction.  A returned Frontier is not evidence that the set is different, only
    that this enclosure does not decide it.

    `D` is 1-D of length n.  `R` is a scalar or broadcastable to it, and must be a bound on
    |D - exact| -- this function does not and cannot check that.

    The interval used here is `[D-R, D+R]`, **unclamped**.  `Enclosure.interval` clamps the
    lower end at zero because a squared distance cannot be negative, and that clamp would
    make this rule strictly stronger -- but this file does not know the kernel, and for an
    inner-product score a negative value is legal, so clamping here would be unsound for a
    score type this rule is meant to serve.  The conservatism it costs was measured rather
    than assumed: across all five benchmark corpora (iid, clustered, MNIST-shaped, real
    MNIST, real SciFact, 300 queries each at k=10) **0 of 4,764,900 enclosure lower bounds
    fell below zero**, so the clamp changes 0 refusals on every corpus measured.
    """
    D = np.asarray(D, dtype=np.float64).ravel()
    if largest:
        # negate and reuse: one rule, one place to be wrong.  The radii are unsigned and
        # ride along unchanged.
        f = topk_determined(-D, R, k, largest=False, ordered=ordered, row=row)
        if f is None:
            return None
        return Frontier(
            row=f.row,
            inside=f.inside,
            outside=f.outside,
            inside_lo=-f.inside_hi,
            inside_hi=-f.inside_lo,
            outside_lo=-f.outside_hi,
            outside_hi=-f.outside_lo,
            gap=f.gap,
            width=f.width,
        )

    R = np.broadcast_to(np.asarray(R, dtype=np.float64), D.shape)
    T = topk_set(D, k, largest=False)
    mask = np.zeros(D.shape, dtype=bool)
    mask[T] = True

    hi = D + R
    lo = D - R
    inside = int(T[np.argmax(hi[T])])
    out_idx = np.flatnonzero(~mask)
    outside = int(out_idx[np.argmin(lo[out_idx])])

    f = Frontier(
        row=row,
        inside=inside,
        outside=outside,
        inside_lo=float(lo[inside]),
        inside_hi=float(hi[inside]),
        outside_lo=float(lo[outside]),
        outside_hi=float(hi[outside]),
        gap=float(D[outside] - D[inside]),
        width=float(R[inside] + R[outside]),
    )
    if not f.determined:
        return f
    if ordered:
        # k-1 adjacent pairs inside T; transitivity does the rest.
        for a, b in zip(T[:-1], T[1:]):
            g = Frontier(
                row=row,
                inside=int(a),
                outside=int(b),
                inside_lo=float(lo[a]),
                inside_hi=float(hi[a]),
                outside_lo=float(lo[b]),
                outside_hi=float(hi[b]),
                gap=float(D[b] - D[a]),
                width=float(R[a] + R[b]),
            )
            if not g.determined:
                return g
    return None


def naive_boundary_pair(
    D: np.ndarray, R: np.ndarray | float, k: int, largest: bool = False
) -> bool:
    """The FALSE theorem: compare only the rank-k and rank-(k+1) enclosures.

    Shipped so `test_boundary_pair_rule_is_unsound_and_ours_is_not` can score against it
    rather than against a re-typed formula.  Never called by anything that certifies.
    """
    D = np.asarray(D, dtype=np.float64).ravel()
    R = np.broadcast_to(np.asarray(R, dtype=np.float64), D.shape)
    order = np.argsort(-D if largest else D, kind="stable")
    a, b = int(order[k - 1]), int(order[k])
    if largest:
        return bool((D[a] - R[a]) > (D[b] + R[b]))
    return bool((D[a] + R[a]) < (D[b] - R[b]))


def worst_corner(D: np.ndarray, R: np.ndarray | float, T: np.ndarray) -> np.ndarray:
    """The box's extremum for this question: members pushed up, non-members pushed down.

    A test that passes on this corner is a PROOF of the rule over the whole box, not a
    sample of it -- no other point of the box can displace a member of T if this one
    cannot.
    """
    D = np.asarray(D, dtype=np.float64).ravel()
    R = np.broadcast_to(np.asarray(R, dtype=np.float64), D.shape)
    v = D - R
    v = v.copy()
    v[np.asarray(T, dtype=np.intp)] = (D + R)[np.asarray(T, dtype=np.intp)]
    return v


def rows_determined(
    D: np.ndarray,
    R: np.ndarray | float,
    k: int,
    largest: bool = False,
    *,
    ordered: bool = False,
) -> list[Frontier | None]:
    """topk_determined over every row of a (m, n) score matrix.  A loop, deliberately.

    ponytail: a Python loop over m rows, ~30 us each.  Vectorise only if a profile ever
    shows it beside the O(mnd) matmul that produced D, which it will not.
    """
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    R = np.asarray(R, dtype=np.float64)
    out: list[Frontier | None] = []
    for i in range(D.shape[0]):
        r = R if R.ndim == 0 else R[i]
        out.append(topk_determined(D[i], r, k, largest, ordered=ordered, row=i))
    return out


def _demo() -> None:
    # the false theorem, and the witness that kills it
    D = np.array([0.0, 1.0, 2.0, 10.0])
    R = np.array([12.0, 0.0, 0.0, 0.0])
    assert naive_boundary_pair(D, R, 2) is True
    f = topk_determined(D, R, 2)
    assert f is not None and f.inside == 0 and f.outside == 2
    witness = np.array([11.0, 1.0, 2.0, 10.0])
    assert np.all(np.abs(witness - D) <= R)
    assert set(topk_set(witness, 2).tolist()) == {1, 2}
    assert set(topk_set(D, 2).tolist()) == {0, 1}

    # constant radii: the two rules agree, so nothing is given up
    Rc = np.full(4, 0.2)
    assert (topk_determined(D, Rc, 2) is None) == naive_boundary_pair(D, Rc, 2)

    # every certified set survives the box's worst corner
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(2000):
        d = rng.standard_normal(12)
        r = rng.random(12) * 0.4
        if topk_determined(d, r, 3) is None:
            T = topk_set(d, 3)
            assert set(topk_set(worst_corner(d, r, T), 3).tolist()) == set(T.tolist())
            checked += 1
    assert checked > 100, checked

    # largest is negate-and-reuse, on HETEROSCEDASTIC radii (constant radii would pass
    # under the unsound rule too, so the test would not bite)
    for _ in range(500):
        d = rng.standard_normal(9)
        r = rng.random(9) ** 3
        a = topk_determined(d, r, 3, largest=True)
        b = topk_determined(-d, r, 3, largest=False)
        assert (a is None) == (b is None)
        if a is None:
            assert set(topk_set(d, 3, largest=True).tolist()) == set(
                topk_set(-d, 3).tolist()
            )

    # the set can be determined while the order inside it is not
    Ds = np.array([0.0, 0.01, 5.0])
    Rs = np.array([0.05, 0.05, 0.05])
    assert topk_determined(Ds, Rs, 2) is None
    assert topk_determined(Ds, Rs, 2, ordered=True) is not None

    print("decide: ok")


if __name__ == "__main__":
    _demo()
