"""The oracle, and the escalation rung it is shared with.

The oracle is scored against a closed form -- integer coordinates, where the answer is a
known integer before any float runs -- and never against separatrix's own arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from separatrix.decide import topk_determined, topk_set
from separatrix.exact import (
    BITS_DOUBLE,
    BITS_SMALL,
    Escalation,
    bits_for,
    escalate,
    escalate_row,
    exact_sq,
    exact_sq_float,
    exact_topk,
    scale,
)
from separatrix.verdict import EXACT_TIE


# -- the oracle ------------------------------------------------------------------------------


def test_oracle_against_a_closed_form():
    """Integer coordinates make the exact squared distance a known integer."""
    n, d = 100_000, 3
    rng = np.random.default_rng(0)
    A = rng.integers(-2048, 2049, size=(n, d)).astype(np.float32)
    B = rng.integers(-2048, 2049, size=(n, d)).astype(np.float32)
    want = ((A.astype(np.int64) - B.astype(np.int64)) ** 2).sum(axis=1)
    S = scale(BITS_SMALL)
    hits = sum(exact_sq(A[i], B[i]) == int(want[i]) * S for i in range(n))
    assert hits == n


def test_the_ldexp_spelling_would_have_overflowed():
    """Why the scaling goes through as_integer_ratio and not math.ldexp.

    The obvious spelling `int(math.ldexp(v, 1074))` returns inf for any |v| > 2**-50, so
    it is unusable for float64 -- silently, and in the direction that produces a wrong
    exact value rather than an error.
    """
    with pytest.raises(OverflowError):
        math.ldexp(1e6, BITS_DOUBLE)
    # and the spelling that ships is exact for the same value
    x = np.array([1e6, 0.0])
    y = np.array([1e6 + 1e-6, 0.0])
    got = exact_sq(x, y, BITS_DOUBLE) / scale(BITS_DOUBLE)
    assert got == pytest.approx(1.0000152290447206e-12, rel=1e-12)
    # the Gram identity in float64 returns exactly 0.0 for that same pair: changing the
    # formula is the fix, changing the precision is not
    assert x[0] ** 2 + y[0] ** 2 - 2 * x[0] * y[0] == 0.0
    assert (x[0] - y[0]) ** 2 == pytest.approx(got, rel=1e-12)


def test_bits_too_small_raises_rather_than_rounding():
    with pytest.raises(ValueError):
        exact_sq(np.array([1e-60]), np.array([0.0]), BITS_SMALL)
    assert exact_sq(np.array([1e-60]), np.array([0.0]), BITS_DOUBLE) > 0


def test_bits_for_matches_the_dtypes_smallest_subnormal():
    assert bits_for(np.float16) == BITS_SMALL
    assert bits_for(np.float32) == BITS_SMALL
    assert bits_for(np.float64) == BITS_DOUBLE
    assert float(np.finfo(np.float32).smallest_subnormal) == 2.0**-BITS_SMALL


def test_exact_topk_is_the_ground_truth_float64_agrees_with_on_the_lattice():
    rng = np.random.default_rng(1)
    X = rng.integers(-64, 65, size=(120, 5)).astype(np.float32)
    for _ in range(20):
        q = rng.integers(-64, 65, size=5).astype(np.float32)
        d64 = ((X.astype(np.float64) - q.astype(np.float64)) ** 2).sum(1)
        assert set(exact_topk(q, X, 4).tolist()) == set(topk_set(d64, 4).tolist())


# -- escalation ------------------------------------------------------------------------------


def _one_d(vals, dtype=np.float32):
    return np.asarray(vals, dtype=dtype).reshape(-1, 1)


def test_escalating_the_named_pair_does_not_lift_the_refusal():
    """The recorded counterexample: s=[1.0, 1.05, 1.06, 5.0], r=[.1, .1, .1, 0], k=2.

    The refusal names the pair (1, 2).  Resolve exactly that pair and
    max_in = max(s0 + r0, s1) = 1.10 still exceeds min_out = 1.06, because index 0's
    enclosure also straddles.  So escalation must resolve the FRONTIER.
    """
    D = np.array([1.0, 1.05, 1.06, 5.0])
    R = np.array([0.1, 0.1, 0.1, 0.0])
    f = topk_determined(D, R, 2)
    assert f is not None and (f.inside, f.outside) == (1, 2)

    # pair-only escalation: indices 1 and 2 become exact, index 0 keeps its interval
    max_in_pair_only = max(D[0] + R[0], D[1])
    min_out_pair_only = D[2]
    assert max_in_pair_only > min_out_pair_only

    # the frontier is wider than the pair
    max_in, min_out = D[0] + R[0], D[2] - R[2]
    inside_frontier = [i for i in (0, 1) if D[i] + R[i] >= min_out]
    outside_frontier = [j for j in (2, 3) if D[j] - R[j] <= max_in]
    assert inside_frontier == [0, 1] and outside_frontier == [2]


def test_escalation_lifts_the_refusal():
    """The same shape, on a corpus where the scores really are the exact distances."""
    q = np.zeros(1, dtype=np.float32)
    X = _one_d([1.0, 1.0246950766, 1.0295630141, 2.2360679775])
    D = np.array([exact_sq_float(q, X[j]) for j in range(4)])
    R = np.array([0.1, 0.1, 0.1, 0.0])
    assert topk_determined(D, R, 2) is not None  # refused before escalation

    e = escalate_row(q, X, D, R, 2)
    assert e.determined and e.reason == ""
    assert sorted(e.indices.tolist()) == [0, 1]
    assert e.n_escalated >= 3, "only the named pair was escalated"
    assert sorted(e.indices.tolist()) == sorted(exact_topk(q, X, 2).tolist())


def test_escalation_reports_when_the_float_set_differed():
    """The third outcome, and the most valuable one.

    A float computation that got the order wrong, inside a valid enclosure.  Returning the
    float set here would contradict the guarantee, which promises the set exact arithmetic
    returns -- so escalation returns the CORRECTED indices and says the set moved.
    """
    q = np.zeros(1, dtype=np.float32)
    X = _one_d([1.0, np.float32(1.0) - np.float32(2.0**-24), 2.5])
    exact = [exact_sq(q, X[j]) for j in range(3)]
    assert exact[1] < exact[0], "the corpus does not have the order this test needs"

    R = np.full(3, 5e-7)
    D = np.array([exact[0] / scale(BITS_SMALL) - 2e-7,
                  exact[1] / scale(BITS_SMALL) + 2e-7,
                  exact[2] / scale(BITS_SMALL)])
    assert np.all(np.abs(D - np.array([v / scale(BITS_SMALL) for v in exact])) <= R)
    assert topk_set(D, 1)[0] == 0  # the float answer

    e = escalate_row(q, X, D, R, 1)
    assert e.determined and e.float_set_differed
    assert e.indices.tolist() == [1] == exact_topk(q, X, 1).tolist()


def test_exact_tie_is_not_certified():
    """No arithmetic anywhere resolves it, and the code says so with its own reason."""
    q = np.zeros(1, dtype=np.float32)
    X = _one_d([1.0, -1.0, 5.0])
    D = np.array([1.0, 1.0, 25.0])
    R = np.array([0.2, 0.2, 0.0])
    e = escalate_row(q, X, D, R, 1)
    assert not e.determined
    assert e.reason == EXACT_TIE
    assert e.tie is not None and set(e.tie) == {0, 1}


def test_exact_tie_on_identical_rows():
    rng = np.random.default_rng(2)
    row = rng.standard_normal(6).astype(np.float32)
    X = np.vstack([row, row, rng.standard_normal((3, 6)).astype(np.float32) * 9])
    q = rng.standard_normal(6).astype(np.float32)
    D = ((X.astype(np.float64) - q.astype(np.float64)) ** 2).sum(1)
    R = np.full(5, 1e-6)
    e = escalate_row(q, X, D, R, 1)
    assert not e.determined and e.reason == EXACT_TIE


def test_escalation_never_contradicts_the_exact_topk():
    """Every closed escalation is re-decided in exact integers; 0 contradictions is the bar.

    The count of refusals that exact arithmetic shows were fine is the pessimism number,
    and it is printed rather than hidden: a refusal is not a finding.
    """
    refused = closed = fine = 0
    for t in range(120):
        rng = np.random.default_rng(500 + t)
        X = (rng.standard_normal((18, 5)) * rng.choice([1.0, 1e3])).astype(np.float32)
        q = rng.standard_normal(5).astype(np.float32)
        x64, q64 = X.astype(np.float64), q.astype(np.float64)
        D = (q64**2).sum() + (x64**2).sum(1) - 2 * (x64 @ q64)
        # radii scaled to the row's own spread, so the corpus refuses often enough for
        # the assertion below to mean something
        R = np.full(18, 0.01 * float(D.max() - D.min()))
        k = 3
        if topk_determined(D, R, k) is None:
            continue
        refused += 1
        e = escalate_row(q, X, D, R, k)
        if not e.determined:
            continue
        closed += 1
        truth = sorted(exact_topk(q, X, k).tolist())
        assert sorted(e.indices.tolist()) == truth, t
        fine += int(not e.float_set_differed)
    assert refused > 10, f"only {refused} refusals; the corpus is too easy to score against"
    assert closed == refused, f"{refused - closed} refusals did not close inside the budget"
    # the pessimism triple, reported rather than asserted away
    print(f"refused {refused}  closed {closed}  confirmed fine {fine}")


def test_escalation_budget_is_a_typed_outcome_not_a_hang():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 4)).astype(np.float32)
    q = rng.standard_normal(4).astype(np.float32)
    D = ((X.astype(np.float64) - q.astype(np.float64)) ** 2).sum(1)
    R = np.full(60, 1e3)  # every enclosure straddles every other
    e = escalate_row(q, X, D, R, 5, max_escalations=4)
    assert isinstance(e, Escalation)
    assert not e.determined and e.reason == "ESCALATION_BUDGET"
    assert e.n_escalated <= 4


def test_escalate_over_rows_matches_the_single_row_call():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((25, 4)).astype(np.float32)
    Q = rng.standard_normal((3, 4)).astype(np.float32)
    D = ((Q[:, None, :].astype(np.float64) - X[None, :, :].astype(np.float64)) ** 2).sum(-1)
    R = np.full((3, 25), 5e-3)
    got = escalate(X, Q, D, R, 2)
    assert set(got) == {0, 1, 2}
    for i in range(3):
        one = escalate_row(Q[i], X, D[i], R[i], 2)
        assert got[i].determined == one.determined
        assert got[i].indices.tolist() == one.indices.tolist()


def test_largest_escalates_the_same_way():
    q = np.zeros(1, dtype=np.float32)
    X = _one_d([1.0, 1.0246950766, 1.0295630141, 0.1])
    D = np.array([exact_sq_float(q, X[j]) for j in range(4)])
    R = np.array([0.0, 0.1, 0.1, 0.0])
    e = escalate_row(q, X, D, R, 2, largest=True)
    assert e.determined
    assert sorted(e.indices.tolist()) == sorted(exact_topk(q, X, 2, largest=True).tolist())
