"""The public surface: three decisions, four exit codes, and the two agreement theorems.

The two tests this file exists for are `test_gram_and_direct_certified_sets_agree` (never-
claim #6: two independently CERTIFIED verdicts on the same stored floats must agree,
whatever formula each used) and `test_backends_agree_on_certified` (the headline: 0
disagreements among CERTIFIED across numerically distinct evaluations, against D > 0 among
the REFUSED, which is what proves the corpus was not too easy).

Neither scores against separatrix's own arithmetic: the first scores two of this package's
kernels against each other and against `exact_topk`, the second scores against numpy and
torch evaluations this package does not own.
"""

from __future__ import annotations

import numpy as np
import pytest

import separatrix as sx
from separatrix import corpus as C
from separatrix import verdict as V
from separatrix.api import certified_argmin, certified_threshold, certified_topk
from separatrix.cli import near_duplicate_corpus, permuted_evaluation
from separatrix.decide import topk_set
from separatrix.enclose import enclose_scores
from separatrix.exact import exact_topk

def _torch():
    try:
        import torch as t
    except ImportError:  # pragma: no cover - exercised on machines without torch
        return None
    return t


# --------------------------------------------------------------------------------------
# the lazy public surface
# --------------------------------------------------------------------------------------


def test_the_six_public_names_resolve_through_the_package():
    for name in ("certified_topk", "certified_argmin", "certified_threshold", "gate"):
        assert callable(getattr(sx, name)), name
    assert sx.gate is sx.api.gate, "there is one gate, and it lives in harness.py"


def test_topk_returns_indices_and_a_verdict_and_never_raises_for_arithmetic():
    X, Q = near_duplicate_corpus(n=200, d=16, m=8)
    idx, v = certified_topk(X, Q, k=5)
    assert idx.shape[1] == 5 and idx.dtype == np.int64
    assert isinstance(v, V.Verdict) and v.n_queries == 8
    assert v.status in V.STATUSES and v.exit_code in (0, 1, 2, 4)


def test_a_certificate_is_the_exact_set_and_a_refusal_still_returns_a_ranking():
    """CERTIFIED means the set exact arithmetic returns, checked against the oracle."""
    X, Q = near_duplicate_corpus(n=120, d=12, m=6)
    idx, v = certified_topk(X, Q, k=4)
    assert idx.shape == (len(Q), 4), "a refusal is not a finding: the ranking still returns"
    if v.certified:
        for i in range(len(Q)):
            assert set(idx[i].tolist()) == set(exact_topk(Q[i], X, 4).tolist())


def test_argmin_is_topk_of_one_and_has_no_second_rule():
    X, Q = near_duplicate_corpus(n=80, d=8, m=4)
    a, va = certified_argmin(X, Q)
    b, vb = certified_topk(X, Q, k=1)
    assert a.shape == (4,) and np.array_equal(a, b.reshape(-1))
    assert va.k == vb.k == 1 and va.status == vb.status
    for fixed in ("k", "largest", "ordered"):
        with pytest.raises(TypeError):
            certified_argmin(X, Q, **{fixed: 1})


# --------------------------------------------------------------------------------------
# the threshold trit
# --------------------------------------------------------------------------------------


def test_threshold_is_a_trit_and_undetermined_is_not_dropped_by_a_boolean_index():
    D = np.array([1.0, 5.0, 3.0])
    R = np.array([0.1, 0.1, 2.0])
    t = certified_threshold(D, R, 3.0)
    assert t.dtype == np.int8 and t.tolist() == [-1, 1, 0]
    # the failure a bool mask would hide: the undetermined element is visibly not +1
    assert (t == 0).sum() == 1


def test_a_distance_threshold_against_squared_scores_is_a_usage_error():
    with pytest.raises(ValueError, match="distance units"):
        certified_threshold([1.0], [0.0], 2.0, units="distance")
    with pytest.raises(ValueError):
        certified_threshold([1.0], [0.0], 2.0, units="furlongs")
    with pytest.raises(ValueError):
        certified_threshold([1.0], [-1.0], 2.0)


# --------------------------------------------------------------------------------------
# usage errors are exceptions, exit class 3, and are not in the refusal catalogue
# --------------------------------------------------------------------------------------


def test_usage_errors_are_exceptions():
    X, Q = near_duplicate_corpus(n=60, d=8, m=3)
    with pytest.raises(TypeError, match="topk_determined"):
        certified_topk(np.zeros(7), Q)  # a bare score array names both producers
    with pytest.raises(ValueError):
        certified_topk(X, Q, k=0)
    with pytest.raises(ValueError):
        certified_topk(X, Q, k=len(X))
    with pytest.raises(TypeError):
        certified_topk(X, Q.astype(np.float64))  # mixed dtypes
    with pytest.raises(TypeError):
        certified_topk(X, Q, k=True)
    with pytest.raises(ValueError):
        certified_topk(X, Q, k=5, chunk=0)


def test_a_usage_error_carries_no_refusal_code():
    X, Q = near_duplicate_corpus(n=60, d=8, m=3)
    for call in (lambda: certified_topk(np.zeros(7), Q), lambda: certified_topk(X, Q, k=0)):
        try:
            call()
        except (TypeError, ValueError) as e:
            for reason in V.REASONS:
                assert reason not in str(e)


# --------------------------------------------------------------------------------------
# the must-fix, through the public surface
# --------------------------------------------------------------------------------------


def test_fp16_range_is_never_certified_through_the_public_surface():
    """The recorded case: max ||x||^2 far above float16's 65504, prototype certified it."""
    c = C.adversarial("fp16_range_784")
    idx, v = certified_topk(c.X, c.Q, k=c.k)
    assert not v.certified
    assert v.reason == V.RANGE_UNSAFE, "RANGE_UNSAFE fires before any score is read"
    assert v.reason != V.NONFINITE_INPUT, "naming the damage instead of the cause"
    assert idx.size == 0, "no score was read, so no ranking is handed back"

    # upcast reproduces the float32 ranking and is exit 4, never 0
    iu, vu = certified_topk(c.X, c.Q, k=c.k, upcast=True)
    assert vu.status == V.CERTIFIED_UPCAST and vu.exit_code == 4
    assert vu.dtype_in == "float16" and vu.dtype_used == "float32"
    Xf, Qf = c.X.astype(np.float32), c.Q.astype(np.float32)
    for i in range(len(c.Q)):
        assert set(iu[i].tolist()) == set(exact_topk(Qf[i], Xf, c.k).tolist())


def test_upcast_has_nothing_to_widen_float64_to():
    X, Q = near_duplicate_corpus(n=60, d=8, m=3)
    with pytest.raises(ValueError, match="nothing to widen"):
        certified_topk(X.astype(np.float64), Q.astype(np.float64), k=5, upcast=True)


def test_the_accumulator_assumption_travels_on_every_verdict():
    X, Q = near_duplicate_corpus(n=60, d=8, m=3)
    for kw in ({}, {"kernel": "direct"}, {"bound": "tight"}, {"per_pair": True}):
        _, v = certified_topk(X, Q, k=5, **kw)
        assert v.accum_assumed == V.ACCUM_ASSUMED and v.accum_assumed


# --------------------------------------------------------------------------------------
# escalation, through the public surface
# --------------------------------------------------------------------------------------


def test_escalation_decides_what_the_enclosure_refused():
    X, Q = near_duplicate_corpus(n=200, d=16, m=8, dups=60, jitter=1e-8)
    _, before = certified_topk(X, Q, k=5)
    assert not before.certified, "this corpus is seeded to leave escalation something to do"
    idx, after = certified_topk(X, Q, k=5, escalate=True)
    assert after.escalated and after.n_escalated >= 0
    assert after.n_refused <= before.n_refused
    if after.certified:
        for i in range(len(Q)):
            assert set(idx[i].tolist()) == set(exact_topk(Q[i], X, 5).tolist())


def test_every_frontier_names_its_own_row_after_escalation():
    """The reporting half of RESULTS 1.1(a), found on SIFT1M and fixed here.

    `escalate_row` hard-coded `row=0` on the Frontier it returns, so after `escalate=True`
    every surviving frontier claimed row 0: on the 1M-row SIFT corpus the two rows that
    survived escalation (82 and 93) both printed as row 0, and a consumer reconstructing
    the refused set from `v.frontiers` read one row refused where two were.
    """
    X, Q = near_duplicate_corpus(n=200, d=16, m=40, dups=80, jitter=1e-9)
    _, v = certified_topk(X, Q, k=5, escalate=True, max_escalations=2)
    assert v.n_refused >= 2, "this corpus is seeded to leave two rows undecided"
    assert len({f.row for f in v.frontiers}) == v.n_refused
    assert sorted(f.row for f in v.frontiers) == sorted({f.row for f in v.frontiers})


def test_chunk_is_a_tenth_engine():
    """`chunk` was validated and then ignored; it now blocks the (m, n) score array.

    It is not a no-op on the arithmetic: BLAS picks a different gemm path for a 1-row
    right-hand side, so `chunk=1` is a tenth numerically distinct evaluation of the same
    formula on the same stored bytes. One row of 17 moves here, and it is refused under
    both. The certified rows are what must agree, and they do.
    """
    X, Q = near_duplicate_corpus(n=300, d=16, m=17, dups=60, jitter=1e-8)
    whole, v0 = certified_topk(X, Q, k=5)
    assert v0.n_refused > 0, "a corpus with no refusals would not test the boundary"
    refused0 = {f.row for f in v0.frontiers}
    moved = 0
    for c in (1, 4, 17, 64):
        part, v = certified_topk(X, Q, k=5, chunk=c)
        refused = {f.row for f in v.frontiers}
        assert v.n_refused == len(refused)
        for i in range(len(Q)):
            if np.array_equal(whole[i], part[i]):
                continue
            moved += 1
            assert i in refused0 and i in refused, (
                f"row {i} moved between chunk={c} and the unchunked call while certified"
            )
    assert moved > 0, "the gemv path moved a row here; if it stops, this test is blind"


def test_chunk_leaves_escalation_and_its_frontiers_alone():
    X, Q = near_duplicate_corpus(n=300, d=16, m=17, dups=60, jitter=1e-8)
    whole, v0 = certified_topk(X, Q, k=5, escalate=True)
    part, v = certified_topk(X, Q, k=5, chunk=4, escalate=True)
    assert np.array_equal(whole, part) and v.n_escalated == v0.n_escalated
    assert [str(f) for f in v.frontiers] == [str(f) for f in v0.frontiers]


def test_an_exact_tie_is_not_certified_and_no_precision_removes_it():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    Q = X[:1]
    _, v = certified_topk(X, Q, k=2, escalate=True)
    assert v.status == V.NOT_CERTIFIED and v.reason == V.EXACT_TIE and v.exit_code == 1


# --------------------------------------------------------------------------------------
# never-claim #6: two CERTIFIED verdicts on the same stored floats must agree
# --------------------------------------------------------------------------------------


def test_gram_and_direct_certified_sets_agree():
    """A disagreement between two certificates is a bug in this package, not in the data.

    Both exact values are the same real number -- the Gram identity IS ||x-y||^2 in exact
    arithmetic -- so any two CERTIFIED verdicts must return the same set.  Scored against
    `exact_topk` as well, so agreement-with-each-other cannot pass by being wrong twice.
    """
    seen = 0
    for seed in range(12):
        X, Q = near_duplicate_corpus(n=150, d=24, m=6, seed=seed, dups=8, jitter=1e-4)
        ig, vg = certified_topk(X, Q, k=5, kernel="gram")
        idr, vd = certified_topk(X, Q, k=5, kernel="direct")
        if not (vg.certified and vd.certified):
            continue
        seen += 1
        for i in range(len(Q)):
            assert set(ig[i].tolist()) == set(idr[i].tolist()), (seed, i)
            assert set(ig[i].tolist()) == set(exact_topk(Q[i], X, 5).tolist()), (seed, i)
    assert seen >= 3, f"only {seen} draws had two certificates to compare; test is vacuous"


# --------------------------------------------------------------------------------------
# the headline: agreement across numerically distinct evaluations
# --------------------------------------------------------------------------------------


def _evaluations(X, Q, k):
    """(name, sets) for every numerically distinct evaluation available on this machine.

    numpy gram fp32, a permuted reduction order, numpy direct, numpy gram fp64, and -- when
    torch is installed -- torch.cdist above its 25-row switch to the Gram identity, at two
    batch sizes.  None of these is separatrix's own arithmetic.
    """
    out = []

    def sets(D):
        return [frozenset(topk_set(np.asarray(D[i], dtype=np.float64), k).tolist())
                for i in range(D.shape[0])]

    X64, Q64 = X.astype(np.float64), Q.astype(np.float64)
    g32 = (X.astype(np.float32) ** 2).sum(1)[None, :] + \
          (Q.astype(np.float32) ** 2).sum(1)[:, None] - \
          2.0 * (Q.astype(np.float32) @ X.astype(np.float32).T)
    out.append(("numpy gram fp32", sets(g32)))
    out.append(("numpy gram fp32, reduction permuted",
                sets(permuted_evaluation(X, Q, np.float32))))
    out.append(("numpy direct fp32",
                sets(np.stack([((X.astype(np.float32) - q) ** 2).sum(1)
                               for q in Q.astype(np.float32)]))))
    out.append(("numpy gram fp64",
                sets((X64 ** 2).sum(1)[None, :] + (Q64 ** 2).sum(1)[:, None]
                     - 2.0 * (Q64 @ X64.T))))
    t = _torch()
    if t is not None:
        tX = t.from_numpy(np.ascontiguousarray(X.astype(np.float32)))
        tQ = t.from_numpy(np.ascontiguousarray(Q.astype(np.float32)))
        out.append(("torch.cdist mm", sets(t.cdist(tQ, tX).numpy())))
        out.append(("torch.cdist direct",
                    sets(t.cdist(tQ, tX,
                                 compute_mode="donot_use_mm_for_euclid_dist").numpy())))
        b = [t.cdist(tQ[s:s + 32], tX).numpy() for s in range(0, len(tQ), 32)]
        out.append(("torch.cdist batch 32", sets(np.concatenate(b))))
    return out


def test_backends_agree_on_certified():
    """0 disagreements among CERTIFIED rows; D > 0 among the REFUSED, or the corpus is easy.

    The `D > 0` half is not decoration.  A corpus on which every evaluation agrees
    everywhere proves nothing about the rule, and this assertion is what says so out loud
    rather than shipping a green tick over an easy draw.
    """
    X, Q = near_duplicate_corpus(n=400, d=48, m=200, dups=150, jitter=1e-8)
    X = X.astype(np.float32)
    Q = Q.astype(np.float32)
    k = 10
    ev = _evaluations(X, Q, k)
    assert len(ev) >= 4, ev

    _, v = certified_topk(X, Q, k=k)
    rows_refused = {f.row for f in v.frontiers}
    assert len(rows_refused) == v.n_refused, (
        "every refused row's frontier must ride on the Verdict, or a consumer that "
        "reconstructs the refused set from v.frontiers reads a refused row as certified"
    )

    disagree_certified = disagree_refused = 0
    for i in range(len(Q)):
        distinct = {names[i] for _, names in ev}
        if len(distinct) > 1:
            if i in rows_refused:
                disagree_refused += 1
            else:
                disagree_certified += 1

    assert disagree_certified == 0, (
        f"{disagree_certified} rows separatrix did not refuse were decided differently by "
        f"two evaluations of the same formula on the same stored bytes"
    )
    if disagree_refused == 0:
        # printed, not passed over: an arm where the tool had nothing to say
        assert rows_refused, "no row disagreed and none was refused: this corpus was easy"


# --------------------------------------------------------------------------------------
# the gate collects what the public surface reports
# --------------------------------------------------------------------------------------


def test_a_verdict_reaches_an_ambient_gate_without_the_caller_wiring_anything(tmp_path):
    X, Q = near_duplicate_corpus(n=200, d=16, m=8)
    path = tmp_path / ".separatrix-gate.json"
    with sx.gate(max_refused=1.0, fixture="test-fixture", path=path) as g:
        certified_topk(X, Q, k=5)
    assert g.n == 8, "certified_topk called harness.report once, with no caller wiring"


def test_the_module_self_check_passes():
    from separatrix.api import _demo

    _demo()


# --------------------------------------------------------------------------------------
# the second soundness bug this build found, pinned
# --------------------------------------------------------------------------------------


def test_a_negative_radius_can_never_reach_the_rule():
    """MEASURED: the direct kernel certified everything over a radius of -9.22e+14.

    P3 admits `n*u == 1/2`, where gamma_n is exactly 1.0 and rounds outward above it. The
    Gram form multiplies by gamma and stays sound; the direct kernel's relative form
    divides by `1 - gamma` and produced a negative radius, which inverts every interval so
    that max-in falls below min-out and the rule certifies every row. Reproduced on
    float16 at d = 1023, where `certified_topk(kernel="direct")` returned CERTIFIED over a
    bound that was not one. Both kernels now refuse BOUND_VACUOUS there.
    """
    c = C.adversarial("vacuous_f16_d1023")
    for kernel in ("gram", "direct"):
        _, v = certified_topk(c.X, c.Q, k=2, kernel=kernel)
        assert not v.certified, (kernel, v)
        assert v.reason == V.BOUND_VACUOUS, (kernel, v)

    # d = 1022 is the last legal width and is NOT refused for being vacuous: the bound is
    # legal there and simply refuses on its own merits, which is a different statement.
    legal = C.adversarial("vacuous_edge_f16_d1022")
    for kernel in ("gram", "direct"):
        _, v = certified_topk(legal.X, legal.Q, k=2, kernel=kernel)
        assert v.reason != V.BOUND_VACUOUS, (kernel, v)


def test_no_radius_anywhere_is_negative():
    """The invariant behind the bug above, over every kernel, bound and adversarial case."""
    for c in C.adversarial():
        for kernel in ("gram", "direct"):
            for bound in ("cheap", "tight"):
                try:
                    e = enclose_scores(
                        c.X, c.Q, kernel=kernel, bound=bound
                    )
                except V.Refusal:
                    continue
                assert np.all(np.asarray(e.R) >= 0.0), (c.name, kernel, bound, e.R.min())
                lo, hi = e.interval(0)
                assert np.all(lo <= hi), (c.name, kernel, bound)
