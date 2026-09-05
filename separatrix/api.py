"""The three decisions, and the gate re-exported so ``separatrix.gate`` resolves.

This file owns no arithmetic.  It calls ``enclose.enclose_scores`` for (D, R), asks
``decide.topk_determined`` the one question there is to ask, optionally hands the refused
rows to ``exact.escalate_row``, and turns the result into a ``Verdict``.  Every number it
prints came off one of those three modules, so a bug here is a presentation bug and never
an unsound one -- the separation ``decide.py``'s docstring insists on, kept at the surface.

**Nothing here raises for an arithmetic outcome.**  A precondition ``Refusal`` raised by
the enclosure is caught and returned as a REFUSED ``Verdict``, because the measured refusal
rates make a raising default a library that is uninstalled the same afternoon.  A *usage*
error -- a bare score array, k out of range, mixed dtypes -- stays an exception, exit class
3, and is deliberately not in the refusal catalogue: a Verdict is a statement about the
caller's data, an exception is a statement about the caller's code, and an agent reading
the JSON must be able to tell "go re-observe" from "go fix your call".

Why the reason on a gram refusal is sometimes GRAM_CANCELLATION
---------------------------------------------------------------

The direct kernel's enclosure is *relative* -- every term of ``sum_l (x_l - y_l)^2`` is
non-negative, so there is no cancellation term -- while the Gram identity's is absolute and
is dominated by the cross term.  So a pair the Gram identity cannot separate is often one
the direct kernel separates easily, and that refusal names a **code change** rather than a
re-observation.  Deciding it for a refused row costs two direct dot products over the
frontier pair, O(d) per refused row and no second matmul, so it is measured rather than
predicted.  It scores the frontier *pair* only, which is why it sets the reason and never a
certificate: certifying needs max-in/min-out over all n, and that is
``decide.topk_determined``'s job.

Self-check:  python -m separatrix.api
"""

from __future__ import annotations

import numpy as np

from . import decide, enclose
from .exact import escalate_row
from .harness import gate, report  # noqa: F401  -- separatrix.gate resolves through here
from .verdict import (
    BOUNDARY_UNDETERMINED,
    CERTIFIED,
    CERTIFIED_UPCAST,
    ESCALATION_BUDGET,
    EXACT_TIE,
    GRAM_CANCELLATION,
    NEXT_ACTION,
    NOT_CERTIFIED,
    REFUSED,
    Frontier,
    Refusal,
    Verdict,
)

__all__ = ["certified_topk", "certified_argmin", "certified_threshold", "gate"]

# EVERY refused row's frontier rides on the Verdict.  An earlier cap of 64 was measured
# to be a bug at the boundary rather than a saving: with 69 refused rows out of 200, the
# 5 rows past the cap read as certified to any consumer that reconstructs the refused set
# from `v.frontiers`, and two evaluations of the same formula then "disagreed on a
# certified row" -- a soundness violation that was not one.  The CLI caps what it *prints*
# with --max-report; a Verdict caps nothing.

# upcast=True widens by exactly one step, so the advice printed on RANGE_UNSAFE ("certify
# the float32 ranking instead") is the thing that actually runs.  float64 has nothing
# wider, and asking for it is a usage error rather than a silent no-op returning exit 4.
WIDER = {"float16": np.float32, "float32": np.float64}


def _refusal_verdict(e: Refusal, **kw) -> Verdict:
    """A precondition Refusal as the Verdict the caller receives.  One conversion."""
    detail = str(e)
    if ": " in detail:
        detail = detail.split(": ", 1)[1]
    return Verdict(
        status=NOT_CERTIFIED if e.exit_code == 1 else REFUSED,
        reason=e.reason,
        detail=detail,
        next_action=e.next_action,
        **kw,
    )


def _direct_separates(q, X, frontier: Frontier, d: int, work_dtype) -> bool:
    """Does the direct kernel separate this frontier pair where the Gram identity did not?

    Two squared-difference sums and two relative radii, in the working dtype the caller
    ran, built the same way ``enclose.direct_radii`` builds them.  This is a measurement of
    the pair, not of the row: it sets a reason code, never a certificate.
    """
    dt = np.dtype(work_dtype)
    g = enclose.gamma(d + 1, dt)
    rel = g / (1.0 - g)
    box = []
    for j in (frontier.inside, frontier.outside):
        diff = (np.asarray(X[j], dtype=dt) - np.asarray(q, dtype=dt)).astype(dt)
        s = float(np.sum((diff * diff).astype(dt), dtype=dt))
        r = float(enclose._inflate(np.array([rel * s + enclose.eta(d, dt)]), d, dt)[0])
        box.append((max(s - r, 0.0), s + r))
    (a_lo, a_hi), (b_lo, b_hi) = box
    return a_hi < b_lo or b_hi < a_lo


def certified_topk(
    X,
    Q,
    k: int = 10,
    *,
    largest: bool = False,
    kernel: str = "gram",
    bound: str = "cheap",
    per_pair: bool = False,
    ordered: bool = False,
    escalate: bool = False,
    max_escalations: int = 64,
    upcast: bool = False,
    chunk: int | None = None,
) -> tuple[np.ndarray, Verdict]:
    """``(idx (m, k) int64, Verdict)``.  CERTIFIED only when every row's set is determined.

    ``X`` is the corpus (n, d), ``Q`` the queries (m, d), both float and both the same
    dtype -- the certificate is about one working dtype, so it is picked explicitly rather
    than left to numpy's promotion rules.  Scores are **squared** euclidean distances under
    both kernels; ``largest=True`` negates and reuses the one rule.

    ``chunk`` bounds peak memory: the scores are an (m, n) float64 array, which is 8 GB
    for 10,000 queries against a 1M-row corpus, and ``chunk=100`` computes them 100 query
    rows at a time for 80 MB instead.  Until this build the argument was validated and
    then ignored, so a caller who passed it got the 8 GB allocation anyway.

    **Chunking is a tenth engine, not a no-op.**  BLAS picks a different gemm path for a
    1-row right-hand side than for a 17-row one, so ``Q @ X.T`` chunked is a numerically
    distinct evaluation of the same formula on the same stored bytes -- measured here, one
    row of 17 moves between ``chunk=1`` and the unchunked call on the near-duplicate
    corpus, and that row was REFUSED under both.  Every CERTIFIED row is identical across
    chunkings, which is the guarantee and not a coincidence:
    ``test_chunk_is_a_tenth_engine`` asserts both halves.

    The indices returned are the float top-k set, corrected to the exact set on any row
    escalation moved -- returning the float set there would contradict the guarantee, which
    promises the set exact arithmetic returns.  A REFUSED verdict still returns indices:
    they are what the caller's own ``argpartition`` would have produced, and the verdict is
    the statement that this enclosure did not determine them.
    """
    from .corpus import as_points  # one input door, and it names both producers

    X = as_points(X, "X")
    Q = as_points(Q, "Q")
    if X.dtype != Q.dtype:
        raise TypeError(
            f"X is {X.dtype} and Q is {Q.dtype}; the certificate is about one working "
            f"dtype, so pick it explicitly rather than letting numpy promote"
        )
    n, d = X.shape
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    k = int(k)
    if k <= 0 or k >= n:
        raise ValueError(f"k must satisfy 0 < k < n = {n}, got {k}")
    if chunk is not None and chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")

    work = None
    status_ok = CERTIFIED
    if upcast:
        wider = WIDER.get(X.dtype.name)
        if wider is None:
            raise ValueError(
                f"upcast=True has nothing to widen {X.dtype.name} to; drop the flag and "
                f"certify the arithmetic your pipeline actually runs"
            )
        work = wider
        status_ok = CERTIFIED_UPCAST

    common = dict(
        kernel=kernel,
        bound=bound,
        per_pair=per_pair,
        k=k,
        largest=largest,
        n_queries=int(Q.shape[0]),
        dtype_in=X.dtype.name,
        dtype_used=np.dtype(work or X.dtype).name,
    )

    m = int(Q.shape[0])
    step = m if chunk is None else min(chunk, m)
    idx = np.empty((m, k), dtype=np.int64)
    frontiers: list[Frontier] = []
    refused_rows: list[int] = []
    n_escalated = 0
    float_set_differed = False
    tied = budgeted = 0
    enc = None

    for b0 in range(0, m, step):
        b1 = min(b0 + step, m)
        try:
            enc = enclose.enclose_scores(
                X, Q[b0:b1], kernel=kernel, bound=bound, per_pair=per_pair, work_dtype=work
            )
        except Refusal as e:
            # No score was read on this block, and a precondition that fails on one block
            # of queries is a statement about the call: there is no ranking to hand back
            # beside the refusal, and a partial one would read as a certificate.
            v = _refusal_verdict(e, n_refused=common["n_queries"], **common)
            report(v)
            return np.empty((0, k), dtype=np.int64), v

        if b0 == 0:
            common.update(
                kernel=enc.kernel,
                bound=enc.bound,
                per_pair=enc.per_pair,
                dtype_used=enc.dtype_used,
                canary=enc.canary,
                accum_assumed=enc.accum_assumed,
            )

        for j in range(enc.D.shape[0]):
            i = b0 + j
            r = enc.R if enc.R.ndim == 0 else enc.R[j]
            f = decide.topk_determined(enc.D[j], r, k, largest, ordered=ordered, row=i)
            idx[i] = np.sort(decide.topk_set(enc.D[j], k, largest=largest))
            if f is None:
                continue
            if not escalate:
                frontiers.append(f)
                refused_rows.append(i)
                continue
            e = escalate_row(
                Q[i],
                X,
                enc.D[j],
                r,
                k,
                largest=largest,
                max_escalations=max_escalations,
                row=i,
            )
            n_escalated += e.n_escalated
            idx[i] = e.indices
            if e.determined:
                float_set_differed |= bool(e.float_set_differed)
                continue
            refused_rows.append(i)
            frontiers.append(e.frontier if e.frontier is not None else f)
            if e.reason == EXACT_TIE:
                tied += 1
            else:
                budgeted += 1

    # The last block's scores are (m_block, n) floats and nothing below needs them.
    dtype_used, kernel_used = common["dtype_used"], common["kernel"]
    enc = None

    if not refused_rows:
        v = Verdict(
            status=status_ok,
            n_refused=0,
            escalated=bool(escalate),
            n_escalated=n_escalated,
            float_set_differed=float_set_differed,
            **common,
        )
        report(v)
        return idx, v

    # Which refusal this is.  GRAM_CANCELLATION is claimed only when the direct kernel
    # separates EVERY refused row's frontier pair; the count prints either way, so the
    # partial case is an exact integer rather than a softened word.
    n_direct = 0
    if kernel_used == "gram" and not budgeted and not tied:
        for i, f in zip(refused_rows, frontiers):
            if _direct_separates(Q[i], X, f, d, dtype_used):
                n_direct += 1

    if budgeted:
        reason = ESCALATION_BUDGET
    elif tied:
        reason = EXACT_TIE
    elif n_direct and n_direct == len(refused_rows):
        reason = GRAM_CANCELLATION
    else:
        reason = BOUNDARY_UNDETERMINED

    detail = (
        f"{len(refused_rows)} of {m} rows have a rank-{k} boundary this enclosure "
        f"does not decide"
    )
    if kernel_used == "gram" and n_direct:
        detail += (
            f"; the direct kernel separates {n_direct} of those {len(refused_rows)} "
            f"frontier pairs"
        )
    if tied:
        detail += f"; {tied} of them are exact ties"
    if budgeted:
        detail += f"; {budgeted} of them exceeded max_escalations={max_escalations}"

    v = Verdict(
        status=NOT_CERTIFIED if reason == EXACT_TIE else REFUSED,
        reason=reason,
        detail=detail,
        next_action=NEXT_ACTION[reason],
        n_refused=len(refused_rows),
        escalated=bool(escalate),
        n_escalated=n_escalated,
        float_set_differed=float_set_differed,
        frontiers=tuple(frontiers),
        **common,
    )
    report(v)
    return idx, v


def certified_argmin(X, Q, **kw) -> tuple[np.ndarray, Verdict]:
    """``certified_topk(k=1, largest=False)`` with the axis squeezed.

    There is no second rule here, so there is no second rule to get wrong.
    """
    for fixed in ("k", "largest", "ordered"):
        if fixed in kw:
            raise TypeError(
                f"certified_argmin does not take {fixed}=; it is certified_topk(k=1, "
                f"largest=False), so call that directly if you want another shape"
            )
    idx, v = certified_topk(X, Q, k=1, largest=False, **kw)
    return idx.reshape(-1), v


def certified_threshold(D, R, t, *, units: str = "score") -> np.ndarray:
    """An int8 trit array: ``+1`` above ``t``, ``-1`` below, **``0`` undetermined**.

    Not a bool mask.  ``X[mask]`` silently consumes every undetermined element as False,
    and "the undetermined indices are a separate return value" is a convention a caller
    drops by writing one idiomatic line of numpy; a trit cannot be dropped without noticing.

    ``t`` is in **score units** -- squared, if the scores are squared -- so the threshold
    itself carries no rounding and ``r_t = 0``.  ``units="distance"`` against squared scores
    is a ``ValueError`` rather than a silent wrong answer at every boundary.
    """
    if units == "distance":
        raise ValueError(
            "t was given in distance units against squared scores; pass t**2, or square "
            "the threshold at the call site so the units are visible where it is written"
        )
    if units != "score":
        raise ValueError(f"units must be 'score' or 'distance', got {units!r}")
    D = np.asarray(D, dtype=np.float64)
    R = np.broadcast_to(np.asarray(R, dtype=np.float64), D.shape)
    if np.any(R < 0):
        raise ValueError("R is a radius and cannot be negative")
    t = float(t)
    if not np.isfinite(t):
        raise ValueError(f"t must be finite, got {t}")
    out = np.zeros(D.shape, dtype=np.int8)
    out[D - R > t] = 1
    out[D + R < t] = -1
    return out


def _demo() -> None:
    from .exact import exact_topk

    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 16)).astype(np.float32)
    Q = X[:5] + np.float32(1e-3) * rng.normal(size=(5, 16)).astype(np.float32)

    idx, v = certified_topk(X, Q, k=5)
    assert idx.shape == (5, 5) and v.n_queries == 5
    assert v.exit_code in (0, 1, 2) and v.accum_assumed

    # a certified set is the exact set: score it against the oracle, not against itself
    if v.certified:
        for i in range(5):
            assert set(idx[i].tolist()) == set(exact_topk(Q[i], X, 5).tolist())

    # argmin is topk(k=1) with the axis squeezed, and takes no second rule
    a, va = certified_argmin(X, Q)
    assert a.shape == (5,) and va.k == 1
    try:
        certified_argmin(X, Q, k=3)
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("certified_argmin accepted k=")

    # usage errors are exceptions, never verdicts
    for bad in (
        lambda: certified_topk(np.zeros(7), Q),
        lambda: certified_topk(X, Q, k=0),
        lambda: certified_topk(X, Q, k=len(X)),
        lambda: certified_topk(X, Q.astype(np.float64)),
        lambda: certified_topk(X.astype(np.float64), Q.astype(np.float64), upcast=True),
        lambda: certified_threshold([1.0], [0.0], 1.0, units="distance"),
    ):
        try:
            bad()
        except (TypeError, ValueError):
            pass
        else:  # pragma: no cover
            raise AssertionError("a usage error did not raise")

    # the float16 range case: RANGE_UNSAFE before any score is read, upcast is exit 4
    big = (rng.random((40, 784)) * 255.0).astype(np.float16)
    _, vr = certified_topk(big, big[:3], k=10)
    assert vr.reason == "RANGE_UNSAFE" and vr.exit_code == 2, vr
    iu, vu = certified_topk(big, big[:3], k=10, upcast=True)
    assert vu.status == CERTIFIED_UPCAST and vu.exit_code == 4, vu
    assert vu.dtype_in == "float16" and vu.dtype_used == "float32"
    assert iu.shape == (3, 10)

    # frame 1 through the public surface: the gram refusal names the code change
    P = np.array([[1e6, 0.0], [1e6 + 1e-6, 0.0], [0.0, 0.0]], dtype=np.float64)
    _, vg = certified_topk(P, P[:1], k=1)
    assert vg.reason == GRAM_CANCELLATION, vg
    _, vd = certified_topk(P, P[:1], k=1, kernel="direct")
    assert vd.status == CERTIFIED, vd

    # trits: undetermined is 0 and cannot be dropped by an idiomatic boolean index
    tr = certified_threshold([1.0, 5.0, 3.0], [0.1, 0.1, 2.0], 3.0)
    assert tr.tolist() == [-1, 1, 0]

    print("api: ok")


if __name__ == "__main__":
    _demo()
