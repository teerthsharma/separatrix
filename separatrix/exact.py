"""The scaled-integer oracle, and the escalation rung it is shared with.

One implementation, two jobs, on purpose: the benchmark's ground truth cannot drift from
the escalation a user actually runs, and the control does not live inside the code it
controls -- nothing here calls `enclose`, and `enclose` never calls anything here.

Why integers and not `fractions.Fraction`
-----------------------------------------

`Fraction` runs a gcd on every ``__add__``, so one exact d=384 dot product is hundreds of
multi-hundred-bit gcds, and the mandatory soundness control would be quietly shrunk to a
sample.  Every float is exactly ``m * 2**e``, so scaling by a fixed power of two is exact
with one shift and no gcd:

    x  ->  int(x * 2**bits)          exact when 2**bits clears the smallest exponent

``bits = 149`` clears every float32 and float16 value (float32's smallest subnormal is
2**-149); ``bits = 1074`` clears every float64.  Correction against the obvious spelling:
``math.ldexp(v, 1074)`` OVERFLOWS for |v| > 2**-50 and silently returns inf, so the scaling
goes through ``float.as_integer_ratio()``, which is exact for every finite float and whose
denominator is always a power of two.

A squared distance scaled by ``2**bits`` per coordinate comes back scaled by
``2**(2*bits)``; ``SCALE(bits)`` is that divisor and two exact values are comparable only
when they carry the same ``bits``.

Escalation resolves the FRONTIER, not the named pair
----------------------------------------------------

Escalating only the pair a refusal names provably does not lift the refusal.  Verified:
``s=[1.0, 1.05, 1.06, 5.0]``, ``r=[0.1, 0.1, 0.1, 0.0]``, k=2 smallest names the pair
(1, 2); resolve exactly that pair and ``max_in = max(s0+r0, s1) = 1.10`` still exceeds
``min_out = 1.06``, because index 0's enclosure also straddles.  So:

    F = {i in T : D_i + R_i >= min_out}  union  {j not in T : D_j - R_j <= max_in}

Escalation has THREE outcomes, and the third is the most valuable thing this package can
produce:

  * the set closes                    -> determined, escalated
  * the exact scores are equal        -> EXACT_TIE; no arithmetic anywhere resolves it
  * the exact set differs from float  -> determined, with the CORRECTED indices, and
                                         `float_set_differed`.  Returning the float set
                                         here would contradict the guarantee, which
                                         promises the set exact arithmetic returns.

Self-check:  python -m separatrix.exact
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decide import topk_set
from .verdict import EXACT_TIE, Frontier

# float32's smallest subnormal is 2**-149; float64's is 2**-1074.
BITS_SMALL = 149
BITS_DOUBLE = 1074


def bits_for(dtype) -> int:
    """The scaling exponent that makes every value of ``dtype`` an exact integer."""
    return BITS_DOUBLE if np.dtype(dtype) == np.float64 else BITS_SMALL


def scale(bits: int) -> int:
    """The divisor a squared distance at ``bits`` carries: 2**(2*bits)."""
    return 1 << (2 * bits)


def _as_int(v: float, bits: int) -> int:
    """int(v * 2**bits), exact.  Raises when ``bits`` cannot clear v's exponent."""
    num, den = float(v).as_integer_ratio()
    e = den.bit_length() - 1
    if e > bits:
        raise ValueError(
            f"bits={bits} cannot represent {v!r} exactly; it needs {e}. "
            f"Pass bits=exact.bits_for(np.float64)."
        )
    return num << (bits - e)


def _from_float(v: float, bits: int, up: bool) -> int:
    """The outward integer image of a float at scale 2**bits: floor down, ceil up.

    Monotone in v, which is what lets a mixed exact/enclosed comparison find its extremes
    with numpy over floats and convert only the two winners.
    """
    num, den = float(v).as_integer_ratio()
    e = den.bit_length() - 1
    if bits >= e:
        return num << (bits - e)
    q, r = divmod(num, 1 << (e - bits))  # divmod floors, which is the `down` direction
    return q + 1 if (r and up) else q


def _to_float(scaled: int, bits: int) -> float:
    """The scaled integer back as a float. Correctly rounded; used only to pre-filter."""
    try:
        return scaled / scale(bits)
    except OverflowError:  # pragma: no cover - a score wider than float64's range
        return float("inf") if scaled > 0 else float("-inf")


def exact_sq(x, y, bits: int = BITS_SMALL) -> int:
    """||x - y||^2, exactly, as an integer scaled by ``2**(2*bits)``.

    Exact for the stored floats -- not for whatever produced them.  This is the ground
    truth every containment number in this package is scored against, and it is the only
    arithmetic here that no float touches.
    """
    ix = [_as_int(v, bits) for v in np.asarray(x).ravel()]
    iy = [_as_int(v, bits) for v in np.asarray(y).ravel()]
    if len(ix) != len(iy):
        raise ValueError(f"x has d={len(ix)}, y has d={len(iy)}")
    return sum((a - b) * (a - b) for a, b in zip(ix, iy))


def exact_sq_float(x, y, bits: int = BITS_SMALL) -> float:
    """``exact_sq`` divided back to a float. Correctly rounded; use only for printing."""
    return exact_sq(x, y, bits) / scale(bits)


def exact_topk(q, X, k: int, largest: bool = False, bits: int = BITS_SMALL) -> np.ndarray:
    """The top-k of one query against X, decided entirely in exact integers.

    O(n*d) big-integer operations.  The control, not the product: at n=5,000 d=384 this is
    minutes, which is exactly why the enclosure exists.
    """
    vals = [exact_sq(q, X[j], bits) for j in range(len(X))]
    order = sorted(range(len(vals)), key=lambda j: (-vals[j], j) if largest else (vals[j], j))
    return np.array(order[:k], dtype=np.int64)


# -- escalation ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Escalation:
    """What resolving the frontier exactly did to one row's decision."""

    determined: bool
    reason: str  # "" when determined; EXACT_TIE or ESCALATION_BUDGET otherwise
    indices: np.ndarray  # the top-k set, corrected if exact arithmetic moved it
    float_set_differed: bool
    n_escalated: int
    frontier: Frontier | None
    tie: tuple[int, int] | None = None


def _mixed_extremes(D, R, T_mask, known, bits):
    """(max_in, in_idx, min_out, out_idx) as integers at scale 2**(2*bits).

    Exact members contribute their exact integer; the rest contribute the outward integer
    image of their float bound.  Both maps are monotone, so numpy finds the float-side
    extreme and only two values are ever converted -- the whole point of doing it this way
    rather than lifting all n indices into big integers on every iteration.
    """
    S = 2 * bits
    hi = D + R
    lo = D - R
    best = []
    for mask, arr, up, better in (
        (T_mask, hi, True, max),
        (~T_mask, lo, False, min),
    ):
        idx = np.flatnonzero(mask)
        cands: list[tuple[int, int]] = []
        soft = np.array([j for j in idx if int(j) not in known], dtype=np.intp)
        if soft.size:
            j = int(soft[np.argmax(arr[soft])] if up else soft[np.argmin(arr[soft])])
            cands.append((_from_float(float(arr[j]), S, up), j))
        hard = [int(j) for j in idx if int(j) in known]
        if hard:
            j = (max if up else min)(hard, key=lambda t: known[t])
            cands.append((known[j], j))
        best.append(better(cands, key=lambda t: t[0]))
    (max_in, in_idx), (min_out, out_idx) = best
    return max_in, in_idx, min_out, out_idx


def escalate_row(
    q,
    X,
    D,
    R,
    k: int,
    *,
    largest: bool = False,
    max_escalations: int = 64,
    bits: int | None = None,
    row: int = 0,
) -> Escalation:
    """Rung 4: resolve the frontier exactly until the set closes, ties, or runs out.

    ``D`` and ``R`` are one row of the enclosure.  ``row`` is that row's index in the
    caller's query block and rides on the returned ``Frontier``: escalation happens one
    row at a time, so this function is the only place that index can come from, and a
    frontier that reports the wrong row is the reporting half of the soundness bug
    RESULTS 1.1(a) records -- a consumer reconstructing the refused set from
    ``v.frontiers`` gets the wrong rows.  Soundness does not depend on how the
    candidate set T is chosen at any iteration -- the max-in/min-out comparison over exact
    integers and outward float images is the proof, and a badly ordered T can only fail to
    close, never certify wrongly.
    """
    D = np.asarray(D, dtype=np.float64).ravel()
    R = np.broadcast_to(np.asarray(R, dtype=np.float64), D.shape).astype(np.float64)
    if bits is None:
        bits = bits_for(np.asarray(X).dtype)
    sgn = -1.0 if largest else 1.0
    n = D.shape[0]
    float_T = set(int(i) for i in topk_set(D, k, largest=largest))

    known: dict[int, int] = {}  # index -> exact scaled integer, sign-flipped for largest
    proxy = D.copy()
    frontier: Frontier | None = None

    for _ in range(max_escalations + 2):
        T = topk_set(proxy, k, largest=largest)
        mask = np.zeros(n, dtype=bool)
        mask[T] = True
        signed_known = {i: (int(-v) if largest else int(v)) for i, v in known.items()}
        max_in, in_idx, min_out, out_idx = _mixed_extremes(
            sgn * D, R, mask, signed_known, bits
        )
        frontier = Frontier(
            row=int(row),
            inside=in_idx,
            outside=out_idx,
            inside_lo=float(D[in_idx] - R[in_idx]),
            inside_hi=float(D[in_idx] + R[in_idx]),
            outside_lo=float(D[out_idx] - R[out_idx]),
            outside_hi=float(D[out_idx] + R[out_idx]),
            gap=float(sgn * (D[out_idx] - D[in_idx])),
            width=float(R[in_idx] + R[out_idx]),
        )
        if max_in < min_out:
            idx = np.array(sorted(int(i) for i in T), dtype=np.int64)
            return Escalation(
                determined=True,
                reason="",
                indices=idx,
                float_set_differed=set(idx.tolist()) != float_T,
                n_escalated=len(known),
                frontier=frontier,
            )
        if in_idx in known and out_idx in known and known[in_idx] == known[out_idx]:
            return Escalation(
                determined=False,
                reason=EXACT_TIE,
                indices=np.array(sorted(int(i) for i in T), dtype=np.int64),
                float_set_differed=False,
                n_escalated=len(known),
                frontier=frontier,
                tie=(in_idx, out_idx),
            )

        # The frontier, in floats with an outward cushion.  _from_float is monotone, so
        # comparing the float bounds against the float images of max_in / min_out -- each
        # nudged one ulp the permissive way -- can only ADD indices to the frontier, never
        # drop one.  An extra index costs one exact dot product; a dropped one would cost
        # the certificate.
        hi = sgn * D + R
        lo = sgn * D - R
        min_out_f = np.nextafter(_to_float(min_out, bits), -np.inf)
        max_in_f = np.nextafter(_to_float(max_in, bits), np.inf)
        want = np.flatnonzero((mask & (hi >= min_out_f)) | (~mask & (lo <= max_in_f)))
        new = [int(i) for i in want if int(i) not in known]
        if not new:
            break
        if len(known) + len(new) > max_escalations:
            break
        for i in new:
            known[i] = exact_sq(q, X[i], bits)
            proxy[i] = known[i] / scale(bits)

    return Escalation(
        determined=False,
        reason="ESCALATION_BUDGET",
        indices=np.array(sorted(int(i) for i in topk_set(proxy, k, largest=largest)),
                         dtype=np.int64),
        float_set_differed=False,
        n_escalated=len(known),
        frontier=frontier,
    )


def escalate(X, Q, D, R, k: int, rows=None, **kw) -> dict[int, Escalation]:
    """``escalate_row`` over the named rows.  A loop; the exact arithmetic is the cost."""
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    R = np.asarray(R, dtype=np.float64)
    rows = range(D.shape[0]) if rows is None else rows
    out = {}
    for i in rows:
        r = R if R.ndim == 0 else R[i]
        out[int(i)] = escalate_row(Q[i], X, D[i], r, k, row=int(i), **kw)
    return out


def _demo() -> None:
    # the oracle against a closed form: integer coordinates make the answer known before
    # any float runs
    rng = np.random.default_rng(0)
    A = rng.integers(-2048, 2048, size=(200, 12)).astype(np.float32)
    B = rng.integers(-2048, 2048, size=(200, 12)).astype(np.float32)
    want = ((A.astype(np.int64) - B.astype(np.int64)) ** 2).sum(axis=1)
    for i in range(200):
        assert exact_sq(A[i], B[i]) == int(want[i]) * scale(BITS_SMALL), i

    # float64 needs the wide scaling, and the ldexp spelling would have overflowed here
    x = np.array([1e6, 0.0])
    y = np.array([1e6 + 1e-6, 0.0])
    b = bits_for(np.float64)
    assert abs(exact_sq(x, y, b) / scale(b) - 1.0000152290447206e-12) < 1e-24
    # ... and the Gram identity, in float64, returns exactly 0.0 for that same pair
    assert x[0] ** 2 + y[0] ** 2 - 2 * x[0] * y[0] == 0.0

    # escalating the NAMED PAIR does not lift the refusal; the frontier does
    Xs = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    D = np.array([1.0, 1.05, 1.06, 5.0])
    R = np.array([0.1, 0.1, 0.1, 0.0])
    # the pair a refusal names is (1, 2); resolving only it leaves max_in = 1.10 > 1.06
    assert max(D[0] + R[0], D[1]) > D[2]
    # and the real thing, on a corpus where the exact values are the scores themselves
    q = np.array([0.0], dtype=np.float32)
    Xr = np.array([[1.0], [1.02469508], [1.02956301], [2.23606798]], dtype=np.float32)
    Dr = np.array([exact_sq_float(q, Xr[j]) for j in range(4)])
    Rr = np.array([0.1, 0.1, 0.1, 0.0])
    e = escalate_row(q, Xr, Dr, Rr, 2)
    assert e.determined and e.n_escalated >= 3, e
    assert sorted(e.indices.tolist()) == [0, 1]

    # an exact tie is not certified at any precision, and says so
    Xt = np.array([[1.0], [-1.0], [5.0]], dtype=np.float32)
    Dt = np.array([1.0, 1.0, 25.0])
    et = escalate_row(np.array([0.0], dtype=np.float32), Xt, Dt, np.array([0.2, 0.2, 0.0]), 1)
    assert not et.determined and et.reason == EXACT_TIE and et.tie is not None

    # escalation never contradicts the exact top-k, and reports it when the float set moved
    moved = 0
    for t in range(60):
        r = np.random.default_rng(100 + t)
        Xf = r.standard_normal((14, 6)).astype(np.float32)
        qf = r.standard_normal(6).astype(np.float32)
        Df = ((qf.astype(np.float64) ** 2).sum() + (Xf.astype(np.float64) ** 2).sum(1)
              - 2 * (Xf.astype(np.float64) @ qf.astype(np.float64)))
        Rf = np.full(14, 3e-3)
        es = escalate_row(qf, Xf, Df, Rf, 3)
        if es.determined:
            assert sorted(es.indices.tolist()) == sorted(exact_topk(qf, Xf, 3).tolist()), t
            moved += int(es.float_set_differed)
    print(f"exact: ok  (float set moved on {moved} of 60 escalated rows)")


if __name__ == "__main__":
    _demo()
