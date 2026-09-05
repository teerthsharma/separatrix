"""The enclosure: unit roundoff, gamma, the preconditions, and the two kernels' radii.

This is the only file whose bugs are *unsound* rather than merely wrong.  Every claim in
it is a line of Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed., 2002),
chapter 3, and the line is named in the docstring that uses it.

  Thm 3.1   |fl(x'y) - x'y| <= gamma_d * sum_i |x_i||y_i|
            The absolute value is on the RIGHT-HAND SIDE, INSIDE THE SUM.  The form
            `2|<x,y>|` is not this bound and is below it exactly under cancellation,
            which is the regime this package exists for: measured 20.7x understatement
            at d=384 on two random unit-norm vectors.
  (3.1)     gamma_n = n*u / (1 - n*u),  u = eps/2.
  sec 2.7   the relative-error model fl(ab) = ab(1+delta) holds only for normal results;
            a product that lands subnormal carries ABSOLUTE error.  Hence `eta`, carried
            unconditionally.

The Gram identity `d^2 = ||x||^2 + ||y||^2 - 2<x,y>` -- the one `torch.cdist` switches to
above 25 rows -- is three such reductions plus two additions and an exact multiply by 2,
so the constant is gamma_{d+2} and the cross term is `2*sum_i|x_i y_i|`.

Two rungs, and Cauchy-Schwarz makes the cheap one free:

    R_cheap = gamma_{d+2} * (||x|| + ||y||)^2          no extra matmul
    R_tight = gamma_{d+2} * (||x||^2 + ||y||^2 + 2<|x|,|y|>)      one extra matmul

`R_cheap >= R_tight` pointwise, so the ladder cannot invert; `test_cheap_dominates_tight`
asserts it, because an inverted ladder is the one way the rungs could silently stop being
bounds.

The direct kernel `sum_l (x_l - y_l)^2` gets a RELATIVE bound: every term is non-negative,
so there is no cancellation term at all.  That asymmetry between the two kernels is the
most valuable thing this package can say.

**The norms are computed in float64.**  Reusing the working-dtype `||x||^2` already sitting
in the Gram identity is the tempting shortcut and it is unsound in the direction that
matters: a norm that rounds low yields a radius that is low, and a low radius is not a
bound.  The float64 norm pass is the same O(nd) pass precondition P2 already requires.

**Why a-priori and not a compensated dot product.**  Ogita-Rump-Oishi `Dot2` gives an
a-posteriori bound orders of magnitude tighter than any gamma_n, and it is the first thing
a numerics reviewer reaches for.  It is elementwise and forfeits the gemm entirely.
gamma_n needs only the norms, so it rides the one BLAS call the scores already cost.  The
choice is throughput, it is a decision rather than an oversight, and `bench.py` runs Dot2
as an arm expecting it to win on coverage and lose on throughput.  Higham & Mary's
probabilistic sqrt(n)*u would cut the width by ~19x at d=384 and is REFUSED: it is a
probability, and no output of this package carries one.

Self-check:  python -m separatrix.enclose
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .verdict import (
    ACCUM_ASSUMED,
    BOUND_VACUOUS,
    NONFINITE_INPUT,
    RANGE_UNSAFE,
    REDUCED_PRECISION_ARITHMETIC,
    Refusal,
)

KERNELS = ("gram", "direct")
BOUNDS = ("cheap", "tight")

# P5, stated where it can be read.  Both black-box probes tried came back blind: the
# ones-vector probe dot(1,1)==n is exact at n=32768 for float16 and float32 alike because
# every partial sum of ones is a power of two, and the eps-tail probe retains 1.000 of the
# tail at every dtype because it cannot separate accumulator width from summation order.
# So the accumulator width is a declared assumption that travels on every Verdict, not a
# checked precondition.
P5_NOTE = (
    "P5 accumulator width is not testable by any probe tried; it is assumed to be the "
    f"storage dtype and printed on every verdict as accum_assumed={ACCUM_ASSUMED!r}."
)


# -- constants ----------------------------------------------------------------------------


def unit_roundoff(dtype) -> float:
    """u = eps/2.  float16 2**-11, float32 2**-24, float64 2**-53."""
    return float(np.finfo(np.dtype(dtype)).eps) / 2.0


def gamma(n: int, dtype=np.float64) -> float:
    """gamma_n = n*u / (1 - n*u), evaluated in float64 and rounded outward.

    Raises Refusal(BOUND_VACUOUS) when n*u > 1/2 (precondition P3).  gamma_n goes
    *negative* for n*u > 1, which would produce negative radii and certify everything; the
    1/2 cut is where the bound stops carrying information rather than where it inverts.
    Fires at d >= 1022 for float16.
    """
    if n <= 0:
        raise ValueError(f"gamma needs a positive reduction length, got {n}")
    u = unit_roundoff(dtype)
    nu = n * u
    if nu > 0.5:
        raise Refusal(
            BOUND_VACUOUS,
            f"n*u = {n} * {u:.6e} = {nu:.6f} exceeds 1/2 for {np.dtype(dtype).name}; "
            f"the a-priori bound carries no information at this reduction length",
        )
    return math.nextafter(nu / (1.0 - nu), math.inf)


def eta(d: int, dtype) -> float:
    """The unconditional underflow term, in score units.

    fl(ab) = ab(1+delta) is false when the product lands subnormal: the error is then
    ABSOLUTE, at most smallest_subnormal/2 per product.  IEEE addition whose exact result
    is subnormal is exact, so only the d products contribute.  Carried with no
    precondition, because a precondition on the smallest magnitude in the array is a
    second thing to get wrong and this costs one float.

    Measured in this repository, d=8, 4000 trials, seed 21, exact scaled-integer ground
    truth (`test_enclosure_contains_under_underflow`):

        regime                        radius alone     radius + eta
        float32 components ~1e-25      4000/4000          0/4000
        float16 components ~3e-4       3078/4000          0/4000
        float32 components ~1  CONTROL    0/4000          0/4000

    The control row is what makes this a measurement and not a patch: the escapes are
    specifically underflow, and eta is not masking a broken bound in the normal regime.
    """
    return 4.0 * (d + 2) * float(np.finfo(np.dtype(dtype)).smallest_subnormal)


def _push(d: int) -> float:
    """The radius's own rounding, derived rather than guessed.

    R is built by a float64 reduction of length d, a sqrt, a sum and a square:
        ||x||^2   relative error <= gamma_d(f64)
        sqrt      <= gamma_d/2 + u
        sum       <= gamma_d/2 + 2u
        square    <= gamma_d + 5u
        scale     <= gamma_d + 6u
    so gamma_{d+2}(f64) + 8u_64 dominates it.  `4*u_64` -- the obvious guess -- is short by
    ~200x at d=784 (4.44e-16 against 8.73e-14).
    """
    return gamma(d + 2, np.float64) + 8.0 * unit_roundoff(np.float64)


def _inflate(R: np.ndarray, d: int, work_dtype) -> np.ndarray:
    """R -> nextafter(R*(1+push) + eta, inf).  The last thing done to every radius."""
    out = R * (1.0 + _push(d)) + eta(d, work_dtype)
    return np.nextafter(out, np.inf)


# -- preconditions --------------------------------------------------------------------------


def check_finite(A: np.ndarray, name: str) -> None:
    """P1.  Names the row and column.  No enclosure is defined over a non-finite entry."""
    bad = ~np.isfinite(A)
    if bad.any():
        i, j = (int(v[0]) for v in np.nonzero(bad))
        raise Refusal(
            NONFINITE_INPUT,
            f"{name}[{i}, {j}] is {A[i, j]}; {int(bad.sum())} of {A.size} entries are "
            f"not finite",
        )


def range_headroom(xnorm: np.ndarray, qnorm: np.ndarray, work_dtype) -> np.ndarray:
    """P2, per query row: (||q_i|| + max_j||x_j||)^2 against finfo(work).max.

    Derived, not a safety factor.  Every intermediate of the Gram identity is dominated by
    this one quantity: ||x||^2, ||q||^2, and every partial sum of the cross term (by
    Cauchy-Schwarz, |sum_{l<=L} x_l q_l| <= sum_l |x_l q_l| <= ||x||*||q||).  Norms in
    float64, so the check itself cannot overflow.

    Returns the per-row value; the caller compares it to finfo(work).max.  Exposed so a
    caller can refuse row by row instead of refusing the call.
    """
    return (qnorm + float(xnorm.max())) ** 2


def check_range(xnorm: np.ndarray, qnorm: np.ndarray, work_dtype) -> None:
    """P2.  THE MUST-FIX.

    MNIST at d=784 has ||x||^2 ~ 5.6e6 against float16's 65504: the Gram intermediates
    overflow, the differences become nan, and an enclosure formed downstream of that
    reports a clean pass over garbage.  The recorded prototype measurement is 0 of 300
    refused while 300 of 300 top-10 sets were wrong.

    This runs BEFORE any score is read, so the refusal is RANGE_UNSAFE and never
    NONFINITE_SCORE -- the difference between naming the cause and naming the damage after
    it arrived.  The advice leads with the damage.
    """
    dt = np.dtype(work_dtype)
    limit = float(np.finfo(dt).max)
    head = range_headroom(xnorm, qnorm, dt)
    bad = head > limit
    if bad.any():
        worst = float(head.max())
        raise Refusal(
            RANGE_UNSAFE,
            f"max ||x||^2 reaches {float(xnorm.max()) ** 2:.2e} and the Gram intermediate "
            f"(||q||+||x||)^2 reaches {worst:.2e} against {dt.name}'s {limit:.2e}, on "
            f"{int(bad.sum())} of {len(head)} query rows; on the recorded case 300 of 300 "
            f"top-10 sets came back different from the float64 ones",
        )


_CANARY_CACHE: dict[tuple[str, str], bool] = {}


def canary(dtype, matmul=None) -> bool:
    """P4.  One 4x4 matmul of entries needing the full declared mantissa.

    A = 1 + eps(dtype) everywhere; B has two rows of ones and two of zeros, so every output
    entry is exactly 2 + 2*eps -- representable in the dtype, and the sum of two identical
    values, so the addition is exact too.  Any arithmetic that rounds its inputs coarser
    than the declared mantissa (TF32's 10 bits, bfloat16 inputs, Apple AMX) collapses
    1+eps to 1 and returns exactly 2.

    One test, no vendor flag taxonomy, and it runs on the CPU path with no GPU.  It tests
    the MULTIPLIER's precision only; accumulator width is P5 and is not testable (see
    P5_NOTE).

    `matmul` is injectable so a test can hand it a synthetic reduced-precision path.
    Returns True when the arithmetic matches the declared dtype.
    """
    dt = np.dtype(dtype)
    eps = float(np.finfo(dt).eps)
    a = np.full((4, 4), 1.0 + eps, dtype=dt)
    b = np.zeros((4, 4), dtype=dt)
    b[0, :] = 1
    b[1, :] = 1
    got = (matmul or np.matmul)(a, b)
    want = 2.0 * (1.0 + eps)
    return bool(np.all(np.asarray(got, dtype=np.float64) == want))


def check_canary(work_dtype, backend: str = "numpy") -> str:
    """P4, cached per (backend, dtype).  Returns a printable result string."""
    key = (backend, np.dtype(work_dtype).name)
    if key not in _CANARY_CACHE:
        _CANARY_CACHE[key] = canary(work_dtype)
    ok = _CANARY_CACHE[key]
    if not ok:
        raise Refusal(
            REDUCED_PRECISION_ARITHMETIC,
            f"the {backend} {np.dtype(work_dtype).name} path rounds its inputs coarser "
            f"than eps = {float(np.finfo(np.dtype(work_dtype)).eps):.6e}",
        )
    return f"{backend}/{np.dtype(work_dtype).name} clean"


# -- the scores ------------------------------------------------------------------------------


def _norms64(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(||a||^2, ||a||) in float64, whatever A is stored as.  One O(nd) pass, blocked.

    The blocking is memory, not speed.  ``A.astype(np.float64)`` on the whole array is a
    second copy at 8 bytes an entry -- 1.0 GB for SIFT1M's 1,000,000 x 128 float32 corpus,
    on top of the 512 MB the corpus already costs -- and it is materialised before the
    scores are, so the peak lands where the caller has least room.  Each row's reduction
    is over its own 128 entries either way, so the values are identical to the unblocked
    pass; ``test_norms64_blocking_is_exact`` asserts bitwise equality at several block
    sizes.
    """
    n, d = A.shape
    step = max(1, int(4e6) // max(1, d))  # ~32 MB of float64 per block
    n2 = np.empty(n, dtype=np.float64)
    for i in range(0, n, step):
        b = A[i : i + step].astype(np.float64, copy=False)
        n2[i : i + step] = np.einsum("ij,ij->i", b, b)
    return n2, np.sqrt(n2)


def gram_scores(X: np.ndarray, Q: np.ndarray, work_dtype) -> np.ndarray:
    """d^2 = ||x||^2 + ||y||^2 - 2<x,y>, evaluated entirely in the working dtype.

    This is what torch.cdist computes above 25 rows and what sklearn's euclidean_distances
    computed before PR 13554 chunk-upcast it.  Returned as float64 for the decision layer:
    the VALUES are the working dtype's, losslessly widened, never recomputed.
    """
    dt = np.dtype(work_dtype)
    Xw = X.astype(dt, copy=False)
    Qw = Q.astype(dt, copy=False)
    xn2 = np.einsum("ij,ij->i", Xw, Xw)
    qn2 = np.einsum("ij,ij->i", Qw, Qw)
    D = qn2[:, None] + xn2[None, :] - 2.0 * (Qw @ Xw.T).astype(dt, copy=False)
    return D.astype(np.float64)


def direct_scores(X: np.ndarray, Q: np.ndarray, work_dtype, chunk: int | None = None) -> np.ndarray:
    """sum_l (x_l - y_l)^2, evaluated in the working dtype, chunked over query rows.

    Not scipy.spatial.distance.cdist: that upcasts to double internally, so it would
    certify a computation the caller's float32 pipeline does not run.  Measured at ~100x
    the Gram path's cost, which is exactly why the Gram path exists.
    """
    dt = np.dtype(work_dtype)
    Xw = X.astype(dt, copy=False)
    Qw = Q.astype(dt, copy=False)
    m, d = Qw.shape
    n = Xw.shape[0]
    if chunk is None:
        chunk = max(1, int(2e7) // max(1, n * d))
    D = np.empty((m, n), dtype=np.float64)
    for i0 in range(0, m, chunk):
        i1 = min(i0 + chunk, m)
        diff = Qw[i0:i1, None, :] - Xw[None, :, :]
        D[i0:i1] = np.einsum("ijk,ijk->ij", diff, diff).astype(np.float64)
    return D


# -- the radii ---------------------------------------------------------------------------------


def gram_radii(
    X: np.ndarray,
    Q: np.ndarray,
    *,
    bound: str = "cheap",
    per_pair: bool = False,
    work_dtype=None,
) -> np.ndarray:
    """The Gram identity's enclosure radius.  Shape (m, 1) per row, (m, n) per pair.

    rung 1  per-row cheap    R_i  = g*(||q_i|| + max_j||x_j||)^2      O(m) memory
    rung 2  per-pair cheap   R_ij = g*(||q_i|| + ||x_j||)^2           O(mn)
    rung 3  per-pair tight   R_ij = g*(||x_j||^2+||q_i||^2+2<|x_j|,|q_i|>)   + one matmul

    Rung 1 exists because a per-pair float64 (m,n) array is 1.6 GB at m=1,000 n=100,000 --
    the only scale where this tool could win.  R is monotone in ||x_j||, so the row-wise
    collapse is valid for every pair in the row.  Measured cost of the collapse: identical
    refusal counts at d=384; 20 -> 30 of 300 at d=784.

    A `bound="tight"` radius with `per_pair=False` is the row max of the per-pair tight
    radius -- still one matmul, still O(m) to return.
    """
    if bound not in BOUNDS:
        raise ValueError(f"bound must be one of {BOUNDS}, got {bound!r}")
    dt = np.dtype(work_dtype if work_dtype is not None else X.dtype)
    d = X.shape[1]
    g = gamma(d + 2, dt)
    xn2, xn = _norms64(X)
    qn2, qn = _norms64(Q)

    if bound == "cheap":
        if per_pair:
            R = g * (qn[:, None] + xn[None, :]) ** 2
        else:
            R = g * (qn[:, None] + float(xn.max())) ** 2
    else:
        absdot = np.abs(Q.astype(np.float64, copy=False)) @ np.abs(
            X.astype(np.float64, copy=False)
        ).T
        R = g * (qn2[:, None] + xn2[None, :] + 2.0 * absdot)
        if not per_pair:
            R = R.max(axis=1, keepdims=True)
    return _inflate(R, d, dt)


def direct_radii(D: np.ndarray, d: int, work_dtype) -> np.ndarray:
    """The direct kernel's enclosure radius: RELATIVE, because nothing cancels.

    Higham Thm 3.1 gives |fl(s) - s| <= gamma_{d+1} * s in terms of the EXACT s.  Solving
    for the computed value, s <= fl/(1-gamma), so the radius in terms of what we hold is
    gamma/(1-gamma) * fl.  The difference is O(gamma^2) and is taken anyway, because a
    bound stated over a quantity you do not have is not a bound.
    """
    dt = np.dtype(work_dtype)
    g = gamma(d + 1, dt)
    if g >= 1.0:
        # MEASURED BUG, not a hypothetical.  P3 admits n*u == 1/2, where gamma_n is exactly
        # 1.0 and is rounded outward to 1.0000000000000002.  The Gram form multiplies by
        # gamma and stays sound (an enormous, useless, POSITIVE radius); this relative form
        # divides by 1 - gamma and produces a NEGATIVE one.  A negative radius inverts every
        # interval, so max-in falls below min-out and the rule CERTIFIES everything.
        # Reproduced on `corpus.adversarial("vacuous_f16_d1023")`: float16 at d = 1023 gives
        # (d+1)u = 0.5 exactly, radius -9.22e+14, and `certified_topk(..., kernel="direct")`
        # returned CERTIFIED over a bound that was not one.
        raise Refusal(
            BOUND_VACUOUS,
            f"gamma_{d + 1} = {g!r} for {dt.name} is not below 1, so the direct kernel's "
            f"relative form gamma/(1-gamma) is not a radius; the a-priori bound carries "
            f"no information at this reduction length",
        )
    R = np.maximum(D, 0.0) * (g / (1.0 - g))
    return _inflate(R, d, dt)


# -- the enclosure ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Enclosure:
    """(D, R) plus everything the verdict must print about how they were made."""

    D: np.ndarray  # (m, n) float64 image of the working-dtype scores
    R: np.ndarray  # (m, 1) or (m, n) float64, |D - exact| <= R entrywise
    kernel: str
    bound: str
    per_pair: bool
    dtype_in: str
    dtype_used: str
    d: int
    gamma: float
    eta: float
    canary: str
    accum_assumed: str = ACCUM_ASSUMED

    def interval(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Row i as (lo, hi), squared scores clamped at zero.

        The clamp is free, valid -- a squared distance is non-negative in exact arithmetic
        -- and it halves the width exactly on the near-coincident pairs this tool is
        loudest about.
        """
        r = np.broadcast_to(self.R[i], self.D[i].shape)
        return np.maximum(self.D[i] - r, 0.0), self.D[i] + r


def enclose_scores(
    X: np.ndarray,
    Q: np.ndarray,
    *,
    kernel: str = "gram",
    bound: str = "cheap",
    per_pair: bool = False,
    work_dtype=None,
    backend: str = "numpy",
) -> Enclosure:
    """Preconditions, then scores, then radii.  Raises Refusal; never returns a warning.

    Order is P1 finite, P3 non-vacuous, P2 range, P4 canary, and only then the matmul.  A
    precondition that runs after the scores is a precondition that certifies garbage: the
    fp16 range case reads NONFINITE_SCORE if P2 runs late, which names the damage instead
    of the cause.
    """
    X = np.ascontiguousarray(X)
    Q = np.ascontiguousarray(Q)
    if X.ndim != 2 or Q.ndim != 2:
        raise ValueError(f"X and Q must be 2-D, got {X.shape} and {Q.shape}")
    if X.shape[1] != Q.shape[1]:
        raise ValueError(f"X has d={X.shape[1]}, Q has d={Q.shape[1]}")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    if X.dtype != Q.dtype:
        raise TypeError(
            f"X is {X.dtype} and Q is {Q.dtype}; the certificate is about one working "
            f"dtype, so pick it explicitly rather than letting numpy promote"
        )
    dtype_in = np.dtype(X.dtype)
    dt = np.dtype(work_dtype) if work_dtype is not None else dtype_in
    d = X.shape[1]

    check_finite(X, "X")
    check_finite(Q, "Q")
    g = gamma(d + 2 if kernel == "gram" else d + 1, dt)  # P3
    _, xn = _norms64(X)
    _, qn = _norms64(Q)
    # P2 covers both kernels: ||x-y||^2 <= (||x||+||y||)^2, so the same headroom bounds
    # every intermediate of the direct sum as well as every intermediate of the Gram
    # identity.  One precondition, one place to be wrong.
    check_range(xn, qn, dt)  # P2, the must-fix
    can = check_canary(dt, backend)  # P4

    if kernel == "gram":
        D = gram_scores(X, Q, dt)
        R = gram_radii(X, Q, bound=bound, per_pair=per_pair, work_dtype=dt)
    else:
        # The direct kernel's radius is relative, so it is a function of D, which is
        # already an (m, n) array.  The row collapse that rung 1 exists for would buy no
        # memory here, so `per_pair` is not offered a choice.
        D = direct_scores(X, Q, dt)
        R = direct_radii(D, d, dt)
        per_pair = True

    return Enclosure(
        D=D,
        R=R,
        kernel=kernel,
        bound=bound if kernel == "gram" else "relative",
        per_pair=per_pair,
        dtype_in=dtype_in.name,
        dtype_used=dt.name,
        d=d,
        gamma=g,
        eta=eta(d, dt),
        canary=can,
    )


def _demo() -> None:
    from .exact import exact_sq

    # the textbook constants, typed out rather than asked of numpy
    assert unit_roundoff(np.float16) == 2.0**-11
    assert unit_roundoff(np.float32) == 2.0**-24
    assert unit_roundoff(np.float64) == 2.0**-53

    # gamma is vacuous exactly where P3 says it is
    assert gamma(386, np.float32) > 0
    try:
        gamma(2050, np.float16)
    except Refusal as r:
        assert r.reason == BOUND_VACUOUS
    else:  # pragma: no cover
        raise AssertionError("a vacuous bound was returned")

    # Higham Thm 3.1: the signed cross term is BELOW the true bound
    rng = np.random.default_rng(0)
    x = rng.standard_normal(384)
    y = rng.standard_normal(384)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    assert abs(x @ y) < np.abs(x) @ np.abs(y)

    # cheap dominates tight, pointwise, by Cauchy-Schwarz
    A = rng.standard_normal((40, 16)).astype(np.float32)
    B = rng.standard_normal((7, 16)).astype(np.float32)
    Rc = gram_radii(A, B, bound="cheap", per_pair=True)
    Rt = gram_radii(A, B, bound="tight", per_pair=True)
    assert np.all(Rc >= Rt), float((Rt - Rc).max())

    # the enclosure contains the exact value, on a corpus built to cancel
    enc = enclose_scores(A, B, kernel="gram", bound="cheap", per_pair=True)
    for i in range(B.shape[0]):
        lo, hi = enc.interval(i)
        for j in range(A.shape[0]):
            s = exact_sq(B[i], A[j]) / float(1 << (2 * 149))
            assert lo[j] <= s <= hi[j], (i, j, lo[j], s, hi[j])

    # P2 is the must-fix: MNIST-shaped rows at float16 refuse before any score is read
    mn = (rng.random((64, 784)) * 255.0).astype(np.float16)
    try:
        enclose_scores(mn, mn[:4], work_dtype=np.float16)
    except Refusal as r:
        assert r.reason == RANGE_UNSAFE, r.reason
    else:  # pragma: no cover
        raise AssertionError("the float16 range case was enclosed")
    # and the refusal was necessary: the scores really do come back non-finite
    with np.errstate(over="ignore", invalid="ignore"):
        bad = gram_scores(mn, mn[:4], np.float16)
    assert not np.isfinite(bad).all()

    # the canary is clean on the real path and fires on a 10-bit one
    assert canary(np.float32) and canary(np.float64) and canary(np.float16)

    def tf32ish(a, b, bits=10):
        def rnd(z):
            m, e = np.frexp(z.astype(np.float64))
            return np.ldexp(np.round(m * 2.0**bits) / 2.0**bits, e)

        return rnd(a) @ rnd(b)

    assert not canary(np.float32, matmul=tf32ish)

    print("enclose: ok")


if __name__ == "__main__":
    _demo()
