"""The enclosure, scored against closed forms and against exact scaled-integer arithmetic.

Nothing here scores against separatrix's own floating point.  Every row of the table in
the spec's section 7 that names `enclose.py` is one test below, and the pass condition is
the one written there: for containment it is **0 escapes**, and one escape withdraws the
package.
"""

from __future__ import annotations

import numpy as np
import pytest

from separatrix import enclose as E
from separatrix.exact import BITS_DOUBLE, BITS_SMALL, exact_sq, scale
from separatrix.verdict import (
    BOUND_VACUOUS,
    NONFINITE_INPUT,
    RANGE_UNSAFE,
    REDUCED_PRECISION_ARITHMETIC,
    Refusal,
)


# -- the constants -------------------------------------------------------------------------


def test_u_is_the_textbook_constant():
    # typed out, not asked of numpy, so a numpy change cannot move the target
    assert E.unit_roundoff(np.float16) == 2.0**-11
    assert E.unit_roundoff(np.float32) == 2.0**-24
    assert E.unit_roundoff(np.float64) == 2.0**-53


@pytest.mark.parametrize(
    "d, dtype, want",
    [
        (384, np.float32, 2.300792e-05),
        (784, np.float32, 4.685145e-05),
        (384, np.float16, 2.322503e-01),
        (784, np.float16, 6.228209e-01),
    ],
)
def test_gamma_table_is_the_published_one(d, dtype, want):
    # gamma_n = n*u/(1-n*u), evaluated by hand here rather than by calling the code twice
    u = {np.float16: 2.0**-11, np.float32: 2.0**-24}[dtype]
    n = d + 2
    by_hand = n * u / (1.0 - n * u)
    got = E.gamma(n, dtype)
    assert got == pytest.approx(by_hand, rel=1e-15)
    assert got == pytest.approx(want, rel=1e-6)
    # rounded outward: never below the real value
    assert got >= by_hand


def test_gamma_is_rounded_outward_and_positive():
    for n, dt in ((386, np.float32), (786, np.float32), (386, np.float16)):
        assert E.gamma(n, dt) > 0


def test_gamma_raises_when_vacuous():
    # (d+2)*u > 1/2 is the cut; float16 at d=2048 is 1.0009
    with pytest.raises(Refusal) as ei:
        E.gamma(2050, np.float16)
    assert ei.value.reason == BOUND_VACUOUS
    assert ei.value.exit_code == 2
    # and the boundary is where the algebra says it is, not one step early
    u = 2.0**-11
    n_ok = int(0.5 / u)
    assert E.gamma(n_ok, np.float16) > 0
    with pytest.raises(Refusal):
        E.gamma(n_ok + 1, np.float16)


# -- the corpora these tests use (adversarial, not random) -----------------------------------


def adversarial_pairs(dtype):
    """Pairs built to cancel, to underflow, and to sit on the preconditions' edges.

    Random normal data at d=384 sits sqrt(d)/d of the way to the bound and never
    approaches the worst case, so a containment test on random data is an
    implementation-bug test rather than a test of the theorem.  This is the theorem's
    corpus.
    """
    dt = np.dtype(dtype)
    rng = np.random.default_rng(7)
    pairs = []
    # near-coincident at a large offset: the cancellation frame
    pairs.append((np.array([1e6, 0.0]), np.array([1e6 + 1e-6, 0.0])))
    pairs.append((np.array([1e3, 1e3]), np.array([1e3 + 1e-3, 1e3])))
    # alternating signs: the cross term's worst case
    v = np.array([(-1.0) ** i for i in range(64)])
    pairs.append((v, v * (1.0 + 1e-5)))
    # 1e16 dynamic range at d=2
    pairs.append((np.array([1e8, 1e-8]), np.array([1e8, -1e-8])))
    # identical rows: an exact tie in the making
    w = rng.standard_normal(32)
    pairs.append((w.copy(), w.copy()))
    # ordinary scale, for contrast
    pairs.append((rng.standard_normal(32), rng.standard_normal(32)))
    if dt == np.float16:
        # float16 cannot hold the large-offset frames without tripping P2; scale them in
        pairs = [(a / 4096.0, b / 4096.0) for a, b in pairs]
    return [(a.astype(dt), b.astype(dt)) for a, b in pairs]


def integer_lattice(dtype, n=64, d=12, cap=None, seed=3):
    """Integer coordinates: ||x||^2, <x,y> and d^2 are exact integers in float64.

    Per-dtype caps, or the float16 column is a domain error rather than a measurement:
    |x_i| <= 2**11 at d=784 gives ||x||^2 <= 3.3e9, which is 5e4 times float16's 65504.
    """
    dt = np.dtype(dtype)
    if cap is None:
        cap = {np.dtype(np.float16): 8, np.dtype(np.float32): 2048, np.dtype(np.float64): 2048}[dt]
    rng = np.random.default_rng(seed)
    return rng.integers(-cap, cap + 1, size=(n, d)).astype(dt)


# -- containment: the one failure that withdraws the package ---------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kernel", ["gram", "direct"])
@pytest.mark.parametrize("bound", ["cheap", "tight"])
def test_enclosure_contains_the_exact_value(dtype, kernel, bound):
    bits = BITS_DOUBLE if np.dtype(dtype) == np.float64 else BITS_SMALL
    escapes = 0
    checked = 0
    for a, b in adversarial_pairs(dtype):
        X = np.atleast_2d(a)
        Q = np.atleast_2d(b)
        enc = E.enclose_scores(X, Q, kernel=kernel, bound=bound, per_pair=True)
        lo, hi = enc.interval(0)
        s = exact_sq(b, a, bits) / scale(bits)
        checked += 1
        escapes += int(not (lo[0] <= s <= hi[0]))
    assert checked == len(adversarial_pairs(dtype))
    assert escapes == 0


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("kernel", ["gram", "direct"])
def test_enclosure_contains_on_the_integer_lattice(dtype, kernel):
    X = integer_lattice(dtype, n=48, d=10)
    Q = integer_lattice(dtype, n=6, d=10, seed=11)
    enc = E.enclose_scores(X, Q, kernel=kernel, per_pair=True)
    escapes = 0
    for i in range(Q.shape[0]):
        lo, hi = enc.interval(i)
        for j in range(X.shape[0]):
            s = exact_sq(Q[i], X[j], BITS_SMALL) / scale(BITS_SMALL)
            escapes += int(not (lo[j] <= s <= hi[j]))
    assert escapes == 0


def _escapes(X, Q, dtype, with_eta: bool) -> int:
    """Containment count with, and without, the underflow term.  The regression's arms."""
    d = X.shape[1]
    enc = E.enclose_scores(X, Q, kernel="gram", bound="cheap", per_pair=True, work_dtype=dtype)
    if with_eta:
        R = enc.R[0]
    else:
        # the radius alone: gamma * (||x||+||y||)^2, no eta, no outward push
        xn = np.linalg.norm(X.astype(np.float64), axis=1)
        qn = np.linalg.norm(Q.astype(np.float64), axis=1)
        R = E.gamma(d + 2, dtype) * (qn[0] + xn) ** 2
    D = enc.D[0]
    bad = 0
    for j in range(X.shape[0]):
        s = exact_sq(Q[0], X[j], BITS_SMALL) / scale(BITS_SMALL)
        if not (D[j] - R[j] <= s <= D[j] + R[j]):
            bad += 1
    return bad


@pytest.mark.parametrize(
    "dtype, magnitude, control",
    [
        (np.float32, 1e-25, False),
        (np.float16, 3e-4, False),
        (np.float32, 1.0, True),  # the control: normal products, eta must not be masking
    ],
)
def test_enclosure_contains_under_underflow(dtype, magnitude, control):
    """fl(ab) = ab(1+delta) is false for subnormal products; eta is what makes it a bound.

    The control row is what makes this a measurement rather than a patch: with normal
    components the radius alone already contains the exact value, so eta is not covering
    for a broken bound in the ordinary regime.
    """
    n = 4000
    rng = np.random.default_rng(21)
    X = (rng.standard_normal((n, 8)) * magnitude).astype(dtype)
    Q = (rng.standard_normal((1, 8)) * magnitude).astype(dtype)

    with_eta = _escapes(X, Q, dtype, with_eta=True)
    assert with_eta == 0, f"{with_eta}/{n} escapes WITH the underflow term"

    without = _escapes(X, Q, dtype, with_eta=False)
    if control:
        assert without == 0, "the normal-magnitude control escaped; eta is masking a bug"
    else:
        assert without > 0, "the subnormal regime did not escape; the regression is inert"


def test_random_data_containment():
    """An IMPLEMENTATION-BUG test, not a test of the theorem.

    Random normal data never approaches the worst case the bound is stated over, so 0
    escapes here is evidence the code matches the formula, not that the formula is a bound.
    """
    rng = np.random.default_rng(5)
    X = rng.standard_normal((120, 24)).astype(np.float32)
    Q = rng.standard_normal((4, 24)).astype(np.float32)
    enc = E.enclose_scores(X, Q, per_pair=True)
    escapes = 0
    for i in range(4):
        lo, hi = enc.interval(i)
        for j in range(120):
            s = exact_sq(Q[i], X[j], BITS_SMALL) / scale(BITS_SMALL)
            escapes += int(not (lo[j] <= s <= hi[j]))
    assert escapes == 0


def test_cheap_dominates_tight():
    """Cauchy-Schwarz: <|x|,|y|> <= ||x||*||y||, so the ladder cannot invert."""
    rng = np.random.default_rng(2)
    for d in (2, 17, 128):
        X = rng.standard_normal((60, d)).astype(np.float32) * rng.random((60, 1))
        Q = rng.standard_normal((9, d)).astype(np.float32)
        c = E.gram_radii(X, Q, bound="cheap", per_pair=True)
        t = E.gram_radii(X, Q, bound="tight", per_pair=True)
        assert np.all(c >= t), float((t - c).max())


def test_row_collapse_is_still_a_bound():
    """Rung 1 is monotone in ||x_j||, so the row-wise radius dominates every pair in it."""
    rng = np.random.default_rng(4)
    X = (rng.standard_normal((80, 16)) * rng.random((80, 1)) * 30).astype(np.float32)
    Q = rng.standard_normal((5, 16)).astype(np.float32)
    row = E.gram_radii(X, Q, per_pair=False)
    pair = E.gram_radii(X, Q, per_pair=True)
    assert row.shape == (5, 1) and pair.shape == (5, 80)
    assert np.all(row >= pair)


# -- P1, P2, P3, P4 --------------------------------------------------------------------------


def test_nonfinite_input_refuses_and_names_the_cell():
    X = np.ones((4, 3), dtype=np.float32)
    X[2, 1] = np.nan
    with pytest.raises(Refusal) as ei:
        E.enclose_scores(X, np.ones((1, 3), dtype=np.float32))
    assert ei.value.reason == NONFINITE_INPUT
    assert "X[2, 1]" in ei.value.detail


def _mnist_shaped(n, d=784, dtype=np.float16, seed=13):
    """Pixel-scaled rows: ||x||^2 lands around 5.6e6, the recorded case's magnitude."""
    rng = np.random.default_rng(seed)
    A = (rng.random((n, d)) < 0.19) * rng.integers(120, 256, size=(n, d))
    return A.astype(dtype)


def test_fp16_range_is_never_certified():
    """THE MUST-FIX.

    The recorded prototype behaviour: 0 of 300 refused while 300 of 300 top-10 sets came
    back different from the float64 ones, because ||x||^2 ~ 5.6e6 overflows float16's
    65504, the Gram intermediates go to inf, the differences go to nan, and an enclosure
    formed downstream of that reports a clean pass over garbage.

    The assertion is that P2 fires BEFORE any score is read -- RANGE_UNSAFE, naming the
    cause, and never a non-finite-score refusal, which would name the damage after it
    arrived.
    """
    X = _mnist_shaped(400)
    Q = _mnist_shaped(30, seed=99)
    xn2 = (X.astype(np.float64) ** 2).sum(1)
    assert xn2.max() > 4.0e6, f"the corpus is not MNIST-shaped: max ||x||^2 = {xn2.max():.2e}"

    with pytest.raises(Refusal) as ei:
        E.enclose_scores(X, Q, work_dtype=np.float16)
    assert ei.value.reason == RANGE_UNSAFE
    assert "float16" in ei.value.detail
    assert "upcast=True" in ei.value.next_action

    # the refusal was necessary, not merely cautious: the unguarded float16 scores are
    # garbage and the top-10 sets they produce are not the float64 ones
    with np.errstate(over="ignore", invalid="ignore"):
        bad = E.gram_scores(X, Q, np.float16)
    assert not np.isfinite(bad).all()
    good = E.gram_scores(X, Q, np.float64)
    wrong = sum(
        set(np.argpartition(np.nan_to_num(bad[i], nan=np.inf), 9)[:10].tolist())
        != set(np.argpartition(good[i], 9)[:10].tolist())
        for i in range(Q.shape[0])
    )
    assert wrong == Q.shape[0], f"{wrong}/{Q.shape[0]} top-10 sets differed"

    # float32 is in range for the same corpus, so this is a dtype limit and not a refusal
    # to work on the data
    enc = E.enclose_scores(X, Q, work_dtype=np.float32)
    assert np.isfinite(enc.D).all() and enc.dtype_used == "float32"


def test_partial_overflow_refuses():
    """Each ||.||^2 is finite in float16; their sum is not.

    The naive "is the input finite" check passes here, which is exactly why P1 is not what
    guards this.
    """
    X = np.array([[200.0, 0.0]], dtype=np.float16)
    Q = np.array([[0.0, 200.0]], dtype=np.float16)
    assert np.isfinite(X).all() and np.isfinite(Q).all()
    assert float((X.astype(np.float64) ** 2).sum()) == 40000.0
    with np.errstate(over="ignore"):
        assert np.float16(40000.0) + np.float16(40000.0) == np.inf
    with pytest.raises(Refusal) as ei:
        E.enclose_scores(X, Q, work_dtype=np.float16)
    assert ei.value.reason == RANGE_UNSAFE


def test_range_headroom_is_per_row():
    """One outlier row must not refuse every query row."""
    X = np.ones((3, 4), dtype=np.float16)
    q = np.zeros((2, 4), dtype=np.float16)
    q[1] = 200.0
    head = E.range_headroom(
        np.linalg.norm(X.astype(np.float64), axis=1),
        np.linalg.norm(q.astype(np.float64), axis=1),
        np.float16,
    )
    limit = float(np.finfo(np.float16).max)
    assert head[0] < limit < head[1]


def test_canary_detects_reduced_precision():
    """P4: one 4x4 matmul, no vendor flag taxonomy, and it runs on the CPU path."""
    for dt in (np.float16, np.float32, np.float64):
        assert E.canary(dt), f"the true {np.dtype(dt).name} path failed its own canary"

    def rounded(bits):
        def mm(a, b):
            def r(z):
                m, e = np.frexp(z.astype(np.float64))
                return np.ldexp(np.round(m * 2.0**bits) / 2.0**bits, e)

            return r(a) @ r(b)

        return mm

    assert not E.canary(np.float32, matmul=rounded(10)), "TF32's 10-bit mantissa went unseen"
    assert not E.canary(np.float64, matmul=rounded(23)), "a float32 multiplier went unseen"
    assert not E.canary(np.float32, matmul=rounded(7)), "a bfloat16 multiplier went unseen"


def test_reduced_precision_refusal_is_typed():
    E._CANARY_CACHE[("synthetic", "float32")] = False
    try:
        with pytest.raises(Refusal) as ei:
            E.check_canary(np.float32, backend="synthetic")
        assert ei.value.reason == REDUCED_PRECISION_ARITHMETIC
    finally:
        E._CANARY_CACHE.pop(("synthetic", "float32"), None)


def test_accumulator_assumption_is_stated_where_it_can_be_read():
    """P5 is not testable by any probe tried, so it is a declared assumption that travels."""
    enc = E.enclose_scores(
        np.ones((4, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)
    )
    assert enc.accum_assumed and enc.accum_assumed in E.P5_NOTE
    # and the probes that came back blind are recorded, so nobody re-runs them hopefully
    ones = np.ones(32768, dtype=np.float16)
    assert float(ones @ ones) == 32768.0  # exact at float16: every partial sum is a power of 2


# -- shape and usage -------------------------------------------------------------------------


def test_mixed_dtypes_are_a_usage_error_not_a_refusal():
    with pytest.raises(TypeError):
        E.enclose_scores(np.ones((3, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float64))


def test_dimension_mismatch_is_a_usage_error():
    with pytest.raises(ValueError):
        E.enclose_scores(np.ones((3, 2), dtype=np.float32), np.ones((1, 5), dtype=np.float32))


def test_gram_and_direct_agree_in_float64_on_the_lattice():
    """Both kernels compute the same real number; on integer data float64 gets it exactly."""
    X = integer_lattice(np.float64, n=32, d=8)
    Q = integer_lattice(np.float64, n=4, d=8, seed=17)
    g = E.gram_scores(X, Q, np.float64)
    d = E.direct_scores(X, Q, np.float64)
    assert np.array_equal(g, d)


def test_interval_clamps_squared_scores_at_zero():
    X = np.array([[1e6, 0.0]], dtype=np.float64)
    Q = np.array([[1e6, 0.0]], dtype=np.float64)
    enc = E.enclose_scores(X, Q, per_pair=True)
    lo, hi = enc.interval(0)
    assert lo[0] == 0.0 and hi[0] > 0.0


def test_norms64_blocking_is_exact():
    """The blocked float64 norm pass returns exactly what the one-shot pass returned.

    Blocking is over rows and each row's reduction is unchanged, so this is bitwise, not
    approximate. It exists because the one-shot pass allocated a full float64 copy of the
    corpus -- 1.0 GB on SIFT1M -- before any score was computed.
    """
    rng = np.random.default_rng(5)
    for dt in (np.float32, np.float64):
        A = (rng.standard_normal((997, 33)) * 1e3).astype(dt)
        want2 = np.einsum("ij,ij->i", A.astype(np.float64), A.astype(np.float64))
        got2, got = E._norms64(A)
        assert np.array_equal(got2, want2)
        assert np.array_equal(got, np.sqrt(want2))
