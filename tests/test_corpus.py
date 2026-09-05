"""The corpora, scored against the oracle and against the certifier they feed.

Two separations are deliberate.  The lattice's ground truth is asserted against
`exact.exact_sq` -- an independent arithmetic -- and not against the integers the
constructor used, so a constructor that quietly built a different corpus than it claimed
is caught.  And the containment test runs on the *shipped* adversarial corpus rather than
on pairs typed into the test, because a corpus that lives only in a test file cannot be
what `bench.py` measures.
"""

from __future__ import annotations

import numpy as np
import pytest

from separatrix.corpus import (
    DELTA_SCHEDULE,
    Corpus,
    CorpusUnavailable,
    adversarial,
    as_points,
    exact_lattice,
    load,
    mnist,
    sift1m,
)
from separatrix.decide import topk_determined, topk_set
from separatrix.enclose import enclose_scores
from separatrix.exact import escalate_row, exact_sq
from separatrix.verdict import BOUND_VACUOUS, EXACT_TIE, RANGE_UNSAFE, Refusal

# -- C1: truth by construction, checked against the oracle ------------------------------------


@pytest.mark.parametrize("dtype,delta", [("float64", 0), ("float32", 65536), ("float16", 1024)])
def test_lattice_distances_are_the_prescribed_integers(dtype, delta):
    """The exact squared distances are integers, and the frontier margin is exactly delta.

    Scored with `exact_sq` at bits=0, which is a different arithmetic from the one the
    constructor used: the constructor adds a four-square decomposition to a sparse +-1
    vector, the oracle squares the stored floats.  Agreement is evidence the corpus is
    what it says it is.
    """
    c = exact_lattice(dtype=np.dtype(dtype), delta=delta, seed=5)
    vals = np.array([exact_sq(c.Q[0], c.X[j], bits=0) for j in range(c.n)], dtype=object)
    ordered = sorted(int(v) for v in vals)
    assert ordered[c.k] - ordered[c.k - 1] == delta
    assert len(set(ordered)) == (c.n - 1 if delta == 0 else c.n)


@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_lattice_truth_is_the_exact_topk(dtype):
    c = exact_lattice(dtype=np.dtype(dtype), delta=1024, seed=1)
    order = sorted(range(c.n), key=lambda j: exact_sq(c.Q[0], c.X[j], bits=0))
    assert set(order[: c.k]) == set(c.truth[0].tolist())
    # the answer is not the first k indices: a corpus whose truth is range(k) would pass
    # a certifier that ignored the data
    assert c.truth[0].tolist() != list(range(c.k))


@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_lattice_survives_its_dtype(dtype):
    """Integer coordinates, stored losslessly, and inside the dtype's Gram headroom."""
    dt = np.dtype(dtype)
    c = exact_lattice(dtype=dt, delta=1024 if dt == np.float16 else 65536, seed=2)
    assert c.X.dtype == dt and c.Q.dtype == dt
    assert np.array_equal(c.X.astype(np.float64), np.rint(c.X.astype(np.float64)))
    assert c.headroom() < float(np.finfo(dt).max)


def test_lattice_refuses_to_build_what_the_dtype_cannot_hold():
    """A corpus that overflows its dtype is a usage error at construction, not a silent
    corpus that P2 refuses later.  The message carries both numbers."""
    with pytest.raises(ValueError) as e:
        exact_lattice(dtype=np.float16, delta=2**20)
    assert "float16" in str(e.value) and "headroom" in str(e.value)


def test_lattice_delta_zero_is_an_exact_tie_and_no_precision_removes_it():
    """delta = 0 puts two exactly equal scores at the frontier: no correct answer exists.

    Every dtype must reach NOT CERTIFIED (EXACT_TIE) through escalation, never a
    certificate -- this is the one refusal no arithmetic anywhere resolves.
    """
    for dtype in (np.float64, np.float32):
        c = exact_lattice(dtype=dtype, delta=0, seed=4)
        e = enclose_scores(c.X, c.Q, kernel="direct")
        assert topk_determined(e.D[0], e.R[0], c.k) is not None
        esc = escalate_row(c.Q[0], c.X, e.D[0], e.R[0], c.k, bits=0)
        assert not esc.determined and esc.reason == EXACT_TIE


def test_lattice_delta_sweep_finds_the_smallest_certifying_margin():
    """Sweeping the schedule gives delta*, and determination is monotone in delta.

    Monotonicity is not free: a wider margin also pushes the outermost point further out,
    which raises the row's cheap radius.  It holds because the radius grows like gamma
    times the margin while the gap grows like the margin, and gamma is 2.3e-05 here.
    """
    seen = []
    for delta in DELTA_SCHEDULE:
        c = exact_lattice(dtype=np.float32, delta=delta, seed=6)
        e = enclose_scores(c.X, c.Q)
        determined = topk_determined(e.D[0], e.R[0], c.k) is None
        if determined:
            assert set(topk_set(e.D[0], c.k).tolist()) == set(c.truth[0].tolist())
        seen.append((delta, determined))

    assert seen[0] == (0, False)
    assert seen[-1][1] is True
    first = next(i for i, (_, ok) in enumerate(seen) if ok)
    assert all(ok for _, ok in seen[first:]), seen
    # and delta* is above the exact tie by more than one schedule step, so the corpus is
    # actually exercising the enclosure rather than certifying everything
    assert first > 0


# -- C2: the adversarial corpus ---------------------------------------------------------------


def test_every_adversarial_case_is_constructible_and_named_once():
    cases = adversarial()
    assert len(cases) >= 10
    assert len({c.name for c in cases}) == len(cases)
    for c in cases:
        assert isinstance(c, Corpus) and c.note and str(c)
        assert adversarial(c.name) is not None
    with pytest.raises(ValueError):
        adversarial("no such case")


@pytest.mark.parametrize("case", [c.name for c in adversarial() if c.expect_reason])
def test_expected_refusals_are_produced_by_the_certifier(case):
    """`expect_reason` is a property of the construction, so the certifier must produce it.

    This is the corpus half of the must-fix: `fp16_range_784` carries RANGE_UNSAFE because
    its pixels overflow float16's Gram intermediates, and the refusal must fire BEFORE any
    score is read.  If this test ever passes for the wrong reason -- NONFINITE_INPUT, say,
    or a certificate -- the corpus stopped being the case it is named for.
    """
    c = adversarial(case)
    with pytest.raises(Refusal) as e:
        enclose_scores(c.X, c.Q)
    assert e.value.reason == c.expect_reason


def test_fp16_range_case_overflows_by_arithmetic_not_by_assertion():
    c = adversarial("fp16_range_784")
    limit = float(np.finfo(np.float16).max)
    n2 = (c.X.astype(np.float64) ** 2).sum(1)
    assert n2.max() > 4e6 and c.headroom() > limit
    # and the unguarded float16 Gram identity really does return garbage on it
    Xw, Qw = c.X, c.Q
    with np.errstate(over="ignore", invalid="ignore"):
        unguarded = (
            np.einsum("ij,ij->i", Qw, Qw)[:, None]
            + np.einsum("ij,ij->i", Xw, Xw)[None, :]
            - np.float16(2.0) * (Qw @ Xw.T)
        )
    assert not np.isfinite(np.asarray(unguarded, dtype=np.float64)).all()


def test_partial_overflow_passes_the_naive_finiteness_check():
    """Each norm is finite in float16 and their sum is not, so `isfinite(X)` is not what
    guards this and the range precondition has to be the thing that does."""
    c = adversarial("partial_overflow_f16")
    assert np.isfinite(c.X).all() and np.isfinite(c.Q).all()
    n2 = (c.X.astype(np.float64) ** 2).sum(1)
    assert (n2 < float(np.finfo(np.float16).max)).all()
    assert c.headroom() > float(np.finfo(np.float16).max)
    with pytest.raises(Refusal) as e:
        enclose_scores(c.X, c.Q)
    assert e.value.reason == RANGE_UNSAFE


def test_the_vacuous_edge_is_one_dimension_wide():
    """d=1022 float16 is the last width the a-priori bound covers; d=1023 is not."""
    with pytest.raises(Refusal) as e:
        enclose_scores(*(lambda c: (c.X, c.Q))(adversarial("vacuous_f16_d1023")))
    assert e.value.reason == BOUND_VACUOUS
    legal = adversarial("vacuous_edge_f16_d1022")
    enc = enclose_scores(legal.X, legal.Q)  # legal, and refuses everything: that is fine
    assert enc.gamma > 0.9  # a 100% relative bound, which is legal and useless
    assert topk_determined(enc.D[0], enc.R[0], legal.k) is not None


def test_frame_one_is_a_formula_failure_and_not_a_precision_failure():
    """Changing the formula is the fix; changing the precision is not.

    In float64 the Gram identity returns exactly 0.0 for a pair the direct sum separates
    at 1.0000152290447206e-12.  So the Gram enclosure cannot decide the pair and the
    direct enclosure can -- which is the whole product in one corpus.
    """
    c = adversarial("cancellation_f64")
    g = enclose_scores(c.X, c.Q, kernel="gram")
    assert g.D[0, 0] == 0.0 and g.D[0, 1] == 0.0
    assert topk_determined(g.D[0], g.R[0], 1) is not None  # the Gram kernel cannot separate

    d = enclose_scores(c.X, c.Q, kernel="direct")
    assert d.D[0, 1] == 1.0000152290447206e-12
    assert topk_determined(d.D[0], d.R[0], 1) is None  # the direct kernel does


def test_frame_one_at_float32_is_a_tie_in_the_data():
    """At float32 the two points are one stored vector, so 0.0 is the correct squared
    distance for the data as it stands and the ambiguity is not the arithmetic's."""
    c = adversarial("cancellation_f32")
    assert np.array_equal(c.X[0], c.X[1])
    assert exact_sq(c.X[0], c.X[1], bits=0) == 0
    e = enclose_scores(c.X, c.Q, kernel="direct")
    esc = escalate_row(c.Q[0], c.X, e.D[0], e.R[0], 1)
    assert not esc.determined and esc.reason == EXACT_TIE


# -- containment: the one failure that withdraws the package ------------------------------------


@pytest.mark.parametrize("kernel", ["gram", "direct"])
@pytest.mark.parametrize("bound", ["cheap", "tight"])
@pytest.mark.parametrize("per_pair", [False, True])
def test_enclosure_contains_the_exact_value_on_the_shipped_corpus(kernel, bound, per_pair):
    """|D - exact| <= R on every adversarial case and on the lattice at three dtypes.

    Scored against `exact_sq`, scaled integers, which no float touches.  One escape
    withdraws the package -- and it must be measured on these corners, because random
    normal data at d=384 sits sqrt(d)/d of the way to the worst case and would make this
    an implementation-bug test instead of a test of the theorem.
    """
    corpora = [c for c in adversarial() if not c.expect_reason]
    corpora += [
        exact_lattice(dtype=dt, delta=1024 if dt == np.float16 else 65536, seed=s)
        for dt, s in ((np.float16, 11), (np.float32, 12), (np.float64, 13))
    ]
    checked, enclosed = 0, 0
    for c in corpora:
        try:
            e = enclose_scores(c.X, c.Q, kernel=kernel, bound=bound, per_pair=per_pair)
        except Refusal:
            continue
        enclosed += 1
        bits = 1074 if c.X.dtype == np.float64 else 149
        for i in range(c.m):
            lo, hi = e.interval(i)
            for j in range(c.n):
                s = exact_sq(c.Q[i], c.X[j], bits=bits) / (1 << (2 * bits))
                assert lo[j] <= s <= hi[j], (c.name, i, j, s, lo[j], hi[j])
                checked += 1
    # 0 escapes is the only passing result; the counts are here so a corpus that quietly
    # started refusing everywhere cannot make this test pass by checking nothing
    assert enclosed >= 6 and checked >= 80, (enclosed, checked)


# -- the input layer ----------------------------------------------------------------------------


def test_a_bare_score_array_is_a_usage_error_naming_both_producers():
    with pytest.raises(TypeError) as e:
        as_points(np.zeros(32), "scores")
    msg = str(e.value)
    assert "certified_topk" in msg and "topk_determined" in msg
    # and it is not a refusal: nothing here is in the catalogue
    assert not isinstance(e.value, Refusal)


def test_as_points_widens_integers_and_rejects_the_rest():
    assert as_points([[1, 2], [3, 4]]).dtype == np.float64
    with pytest.raises(TypeError):
        as_points(np.zeros((2, 2), dtype=np.complex128))
    with pytest.raises(ValueError):
        as_points(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        as_points(np.zeros((0, 4)))


def test_load_reads_npy_and_npz_and_will_not_guess(tmp_path):
    A = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(tmp_path / "a.npy", A)
    assert np.array_equal(load(tmp_path / "a.npy"), A)

    np.savez(tmp_path / "one.npz", corpus=A)
    assert np.array_equal(load(tmp_path / "one.npz"), A)

    np.savez(tmp_path / "two.npz", corpus=A, queries=A[:1])
    with pytest.raises(ValueError) as e:
        load(tmp_path / "two.npz")
    assert "corpus" in str(e.value) and "queries" in str(e.value)
    assert np.array_equal(load(tmp_path / "two.npz", key="queries"), A[:1])
    with pytest.raises(ValueError):
        load(tmp_path / "two.npz", key="nope")
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.npy")

    np.save(tmp_path / "scores.npy", np.zeros(9, dtype=np.float32))
    with pytest.raises(TypeError):
        load(tmp_path / "scores.npy")


def test_load_names_a_corrupt_file_instead_of_crashing(tmp_path):
    """A stranger's file, not a stranger's code: every one of these is a `ValueError`
    naming the path, never the bare exception type `numpy.load` happens to raise inside.

    Reproduced by construction, not guessed: a zero-byte file raises `EOFError` (numpy's
    own message for "no data left"), garbage bytes with a `.npz` extension raise
    `zipfile.BadZipFile` ("not a zip file"), and a zip whose member CRC-32 fails --
    checked lazily, when the array is pulled out of the archive, not when the file is
    opened -- raises `zipfile.BadZipFile` again from a different call site.
    """
    empty = tmp_path / "empty.npy"
    empty.write_bytes(b"")
    with pytest.raises(ValueError) as e:
        load(empty)
    assert "empty.npy" in str(e.value)

    not_a_zip = tmp_path / "garbage.npz"
    not_a_zip.write_bytes(b"PK\x03\x04 zip magic, but everything after it is garbage")
    with pytest.raises(ValueError) as e:
        load(not_a_zip)
    assert "garbage.npz" in str(e.value)

    import zipfile

    bad_crc = tmp_path / "bad_crc.npz"
    import io as _io

    import numpy.lib.format as fmt

    buf = _io.BytesIO()
    fmt.write_array(buf, np.zeros((20, 20), dtype=np.float64))
    payload = buf.getvalue()
    with zipfile.ZipFile(bad_crc, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("arr_0.npy", payload)
    raw = bytearray(bad_crc.read_bytes())
    mid = len(raw) // 2
    for i in range(mid, mid + 20):
        raw[i] ^= 0xFF
    bad_crc.write_bytes(bytes(raw))
    with pytest.raises(ValueError) as e:
        load(bad_crc)
    assert "bad_crc.npz" in str(e.value)


def test_load_handles_unicode_and_spaces_in_the_path(tmp_path):
    """Windows and POSIX both accept these bytes in a filename; the loader must too."""
    A = np.arange(12, dtype=np.float32).reshape(3, 4)
    p = tmp_path / "café corpus ☃.npy"
    np.save(p, A)
    assert np.array_equal(load(p), A)
    assert np.array_equal(load(str(p)), A)


def test_exact_lattice_refuses_an_oversized_prescribed_distance_instead_of_hanging():
    """`_four_squares`'s search cost is not bounded by sqrt(S) in the worst case (see
    `MAX_LATTICE_S`'s docstring); a caller-supplied `delta` large enough to reach that
    regime must be a fast `ValueError`, not a multi-minute search.
    """
    with pytest.raises(ValueError) as e:
        exact_lattice(delta=10**15, dtype=np.float64)
    assert "MAX_LATTICE_S" in str(e.value)
    # and the documented schedule -- the only range anything in this repository uses --
    # is nowhere near the cap
    assert max(DELTA_SCHEDULE) < 2**30


# -- C3: the download, skipped when the machine cannot draw it ------------------------------------


def test_mnist_when_cached():
    try:
        c = mnist(n=200, m=8, k=10)
    except CorpusUnavailable as e:
        pytest.skip(f"mnist unavailable: {e}")
    assert c.X.shape == (200, 784) and c.Q.shape == (8, 784) and c.X.dtype == np.float32
    # raw pixel scale, not normalised: this is where float16 loses and the reason the
    # corpus is not scaled to [0, 1]
    n2 = (c.X.astype(np.float64) ** 2).sum(1)
    assert n2.max() > 1e6
    assert mnist(n=200, m=8, dtype=np.float16).headroom() > float(np.finfo(np.float16).max)


def test_sift1m_when_cached():
    """The only corpus whose `truth` was computed outside this repository.

    Skipped when the 516 MB base is not cached. The two assertions that matter are the
    alignment ones: the published ground truth is indexed by position in the query file, so
    a loader that sampled or permuted the queries would break the only external control in
    the repository while still returning a plausible corpus.
    """
    import os

    if os.environ.get("SEPARATRIX_NO_DOWNLOAD"):
        pytest.skip("SEPARATRIX_NO_DOWNLOAD is set")
    try:
        c = sift1m(m=4, k=10)
    except CorpusUnavailable as e:
        pytest.skip(f"sift1m unavailable: {e}")
    assert c.X.shape == (1_000_000, 128) and c.Q.shape == (4, 128)
    assert c.X.dtype == np.float32 and c.truth.shape == (4, 10)
    head = c.X[:1000].astype(np.float64)
    assert np.array_equal(head, np.floor(head)), "SIFT components are integers 0..255"
    # the published neighbour of query 0 is its nearest in exact integer arithmetic
    q = c.Q[0].astype(np.int64)
    d0 = ((c.X[c.truth[0, 0]].astype(np.int64) - q) ** 2).sum()
    for j in c.truth[0, 1:]:
        assert d0 <= ((c.X[j].astype(np.int64) - q) ** 2).sum()
    # float16 fails the range precondition on these bytes, as it does on MNIST
    assert sift1m(m=4, dtype=np.float16).headroom() > float(np.finfo(np.float16).max)
