"""The four corpora, and the one function that turns a file into the object the core eats.

Shipped inside the package rather than kept as a test helper, because two of the four need
no network and carry the tables that must always reproduce.  `bench.py` and the test suite
draw from the same constructors, so the benchmark's ground truth cannot drift from the
suite's.

    C1  exact_lattice   integer coordinates: the exact top-k is known BEFORE any float runs
    C2  adversarial     the demo frames, the range cases, the underflow cases.  No network
    C3  mnist           5,000 x 784, raw 0..255 pixel scale, downloaded on demand
    C4  scifact         BEIR SciFact + all-MiniLM-L6-v2 at d=384, downloaded and encoded

One return shape for all four: `Corpus`.  C1 is the only one carrying `truth`, because it
is the only one whose answer is known by construction rather than by an oracle; the others
leave it None and let `separatrix.exact` be the oracle where an oracle is affordable.

C3 and C4 raise `CorpusUnavailable` when the download or the optional encoder is missing.
That is not a refusal and not a usage error -- it is "this machine cannot draw this corpus"
-- so tests skip on it and `bench.py` names in its own output which corpora it could not
load.  No number in README.md may depend on either.

Self-check:  python -m separatrix.corpus
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import isqrt
from pathlib import Path

import numpy as np

from .verdict import BOUND_VACUOUS, RANGE_UNSAFE

CACHE = Path(__file__).resolve().parent.parent / ".donotcommit"

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

# The delta schedule of the exact lattice: the exact integer margin placed at the rank-k
# frontier.  0 is an exact tie, where no correct answer exists and every configuration
# must return NOT CERTIFIED (EXACT_TIE) rather than a certificate.
DELTA_SCHEDULE = (0, 1, 2) + tuple(2**j for j in range(2, 21))


class CorpusUnavailable(RuntimeError):
    """This machine cannot draw this corpus: no cache, no network, or no encoder.

    Deliberately not a `Refusal` and not a usage error.  A refusal is a statement about
    the caller's data and a usage error is a statement about the caller's code; this is a
    statement about the machine, and it must not reach either catalogue.
    """


# -- the one return shape -------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """(X, Q, k) plus whatever is known about the answer, and how it is known.

    `truth` is the exact top-k of every query row, ascending in exact score, or None when
    the answer is not known by construction.  `expect_reason` is a refusal code that the
    *construction* forces -- not a prediction and not a measurement: the fp16 range case
    overflows by arithmetic that is checked in `_demo`, and the d=1023 fp16 case is
    vacuous by the gamma algebra.  Cases whose outcome depends on the bound's tightness
    carry "" and are scored, never asserted.
    """

    name: str
    X: np.ndarray  # (n, d)
    Q: np.ndarray  # (m, d), same dtype as X
    k: int
    truth: np.ndarray | None = None  # (m, k) int64
    note: str = ""
    expect_reason: str = ""

    def __post_init__(self) -> None:
        if self.X.ndim != 2 or self.Q.ndim != 2:
            raise ValueError(f"{self.name}: X {self.X.shape} and Q {self.Q.shape} must be 2-D")
        if self.X.shape[1] != self.Q.shape[1]:
            raise ValueError(f"{self.name}: X has d={self.X.shape[1]}, Q has d={self.Q.shape[1]}")
        if self.X.dtype != self.Q.dtype:
            raise TypeError(f"{self.name}: X is {self.X.dtype} and Q is {self.Q.dtype}")
        if not (0 < self.k < self.X.shape[0]):
            raise ValueError(f"{self.name}: k must satisfy 0 < k < n, got {self.k}")
        if self.truth is not None and self.truth.shape != (self.Q.shape[0], self.k):
            raise ValueError(f"{self.name}: truth must be (m, k) = {(self.Q.shape[0], self.k)}")

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def m(self) -> int:
        return int(self.Q.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def dtype(self) -> str:
        return self.X.dtype.name

    def headroom(self) -> float:
        """max_i (||q_i|| + max_j||x_j||)^2, in float64.  What P2 compares to finfo.max."""
        x64 = self.X.astype(np.float64, copy=False)
        q64 = self.Q.astype(np.float64, copy=False)
        xn = float(np.sqrt(np.einsum("ij,ij->i", x64, x64)).max())
        qn = np.sqrt(np.einsum("ij,ij->i", q64, q64))
        return float(((qn + xn) ** 2).max())

    def __str__(self) -> str:
        head = (
            f"{self.name}: n={self.n} m={self.m} d={self.d} k={self.k} {self.dtype}  "
            f"headroom {self.headroom():.3e} of {float(np.finfo(self.X.dtype).max):.3e}"
        )
        if self.expect_reason:
            head += f"  expects {self.expect_reason}"
        return head + (f"\n  {self.note}" if self.note else "")


# -- the input layer --------------------------------------------------------------------------


def as_points(A, name: str = "X") -> np.ndarray:
    """A caller's array as an (n, d) float array, or a usage error saying why not.

    A bare 1-D score array is the one input that must fail loudly.  An enclosure is a
    function of the computation, not of its result: a float vector that already went
    through somebody's matmul carries no error bound and never can.  So this raises
    `TypeError` naming the two producers, exit class 3 -- never a Verdict.  Mixing the two
    would mean the JSON an agent consumes cannot tell "go re-observe" from "go fix your
    call", and the first thing a new user sees would be a refusal banner for a typo.
    """
    A = np.asarray(A)
    if A.dtype.kind in "iub":
        A = A.astype(np.float64)
    if A.dtype.kind != "f":
        raise TypeError(f"{name} must be a real float array, got dtype {A.dtype}")
    if A.ndim == 1:
        raise TypeError(
            f"{name} is 1-D with {A.size} entries, which reads as a score array. separatrix "
            f"encloses a computation, not its output: a bare score vector carries no error "
            f"bound. Pass the points -- certified_topk(X, Q, k) -- or bring your own "
            f"enclosure to decide.topk_determined(D, R, k)."
        )
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, d), got shape {A.shape}")
    if A.size == 0:
        raise ValueError(f"{name} is empty, shape {A.shape}")
    return np.ascontiguousarray(A)


def load(path, name: str | None = None, key: str | None = None) -> np.ndarray:
    """One `.npy` or `.npz` file as an (n, d) float array.

    `.npz` with one array takes it; with several, `key` names it and the error lists the
    keys rather than guessing.  Guessing which of three arrays in an archive is the corpus
    is exactly the class of silent wrong answer this package exists to attack.
    """
    p = Path(path)
    label = name or p.name
    if not p.exists():
        raise FileNotFoundError(f"{label}: {p} does not exist")
    obj = np.load(p, allow_pickle=False)
    if isinstance(obj, np.ndarray):
        return as_points(obj, label)
    with obj as z:
        keys = list(z.files)
        if key is not None:
            if key not in keys:
                raise ValueError(f"{label}: {p} has no array {key!r}; it has {keys}")
            return as_points(z[key], f"{label}[{key}]")
        if len(keys) != 1:
            raise ValueError(
                f"{label}: {p} holds {len(keys)} arrays {keys}; name one with key= rather "
                f"than letting the loader guess which is the corpus"
            )
        return as_points(z[keys[0]], f"{label}[{keys[0]}]")


# -- C1: the exact lattice ------------------------------------------------------------------------


def _two_squares(n: int) -> tuple[int, int] | None:
    a = isqrt(n)
    while a >= 0:
        b = isqrt(n - a * a)
        if b * b == n - a * a:
            return a, b
        a -= 1
    return None


def _four_squares(S: int) -> tuple[int, int, int, int]:
    """Lagrange: every non-negative integer is a sum of four squares.  Greedy, then search.

    The greedy first term leaves a remainder of at most 2*isqrt(S), so the inner search is
    over a few dozen values however large S is.  Exhaustive over (a, b), so termination is
    Lagrange's theorem and not a heuristic.
    """
    if S < 0:
        raise ValueError(f"a squared distance cannot be {S}")
    a = isqrt(S)
    while a >= 0:
        r = S - a * a
        b = isqrt(r)
        while b >= 0:
            two = _two_squares(r - b * b)
            if two is not None:
                return (a, b) + two
            b -= 1
        a -= 1
    raise AssertionError(f"no four-square decomposition of {S}")  # pragma: no cover


# Defaults per dtype.  `cap` bounds the random shared coordinates; the tuning slots are
# bounded by the prescribed distances instead, and the headroom check is what actually
# guards the dtype.  float16's schedule stops at 2**12 because the headroom does.
_LATTICE = {
    "float16": dict(cap=8, base=256, spread=1024, max_delta=2**12),
    "float32": dict(cap=2048, base=4096, spread=65536, max_delta=2**20),
    "float64": dict(cap=2048, base=4096, spread=65536, max_delta=2**20),
}


def exact_lattice(
    *,
    n: int = 16,
    d: int = 32,
    k: int = 4,
    delta: int = 65536,
    dtype=np.float32,
    seed: int = 0,
    cap: int | None = None,
    base: int | None = None,
    spread: int | None = None,
) -> Corpus:
    """Truth by construction: integer coordinates, an exactly prescribed frontier margin.

    The query is one point `q` with large random integer coordinates.  Every corpus point
    is `q + v_j`, so the exact squared distance is `||v_j||^2` -- an integer, known before
    any float runs -- while `||q||` and `||x_j||` are both large.  That is the adversary in
    one line: the Gram identity subtracts two quantities of size `||q||^2` to produce a
    difference of size `||v||^2`, and the enclosure width scales with the former.

    `v_j` is a sparse +-1 vector over the first `d-4` coordinates plus a four-square
    decomposition over the last four, so `||v_j||^2` hits any prescribed non-negative
    integer exactly.  The schedule puts the k smallest at `base + i*spread` and the rest at
    `base + (k-1)*spread + delta + i*spread`, so the rank-k frontier sits at exactly
    `delta` and every other pair is `spread` apart.

    `delta = 0` is an exact tie: no correct answer exists, and every configuration must
    return NOT CERTIFIED (EXACT_TIE) rather than a certificate.  Sweeping `delta` over
    `DELTA_SCHEDULE` gives the smallest margin this package certifies, against the largest
    margin at which the float decision was actually wrong.

    One query per draw, deliberately: a shared X cannot carry a prescribed frontier for two
    different queries, so the sample comes from seeds rather than from rows.
    """
    dt = np.dtype(dtype)
    if dt.name not in _LATTICE:
        raise ValueError(f"exact_lattice supports {sorted(_LATTICE)}, got {dt.name}")
    cfg = _LATTICE[dt.name]
    cap = int(cfg["cap"] if cap is None else cap)
    base = int(cfg["base"] if base is None else base)
    spread = int(cfg["spread"] if spread is None else spread)
    if d < 8:
        raise ValueError(f"exact_lattice needs d >= 8 (four tuning slots), got {d}")
    if not (0 < k < n):
        raise ValueError(f"k must satisfy 0 < k < n, got k={k}, n={n}")
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")

    rng = np.random.default_rng(seed)
    q = rng.integers(-cap, cap + 1, size=d, dtype=np.int64)
    q[-4:] = 0

    S = [base + i * spread for i in range(k)]
    S += [base + (k - 1) * spread + delta + i * spread for i in range(n - k)]

    X = np.empty((n, d), dtype=np.int64)
    order = rng.permutation(n)  # so the answer is not the first k indices
    for rank, j in enumerate(order):
        v = np.zeros(d, dtype=np.int64)
        nz = rng.choice(d - 4, size=min(4, d - 4), replace=False)
        v[nz] = rng.choice(np.array([-1, 1]), size=len(nz))
        rest = S[rank] - int(v @ v)
        if rest < 0:
            raise ValueError(
                f"base={base} is below the sparse part's norm {int(v @ v)}; raise base"
            )
        v[-4:] = _four_squares(rest)
        X[j] = q + v

    truth = np.array([[int(order[i]) for i in range(k)]], dtype=np.int64)

    Xf = X.astype(dt)
    qf = q.astype(dt).reshape(1, d)
    if not (np.array_equal(Xf.astype(np.int64), X) and np.array_equal(qf.astype(np.int64)[0], q)):
        raise ValueError(
            f"the lattice does not survive {dt.name}: max |coordinate| is "
            f"{int(np.abs(X).max())}, above the {dt.name} integer grid. Lower cap, delta or d."
        )
    c = Corpus(
        name=f"exact_lattice(d={d},k={k},delta={delta},{dt.name},seed={seed})",
        X=Xf,
        Q=qf,
        k=k,
        truth=truth,
        note=(
            f"integer coordinates, |x_i| <= {int(np.abs(X).max())}; exact frontier margin "
            f"{delta}; every other pair {spread} apart"
        ),
    )
    limit = float(np.finfo(dt).max)
    if c.headroom() > limit:
        raise ValueError(
            f"this lattice overflows {dt.name}: headroom {c.headroom():.3e} against "
            f"{limit:.3e}. Lower delta (float16 stops at {cfg['max_delta']}), cap or d."
        )
    return c


# -- C2: the adversarial corpus --------------------------------------------------------------------


def _pair(x, y, dtype, name, k=1, note="", expect_reason="", q=None):
    """A two-point corpus: X = [x, y], the query is x unless given.  The smallest shape
    that has a rank-k boundary at all."""
    dt = np.dtype(dtype)
    X = np.array([x, y], dtype=dt)
    Q = np.array([x if q is None else q], dtype=dt)
    return Corpus(name=name, X=X, Q=Q, k=k, note=note, expect_reason=expect_reason)


def adversarial(name: str | None = None) -> list[Corpus] | Corpus:
    """Every case where a baseline loses, as corpora.  No network, tiny, and the one place
    the containment test is a test of the theorem rather than of the implementation.

    Random normal data at d=384 sits `sqrt(d)/d` of the way to the worst case and never
    approaches it, so a containment test on random data is an implementation-bug test.
    These are the corners.
    """
    out: list[Corpus] = []

    # Frame 1, in float64.  At float32 the two points ARE the same stored vector
    # (np.float32(1e6 + 1e-6) == np.float32(1e6)), so 0.0 is the correct squared distance
    # for the data as it stands; the frame's claim holds only where the inputs differ.
    # Here the Gram identity returns exactly 0.0 for both points while the direct sum
    # returns 1.0000152290447206e-12.  Changing the formula is the fix; changing the
    # precision is not.
    out.append(
        _pair(
            [1e6, 0.0],
            [1e6 + 1e-6, 0.0],
            np.float64,
            "cancellation_f64",
            note="the Gram identity returns 0.0 for both points in float64; the direct sum "
            "separates them at 1.0000152290447206e-12",
        )
    )
    # The honest companion: at float32 the pair collapses to one stored vector, so the
    # exact scores really are equal and the answer is a tie in the data, not in the
    # arithmetic.  Verified in _demo rather than asserted here.
    out.append(
        _pair(
            [1e6, 0.0],
            [1e6 + 1e-6, 0.0],
            np.float32,
            "cancellation_f32",
            note="float32(1e6 + 1e-6) == float32(1e6): the two rows are one stored vector "
            "and the exact scores are equal. A tie in the data, not in the arithmetic",
        )
    )

    # P2, the must-fix, at MNIST's shape and pixel scale: ||x||^2 reaches ~5.6e6 against
    # float16's 65504.  The Gram intermediates overflow, the differences become non-finite,
    # and an enclosure formed downstream of that reports a clean pass over garbage.
    rng = np.random.default_rng(7)
    px = rng.integers(0, 256, size=(8, 784)).astype(np.float16)
    out.append(
        Corpus(
            name="fp16_range_784",
            X=px,
            Q=px[:1].copy(),
            k=3,
            note="raw 0..255 pixels at d=784: ||x||^2 reaches 1.789e7 against float16's "
            "6.55e4. Real MNIST's median is 5.48e6 on the 5,000-row draw at seed 0",
            expect_reason=RANGE_UNSAFE,
        )
    )
    # Each norm is finite in float16 and their sum is not: the naive "is the input finite"
    # check passes, and must not be what guards this.
    out.append(
        _pair(
            [200.0, 0.0],
            [0.0, 200.0],
            np.float16,
            "partial_overflow_f16",
            note="||x||^2 = ||y||^2 = 40000, both finite in float16; the Gram intermediate "
            "(||q||+||x||)^2 = 1.6e5 is not",
            expect_reason=RANGE_UNSAFE,
        )
    )

    # P3: (d+2)u > 1/2 makes gamma_n carry no information, and past n*u > 1 it goes
    # negative, which would certify everything.  float16 crosses at d = 1023.
    small = rng.standard_normal((4, 1023)).astype(np.float16) * np.float16(0.01)
    out.append(
        Corpus(
            name="vacuous_f16_d1023",
            X=small,
            Q=small[:1].copy(),
            k=2,
            note="d=1023 float16: (d+2)u = 0.5005 > 1/2, one dimension past the domain of "
            "the a-priori bound. d=1022 is the last legal width",
            expect_reason=BOUND_VACUOUS,
        )
    )
    legal = small[:, :1022].copy()
    out.append(
        Corpus(
            name="vacuous_edge_f16_d1022",
            X=legal,
            Q=legal[:1].copy(),
            k=2,
            note="d=1022 float16: (d+2)u = 0.5 exactly, the last width the bound covers. "
            "The bound here is 100% relative and refuses; that is legal, not a failure",
        )
    )

    # Underflow: the relative error model fl(ab) = ab(1+delta) is false when the product
    # lands subnormal, and eta is what covers it.  Both regimes measured in enclose.eta.
    tiny32 = (rng.standard_normal((8, 8)) * 1e-25).astype(np.float32)
    out.append(
        Corpus(
            name="subnormal_f32",
            X=tiny32,
            Q=tiny32[:1].copy(),
            k=2,
            note="components ~1e-25: every product lands subnormal in float32, where the "
            "relative model does not hold and only the additive eta bounds the error",
        )
    )
    tiny16 = (rng.standard_normal((8, 8)) * 3e-4).astype(np.float16)
    out.append(
        Corpus(
            name="subnormal_f16",
            X=tiny16,
            Q=tiny16[:1].copy(),
            k=2,
            note="components ~3e-4, below sqrt(smallest_normal) = 0.0078 for float16 -- "
            "reachable by ordinary data: a unit-norm d=16384 embedding has components "
            "of exactly that size",
        )
    )

    # Maximum cancellation in the cross term: alternating signs make sum_i |x_i y_i| far
    # larger than |<x, y>|, which is exactly the 20.7x understatement the naive bound
    # would have taken.
    sign = np.where(np.arange(384) % 2 == 0, 1.0, -1.0)
    alt = (rng.standard_normal((8, 384)) * sign).astype(np.float32)
    alt[1] = -alt[0]
    out.append(
        Corpus(
            name="alternating_f32",
            X=alt,
            Q=alt[:1].copy(),
            k=3,
            note="alternating signs at d=384: sum_i |x_i y_i| runs ~20x |<x, y>|, which is "
            "the gap between Higham's bound and the one that drops the absolute value",
        )
    )

    # 1e16 of dynamic range inside one vector: the small coordinate is below the rounding
    # of the large one, so the reduction order decides the low bits of the score.
    out.append(
        _pair(
            [1e8, 1e-8],
            [1e8, 2e-8],
            np.float64,
            "dynamic_range_f64",
            note="1e16 of dynamic range in d=2: the second coordinate is below the rounding "
            "of the first, so the decision lives entirely in the discarded bits",
        )
    )

    if name is None:
        return out
    for c in out:
        if c.name == name:
            return c
    raise ValueError(f"no adversarial case {name!r}; have {[c.name for c in out]}")


# -- C3, C4: the downloads -----------------------------------------------------------------


def _cached(filename: str, url: str) -> Path:
    """The cache path, downloading once.  Never a silent partial file."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / filename
    if p.exists() and p.stat().st_size > 0:
        return p
    if os.environ.get("SEPARATRIX_NO_DOWNLOAD"):
        raise CorpusUnavailable(f"{filename} is not cached and SEPARATRIX_NO_DOWNLOAD is set")
    import urllib.error
    import urllib.request

    tmp = p.with_suffix(p.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:  # noqa: S310
            f.write(r.read())
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        tmp.unlink(missing_ok=True)
        raise CorpusUnavailable(f"could not download {url}: {e}") from e
    tmp.replace(p)
    return p


def mnist(*, n: int = 5000, m: int = 300, k: int = 10, dtype=np.float32, seed: int = 0) -> Corpus:
    """MNIST at raw 0..255 pixel scale, 5,000 x 784 by default.

    Not normalised, deliberately: the raw scale is where float16 overflows, which is the
    case this package exists for.  Measured on the n=5,000 m=300 draw at seed 0:
    ||x||^2 has median 5.482e6 and max 1.489e7, and the float16 Gram headroom reaches
    5.427e7 against float16's 6.55e4.  Normalising would hide the only real-data range
    failure in the repository.

    Downloaded once to `.donotcommit/mnist.npz` (~11 MB).  Raises `CorpusUnavailable` with
    no cache and no network.
    """
    p = _cached("mnist.npz", MNIST_URL)
    with np.load(p, allow_pickle=False) as z:
        imgs = z["x_train"]
    flat = imgs.reshape(len(imgs), -1)
    if len(flat) < n + m:
        raise CorpusUnavailable(f"{p} holds {len(flat)} images, need {n + m}")
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(flat))[: n + m]
    dt = np.dtype(dtype)
    X = flat[pick[:n]].astype(dt)
    Q = flat[pick[n:]].astype(dt)
    return Corpus(
        name=f"mnist(n={n},m={m},{dt.name})",
        X=X,
        Q=Q,
        k=k,
        note="raw 0..255 pixels, d=784, not normalised; max ||x||^2 "
        f"{float((X.astype(np.float64) ** 2).sum(1).max()):.3e}",
    )


def scifact(
    *, n: int | None = None, m: int = 300, k: int = 10, dtype=np.float32, seed: int = 0
) -> Corpus:
    """BEIR SciFact encoded with all-MiniLM-L6-v2: 5,183 normalised rows at d=384.

    Needs `datasets` and `sentence-transformers` (the `bench` extra) on the first call
    only; the embeddings are cached to `.donotcommit/scifact-minilm.npy` and every later
    call is numpy alone.

    The prediction is on record before the run and it is not the optimistic one: with
    ||x|| = ||q|| = 1 the cheap per-pair radius is `gamma_386 * 4` for EVERY pair, so on
    this corpus `bound="cheap"` carries zero per-pair information and the shuffled-
    enclosure control is a mathematical no-op.  Only `bound="tight"` says anything here.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "scifact-minilm.npy"
    if cache.exists():
        emb = np.load(cache)
    else:
        if os.environ.get("SEPARATRIX_NO_DOWNLOAD"):
            raise CorpusUnavailable("scifact is not cached and SEPARATRIX_NO_DOWNLOAD is set")
        try:
            from datasets import load_dataset
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise CorpusUnavailable(
                f"scifact needs the bench extra (datasets, sentence-transformers): {e}"
            ) from e
        try:
            ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
            texts = [f"{t} {x}".strip() for t, x in zip(ds["title"], ds["text"])]
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception as e:  # network, hub, or model load
            raise CorpusUnavailable(f"could not build scifact: {e}") from e
        emb = np.asarray(emb, dtype=np.float32)
        np.save(cache, emb)
    dt = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(emb))
    m = min(m, len(emb) // 2)
    n = len(emb) - m if n is None else min(n, len(emb) - m)
    X = emb[pick[:n]].astype(dt)
    Q = emb[pick[len(emb) - m :]].astype(dt)
    return Corpus(
        name=f"scifact(n={n},m={m},{dt.name})",
        X=X,
        Q=Q,
        k=k,
        note="all-MiniLM-L6-v2, normalised rows at d=384; the cheap per-pair radius is "
        "constant here by construction, so only bound='tight' carries per-pair information",
    )


# -- self-check ---------------------------------------------------------------------------------


def _demo() -> None:
    from .exact import exact_sq

    # C1: the exact distances are the prescribed integers, before any float runs
    for dtype, delta in (("float32", 65536), ("float16", 1024), ("float64", 0)):
        c = exact_lattice(dtype=np.dtype(dtype), delta=delta, seed=3)
        q = c.Q[0]
        vals = sorted(exact_sq(q, c.X[j], bits=0) for j in range(c.n))
        assert all(float(v).is_integer() for v in vals), dtype
        assert vals[c.k] - vals[c.k - 1] == delta, (dtype, delta, vals[c.k] - vals[c.k - 1])
        # and the truth is the k smallest, by construction
        order = sorted(range(c.n), key=lambda j: exact_sq(q, c.X[j], bits=0))
        assert set(order[: c.k]) == set(c.truth[0].tolist()), dtype
        assert c.headroom() < float(np.finfo(c.X.dtype).max)

    # the schedule reaches an exact tie and a margin wider than any enclosure here
    assert DELTA_SCHEDULE[0] == 0 and DELTA_SCHEDULE[-1] == 2**20

    # C2: the two float32 rows of cancellation_f32 really are one stored vector
    c32 = adversarial("cancellation_f32")
    assert np.float32(1e6 + 1e-6) == np.float32(1e6)
    assert np.array_equal(c32.X[0], c32.X[1])
    # and in float64 the Gram identity returns 0.0 for a pair the direct sum separates
    c64 = adversarial("cancellation_f64")
    x, y = c64.X[0].astype(np.float64), c64.X[1].astype(np.float64)
    assert x @ x + y @ y - 2.0 * (x @ y) == 0.0
    assert float((x - y) @ (x - y)) == 1.0000152290447206e-12

    # the range case overflows float16 by arithmetic, not by assertion
    rng_case = adversarial("fp16_range_784")
    assert rng_case.headroom() > float(np.finfo(np.float16).max)
    assert adversarial("partial_overflow_f16").headroom() > float(np.finfo(np.float16).max)
    # while each individual norm is finite there
    po = adversarial("partial_overflow_f16")
    assert np.isfinite((po.X.astype(np.float64) ** 2).sum(1)).all()

    # P3's edge, by the gamma algebra rather than by a table
    u16 = float(np.finfo(np.float16).eps) / 2
    assert (1023 + 2) * u16 > 0.5 >= (1022 + 2) * u16

    # every case is constructible and prints
    cases = adversarial()
    assert len({c.name for c in cases}) == len(cases)
    for c in cases:
        assert str(c)

    # the input layer: a bare score array is a usage error naming the two producers
    try:
        as_points(np.zeros(10), "scores")
    except TypeError as e:
        assert "topk_determined" in str(e) and "certified_topk" in str(e)
    else:  # pragma: no cover
        raise AssertionError("a bare score array was accepted as points")
    assert as_points(np.zeros((3, 4), dtype=np.int32)).dtype == np.float64

    print(f"corpus: ok ({len(cases)} adversarial cases)")


if __name__ == "__main__":
    _demo()
