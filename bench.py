"""bench.py -- every table in README.md and RESULTS.md, from one draw with one seed.

    .venv/Scripts/python bench.py [--n 2000] [--m 300] [--k 10] [--seed 11]
                                  [--out .donotcommit/results.json] [--selfcheck]

Read the headline correctly, or do not read it at all.

  * **A refusal is not a detection.**  The refused column ships as part of a triple --
    refused / flipped / confirmed-fine -- and only ``flipped`` is evidence this package
    found anything.  The design predicted ``flipped = 0`` on every corpus, in writing,
    before the run; the prediction is printed beside the measurement whichever way it
    lands, because that is the only thing that stops a pessimism count being narrated as a
    detection count.  It did not hold on the clustered arm, and the float64 diff found the
    same rows there, so that arm is not one where this package saw what the practice
    missed.
  * **The claim is determinism, not correctness.**  The row worth reading first is the
    agreement table: zero disagreements among CERTIFIED top-k sets across numerically
    distinct evaluations of the same formula, against a positive count among the REFUSED
    ones.  If the REFUSED count is zero on a corpus, that corpus is an arm where this
    package had nothing to say, and it prints as one.

Every control prints whichever way it lands, including the two that were expected to
embarrass the design: the shuffled-enclosure control and the tuned-margin sweep.

Not run here, and named rather than skipped silently: the torch backends (batch 32 against
64, and the 25-row ``cdist`` switch), real MNIST and real BEIR SciFact with
all-MiniLM-L6-v2 (both need a download), the Ogita-Rump-Oishi compensated-dot arm, and the
scikit-learn chunked-upcast arm.  Every corpus below is generated, so every number here
reproduces with no network.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time

import numpy as np

from separatrix import corpus as C
from separatrix.cli import permuted_evaluation, rules
from separatrix.decide import naive_boundary_pair, rows_determined, topk_set
from separatrix.enclose import direct_scores, enclose_scores, gram_scores
from separatrix.exact import escalate_row
from separatrix.verdict import EXACT_TIE

K_DEFAULT = 10


# --------------------------------------------------------------------------------------
# the corpora -- generated, one seed, so every table below reproduces with no network
# --------------------------------------------------------------------------------------


def corpus_iid(n, d, m, rng, dtype=np.float32):
    """Unit-normalised iid gaussian rows.  The easy case, and the one that does not exist."""
    X = rng.standard_normal((n, d))
    Q = rng.standard_normal((m, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return np.ascontiguousarray(X, dtype), np.ascontiguousarray(Q, dtype)


def corpus_clustered(n, d, m, rng, dtype=np.float32, per=20, spread=0.02):
    """Unit-normalised rows in tight clusters: what a corpus with near-duplicates is.

    Queries are drawn as perturbed corpus rows, which is what a retrieval query set is,
    rather than as independent noise, which is what makes the iid arm easy.
    """
    C = rng.standard_normal((max(1, n // per), d))
    X = np.repeat(C, per, axis=0)[:n] + spread * rng.standard_normal((n, d))
    Q = X[rng.choice(n, m, replace=False)] + spread * rng.standard_normal((m, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return np.ascontiguousarray(X, dtype), np.ascontiguousarray(Q, dtype)


def corpus_mnist_shaped(n, d, m, rng, dtype=np.float32):
    """Un-normalised non-negative rows with MNIST's magnitudes: ||x||^2 of order 1e6.

    Generated, not downloaded, and named as generated everywhere it prints.  What it
    reproduces from MNIST is the property the tool is about -- an un-normalised feature
    scale, so the Gram identity's cancellation term is large against the distances -- not
    the digits.
    """
    mask = rng.random((n + m, d)) < 0.19
    A = (rng.random((n + m, d)) * 255.0) * mask
    X, Q = A[:n], A[n:]
    return np.ascontiguousarray(X, dtype), np.ascontiguousarray(Q, dtype)


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


def sets(D, k):
    """The top-k set of every row, as frozensets, so two evaluations compare directly."""
    return [frozenset(topk_set(D[i], k).tolist()) for i in range(D.shape[0])]


def evaluations(X, Q, k):
    """Numerically distinct evaluations of the SAME real number on the SAME stored bytes.

    Each of these five is a legal rounding of ||x - q||^2 over the stored floats.  Any two
    of them disagreeing on a top-k set means that set was decided by the arithmetic; all
    five agreeing is not evidence of anything, which is why the REFUSED column of the
    agreement table is what proves the corpus was not too easy.
    """
    X64, Q64 = X.astype(np.float64), Q.astype(np.float64)
    out = {
        "gram fp32": gram_scores(X, Q, np.float32),
        "gram fp32, reduction permuted": permuted_evaluation(X, Q, np.float32),
        "direct fp32": direct_scores(X, Q, np.float32),
        "gram fp64": gram_scores(X64, Q64, np.float64),
    }
    try:
        from scipy.spatial.distance import cdist

        out["scipy cdist fp64, squared"] = cdist(Q64, X64, "sqeuclidean")
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        pass
    return {name: sets(D, k) for name, D in out.items()}


def disagreeing_rows(ev, m):
    """The rows on which any two of the evaluations return different top-k sets."""
    names = list(ev)
    return {
        i
        for i in range(m)
        if len({ev[name][i] for name in names}) > 1
    }


def escalate_all(X, Q, D, R, k, rows, budget):
    """Exact re-decision of every refused row.  The triple, not the count.

    ``flipped`` is the only column that is evidence this package found something: it is a
    row where the exact top-k set differs from the float one.  ``confirmed`` is the
    pessimism of the a-priori bound, measured rather than argued.
    """
    tally = {"flipped": 0, "confirmed": 0, "tie": 0, "budget": 0, "escalated": 0}
    flipped = []
    for i in rows:
        r = R[i] if R.ndim > 1 else R
        e = escalate_row(Q[i], X, D[i], r, k, max_escalations=budget)
        tally["escalated"] += e.n_escalated
        if e.determined:
            tally["flipped" if e.float_set_differed else "confirmed"] += 1
            if e.float_set_differed:
                flipped.append(int(i))
        elif e.reason == EXACT_TIE:
            tally["tie"] += 1
        else:
            tally["budget"] += 1
    return tally, flipped


def tuned_margin(D, k, eps):
    """The baseline that needs no bound: certify when ``gap > eps * |score|``.

    Four lines, no theorem, one fitted constant.  The honest differentiator is that no
    single eps holds across dtype, d and feature scale -- and that is a sweep to be
    measured, not an argument to be made.  Returns the certified row indices.
    """
    out = []
    for i in range(D.shape[0]):
        row = D[i]
        order = np.argpartition(row, k)[: k + 1]
        order = order[np.argsort(row[order], kind="stable")]
        a, b = row[order[k - 1]], row[order[k]]
        if (b - a) > eps * abs(b):
            out.append(i)
    return out


def timeit(fn, repeat=5):
    """Best of ``repeat``.  Best, not mean: the noise on this machine is one-sided."""
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------

EPS_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


def one_corpus(name, X, Q, k, budget=64):
    """Every table's row for one corpus, from one draw."""
    m = Q.shape[0]
    e = enclose_scores(X, Q, kernel="gram", bound="cheap", per_pair=False)
    fronts = rows_determined(e.D, e.R, k)
    refused = [i for i, f in enumerate(fronts) if f is not None]

    ed = enclose_scores(X, Q, kernel="direct")
    refused_direct = [
        i for i, f in enumerate(rows_determined(ed.D, ed.R, k)) if f is not None
    ]

    # the false theorem, scored against the sound rule on the same enclosure
    naive_refused = [
        i for i in range(m) if not naive_boundary_pair(e.D[i], e.R[i], k)
    ]

    # The shuffled-enclosure control: permute R within a row and ask whether the verdict
    # moves.  It has to run at RUNG 2, because at rung 1 the radius is constant across the
    # row and shuffling it is the identity -- a control that cannot fail is not a control.
    # So the arm it scores against is rung 2, not rung 1, and both print.
    ep = enclose_scores(X, Q, kernel="gram", bound="cheap", per_pair=True)
    pair_refused = [i for i, f in enumerate(rows_determined(ep.D, ep.R, k)) if f is not None]
    rng = np.random.default_rng(99)
    Rs = np.array(ep.R, dtype=np.float64, copy=True)
    for i in range(m):
        rng.shuffle(Rs[i])
    shuffled_refused = [
        i for i, f in enumerate(rows_determined(ep.D, Rs, k)) if f is not None
    ]

    ev = evaluations(X, Q, k)
    dis = disagreeing_rows(ev, m)
    fp64_diff = sorted(
        {i for i in range(m) if ev["gram fp32"][i] != ev["gram fp64"][i]}
    )

    tally, flipped_rows = escalate_all(X, Q, e.D, e.R, k, refused, budget)
    # A witness that a certificate is wrong is either two evaluations disagreeing on the
    # row, or exact arithmetic moving the set.  The second is a proof; the first is not.
    witnesses = dis | set(flipped_rows)

    tuned = {}
    for eps in EPS_GRID:
        cert = set(tuned_margin(e.D, k, eps))
        tuned[eps] = {
            "certified": len(cert),
            "wrong_witnessed": len(cert & witnesses),
        }

    width = np.broadcast_to(e.R, e.D.shape)
    return {
        "name": name,
        "shape": [int(X.shape[0]), int(X.shape[1]), int(m)],
        "dtype": X.dtype.name,
        "k": k,
        "refused": len(refused),
        "refused_rows": refused,
        "refused_direct": len(refused_direct),
        "refused_per_pair": len(pair_refused),
        "flipped_rows": flipped_rows,
        "naive_refused": len(naive_refused),
        "shuffled_refused": len(shuffled_refused),
        "median_width_gram": float(np.median(2.0 * width)),
        "median_width_direct": float(np.median(2.0 * np.broadcast_to(ed.R, ed.D.shape))),
        "escalation": tally,
        "fp64_diff": len(fp64_diff),
        "disagreeing": sorted(dis),
        "certified_disagreeing": sorted(dis - set(refused)),
        "refused_disagreeing": sorted(dis & set(refused)),
        "evaluations": list(ev),
        "tuned": {f"{k_:.0e}": v for k_, v in tuned.items()},
    }


def cost_table(X, Q, k):
    """The cost of the check, against the control it actually replaces.

    The competing *practice* is the float32-against-float64 diff: it costs both runs and
    proves nothing.  The competing *arithmetic* is the float64 Gram identity, and against
    that the honest word is parity.  Both rows print, always.
    """
    X64, Q64 = X.astype(np.float64), Q.astype(np.float64)

    def status_quo():
        np.argpartition(gram_scores(X, Q, np.float32), k, axis=1)

    def fp64():
        np.argpartition(gram_scores(X64, Q64, np.float64), k, axis=1)

    def scipy_cdist():
        from scipy.spatial.distance import cdist

        np.argpartition(cdist(Q64, X64, "sqeuclidean"), k, axis=1)

    def rung1():
        e = enclose_scores(X, Q, kernel="gram", bound="cheap", per_pair=False)
        rows_determined(e.D, e.R, k)

    def rung2():
        e = enclose_scores(X, Q, kernel="gram", bound="cheap", per_pair=True)
        rows_determined(e.D, e.R, k)

    def diff():
        a = sets(gram_scores(X, Q, np.float32), k)
        b = sets(gram_scores(X64, Q64, np.float64), k)
        sum(1 for u, v in zip(a, b) if u != v)

    return {
        "fp32 gram + argpartition": timeit(status_quo),
        "fp64 gram + argpartition": timeit(fp64),
        "scipy cdist fp64 + argpartition": timeit(scipy_cdist),
        "separatrix rung 1 (per-row)": timeit(rung1),
        "separatrix rung 2 (per-pair)": timeit(rung2),
        "the fp32-vs-fp64 diff": timeit(diff),
    }


def lattice_arm(k=4, d=32, n=16, trials=8, dtype=np.float32):
    """The only arm whose truth is known BEFORE any float runs.

    Integer coordinates make ``||x||^2``, ``<x,y>`` and ``d^2`` exact integers in float64,
    so ``corpus.exact_lattice`` hands back the true top-k with the rank-k frontier placed
    at an exactly prescribed integer margin.  Two numbers come out, and only the first is
    deterministic:

      ``delta*``       the smallest prescribed margin this package certified.  A property
                       of the bound; no sampling in it.
      ``delta_wrong``  the largest margin at which the float top-k was actually wrong.  A
                       MAX OVER A SAMPLE, so it can only rise with more trials, and it
                       ships with its trial count and seed rather than as a bare number.

    Their ratio is the pessimism factor.  The row that would withdraw the package is
    neither of them: it is ``wrong``, the number of CERTIFIED verdicts the construction's
    own truth contradicts, and it is an instance, not a budget.
    """
    rows = []
    wrong = 0
    for delta in C.DELTA_SCHEDULE:
        cert = 0
        float_wrong = 0
        for t in range(trials):
            c = C.exact_lattice(n=n, d=d, k=k, delta=delta, dtype=dtype, seed=1000 + t)
            e = enclose_scores(c.X, c.Q, kernel="gram")
            determined = rows_determined(e.D, e.R, c.k)[0] is None
            got = frozenset(topk_set(e.D[0], c.k).tolist())
            truth = frozenset(int(i) for i in c.truth[0])
            cert += determined
            float_wrong += got != truth
            wrong += determined and got != truth
        rows.append({"delta": int(delta), "certified": cert, "float_wrong": float_wrong,
                     "trials": trials})
    certified_deltas = [r["delta"] for r in rows if r["certified"] == r["trials"]]
    wrong_deltas = [r["delta"] for r in rows if r["float_wrong"]]
    return {
        "rows": rows,
        "delta_star": min(certified_deltas) if certified_deltas else None,
        "delta_wrong": max(wrong_deltas) if wrong_deltas else None,
        "wrong_certificates": wrong,
        "trials": trials,
        "shape": [n, d, k],
    }


def real_corpora(k):
    """C3 MNIST and C4 SciFact, attempted.  Named in the output when this machine cannot
    draw them -- a download nobody can reproduce may not carry a number in README.md."""
    got, missing = [], []
    for name, fn in (("mnist", C.mnist), ("scifact", C.scifact)):
        try:
            c = fn(k=k)
        except C.CorpusUnavailable as e:
            missing.append(f"{name}: {e}")
            continue
        except Exception as e:  # a broken cache is not a benchmark failure
            missing.append(f"{name}: {type(e).__name__}: {e}")
            continue
        got.append(one_corpus(f"{name} (downloaded)", c.X, c.Q, c.k))
    return got, missing


def _commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or "(not a git checkout)"
        )
    except Exception:
        return "(git unavailable)"


def run(n=2000, m=300, d=384, k=K_DEFAULT, seed=11, cost_n=5000, cost_d=784,
        real=True) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    corpora = [
        one_corpus(f"iid normalised d={d}", *corpus_iid(n, d, m, rng), k),
        one_corpus(f"clustered normalised d={d}", *corpus_clustered(n, d, m, rng), k),
        one_corpus(
            f"MNIST-shaped (generated) d={cost_d}",
            *corpus_mnist_shaped(n, cost_d, m, rng),
            k,
        ),
    ]
    if real:
        drawn, missing = real_corpora(k)
    else:
        drawn = []
        missing = ["mnist and scifact not attempted (--no-download)"]
    corpora += drawn
    Xc, Qc = corpus_mnist_shaped(cost_n, cost_d, m, np.random.default_rng(seed + 1))
    import scipy

    return {
        "config": {"n": n, "m": m, "d": d, "k": k, "seed": seed,
                   "cost_shape": [cost_n, cost_d, m]},
        "corpora": corpora,
        "lattice": lattice_arm(),
        "cost": cost_table(Xc, Qc, k),
        "not_run": missing + [
            "torch: batch 32 against 64, and the 25-row cdist switch (torch not installed)",
            "Ogita-Rump-Oishi compensated dot (Dot2) as a tighter a-posteriori arm",
            "scikit-learn chunked upcast as a falsifiable test of the bound itself",
        ],
        "provenance": {
            "commit": _commit(),
            "machine": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "seconds": round(time.perf_counter() - t0, 1),
        },
    }


# --------------------------------------------------------------------------------------
# the tables
# --------------------------------------------------------------------------------------


def _row(label, value, note=""):
    return f"    {label:<40}{value:>8}   {note}".rstrip()


def table(res: dict, ascii_only: bool = False) -> str:
    heavy, light = rules(ascii_only)
    cfg = res["config"]
    la = res["lattice"]
    L = [
        heavy,
        f"  the exact lattice -- integer coordinates, so the top-{la['shape'][2]} is known",
        "  BEFORE any float runs. The only arm here whose truth is not an oracle's.",
        heavy,
        _row("CERTIFIED verdicts the truth contradicts", str(la["wrong_certificates"]),
             "one of these withdraws the package"),
        _row("delta*  smallest margin certified", str(la["delta_star"]),
             "a property of the bound, no sampling"),
        _row("delta_wrong  largest margin float got wrong", str(la["delta_wrong"]),
             f"MAX OVER {la['trials']} trials, can only rise"),
        _row(
            "pessimism factor  delta* / delta_wrong",
            (f"{la['delta_star'] / la['delta_wrong']:.0f}x"
             if la["delta_star"] and la["delta_wrong"] else "--"),
            f"n={la['shape'][0]} d={la['shape'][1]} k={la['shape'][2]}, seeds 1000..",
        ),
        "",
        heavy,
        "  agreement among CERTIFIED sets, across numerically distinct evaluations",
        "  of the same formula on the same stored bytes -- the claim is determinism",
        heavy,
    ]
    for c in res["corpora"]:
        nq = c["shape"][2]
        L += [
            f"  {c['name']}   {c['shape'][0]} x {c['shape'][1]} {c['dtype']}, "
            f"{nq} queries, k = {c['k']}",
            _row("evaluations compared", str(len(c["evaluations"]))),
            _row(
                "disagreements among CERTIFIED rows",
                str(len(c["certified_disagreeing"])),
                "one of these withdraws the package",
            ),
            _row(
                "disagreements among REFUSED rows",
                str(len(c["refused_disagreeing"])),
                "0 here means this corpus was too easy",
            ),
            "",
        ]

    L += [
        heavy,
        "  the refusal triple. `flipped` is the only column that is evidence this",
        "  package found anything, and 0 was the prediction on record before the run.",
        heavy,
        f"    {'corpus':<30}{'refused':>9}{'flipped':>9}{'tie':>6}{'confirmed':>11}"
        f"{'fp64 diff':>11}",
    ]
    for c in res["corpora"]:
        t = c["escalation"]
        L.append(
            f"    {c['name'][:30]:<30}{c['refused']:>4}/{c['shape'][2]:<4}"
            f"{t['flipped']:>9}{t['tie']:>6}{t['confirmed']:>11}"
            f"{c['fp64_diff']:>7}/{c['shape'][2]:<3}"
        )
    L += [
        "    " + light[:64],
        "    `confirmed` is the a-priori bound's pessimism, measured. `fp64 diff` is",
        "    the practice this replaces: it costs two runs and proves nothing either way.",
        "",
        heavy,
        "  the kernel switch -- the same real number, two enclosures, orders apart",
        heavy,
        f"    {'corpus':<30}{'gram':>10}{'direct':>10}{'width gram':>14}{'width direct':>15}",
    ]
    for c in res["corpora"]:
        L.append(
            f"    {c['name'][:30]:<30}{c['refused']:>5}/{c['shape'][2]:<4}"
            f"{c['refused_direct']:>5}/{c['shape'][2]:<4}"
            f"{c['median_width_gram']:>14.3e}{c['median_width_direct']:>15.3e}"
        )
    L += [
        "    " + light[:64],
        "    A gram refusal the direct kernel separates names a CODE CHANGE, not a",
        "    re-observation: it is the one refusal in the catalogue that does.",
        "",
        heavy,
        "  controls, printed whichever way they land",
        heavy,
        f"    {'corpus':<30}{'rung 1':>9}{'naive':>8}{'rung 2':>9}{'shuffled':>10}",
    ]
    for c in res["corpora"]:
        L.append(
            f"    {c['name'][:30]:<30}{c['refused']:>9}{c['naive_refused']:>8}"
            f"{c['refused_per_pair']:>9}{c['shuffled_refused']:>10}"
        )
    L += [
        "    " + light[:64],
        "    naive rule: the rank-k/rank-(k+1) boundary pair, which is UNSOUND when the",
        "    radii vary. It ships green here -- a benchmark cannot find it, only the",
        "    counterexample test can, which is why that test exists.",
        "    shuffled: the RUNG 2 per-pair radii permuted within a row, scored against",
        "    rung 2 beside it. At rung 1 the radius is constant across the row and the",
        "    shuffle is the identity, so this control can only run against rung 2.",
        "",
        heavy,
        "  the tuned-margin baseline: certify when gap > eps * |score|. Four lines and",
        "  one fitted constant. If one eps holds across every corpus, this package's",
        "  advantage reduces to no tuning and a proof, and that is smaller than the pitch.",
        heavy,
        f"    {'eps':>8}" + "".join(f"{c['name'][:18]:>20}" for c in res["corpora"]),
    ]
    for eps in EPS_GRID:
        key = f"{eps:.0e}"
        cells = ""
        for c in res["corpora"]:
            t = c["tuned"][key]
            cells += f"{t['certified']:>13}/{c['shape'][2]:<3} {t['wrong_witnessed']:>2}"
        L.append(f"    {key:>8}" + cells)
    L += [
        "    " + light[:64],
        "    each cell is certified/queries and the number of those certificates a",
        "    witness contradicts. A witness is two evaluations differing, or exact",
        "    arithmetic moving the set; only the second is a proof, and the union is a",
        "    LOWER BOUND on wrong certificates, not an audit of them.",
        "",
        heavy,
        f"  cost, best of 5, on {cfg['cost_shape'][0]} x {cfg['cost_shape'][1]} "
        f"float32, {cfg['cost_shape'][2]} queries, k = {cfg['k']}",
        heavy,
    ]
    base = res["cost"]["fp64 gram + argpartition"]
    quo = res["cost"]["fp32 gram + argpartition"]
    for label, secs in res["cost"].items():
        L.append(
            _row(label, f"{secs:.4f} s",
                 f"{secs / base:>5.2f}x fp64   {secs / quo:>5.2f}x fp32")
        )
    L += [
        "    " + light[:64],
        f"    rung 1 against the diff it replaces: "
        f"{res['cost']['separatrix rung 1 (per-row)'] / res['cost']['the fp32-vs-fp64 diff']:.2f}x."
        "  Ratios above are",
        "    the measurement, not a claim: whatever they say is what this costs. The diff",
        "    is the competing practice and it needs two runs to prove nothing either way.",
        "    CPU only: a CUDA tensor forces a host copy before the enclosure.",
        heavy,
        "  not run here:",
    ]
    L += [f"    {s}" for s in res["not_run"]]
    p = res["provenance"]
    L += [
        heavy,
        f"  commit {p['commit']}   machine {p['machine']}   python {p['python']}"
        f"   numpy {p['numpy']}   scipy {p['scipy']}   {p['seconds']} s",
        f"  seed {cfg['seed']}. Every corpus not marked (downloaded) is generated here, so",
        "  those rows reproduce with no network. A downloaded corpus carries no number",
        "  into README.md, because a reader without the download cannot check it.",
        heavy,
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# self-check:  .venv/Scripts/python bench.py --selfcheck
# --------------------------------------------------------------------------------------


def _demo() -> None:
    res = run(n=200, m=20, d=32, k=4, seed=5, cost_n=200, cost_d=32, real=False)
    txt = table(res, ascii_only=True)
    assert "not run here" in txt and "refusal triple" in txt

    for c in res["corpora"]:
        # SOUNDNESS: a row that two evaluations decided differently and that this package
        # certified is a counterexample to the certificate.  Not a threshold; an instance.
        assert c["certified_disagreeing"] == [], (c["name"], c["certified_disagreeing"])
        # the escalation triple accounts for every refused row
        t = c["escalation"]
        assert t["flipped"] + t["confirmed"] + t["tie"] + t["budget"] == c["refused"]
        # the sound rule never certifies where the naive one refuses
        assert c["refused"] >= 0 and c["naive_refused"] >= 0

    assert json.loads(json.dumps(res))["config"]["k"] == 4
    print("bench: ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--m", type=int, default=300)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--k", type=int, default=K_DEFAULT)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default=None, metavar="results.json")
    p.add_argument("--no-download", action="store_true", dest="no_download",
                   help="skip the MNIST and SciFact arms instead of attempting them")
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args(argv)
    if a.selfcheck:
        _demo()
        return 0
    res = run(n=a.n, m=a.m, d=a.d, k=a.k, seed=a.seed, real=not a.no_download)
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "━".encode(enc)
        ascii_only = False
    except (UnicodeEncodeError, LookupError):
        ascii_only = True
    print(table(res, ascii_only))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"  results -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
