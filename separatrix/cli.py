"""separatrix/cli.py -- ``separatrix check`` / ``demo`` / ``probe``.

    separatrix check --corpus X.npy --queries Q.npy --k 10
                     [--kernel gram|direct] [--bound cheap|tight] [--per-pair]
                     [--escalate] [--upcast] [--ordered] [--largest]
                     [--max-refused FRAC] [--json] [--max-report 20]
    separatrix demo  [--frame cancellation|batch|preview]
    separatrix probe

    exit 0 CERTIFIED   1 NOT CERTIFIED   2 REFUSED   3 usage   4 CERTIFIED_UPCAST

Every field this file prints comes off the ``Verdict``, the ``Enclosure`` or the
``Frontier``.  The only strings the CLI owns are the labels, so no example block anywhere
in this repository is hand-typed and none of them can drift from what the code returns.

``--max-refused`` runs the *gate*, not the certificate: it uses ``harness.within_budget``,
the same comparison ``separatrix.harness.gate`` uses, so the CLI in CI and the pytest gate
cannot disagree about whether one corpus passes.  When it is given, a REFUSED run whose
undetermined fraction is inside the budget exits 0 and the gate line says so.  It does not
touch exit 1: an exact tie is not a budget question.

``probe`` is a diagnostic and never an input to a certificate.

The demo corpus lives here rather than in a module of its own, and ``bench.py`` imports
``near_duplicate_corpus`` from here: one definition, two callers.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

import numpy as np

from .decide import rows_determined, topk_set
from .enclose import (
    BOUNDS,
    KERNELS,
    canary,
    enclose_scores,
    gamma,
    gram_scores,
    unit_roundoff,
)
from .exact import BITS_DOUBLE, exact_sq_float
from .harness import within_budget
from .verdict import (
    BOUNDARY_UNDETERMINED,
    CERTIFIED,
    EXIT,
    EXIT_USAGE,
    GRAM_CANCELLATION,
    NEXT_ACTION,
    REFUSED,
    Refusal,
    Verdict,
)

WIDTH = 78
SCHEMA = "separatrix/1"


# --------------------------------------------------------------------------------------
# the demo corpus: near-duplicate pairs in a corpus that is otherwise easy
# --------------------------------------------------------------------------------------


def near_duplicate_corpus(
    n=400, d=64, m=60, seed=3, dups=40, jitter=3e-7, dtype=np.float32
):
    """A unit-norm corpus where ``dups`` points have a near-duplicate at ``jitter``.

    The mixture is the point.  A corpus of nothing but near-duplicates refuses every row,
    and then "the disagreements are among the refused rows" is true by construction and
    measures nothing.  Here most rows are determined and the frame has something to lose.

    Returns ``(X, Q)`` contiguous in ``dtype``.
    """
    dups = min(dups, n // 2)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:dups] = X[dups : 2 * dups] + jitter * rng.standard_normal((dups, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = rng.standard_normal((m, d))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return np.ascontiguousarray(X, dtype), np.ascontiguousarray(Q, dtype)


def permuted_evaluation(X, Q, work_dtype, seed=7):
    """The same formula on the same stored bytes, with the reduction reordered.

    Permuting the ``d`` columns of X and Q identically leaves every exact score unchanged
    and changes every rounding.  It is the numerically distinct second evaluation the
    certificate's corollary is about, and unlike ``torch.cdist`` at two batch sizes it
    needs nothing installed.
    """
    p = np.random.default_rng(seed).permutation(X.shape[1])
    Xp = np.ascontiguousarray(X[:, p])
    Qp = np.ascontiguousarray(Q[:, p])
    return gram_scores(Xp, Qp, work_dtype)


# --------------------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------------------


def _ascii_only(stream=None) -> bool:
    enc = getattr(stream or sys.stdout, "encoding", None) or "ascii"
    try:
        "━│".encode(enc)
    except (UnicodeEncodeError, LookupError):
        return True
    return False


def rules(ascii_only: bool):
    """(heavy, light) horizontal rules.  ASCII when stdout cannot carry box drawing."""
    return ("=" * WIDTH, "-" * WIDTH) if ascii_only else ("━" * WIDTH, "─" * WIDTH)


def _wrap(label: str, value: str):
    body = textwrap.wrap(value, WIDTH - 15) or [""]
    return [f"  {label:<11} {body[0]}"] + [f"  {'':<11} {b}" for b in body[1:]]


def block(v: Verdict, *, ascii_only: bool = False, max_report: int = 20) -> str:
    """The verdict block, inside rules.  Generated from the Verdict, never hand-typed.

    The refusal half of the block is the product: a typed code, the named boundary with
    both intervals, the exact deficit, and the next action.  A refusal that names only a
    status is a status; a refusal that names the pair is something to act on.
    """
    heavy, light = rules(ascii_only)
    head = v.status if not v.reason else f"{v.status} ({v.reason})"
    counts = (
        f"{v.n_queries - v.n_refused}/{v.n_queries} determined" if v.n_queries else ""
    )
    rows = [heavy, f"  {head}{counts.rjust(max(2, WIDTH - 4 - len(head)))}".rstrip(), heavy]

    if v.detail:
        rows += _wrap("detail", v.detail)
    cfg = []
    if v.kernel:
        cfg.append(f"kernel {v.kernel}")
    if v.bound:
        cfg.append(f"bound {v.bound}")
    cfg.append("per-pair" if v.per_pair else "per-row")
    if v.k:
        cfg.append(f"k {v.k}" + ("  largest" if v.largest else ""))
    if cfg:
        rows += _wrap("computed", "   ".join(cfg))
    if v.dtype_in:
        used = v.dtype_used or v.dtype_in
        line = f"{v.dtype_in} -> {used}" if used != v.dtype_in else v.dtype_in
        rows += _wrap("dtype", line)
    if v.canary:
        rows += _wrap("canary", v.canary)
    rows += _wrap("accumulator", f"assumed {v.accum_assumed} (P5 is not testable)")
    if v.escalated or v.n_escalated:
        rows += _wrap(
            "escalated",
            f"{v.n_escalated} exact scores; the float set "
            + ("differed" if v.float_set_differed else "was unchanged"),
        )

    if v.frontiers:
        rows += [light]
        for f in v.frontiers[:max_report]:
            rows += _wrap("boundary", str(f))
        if len(v.frontiers) > max_report:
            rows += _wrap(
                "",
                f"{len(v.frontiers) - max_report} further boundaries not shown "
                f"(--max-report {max_report})",
            )
    if v.next_action:
        rows += [light]
        rows += _wrap("next", v.next_action)
    rows += [heavy]
    return "\n".join(rows)


def payload(v: Verdict, max_report: int = 20) -> dict:
    """The JSON body.  ``schema`` is the first key, and every value comes off the Verdict."""
    return {
        "schema": SCHEMA,
        "status": v.status,
        "exit": v.exit_code,
        "reason": v.reason,
        "detail": v.detail,
        "next_action": v.next_action,
        "kernel": v.kernel,
        "bound": v.bound,
        "per_pair": v.per_pair,
        "k": v.k,
        "largest": v.largest,
        "n_queries": v.n_queries,
        "n_refused": v.n_refused,
        "dtype_in": v.dtype_in,
        "dtype_used": v.dtype_used,
        "accum_assumed": v.accum_assumed,
        "canary": v.canary,
        "escalated": v.escalated,
        "n_escalated": v.n_escalated,
        "float_set_differed": v.float_set_differed,
        "frontiers": [
            {
                "row": f.row,
                "inside": f.inside,
                "outside": f.outside,
                "inside_interval": [f.inside_lo, f.inside_hi],
                "outside_interval": [f.outside_lo, f.outside_hi],
                "gap": f.gap,
                "width": f.width,
                "deficit": f.deficit,
            }
            for f in v.frontiers[:max_report]
        ],
        "frontiers_omitted": max(0, len(v.frontiers) - max_report),
    }


def refusal_verdict(e: Refusal, **kw) -> Verdict:
    """A precondition Refusal, as the Verdict the CLI prints.  One conversion, one place."""
    status = "NOT CERTIFIED" if e.exit_code == 1 else REFUSED
    return Verdict(
        status=status,
        reason=e.reason,
        detail=str(e).split(": ", 1)[-1] if ": " in str(e) else "",
        next_action=e.next_action,
        **kw,
    )


# --------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------


def load_array(path: str) -> np.ndarray:
    """One 2-D float array out of a ``.npy`` or a single-array ``.npz``.

    Routed through ``corpus.load`` so the package has exactly one input door: a bare 1-D
    score array then gets the same TypeError naming both producers whether it arrived
    through the CLI or through ``certified_topk``.  ``allow_int=False`` because the
    certificate is about a float dtype the caller's pipeline actually ran.
    """
    from .corpus import load

    return load(path, name=path, allow_int=False)


def _decisions():
    """``separatrix.certified_topk``, or None when the decision surface is absent."""
    try:
        from . import api
    except ImportError:
        return None
    return getattr(api, "certified_topk", None)


def cmd_check(args, out) -> int:
    topk = _decisions()
    if topk is None:
        print(
            "the decision surface (separatrix.api) is not present in this checkout, so "
            "`check` has nothing to call. `separatrix demo` and `separatrix probe` run on "
            "the core modules and need no such import.",
            file=out,
        )
        return EXIT_USAGE
    try:
        X = load_array(args.corpus)
        Q = load_array(args.queries)
    except (OSError, TypeError, ValueError) as e:
        # All three are exit class 3: a bad path, a bare score array, a bad shape.  None
        # of them is a Verdict, because none of them is a statement about the data.
        print(str(e), file=out)
        return EXIT_USAGE

    try:
        _, v = topk(
            X,
            Q,
            k=args.k,
            largest=args.largest,
            kernel=args.kernel,
            bound=args.bound,
            per_pair=args.per_pair,
            ordered=args.ordered,
            escalate=args.escalate,
            upcast=args.upcast,
        )
    except Refusal as e:  # a precondition, raised before any score was read
        v = refusal_verdict(
            e, k=args.k, kernel=args.kernel, bound=args.bound, dtype_in=X.dtype.name
        )
    except (TypeError, ValueError) as e:  # a usage error, exit class 3, never a verdict
        print(str(e), file=out)
        return EXIT_USAGE

    code = v.exit_code
    gate_line = ""
    if args.max_refused is not None and v.n_queries:
        observed = v.n_refused / v.n_queries
        ok = within_budget(observed, args.max_refused)
        gate_line = (
            f"gate: {v.n_refused}/{v.n_queries} undetermined ({observed:.4f}) against a "
            f"budget of {args.max_refused:.4f} -> {'pass' if ok else 'fail'}"
        )
        if code == EXIT[REFUSED]:
            code = 0 if ok else EXIT[REFUSED]

    if args.json:
        body = payload(v, args.max_report)
        body["corpus"] = args.corpus
        body["queries"] = args.queries
        body["exit"] = code
        if gate_line:
            body["gate"] = gate_line
        print(json.dumps(body, indent=2), file=out)
    else:
        print(block(v, ascii_only=_ascii_only(out), max_report=args.max_report), file=out)
        if gate_line:
            print("  " + gate_line, file=out)
    return code


# --------------------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------------------


def frame_cancellation(out, ascii_only=False) -> None:
    """Two points 1e-6 apart at coordinate magnitude 1e6, in float64.

    At float32 the two points *are* the same stored vector -- ``np.float32(1e6 + 1e-6) ==
    np.float32(1e6)`` -- so 0.0 is the correct squared distance for the data as it stands
    and there is nothing to refuse.  The frame therefore runs in float64, where the inputs
    are distinct and the Gram identity still returns exactly 0.0.

    Changing the formula is the fix.  Changing the precision is not.
    """
    heavy, light = rules(ascii_only)
    X = np.array([[1e6, 0.0], [1e6 + 1e-6, 0.0]], dtype=np.float64)
    Q = X[:1]
    g = enclose_scores(X, Q, kernel="gram")
    dr = enclose_scores(X, Q, kernel="direct")
    from .decide import topk_determined

    fg = topk_determined(g.D[0], g.R[0], 1)
    fd = topk_determined(dr.D[0], dr.R[0], 1)

    print(heavy, file=out)
    print("  frame 1 -- cancellation.  x = (1e6, 0)   y = (1e6 + 1e-6, 0)   float64", file=out)
    print(heavy, file=out)
    print(f"  float32 collapses the pair:  np.float32(1e6+1e-6) == np.float32(1e6) is "
          f"{bool(np.float32(1e6 + 1e-6) == np.float32(1e6))}", file=out)
    print(f"  gram identity  ||x||^2+||y||^2-2<x,y>   = {float(g.D[0, 1])!r}", file=out)
    print(f"  direct sum     sum_l (x_l - y_l)^2      = {float(dr.D[0, 1])!r}", file=out)
    print(f"  exact, scaled integers                  = "
          f"{exact_sq_float(X[0], X[1], BITS_DOUBLE)!r}", file=out)
    print(light, file=out)
    for name, e, f in (("gram", g, fg), ("direct", dr, fd)):
        status = "determined" if f is None else "not determined"
        print(f"  {name:<7} radius {float(np.ravel(e.R)[0]):.6e}   nearest-neighbour "
              f"decision {status}", file=out)
    v = Verdict(
        status=REFUSED,
        reason=GRAM_CANCELLATION if fd is None else BOUNDARY_UNDETERMINED,
        next_action=NEXT_ACTION[GRAM_CANCELLATION if fd is None else BOUNDARY_UNDETERMINED],
        kernel=g.kernel,
        bound=g.bound,
        per_pair=g.per_pair,
        k=1,
        n_queries=1,
        n_refused=1,
        dtype_in=g.dtype_in,
        dtype_used=g.dtype_used,
        canary=g.canary,
        frontiers=(fg,) if fg is not None else (),
    )
    print(block(v, ascii_only=ascii_only), file=out)


def frame_batch(out, ascii_only=False) -> None:
    """Two numerically distinct evaluations of one formula on one set of stored bytes.

    ``torch.cdist`` at batch 32 against batch 64 is the frame as designed and it needs
    torch.  With torch absent the stand-in permutes the reduction order instead, which is
    the same class of change -- and it is **labelled a stand-in**, because a stand-in
    quoted as the measurement is how a benchmark becomes a story.

    separatrix does not fix this frame and does not claim to.  ``gamma_{d+2}`` is
    deliberately invariant to reduction order: the enclosure says the boundary was never
    determined, and two runs agreeing would not have made it so.
    """
    heavy, _ = rules(ascii_only)
    X, Q = near_duplicate_corpus()
    k = 5
    print(heavy, file=out)
    print("  frame 2 -- one formula, one set of stored bytes, two evaluations", file=out)
    print(heavy, file=out)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  torch is not installed, so the batch-32-against-64 frame is skipped.",
              file=out)
        print("  STAND-IN, not the torch measurement: the reduction order is permuted "
              "instead.", file=out)
    D1 = gram_scores(X, Q, np.float32)
    D2 = permuted_evaluation(X, Q, np.float32)
    dis = [
        i
        for i in range(Q.shape[0])
        if set(topk_set(D1[i], k).tolist()) != set(topk_set(D2[i], k).tolist())
    ]
    print(f"  {len(dis)} of {Q.shape[0]} top-{k} sets differ between the two evaluations: "
          f"{dis}", file=out)
    print("  Both are correct roundings of the same formula. Neither run is the answer.",
          file=out)


def frame_preview(out, ascii_only=False) -> int:
    """Name the undetermined boundaries BEFORE either run, then run both.

    This is the frame no a-posteriori method can do.  A float64 diff, two batch sizes and
    stochastic rounding all find what *moved*; none of them can say a boundary was never
    determined on a row where the two runs happened to agree.

    The assertion at the end is the one that matters, and it is a soundness check rather
    than a demonstration: a row that disagreed between two evaluations and was **not**
    named would be a counterexample to the certificate.
    """
    heavy, light = rules(ascii_only)
    X, Q = near_duplicate_corpus()
    k = 5
    e = enclose_scores(X, Q, kernel="gram")
    fronts = rows_determined(e.D, e.R, k)
    named = [i for i, f in enumerate(fronts) if f is not None]

    D2 = permuted_evaluation(X, Q, np.float32)
    dis = [
        i
        for i in range(Q.shape[0])
        if set(topk_set(e.D[i], k).tolist()) != set(topk_set(D2[i], k).tolist())
    ]
    outside = sorted(set(dis) - set(named))

    print(heavy, file=out)
    print("  frame 3 -- named before either run", file=out)
    print(heavy, file=out)
    print(f"  corpus {X.shape[0]} x {X.shape[1]} {X.dtype.name}, {Q.shape[0]} queries, "
          f"k = {k}, 40 near-duplicate pairs at 3e-07", file=out)
    print(f"  named undetermined, from one evaluation:  {len(named)} of {Q.shape[0]}  "
          f"{named}", file=out)
    print(f"  actually differed, over two evaluations:   {len(dis)} of {Q.shape[0]}  "
          f"{dis}", file=out)
    print(f"  differed and NOT named:                    {len(outside)}  {outside}", file=out)
    print(light, file=out)
    if outside:
        print("  A row that differed and was not named contradicts the certificate.", file=out)
        return EXIT[REFUSED]
    print("  Every row that differed was named in advance, from a single run.", file=out)
    print("  The rows named and not differing are the a-priori bound's pessimism,", file=out)
    print("  and `escalate=True` decides which of them were determined after all.", file=out)
    if named:
        print(light, file=out)
        for line in _wrap("boundary", str(fronts[named[0]])):
            print(line, file=out)
    return 0


FRAMES = {"cancellation": frame_cancellation, "batch": frame_batch, "preview": frame_preview}


def cmd_demo(args, out) -> int:
    ascii_only = _ascii_only(out)
    names = [args.frame] if args.frame else list(FRAMES)
    code = 0
    for i, name in enumerate(names):
        if i:
            print(file=out)
        got = FRAMES[name](out, ascii_only)
        code = code or (got or 0)
    return code


# --------------------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------------------


def cmd_probe(args, out) -> int:
    """A diagnostic.  Never an input to a certificate."""
    heavy, light = rules(_ascii_only(out))
    print(heavy, file=out)
    print("  separatrix probe -- what this machine's arithmetic is, measured now", file=out)
    print(heavy, file=out)
    print(f"  {'dtype':<10}{'u = eps/2':>14}{'canary':>10}   gamma_{{d+2}} at d = 384 / 784",
          file=out)
    for name in ("float16", "float32", "float64"):
        dt = np.dtype(name)
        gs = []
        for d in (384, 784):
            try:
                gs.append(f"{gamma(d + 2, dt):.6e}")
            except Refusal as e:
                gs.append(e.reason)
        ok = "clean" if canary(dt) else "coarse"
        print(f"  {name:<10}{unit_roundoff(dt):>14.6e}{ok:>10}   {gs[0]}  {gs[1]}", file=out)
    print(light, file=out)
    print("  P4 tests the multiplier only. P5, the accumulator width, is not testable by",
          file=out)
    print("  any probe tried and travels on every verdict as an assumption.", file=out)
    try:
        import torch

        print(light, file=out)
        print(f"  torch {torch.__version__}", file=out)
        print(f"    allow_tf32 (matmul)          {torch.backends.cuda.matmul.allow_tf32}",
              file=out)
        print(f"    allow_tf32 (cudnn)           {torch.backends.cudnn.allow_tf32}", file=out)
        print(f"    float32_matmul_precision     {torch.get_float32_matmul_precision()}",
              file=out)
        print("    a flag that is not 'highest' raises REDUCED_PRECISION_ARITHMETIC on the",
              file=out)
        print("    canary; the flags are read at call time, never cached.", file=out)
    except ImportError:
        print("  torch is not installed; its precision flags are not read.", file=out)
    print(heavy, file=out)
    return 0


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="separatrix",
        description="Which of your top-k decisions were determined by your data, and "
        "which by the rounding of the kernel you called.",
    )
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("check", help="certify a top-k over a corpus and a query set")
    c.add_argument("--corpus", required=True, metavar="X.npy")
    c.add_argument("--queries", required=True, metavar="Q.npy")
    c.add_argument("--k", type=int, default=10)
    c.add_argument("--largest", action="store_true", help="top-k largest, not smallest")
    c.add_argument("--kernel", choices=KERNELS, default="gram")
    c.add_argument("--bound", choices=BOUNDS, default="cheap")
    c.add_argument("--per-pair", action="store_true", dest="per_pair")
    c.add_argument("--ordered", action="store_true")
    c.add_argument("--escalate", action="store_true")
    c.add_argument("--upcast", action="store_true")
    c.add_argument("--max-refused", type=float, default=None, dest="max_refused")
    c.add_argument("--max-report", type=int, default=20, dest="max_report")
    c.add_argument("--json", action="store_true")

    d = sub.add_parser("demo", help="the three frames, no network and no download")
    d.add_argument("--frame", choices=sorted(FRAMES), default=None)

    sub.add_parser("probe", help="this machine's unit roundoff, canary and torch flags")
    return p


def main(argv=None, out=None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = out or sys.stdout
    if args.cmd is None:
        parser.print_help(out)
        return EXIT_USAGE
    if args.cmd == "check":
        if args.max_refused is not None and not 0.0 <= args.max_refused <= 1.0:
            print(f"--max-refused is a fraction in [0, 1]; got {args.max_refused}", file=out)
            return EXIT_USAGE
        return cmd_check(args, out)
    if args.cmd == "demo":
        return cmd_demo(args, out)
    return cmd_probe(args, out)


# --------------------------------------------------------------------------------------
# self-check:  python -m separatrix.cli
# --------------------------------------------------------------------------------------


def _demo() -> None:
    import io

    # the corpus is reproducible, and the frame has something to lose: most rows are
    # determined, so "the disagreements are among the refused" is not true by construction
    X, Q = near_duplicate_corpus()
    X2, Q2 = near_duplicate_corpus()
    assert np.array_equal(X, X2) and np.array_equal(Q, Q2), "the demo corpus is not seeded"
    e = enclose_scores(X, Q, kernel="gram")
    named = [i for i, f in enumerate(rows_determined(e.D, e.R, 5)) if f is not None]
    assert 0 < len(named) < Q.shape[0], (len(named), Q.shape[0])

    # every printed field comes off the Verdict, and the block fits its rules
    v = Verdict(
        status=REFUSED,
        reason=BOUNDARY_UNDETERMINED,
        next_action=NEXT_ACTION[BOUNDARY_UNDETERMINED],
        kernel="gram",
        bound="cheap",
        k=10,
        n_queries=300,
        n_refused=19,
        dtype_in="float32",
        dtype_used="float32",
        canary="numpy/float32 clean",
    )
    txt = block(v, ascii_only=True)
    assert "=" * WIDTH in txt and "281/300 determined" in txt
    assert NEXT_ACTION[BOUNDARY_UNDETERMINED][:24] in txt
    assert max(len(line) for line in txt.splitlines()) <= WIDTH + 1

    body = payload(v)
    assert list(body)[0] == "schema" and body["schema"] == SCHEMA
    assert json.loads(json.dumps(body))["exit"] == 2

    # a certificate prints without a next action and exits 0
    ok = Verdict(status=CERTIFIED, kernel="gram", bound="cheap", k=10, n_queries=300)
    assert "next" not in block(ok, ascii_only=True)
    assert payload(ok)["exit"] == 0

    # the three frames run with no network, no download and no torch
    buf = io.StringIO()
    assert main(["demo"], out=buf) == 0
    text = buf.getvalue()
    assert "frame 1" in text and "frame 2" in text and "frame 3" in text
    assert "differed and NOT named:                    0" in text

    buf = io.StringIO()
    assert main(["probe"], out=buf) == 0
    assert "float32" in buf.getvalue() and "accumulator" in buf.getvalue()

    buf = io.StringIO()
    assert main([], out=buf) == EXIT_USAGE

    print("cli: ok")


if __name__ == "__main__":
    _demo()
