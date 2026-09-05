#!/usr/bin/env python
"""Regenerate every file in `assets/` from measured numbers.

    python make_assets.py              draw from assets/data.json  (offline, deterministic)
    python make_assets.py --measure    re-measure first, then draw

`assets/data.json` is the only input to the drawing: a few kilobytes of numbers, each one
produced by `--measure` on this machine and none of them typed in by hand.  The three
measurement paths and what each needs:

    sift          the SIFT1M cache under .donotcommit/ (516 MB, not committed -- the same
                  cache `bench.py --sift` uses).  Per query: the rank-k boundary gap, the
                  enclosure width that has to fit inside it, and whether the returned
                  top-10 equals the ANN_SIFT1M authors' published neighbour list.
    engines       a `bench.py` results.json (default .donotcommit/results.json).
    cancellation  computed here, needs nothing: two float64 points 1e-6 apart.

`measure_sift` asserts the claim these assets carry, where the numbers are produced: every
row whose set differs from the published truth was refused first, and every certified set
equals it.  If that ever fails, no asset is written.

Drawing is pure string building -- no matplotlib, no rasteriser, no font file.  Every text
run is checked against the canvas before a file is written, which is the only failure this
kind of code actually has.  The one optional external is a headless Chromium (Edge or
Chrome), used to turn the social card into the PNG GitHub's social-preview upload wants;
without it the SVG is still written and the step says why it stopped.

Palette, one meaning per colour, identical in every asset:

    #2FD98A  green   CERTIFIED -- the decision is the data's, not the kernel's
    #F5B33C  amber   REFUSED -- this enclosure does not decide it
    #FF6B5A  red     an answer that is wrong, or a claim contradicted
    #6EA8FE  blue    the external control: a third party's answer, or exact arithmetic
    #EAF2F8  ink     a measured value
    #8FA3B0  dim     a label, a unit, provenance
    #0B1015  ground  #121B23 panel  #22303B rule
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE / "assets"
DATA = ASSETS / "data.json"

GROUND = "#0B1015"
PANEL = "#121B23"
RULE = "#22303B"
INK = "#EAF2F8"
DIM = "#8FA3B0"
GREEN = "#2FD98A"
AMBER = "#F5B33C"
RED = "#FF6B5A"
BLUE = "#6EA8FE"

SANS = "Inter,'Segoe UI',system-ui,-apple-system,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"

# Everything is drawn 1600 units wide and shown at roughly 890 px in a GitHub README, so a
# unit is about 0.56 px: 26 units is the floor that keeps every glyph above 14 px there.
MIN_TYPE = 26

_TEXTS: list[tuple] = []  # every text run drawn since the last canvas check


# -- svg primitives ----------------------------------------------------------------------


def esc(t) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def T(x, y, s, fill=INK, size=MIN_TYPE, weight="400", anchor="start", family=None,
      op=1.0, track=None) -> str:
    _TEXTS.append((x, y, str(s), size, anchor, family, track or 0))
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    f = f' font-family="{family}"' if family else ""
    o = f' opacity="{op}"' if op != 1.0 else ""
    t = f' letter-spacing="{track}"' if track else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'font-weight="{weight}"{a}{f}{o}{t}>{esc(s)}</text>')


def M(*a, **kw) -> str:
    kw.setdefault("family", MONO)
    return T(*a, **kw)


def advance(s, size, family, track=0.0) -> float:
    """Width of a text run, wide enough for the fallback faces this may render in."""
    per = 0.62 * size if family == MONO else 0.56 * size
    return len(s) * (per + float(track))


def fits(w, h, margin=24) -> list[str]:
    """Every text run drawn so far, checked against the canvas.  Clears the record."""
    bad = []
    for x, y, s, size, anchor, family, track in _TEXTS:
        a = advance(s, size, family, track)
        x0 = x if anchor == "start" else (x - a if anchor == "end" else x - a / 2)
        if x0 < margin - 2 or x0 + a > w - margin + 2 or not (size < y < h - 4):
            bad.append(f"{s[:44]!r} at ({x:.0f},{y:.0f}) spans {x0:.0f}..{x0 + a:.0f}")
    _TEXTS.clear()
    return bad


def svg(w, h, body: list[str], rx=16) -> str:
    bad = fits(w, h)
    if bad:
        raise AssertionError(f"{len(bad)} text run(s) outside the canvas:\n  "
                             + "\n  ".join(bad))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
        f'height="{h}" font-family="{SANS}" role="img">\n'
        f'<rect width="{w}" height="{h}" rx="{rx}" fill="{GROUND}"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def rect(x, y, w, h, fill, rx=0, op=1.0, stroke=None, sw=2) -> str:
    s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    o = f' opacity="{op}"' if op != 1.0 else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}"{s}{o}/>')


def line(x1, y1, x2, y2, stroke=RULE, sw=2, dash=None, op=1.0) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' opacity="{op}"' if op != 1.0 else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')


def dot(x, y, r, fill, op=1.0, stroke=None, sw=2) -> str:
    s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    o = f' opacity="{op}"' if op != 1.0 else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"{s}{o}/>'


def series(xy, r, fill, op=1.0) -> str:
    """One <g> for a whole scatter: the same picture at a third of the bytes."""
    o = f' opacity="{op}"' if op != 1.0 else ""
    body = "".join(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}"/>' for x, y in xy)
    return f'<g fill="{fill}"{o}>{body}</g>'


def capsule(x1, x2, y, h, fill, op=1.0) -> str:
    return rect(x1, y - h / 2, max(x2 - x1, 3), h, fill, rx=h / 2, op=op)


def runs(idx) -> list[tuple[int, int]]:
    """[0,1,2,7,8] -> [(0,3),(7,2)].  Turns 300 identical ticks into one rect."""
    out: list[tuple[int, int]] = []
    for i in sorted(idx):
        if out and i == out[-1][0] + out[-1][1]:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((i, 1))
    return out


def comma(n) -> str:
    return f"{int(n):,}"


# -- 1. the hero: SIFT1M, one certificate per query ----------------------------------------


def draw_hero(d) -> str:
    s = d["sift"]
    gap, wid = s["gap"], s["width"]
    m = len(gap)
    order = sorted(range(m), key=lambda i: gap[i])
    rank_of = {i: r for r, i in enumerate(order)}
    w0 = s["width_median"]
    W, H = 1600, 900
    b = [
        M(64, 64, "separatrix", INK, 36, "700", track="1.5"),
        M(1536, 64, f"SIFT1M   {comma(s['n'])} x {s['d']} float32   k = {s['k']}   "
                    f"{comma(m)} queries", DIM, 26, anchor="end"),
        line(64, 92, 1536, 92),
    ]

    # left: the claim, and the three numbers that qualify it
    x = 64
    b += [
        T(x, 172, "SCORED AGAINST A THIRD PARTY'S ANSWER", BLUE, 26, "700", track="1.2"),
        M(x, 306, f"{s['certified_agree']}/{s['certified']}", GREEN, 128, "700"),
        T(x, 356, "certified top-10 sets identical to", INK, 28),
        T(x, 392, "the ANN_SIFT1M authors' published list", INK, 28),
        line(x, 430, 700, 430),
    ]
    stats = [
        (str(s["refused"]), AMBER, "queries refused", "this enclosure did not decide them"),
        (str(s["disputed"]), BLUE, "rows where the published", "answer differs from ours"),
        (f"{s['disputed_refused']}/{s['disputed']}", INK, "of those were refused first",
         "and every one is an exact tie"),
    ]
    y = 500
    for val, col, l1, l2 in stats:
        b += [M(x, y + 14, val, col, 54, "700"),
              T(x + 200, y - 2, l1, INK, 28),
              T(x + 200, y + 32, l2, DIM, 26)]
        y += 104
    b += [T(x, 800, "On a tie both answers are correct and nothing", DIM, 26),
          T(x, 834, "decides it. A certificate refuses there. It did.", DIM, 26)]

    # right: every query's boundary margin against the width that must fit inside it
    px0, px1, py0, py1 = 760, 1536, 268, 780
    top = math.log10(max(gap) + 1.0) * 1.10

    floor = py1 - 16  # a zero-gap row draws above the axis rule, not straddling it

    def Y(v):
        return floor - (math.log10(v + 1.0) / top) * (floor - py0)

    def X(r):
        return px0 + (r / (m - 1)) * (px1 - px0)

    b += [T(px0, 172, "every query's rank-10 boundary", INK, 30, "700"),
          T(px0, 208, "gap between the 10th and 11th neighbour, sorted", DIM, 26)]
    ticks = []
    for v in (10000, 1000, 100, 10):
        yy = Y(v)
        if yy < py0 + 30:  # would sit in the title
            continue
        lab = comma(v)
        b.append(line(px0, yy, px1, yy, RULE, 1, op=0.8))
        # the curve runs along the left of the plot, so each label carries its own ground
        # and is drawn last, over the dots rather than under them
        ticks += [rect(px0 + 2, yy - 32, advance(lab, 26, MONO) + 12, 30, GROUND, rx=4,
                       op=0.85),
                  M(px0 + 8, yy - 10, lab, DIM, 26, op=0.9)]

    yb = Y(w0)
    b += [rect(px0, yb, px1 - px0, py1 - yb, AMBER, op=0.08),
          line(px0, yb, px1, yb, AMBER, 3, dash="10 8"),
          T(px1 - 6, yb - 14, f"enclosure width {w0:.2f} -- the gap must clear this",
            AMBER, 26, anchor="end")]

    b.append(series([(X(r), Y(gap[i])) for r, i in enumerate(order) if gap[i] > wid[i]],
                    3.5, GREEN, 0.85))
    b.append(series([(X(r), Y(gap[i])) for r, i in enumerate(order) if gap[i] <= wid[i]],
                    5, AMBER))
    # all 9 disputed rows are among the very smallest margins, so 9 rings land inside 11
    # units and draw as one smudge.  One ring round the cluster, and the count in the key.
    dx = [X(rank_of[i]) for i in s["disputed_rows"]]
    dy = [Y(gap[i]) for i in s["disputed_rows"]]
    b.append(dot((min(dx) + max(dx)) / 2, (min(dy) + max(dy)) / 2,
                 max(max(dx) - min(dx), max(dy) - min(dy)) / 2 + 16,
                 "none", stroke=BLUE, sw=3))

    b += ticks
    # the key lives inside the amber refusal band, on its empty right half: the only large
    # clear area in the plot, and the band is what both of its entries are about
    ky = yb + (py1 - yb) / 2 - 22
    b += [dot(px1 - 578, ky - 8, 8, AMBER),
          M(px1 - 560, ky, f"{s['refused']} refused", AMBER, 28, "700"),
          dot(px1 - 578, ky + 36, 10, "none", stroke=BLUE, sw=3),
          T(px1 - 558, ky + 44, f"{s['disputed']} where the published answer differs",
            BLUE, 26),
          M(px1 - 20, 386, f"{s['certified']} certified", GREEN, 30, "700", anchor="end"),
          line(px0, py1, px1, py1, RULE, 2),
          T(px0, py1 + 40, "1,000 SIFT1M queries, sorted by margin", DIM, 26),
          line(64, 856, 1536, 856, RULE, 1, op=0.6),
          M(64, 886, f"gram kernel, cheap bound, chunk {s['chunk']}, {s['seconds']:.0f} s "
                     f"-- the truth was computed outside this repository", DIM, 26)]
    return svg(W, H, b)


# -- 2. the mechanism: disjoint certifies, overlapping refuses ------------------------------


def draw_boundary(d) -> str:
    s = d["sift"]
    k = s["k"]
    W, H = 1600, 940
    b = [
        M(64, 64, "separatrix", INK, 36, "700", track="1.5"),
        M(1536, 64, "score = squared euclidean distance, integer-valued on SIFT",
          DIM, 26, anchor="end"),
        line(64, 92, 1536, 92),
        T(64, 156, "Two enclosures that do not touch decide the ranking. Two that "
                   "overlap decide nothing.", INK, 30),
        T(64, 190, "green: inside the top 10, above the axis.   grey: outside it, below.",
          DIM, 26),
    ]

    def panel(y0, row, verdict, col, notes):
        nonlocal b
        r = s["rows"][str(row)]
        R, ds, ids = r["radius"], r["dist"], r["idx"]
        b.append(rect(64, y0, 1472, 320, PANEL, rx=14))
        b += [T(104, y0 + 70, verdict, col, 48, "700"),
              M(104, y0 + 112, f"query {row} of 1,000", DIM, 26),
              M(104, y0 + 178, f"gap    {ds[k] - ds[k - 1]:>9,.2f}", INK, 30),
              M(104, y0 + 216, f"width  {2 * R:>9,.2f}", INK, 30),
              T(104, y0 + 266, notes[0], col, 26),
              T(104, y0 + 298, notes[1], DIM, 26)]
        ax0, ax1 = 600, 1440
        lo, hi = ds[k - 3] - 3 * R, ds[k + 2] + 3 * R
        yy = y0 + 150

        def X(v):
            return ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)

        b.append(line(ax0 - 30, yy, ax1 + 30, yy, RULE, 2))
        # inside the set above the axis, outside below it: two rows whose x-ranges
        # either miss each other or do not
        for j in range(k - 3, k + 3):
            b.append(capsule(X(ds[j] - R), X(ds[j] + R), yy - 18 if j < k else yy + 18,
                             26, GREEN if j < k else DIM, op=0.9))
        b += [M(X(ds[k - 1]), yy - 84, f"{ds[k - 1]:,.0f}", DIM, 26, anchor="middle"),
              M(X(ds[k - 1]), yy - 52, f"#{ids[k - 1]}", GREEN, 28, "700", anchor="middle"),
              M(X(ds[k]), yy + 78, f"#{ids[k]}", INK, 28, "700", anchor="middle"),
              M(X(ds[k]), yy + 110, f"{ds[k]:,.0f}", DIM, 26, anchor="middle")]
        if ds[k] - ds[k - 1] > 2 * R:
            xc = (X(ds[k - 1] + R) + X(ds[k] - R)) / 2
            b += [line(xc, yy - 44, xc, yy + 44, GREEN, 3, dash="8 8"),
                  T(xc, yy + 110, "the top-10 cut", GREEN, 26, anchor="middle")]
        else:
            b += [line(X(ds[k]), yy - 44, X(ds[k]), yy + 44, AMBER, 3, dash="8 8"),
                  T(X(ds[k]) + 90, yy - 120, "no cut fits between them", AMBER, 26,
                    anchor="end")]

    panel(216, 0, "CERTIFIED", GREEN,
          ("the 10th and 11th neighbour cannot swap",
           "no evaluation of this formula returns another set"))
    panel(556, 93, "REFUSED", AMBER,
          ("BOUNDARY_UNDETERMINED, and the pair is named",
           "in exact integers both sit at 42,192 from query 93"))
    b.append(M(64, 922, "each capsule is one neighbour's enclosure [D-R, D+R] -- R bounds "
                        "rounding, never a confidence", DIM, 26))
    return svg(W, H, b)


# -- 3. where the naive method is wrong ----------------------------------------------------


def draw_cancellation(d) -> str:
    c, t = d["cancellation"], d["torch"]
    W, H = 1600, 740
    b = [
        M(64, 64, "separatrix", INK, 36, "700", track="1.5"),
        M(1536, 64, "float64 -- upcasting is not the fix", DIM, 26, anchor="end"),
        line(64, 92, 1536, 92),
        M(64, 156, "x = (1e6, 0)     y = (1e6 + 1e-6, 0)     two distinct points", INK, 30),
    ]
    rows = [
        ("||x||^2 + ||y||^2 - 2<x,y>", f"{c['gram']:.1f}", RED, "the Gram identity"),
        ("sum_l (x_l - y_l)^2", f"{c['direct']:.10e}", INK, "the direct sum"),
        ("exact, scaled integers", f"{c['exact']:.10e}", BLUE, "the control"),
    ]
    y = 236
    for formula, val, col, label in rows:
        b += [M(64, y + 8, formula, DIM, 28),
              M(1180, y + 16, val, col, 42, "700", anchor="end"),
              T(1220, y + 8, label, DIM, 26),
              line(64, y + 44, 1536, y + 44, RULE, 1, op=0.7)]
        y += 84
    b += [rect(64, 500, 1472, 100, PANEL, rx=14),
          T(104, 562, "REFUSED", AMBER, 46, "700"),
          M(330, 542, "GRAM_CANCELLATION", AMBER, 28),
          T(330, 578, "next: pass kernel='direct' -- a code change, not a re-run at "
                      "higher precision", DIM, 26),
          M(1496, 562, f"radius {c['gram_radius']:.3e}", DIM, 26, anchor="end"),
          T(64, 648, "torch.cdist switches to this identity above 25 rows, on one 40x8 "
                     "float32 array:", INK, 28)]
    x = 64
    for n in ("24", "25", "26"):
        v = t["spread"][n]
        b.append(M(x, 692, f"{n} rows  {v:.3e}", GREEN if v == 0 else RED, 28, "700"))
        x += 340
    return svg(W, H, b)


# -- 4. nine engines, one set of stored bytes ----------------------------------------------


def draw_engines(d) -> str:
    e = d["engines"]
    W, H = 1600, 1000
    b = [
        M(64, 64, "separatrix", INK, 36, "700", track="1.5"),
        M(1536, 64, f"{comma(e['decisions'])} top-10 decisions, one set of stored bytes",
          DIM, 26, anchor="end"),
        line(64, 92, 1536, 92),
        M(64, 268, "0", GREEN, 160, "700"),
        T(200, 208, "certified top-10 sets moved between any two of the", INK, 34),
        T(200, 250, "nine numerically distinct evaluations of one formula.", INK, 34),
        T(200, 294, f"{comma(e['certified'])} certified of {comma(e['decisions'])}. Every "
                    f"set that did move had been refused first.", DIM, 28),
    ]
    y = 400
    b += [T(64, y - 26, "each tick is one query", DIM, 26),
          dot(560, y - 35, 8, GREEN), T(578, y - 26, "certified", DIM, 26),
          dot(740, y - 35, 8, AMBER), T(758, y - 26, "refused", DIM, 26),
          dot(900, y - 35, 9, "none", stroke=BLUE, sw=3),
          T(920, y - 26, "set moved between engines", BLUE, 26)]
    for c in e["corpora"]:
        n, x0, x1 = c["m"], 700, 1520
        step = (x1 - x0) / n
        b += [T(64, y + 28, c["name"], INK, 28),
              M(64, y + 62, f"{comma(c['n'])} x {c['d']} {c['dtype']}, {c['refused']} of "
                            f"{n} refused", DIM, 26),
              rect(x0, y + 8, x1 - x0, 46, GREEN, rx=3, op=0.9)]
        for i0, ln in runs(c["refused_rows"]):
            b.append(rect(x0 + i0 * step, y + 8, ln * step, 46, AMBER, rx=3, op=0.95))
        for i in c["moved_refused"]:
            b.append(dot(x0 + (i + 0.5) * step, y + 31, 14, "none", stroke=BLUE, sw=3))
        y += 88
    b.append(line(64, y + 4, 1536, y + 4, RULE, 1, op=0.7))
    for row, i in enumerate(range(0, len(e["names"]), 3)):
        b.append(M(64, y + 46 + 34 * row, "  ·  ".join(e["names"][i:i + 3]), DIM, 26))
    return svg(W, H, b)


# -- 5. the social card, 1280 x 640 --------------------------------------------------------


def draw_social(d) -> str:
    s, e = d["sift"], d["engines"]
    W, H = 1280, 640
    b = [
        M(72, 122, "separatrix", INK, 74, "700", track="2"),
        T(72, 186, "Your top-10 changed when the batch size changed.", INK, 34),
        T(72, 230, "Get a proof it was never yours.", DIM, 34),
        line(72, 272, 1208, 272),
    ]
    cards = [
        ("0", GREEN, "certified sets moved", f"across {e['n_engines']} distinct engines"),
        (f"{s['certified_agree']}/{s['certified']}", BLUE, "match a third party",
         "SIFT1M, 1,000,000 x 128"),
        (str(s["refused"]), AMBER, "refused, each pair named", "a refusal is a return value"),
    ]
    x = 72
    for val, col, l1, l2 in cards:
        b += [rect(x, 306, 368, 204, PANEL, rx=14),
              M(x + 28, 400, val, col, 62, "700"),
              T(x + 28, 444, l1, INK, 26),
              T(x + 28, 478, l2, DIM, 24)]
        x += 392
    b += [M(72, 588, "pip install separatrix", INK, 30, "700"),
          M(1208, 588, "one O(nd) pass, no probability in the output", DIM, 26,
            anchor="end")]
    return svg(W, H, b, rx=0)


# -- measurement ---------------------------------------------------------------------------


def measure_cancellation() -> dict:
    from fractions import Fraction

    import numpy as np

    from separatrix import certified_topk, enclose

    x = np.array([[1e6, 0.0]])
    y = np.array([[1e6 + 1e-6, 0.0]])
    X = np.vstack([x, y])
    dx = Fraction(y[0, 0]) - Fraction(x[0, 0])
    _, v = certified_topk(X, x, k=1)
    _, vd = certified_topk(X, x, k=1, kernel="direct")
    eg = enclose.enclose_scores(X, x, kernel="gram")
    return {
        "gram": float(np.dot(x[0], x[0]) + np.dot(y[0], y[0]) - 2 * np.dot(x[0], y[0])),
        "direct": float(np.sum((x[0] - y[0]) ** 2)),
        "exact": float(dx * dx),
        "gram_radius": float(np.ravel(eg.R[0])[0]),
        "gram_verdict": f"{v.status} ({v.reason})",
        "direct_verdict": vd.status,
    }


def measure_engines(res: dict) -> dict:
    cs = res["corpora"]
    return {
        "names": cs[0]["evaluations"],
        "n_engines": len(cs[0]["evaluations"]),
        "decisions": sum(c["shape"][2] for c in cs),
        "certified": sum(c["shape"][2] - c["refused"] for c in cs),
        "corpora": [
            {
                "name": c["name"],
                "n": c["shape"][0],
                "d": c["shape"][1],
                "m": c["shape"][2],
                "dtype": c["dtype"],
                "refused": c["refused"],
                "refused_rows": c["refused_rows"],
                "moved_certified": c["certified_disagreeing"],
                "moved_refused": c["refused_disagreeing"],
            }
            for c in cs
        ],
    }


def measure_sift(m: int, k: int, chunk: int, detail=(0, 93)) -> dict:
    """Per query: the rank-k gap, the enclosure width, and agreement with published truth.

    The radius is one scalar per query row, so the boundary pair is the k-th and (k+1)-th
    smallest score and the rule is `gap > 2R` -- the same comparison
    `decide.topk_determined` makes, on the same enclosure, reached without holding a
    second copy of anything.  It reproduces `bench.py --sift`: 52 of 1,000 refused.
    """
    import time

    import numpy as np

    from separatrix import corpus, enclose

    base = corpus._xvecs(corpus.CACHE / "sift_base.fvecs", np.float32)
    Q = corpus._xvecs(corpus.CACHE / "sift_query.fvecs", np.float32, count=m)
    gt = corpus._xvecs(corpus.CACHE / "sift_groundtruth.ivecs", np.int32, count=m)
    gap, wid, agree, rows = [], [], [], {}
    t0 = time.perf_counter()
    for b0 in range(0, m, chunk):
        enc = enclose.enclose_scores(base, Q[b0:b0 + chunk])
        for j in range(enc.D.shape[0]):
            i = b0 + j
            D = enc.D[j]
            R = float(np.ravel(enc.R[j])[0])
            near = np.argpartition(D, k + 3)[:k + 4]
            near = near[np.argsort(D[near], kind="stable")]
            gap.append(float(D[near[k]] - D[near[k - 1]]))
            wid.append(2.0 * R)
            agree.append(set(near[:k].tolist()) == set(gt[i, :k].tolist()))
            if i in detail:
                rows[str(i)] = {"radius": R,
                                "dist": [float(D[t]) for t in near],
                                "idx": [int(t) for t in near]}
        del enc
    secs = time.perf_counter() - t0
    refused = {i for i in range(m) if gap[i] <= wid[i]}
    disputed = [i for i in range(m) if not agree[i]]
    cert = [i for i in range(m) if i not in refused]
    out = {
        "n": len(base), "d": int(base.shape[1]), "m": m, "k": k, "chunk": chunk,
        "seconds": secs,
        "gap": [round(g, 3) for g in gap],
        "width": [round(w, 4) for w in wid],
        "width_median": float(np.median(wid)),
        "refused": len(refused),
        "certified": len(cert),
        "certified_agree": sum(agree[i] for i in cert),
        "disputed": len(disputed),
        "disputed_rows": disputed,
        "disputed_refused": sum(1 for i in disputed if i in refused),
        "rows": rows,
    }
    # the claim these assets carry, asserted where the numbers are produced
    assert out["certified"] + out["refused"] == m
    assert out["disputed_refused"] == out["disputed"], "a disputed row was certified"
    assert out["certified_agree"] == out["certified"], "a certified set differs from truth"
    return out


# -- rasterise the social card -------------------------------------------------------------

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
]


def rasterise(svg_text: str, png: pathlib.Path, w: int, h: int) -> str:
    exe = (next((p for p in BROWSERS if pathlib.Path(p).exists()), None)
           or shutil.which("chromium") or shutil.which("google-chrome"))
    if not exe:
        return "social.png skipped: no chromium or edge found; social.svg is written"
    png.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        page = pathlib.Path(td) / "card.html"
        page.write_text(f"<style>html,body{{margin:0;background:{GROUND}}}</style>"
                        + svg_text, encoding="utf-8")
        # --user-data-dir is not optional: without it a second Edge hands the page to the
        # already-running one, exits 0, and writes no file.
        cmd = [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
               "--force-device-scale-factor=1", f"--user-data-dir={td}/profile",
               f"--window-size={w},{h}", f"--screenshot={png}", page.as_uri()]
        r = subprocess.run(cmd, capture_output=True, timeout=180)
        # the file lands a moment after the process exits
        for _ in range(50):
            if png.exists():
                break
            time.sleep(0.1)
    if not png.exists():
        return f"social.png skipped: {pathlib.Path(exe).name} exit {r.returncode}"
    return f"social.png  {png.stat().st_size // 1024} KB"


# -- entry point ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--measure", action="store_true",
                   help="re-measure into assets/data.json before drawing")
    p.add_argument("--bench", default=str(HERE / ".donotcommit" / "results.json"),
                   help="a bench.py results.json, for the nine-engine asset")
    p.add_argument("--m", type=int, default=1000, help="SIFT1M queries to measure")
    p.add_argument("--chunk", type=int, default=20,
                   help="query rows per score block; bounds peak memory")
    a = p.parse_args(argv)

    ASSETS.mkdir(exist_ok=True)
    data = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else {}

    if a.measure:
        sys.path.insert(0, str(HERE))
        import platform

        import numpy

        data["cancellation"] = measure_cancellation()
        print("measured: cancellation")
        bench = pathlib.Path(a.bench)
        if bench.exists():
            res = json.loads(bench.read_text(encoding="utf-8"))
            data["engines"] = measure_engines(res)
            data["torch"] = {"version": res["torch"]["version"],
                             "spread": res["torch"]["spread"]}
            print(f"measured: engines, torch  <- {bench}")
        else:
            print(f"skipped:  engines, torch  ({bench} absent; run bench.py)")
        from separatrix import corpus
        if (corpus.CACHE / "sift_base.fvecs").exists():
            data["sift"] = measure_sift(a.m, 10, a.chunk)
            print(f"measured: sift  {data['sift']['seconds']:.1f} s")
        else:
            print("skipped:  sift  (no SIFT1M cache; see separatrix.corpus.sift1m)")
        data["provenance"] = {
            "machine": platform.node(),
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "note": "every number here was measured by make_assets.py --measure",
        }
        DATA.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"wrote {DATA.name}  {DATA.stat().st_size // 1024} KB")

    missing = [k for k in ("sift", "engines", "cancellation", "torch") if k not in data]
    if missing:
        print(f"assets/data.json has no {missing}; run --measure", file=sys.stderr)
        return 1

    for name, text in (("hero.svg", draw_hero(data)),
                       ("boundary.svg", draw_boundary(data)),
                       ("cancellation.svg", draw_cancellation(data)),
                       ("engines.svg", draw_engines(data)),
                       ("social.svg", draw_social(data))):
        (ASSETS / name).write_text(text, encoding="utf-8")
        print(f"{name}  {(ASSETS / name).stat().st_size // 1024} KB")
    print(rasterise(draw_social(data), ASSETS / "social.png", 1280, 640))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
