"""The interface layer: the block, the JSON, the five exit codes, the three frames.

Nothing here scores against separatrix's own arithmetic.  The block is scored against the
Verdict it was built from -- every field must appear, and no field may appear that the
Verdict does not carry -- and the exit codes are scored against the table in `verdict.EXIT`
rather than against re-typed integers.

`check` is the one subcommand that needs the decision surface (`separatrix.api`).  Its
*rendering and exit mapping* are tested here with a stub decision function, because those
are the CLI's own logic; the integration test that calls the real one skips when the module
is absent.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

import separatrix.verdict as V
from separatrix import cli
from separatrix.harness import within_budget


def mk(status=V.REFUSED, **kw):
    base = dict(
        kernel="gram",
        bound="cheap",
        k=10,
        n_queries=300,
        n_refused=19,
        dtype_in="float32",
        dtype_used="float32",
        canary="numpy/float32 clean",
    )
    base.update(kw)
    if status in (V.REFUSED, V.NOT_CERTIFIED) and "reason" not in base:
        reason = V.EXACT_TIE if status == V.NOT_CERTIFIED else V.BOUNDARY_UNDETERMINED
        base["reason"] = reason
        base["next_action"] = V.NEXT_ACTION[reason]
    return V.Verdict(status=status, **base)


def run(argv):
    buf = io.StringIO()
    code = cli.main(argv, out=buf)
    return code, buf.getvalue()


# --------------------------------------------------------------------------------------
# the block
# --------------------------------------------------------------------------------------


def test_the_block_is_generated_from_the_verdict():
    """Every printed field comes off the Verdict, so no example block can drift."""
    f = V.Frontier(7, 3, 91, 1.0, 1.2, 1.1, 1.4, gap=0.1, width=0.5)
    v = mk(frontiers=(f,))
    txt = cli.block(v, ascii_only=True)
    for piece in (
        v.status,
        v.reason,
        v.kernel,
        v.bound,
        v.canary,
        v.accum_assumed,
        "281/300 determined",
        "row 7",
        "#3",
        "#91",
        v.next_action[:30],
    ):
        assert piece in txt, piece


def test_the_block_fits_inside_its_own_rules():
    v = mk(frontiers=tuple(V.Frontier(i, 1, 2, 0.0, 1.0, 0.5, 1.5, gap=0.5, width=1.0)
                           for i in range(3)))
    txt = cli.block(v, ascii_only=True)
    assert max(len(line) for line in txt.splitlines()) <= cli.WIDTH + 1
    assert txt.splitlines()[0] == "=" * cli.WIDTH


def test_a_certificate_prints_no_next_action_and_no_boundary():
    txt = cli.block(mk(V.CERTIFIED, reason="", next_action="", n_refused=0), ascii_only=True)
    assert "next" not in txt and "boundary" not in txt


def test_the_refusal_block_names_the_pair_the_intervals_and_the_deficit():
    """A refusal that names only a status is a status.  This one names an object."""
    f = V.Frontier(2, 11, 12, 1.0, 1.5, 1.2, 1.7, gap=0.2, width=1.0)
    txt = cli.block(mk(frontiers=(f,)), ascii_only=True)
    assert "#11" in txt and "#12" in txt
    assert "gap" in txt and "width" in txt and "deficit" in txt
    assert f"{f.deficit:.6e}" in txt


def test_max_report_caps_the_boundaries_and_says_how_many_it_hid():
    fs = tuple(
        V.Frontier(i, 1, 2, 0.0, 1.0, 0.5, 1.5, gap=0.5, width=1.0) for i in range(25)
    )
    txt = cli.block(mk(frontiers=fs), ascii_only=True, max_report=4)
    assert "21 further boundaries not shown" in txt
    assert txt.count("row 0:") == 1 and "row 9:" not in txt


def test_the_accumulator_assumption_is_on_every_block():
    """P5 is an assumption, and it travels where a consumer can see it."""
    for status in (V.CERTIFIED, V.CERTIFIED_UPCAST, V.NOT_CERTIFIED, V.REFUSED):
        cert = status in (V.CERTIFIED, V.CERTIFIED_UPCAST)
        kw = dict(reason="", next_action="") if cert else {}
        txt = cli.block(mk(status, **kw), ascii_only=True)
        assert V.ACCUM_ASSUMED in txt and "P5" in txt


def test_upcast_prints_both_dtypes_because_the_pipeline_runs_the_narrow_one():
    txt = cli.block(
        mk(V.CERTIFIED_UPCAST, reason="", next_action="", dtype_in="float16",
           dtype_used="float32"),
        ascii_only=True,
    )
    assert "float16 -> float32" in txt


# --------------------------------------------------------------------------------------
# the JSON
# --------------------------------------------------------------------------------------


def test_schema_is_the_first_key():
    body = cli.payload(mk())
    assert list(body)[0] == "schema" and body["schema"] == "separatrix/1"
    assert json.loads(json.dumps(body))["schema"] == "separatrix/1"


def test_the_json_carries_the_frontier_as_numbers_not_as_a_sentence():
    f = V.Frontier(1, 4, 5, 0.9, 1.1, 1.05, 1.25, gap=0.15, width=0.4)
    body = cli.payload(mk(frontiers=(f,)))
    got = body["frontiers"][0]
    assert got["inside"] == 4 and got["outside"] == 5
    assert got["inside_interval"] == [0.9, 1.1]
    assert got["deficit"] == pytest.approx(-0.25)


def test_the_json_says_how_many_boundaries_it_omitted():
    fs = tuple(V.Frontier(i, 1, 2, 0.0, 1.0, 0.5, 1.5, gap=0.5, width=1.0) for i in range(9))
    assert cli.payload(mk(frontiers=fs), max_report=2)["frontiers_omitted"] == 7


# --------------------------------------------------------------------------------------
# exit codes
# --------------------------------------------------------------------------------------


def _stub(monkeypatch, verdict):
    monkeypatch.setattr(cli, "_decisions", lambda: (lambda X, Q, **kw: (None, verdict)))


@pytest.fixture
def npys(tmp_path):
    X, Q = cli.near_duplicate_corpus(n=60, d=8, m=5)
    xp, qp = tmp_path / "X.npy", tmp_path / "Q.npy"
    np.save(xp, X)
    np.save(qp, Q)
    return str(xp), str(qp)


@pytest.mark.parametrize(
    "status,code",
    [
        (V.CERTIFIED, 0),
        (V.NOT_CERTIFIED, 1),
        (V.REFUSED, 2),
        (V.CERTIFIED_UPCAST, 4),
    ],
)
def test_every_status_maps_to_its_documented_exit_code(monkeypatch, npys, status, code):
    cert = status in (V.CERTIFIED, V.CERTIFIED_UPCAST)
    kw = {"reason": "", "next_action": "", "n_refused": 0} if cert else {}
    _stub(monkeypatch, mk(status, **kw))
    got, text = run(["check", "--corpus", npys[0], "--queries", npys[1]])
    assert got == code == V.EXIT[status]
    assert status in text


def test_a_usage_error_is_exit_three_and_never_a_verdict(tmp_path, npys):
    """Exit class 3 is a statement about the caller's code, not about the caller's data."""
    bad = tmp_path / "1d.npy"
    np.save(bad, np.zeros(7))
    ints = tmp_path / "ints.npy"
    np.save(ints, np.zeros((4, 3), dtype=np.int32))
    both = tmp_path / "two.npz"
    np.savez(both, a=np.zeros((4, 3)), b=np.zeros((4, 3)))
    empty = tmp_path / "empty.npy"
    empty.write_bytes(b"")
    corrupt_npz = tmp_path / "corrupt.npz"
    corrupt_npz.write_bytes(b"PK\x03\x04 zip magic, everything after it is garbage")
    a_directory = tmp_path / "a_directory.npy"
    a_directory.mkdir()

    for argv in (
        [],
        ["check", "--corpus", str(bad), "--queries", npys[1]],
        ["check", "--corpus", str(ints), "--queries", npys[1]],
        ["check", "--corpus", str(both), "--queries", npys[1]],
        ["check", "--corpus", str(tmp_path / "nope.npy"), "--queries", npys[1]],
        ["check", "--corpus", str(tmp_path / "nested" / "nope.npy"), "--queries", npys[1]],
        ["check", "--corpus", npys[0], "--queries", npys[1], "--max-refused", "1.5"],
        ["check", "--corpus", str(empty), "--queries", npys[1]],
        ["check", "--corpus", str(corrupt_npz), "--queries", npys[1]],
        ["check", "--corpus", str(a_directory), "--queries", npys[1]],
    ):
        code, text = run(argv)
        assert code == V.EXIT_USAGE, argv
        for reason in V.REASONS:
            assert reason not in text, (argv, reason)


def test_a_precondition_refusal_is_rendered_as_a_refusal_not_a_traceback(monkeypatch, npys):
    def boom(X, Q, **kw):
        raise V.Refusal(V.RANGE_UNSAFE, "max ||x||^2 reaches 5.6e+06 against 6.55e+04")

    monkeypatch.setattr(cli, "_decisions", lambda: boom)
    code, text = run(["check", "--corpus", npys[0], "--queries", npys[1]])
    assert code == 2 and V.RANGE_UNSAFE in text
    assert "5.6e+06" in text and "upcast=True" in text


def test_check_says_so_when_the_decision_surface_is_absent(monkeypatch, npys):
    monkeypatch.setattr(cli, "_decisions", lambda: None)
    code, text = run(["check", "--corpus", npys[0], "--queries", npys[1]])
    assert code == V.EXIT_USAGE
    assert "separatrix.api" in text and "demo" in text


# --------------------------------------------------------------------------------------
# --max-refused: the gate, run from the CLI
# --------------------------------------------------------------------------------------


def test_max_refused_turns_a_run_inside_the_budget_green(monkeypatch, npys):
    _stub(monkeypatch, mk(n_queries=300, n_refused=19))  # 6.3%
    code, text = run(["check", "--corpus", npys[0], "--queries", npys[1],
                      "--max-refused", "0.10"])
    assert code == 0 and "-> pass" in text
    assert V.REFUSED in text, "the status is still REFUSED; only the gate passed"


def test_max_refused_leaves_a_run_over_the_budget_red(monkeypatch, npys):
    _stub(monkeypatch, mk(n_queries=300, n_refused=36))  # 12%
    code, text = run(["check", "--corpus", npys[0], "--queries", npys[1],
                      "--max-refused", "0.10"])
    assert code == 2 and "-> fail" in text


def test_the_cli_and_the_gate_cannot_disagree_about_one_corpus():
    """Both call `harness.within_budget`; there is one comparison, in one place."""
    for n, refused, budget in ((300, 19, 0.10), (300, 36, 0.10), (300, 30, 0.10)):
        assert within_budget(refused / n, budget) == (refused / n <= budget)


def test_max_refused_does_not_touch_an_exact_tie(monkeypatch, npys):
    """An exact tie is not a budget question: no precision and no budget removes it."""
    _stub(monkeypatch, mk(V.NOT_CERTIFIED, n_refused=1, n_queries=300))
    code, _ = run(["check", "--corpus", npys[0], "--queries", npys[1],
                   "--max-refused", "0.99"])
    assert code == 1


def test_json_carries_the_gate_line_and_the_gated_exit(monkeypatch, npys):
    _stub(monkeypatch, mk(n_queries=300, n_refused=19))
    code, text = run(["check", "--corpus", npys[0], "--queries", npys[1],
                      "--max-refused", "0.10", "--json"])
    body = json.loads(text)
    assert code == 0 and body["exit"] == 0
    assert body["status"] == V.REFUSED and "pass" in body["gate"]
    assert body["corpus"] == npys[0]


# --------------------------------------------------------------------------------------
# demo and probe
# --------------------------------------------------------------------------------------


def test_the_demo_corpus_is_seeded_and_leaves_the_frame_something_to_lose():
    from separatrix.decide import rows_determined
    from separatrix.enclose import enclose_scores

    X, Q = cli.near_duplicate_corpus()
    assert np.array_equal(X, cli.near_duplicate_corpus()[0])
    e = enclose_scores(X, Q, kernel="gram")
    named = [i for i, f in enumerate(rows_determined(e.D, e.R, 5)) if f is not None]
    assert 0 < len(named) < Q.shape[0], (
        "a corpus that refuses every row makes frame 3 true by construction"
    )


def test_every_frame_runs_with_no_network_and_no_download():
    for frame in sorted(cli.FRAMES):
        code, text = run(["demo", "--frame", frame])
        assert code == 0, (frame, text)
        assert text.strip()


def test_frame_three_names_every_row_two_evaluations_decided_differently():
    """The soundness claim, as a test: a row that differed and was not named is a
    counterexample to the certificate, and `frame_preview` returns nonzero on one."""
    code, text = run(["demo", "--frame", "preview"])
    assert code == 0
    assert "differed and NOT named:                    0" in text


def test_frame_one_shows_that_the_formula_is_the_fix_and_precision_is_not():
    code, text = run(["demo", "--frame", "cancellation"])
    assert code == 0
    assert V.GRAM_CANCELLATION in text
    assert "float64" in text and "0.0" in text
    assert "kernel='direct'" in text


def test_frame_two_labels_the_stand_in_when_torch_is_absent():
    try:
        import torch  # noqa: F401
    except ImportError:
        _, text = run(["demo", "--frame", "batch"])
        assert "STAND-IN" in text and "skipped" in text
    else:  # pragma: no cover - this machine has no torch
        _, text = run(["demo", "--frame", "batch"])
        assert text.strip()


def test_probe_reports_this_machine_and_is_never_an_input_to_a_certificate():
    code, text = run(["probe"])
    assert code == 0
    for name in ("float16", "float32", "float64"):
        assert f"{np.finfo(np.dtype(name)).eps / 2:.6e}" in text
    assert "not testable" in text


# --------------------------------------------------------------------------------------
# the honesty rules, over what this layer actually prints
# --------------------------------------------------------------------------------------


def test_no_banned_phrase_in_anything_this_layer_prints():
    printed = [cli.block(mk(s, **({"reason": "", "next_action": ""}
                                  if s.startswith("CERTIFIED") else {})), ascii_only=True)
               for s in (V.CERTIFIED, V.CERTIFIED_UPCAST, V.NOT_CERTIFIED, V.REFUSED)]
    for frame in sorted(cli.FRAMES):
        printed.append(run(["demo", "--frame", frame])[1])
    printed.append(run(["probe"])[1])
    printed.append(run([])[1])
    printed.append(json.dumps(cli.payload(mk())))
    for text in printed:
        assert V.scan(text) == [], V.scan(text)


def test_the_module_self_check_passes():
    cli._demo()


# --------------------------------------------------------------------------------------
# the integration test the spec names.  Skips until the decision surface exists, because
# `check` is the one subcommand that calls it; everything above tests this layer's own
# logic against a stub and runs unconditionally.
# --------------------------------------------------------------------------------------


def test_cli_exit_codes_and_json(tmp_path):
    pytest.importorskip("separatrix.api")
    X, Q = cli.near_duplicate_corpus(n=200, d=32, m=40)
    xp, qp = tmp_path / "X.npy", tmp_path / "Q.npy"
    np.save(xp, X)
    np.save(qp, Q)
    argv = ["check", "--corpus", str(xp), "--queries", str(qp), "--k", "5"]

    code, text = run(argv)
    assert code in set(V.EXIT.values()), text
    assert any(s in text for s in V.STATUSES)

    code, text = run(argv + ["--json"])
    body = json.loads(text)
    assert list(body)[0] == "schema" and body["exit"] == code
    assert body["k"] == 5 and body["n_queries"] == Q.shape[0]

    # the gate, over a real refused fraction: a budget above it is green, below it is red
    frac = body["n_refused"] / max(1, body["n_queries"])
    if body["status"] == V.REFUSED and 0 < frac < 1:
        assert run(argv + ["--max-refused", f"{min(1.0, frac + 0.01):.4f}"])[0] == 0
        assert run(argv + ["--max-refused", f"{max(0.0, frac - 0.01):.4f}"])[0] == 2

    # a usage error through the real surface is still exit class 3 and still not a refusal
    code, text = run(argv[:-1] + ["0"])
    assert code == V.EXIT_USAGE
    assert not any(r in text for r in V.REASONS)


def test_cli_reads_unicode_and_space_bearing_paths(tmp_path):
    """Windows and POSIX both accept these bytes in a filename; `check` must too."""
    pytest.importorskip("separatrix.api")
    X, Q = cli.near_duplicate_corpus(n=60, d=8, m=5)
    xp = tmp_path / "corpus café ☃.npy"
    qp = tmp_path / "sub dir with spaces" / "queries.npy"
    qp.parent.mkdir()
    np.save(xp, X)
    np.save(qp, Q)
    code, text = run(["check", "--corpus", str(xp), "--queries", str(qp), "--k", "5"])
    assert code in set(V.EXIT.values()), text
    assert any(s in text for s in V.STATUSES)


def test_cli_names_a_zero_row_corpus_instead_of_crashing(tmp_path):
    """A `.npy` with n=0 is `size == 0`; `as_points` already refuses it -- this is the
    path a stranger takes through the CLI door rather than through `as_points` directly."""
    pytest.importorskip("separatrix.api")
    zero = tmp_path / "zero.npy"
    np.save(zero, np.zeros((0, 8), dtype=np.float32))
    q = tmp_path / "q.npy"
    np.save(q, np.zeros((3, 8), dtype=np.float32))
    code, text = run(["check", "--corpus", str(zero), "--queries", str(q)])
    assert code == V.EXIT_USAGE, text
    assert "empty" in text
