"""The build gate.

The two properties that decide whether a gate survives contact with a CI matrix:

  * it has no default budget, so nobody inherits a number they did not choose;
  * it **refuses to compare** across a change in the arithmetic rather than going red,
    because a refused fraction measured under a different dtype, kernel, rung or canary
    has a different denominator, and a gate that quietly measures the wrong denominator is
    the same class of failure this package exists to attack.

Everything here is scored against constructed Verdicts.  The gate never sees an array, so
none of it depends on separatrix's own arithmetic being right.
"""

from __future__ import annotations

import json

import pytest

import separatrix.verdict as V
from separatrix import harness as H


def mk(n=300, refused=19, **kw):
    base = dict(
        kernel="gram",
        bound="cheap",
        per_pair=False,
        k=10,
        largest=False,
        dtype_in="float32",
        dtype_used="float32",
        canary="numpy/float32 clean",
    )
    base.update(kw)
    if refused:
        return V.Verdict(
            status=V.REFUSED,
            reason=V.BOUNDARY_UNDETERMINED,
            next_action=V.NEXT_ACTION[V.BOUNDARY_UNDETERMINED],
            n_queries=n,
            n_refused=refused,
            **base,
        )
    return V.Verdict(status=V.CERTIFIED, n_queries=n, n_refused=0, **base)


def write_record(tmp_path, verdict, refused, fixture="fx"):
    p = tmp_path / H.GATE_FILE
    p.write_text(
        json.dumps(
            {
                "schema": H.GATE_SCHEMA,
                "fixtures": {fixture: {"digest": H.digest(verdict), "refused": refused}},
            }
        ),
        encoding="utf-8",
    )
    return str(p)


# --------------------------------------------------------------------------------------
# the budget
# --------------------------------------------------------------------------------------


def test_there_is_no_default_budget():
    with pytest.raises(TypeError):
        H.gate(fixture="fx").__enter__()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        H.Gate(None, "fx")
    with pytest.raises(ValueError):
        H.Gate(1.5, "fx")


def test_a_fixture_name_is_required_because_a_fraction_with_no_name_compares_to_nothing():
    with pytest.raises(ValueError):
        H.Gate(0.1, "")


def test_inside_the_budget_passes_and_outside_it_raises():
    with H.gate(0.10, "fx") as g:
        H.report(mk(300, 19))
    assert g.report().status == H.PASS

    with pytest.raises(AssertionError) as e:
        with H.gate(0.10, "fx") as g:
            H.report(mk(300, 36))
    assert "over the budget" in str(e.value)


def test_the_budget_boundary_is_inclusive_and_is_the_cli_comparison():
    assert H.within_budget(0.10, 0.10) and not H.within_budget(0.1000001, 0.10)
    with H.gate(0.10, "fx") as g:
        g.record(mk(100, 10))
    assert g.report().status == H.PASS


def test_nothing_recorded_is_not_a_pass():
    """An empty gate is a gate that did not run, and it says so instead of going green."""
    with H.gate(0.0, "fx") as g:
        pass
    assert g.report().status == H.NOT_COMPARED
    assert "ContextVar" in g.report().detail


# --------------------------------------------------------------------------------------
# the digest
# --------------------------------------------------------------------------------------


def test_gate_digest_mismatch_refuses():
    """THE gate property: an arithmetic change refuses to compare, it does not fail."""
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        base = mk(300, 19)
        path = write_record(td, base, 19 / 300)

        with H.gate(0.50, "fx", path) as g:  # same digest, same fraction
            g.record(base)
        assert g.report().status == H.PASS

        for changed in (
            dict(dtype_used="float64"),
            dict(kernel="direct"),
            dict(bound="tight"),
            dict(per_pair=True),
            dict(k=5),
            dict(canary="numpy/float32 coarse"),
        ):
            with H.gate(0.0, "fx", path) as g:  # a 100% refusal under a 0% budget
                g.record(mk(300, 300, **changed))
            rep = g.report()
            assert rep.status == H.NOT_COMPARED, changed
            assert "refuses to compare" in rep.detail
            assert rep.recorded_digest != rep.digests[0]


def test_a_rise_under_the_same_digest_fails_even_inside_a_loose_budget():
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        base = mk(300, 19)
        path = write_record(td, base, 19 / 300)
        with pytest.raises(AssertionError) as e:
            with H.gate(0.90, "fx", path):
                H.report(mk(300, 25))
        assert "rose from" in str(e.value)


def test_two_arithmetic_configurations_in_one_gate_have_no_single_denominator():
    with H.gate(0.0, "fx") as g:
        g.record(mk(100, 0))
        g.record(mk(100, 50, kernel="direct"))
    rep = g.report()
    assert rep.status == H.NOT_COMPARED and "denominator" in rep.detail


def test_the_digest_is_over_the_arithmetic_and_not_over_the_data():
    """Two runs on different corpora, same arithmetic, must compare."""
    assert H.digest(mk(300, 19)) == H.digest(mk(9000, 4000))
    assert H.digest(mk(300, 19)) != H.digest(mk(300, 19, dtype_used="float64"))


# --------------------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------------------


def test_the_gate_never_writes_the_file_it_read():
    """A gate that records its own budget passes by construction on the run that wrote it."""
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        base = mk(300, 19)
        path = write_record(td, base, 19 / 300)
        before = pathlib.Path(path).read_bytes()
        with H.gate(0.50, "fx", path) as g:
            g.record(base)
        assert pathlib.Path(path).read_bytes() == before
        assert sorted(p.name for p in td.iterdir()) == [H.GATE_FILE]


def test_a_missing_or_broken_record_file_is_not_an_error(tmp_path):
    assert H.read_record("fx", tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert H.read_record("fx", broken) is None
    with H.gate(0.10, "fx", str(broken)) as g:
        g.record(mk(300, 19))
    assert g.report().status == H.PASS and g.report().recorded is None


# --------------------------------------------------------------------------------------
# scope, and what it records
# --------------------------------------------------------------------------------------


def test_report_outside_a_gate_is_a_no_op():
    """The library never depends on a gate being open."""
    H.report(mk(300, 19))


def test_gates_nest_and_the_inner_one_does_not_leak():
    with H.gate(1.0, "outer") as outer:
        H.report(mk(100, 0))
        with H.gate(1.0, "inner") as inner:
            H.report(mk(100, 50))
        H.report(mk(100, 0))
    assert inner.n == 100 and inner.n_refused == 50
    assert outer.n == 200 and outer.n_refused == 0


def test_a_gate_records_verdicts_and_says_so_when_handed_something_else():
    with H.gate(1.0, "fx") as g:
        with pytest.raises(TypeError) as e:
            g.record(0.19)
    assert "certified_topk" in str(e.value)


def test_the_scope_note_travels_on_the_report():
    with H.gate(1.0, "fx") as g:
        g.record(mk(300, 19))
    assert "xdist" in g.report().scope and "per-process" in g.report().scope


def test_the_report_prints_the_counts_and_the_budget():
    with H.gate(0.10, "fx") as g:
        g.record(mk(300, 19))
    text = str(g.report())
    assert "19/300" in text and "0.1000" in text and H.PASS in text


def test_no_banned_phrase_in_anything_the_gate_prints():
    texts = []
    with H.gate(0.10, "fx") as g:
        g.record(mk(300, 19))
    texts.append(str(g.report()))
    with pytest.raises(AssertionError) as e:
        with H.gate(0.0, "fx") as g:
            g.record(mk(300, 1))
    texts.append(str(g.report()))
    texts.append(str(e.value))
    with H.gate(0.0, "fx") as g:
        g.record(mk(100, 0))
        g.record(mk(100, 50, kernel="direct"))
    texts.append(str(g.report()))
    texts.append(H.SCOPE_NOTE)
    for t in texts:
        assert V.scan(t) == [], V.scan(t)


def test_the_module_self_check_passes():
    H._demo()
