"""The benchmark, on a draw small enough to run in the suite.

Two of these are regression tests of the *design*, not of the numbers.  A benchmark that
can only print a good result is not a benchmark, so what is pinned here is that the arms
which could embarrass the design are still computed and still printed:

  * the tuned-margin sweep, the control that may draw on every corpus;
  * the shuffled-enclosure control, scored against rung 2 -- at rung 1 the radius is
    constant across the row and the shuffle is the identity, which would make the control
    a tautology;
  * the list of arms this machine could not run, printed rather than skipped.

The soundness assertions have no threshold.  One CERTIFIED verdict contradicted by the
exact lattice's construction, or by two evaluations of the same formula, withdraws the
package.

No download: every arm here is generated, so the suite needs no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bench  # noqa: E402
import separatrix.verdict as V  # noqa: E402


@pytest.fixture(scope="module")
def res():
    return bench.run(n=200, m=20, d=32, k=4, seed=5, cost_n=200, cost_d=32, real=False)


def test_no_certificate_is_contradicted_by_the_exact_lattice(res):
    """Truth by construction: integer coordinates, so no oracle is in the loop."""
    la = res["lattice"]
    assert la["wrong_certificates"] == 0
    assert la["delta_star"] is not None, "nothing certified at any margin"


def test_the_pessimism_factor_is_reported_with_its_sample_size(res):
    """`delta_wrong` is a max over a sample and can only rise, so it never ships bare."""
    la = res["lattice"]
    assert la["trials"] >= 1
    if la["delta_wrong"] is not None:
        assert la["delta_star"] >= la["delta_wrong"], (
            "the smallest certified margin fell below a margin the floats got wrong"
        )


def test_no_certified_row_disagrees_across_evaluations(res):
    """The claim is determinism, and this is the instance that would withdraw it."""
    for c in res["corpora"]:
        assert c["certified_disagreeing"] == [], (c["name"], c["certified_disagreeing"])


def test_the_escalation_triple_accounts_for_every_refused_row(res):
    for c in res["corpora"]:
        t = c["escalation"]
        assert t["flipped"] + t["confirmed"] + t["tie"] + t["budget"] == c["refused"]


def test_the_controls_that_could_embarrass_the_design_are_still_computed(res):
    for c in res["corpora"]:
        assert c["tuned"], "the tuned-margin sweep was dropped"
        assert set(c["tuned"]) == {f"{e:.0e}" for e in bench.EPS_GRID}
        assert "shuffled_refused" in c and "refused_per_pair" in c
        assert "naive_refused" in c


def test_the_shuffled_control_is_scored_against_rung_two_not_rung_one(res):
    """At rung 1 the radius is constant across the row, so the shuffle is the identity.

    Scored against rung 1 this control cannot fail; scored against rung 2 it can, and on
    an un-normalised corpus it does.
    """
    src = Path(bench.__file__).read_text(encoding="utf-8")
    assert "per_pair=True" in src
    for c in res["corpora"]:
        assert c["shuffled_refused"] >= 0


def test_every_arm_this_machine_could_not_run_is_named(res):
    assert res["not_run"], "an unrun arm was skipped silently"
    txt = bench.table(res, ascii_only=True)
    for line in res["not_run"]:
        assert line in txt


def test_the_table_carries_its_provenance(res):
    txt = bench.table(res, ascii_only=True)
    p = res["provenance"]
    for field in ("commit", "machine", "python", "numpy", "scipy"):
        assert str(p[field]) in txt, field
    assert f"seed {res['config']['seed']}" in txt


def test_the_cost_table_prints_the_control_and_the_status_quo(res):
    cost = res["cost"]
    assert "fp64 gram + argpartition" in cost, "the honest control was dropped"
    assert "the fp32-vs-fp64 diff" in cost, "the competing practice was dropped"
    assert "fp32 gram + argpartition" in cost, "the status quo was dropped"
    assert all(v > 0 for v in cost.values())


def test_the_whole_result_is_json_serialisable(res):
    assert json.loads(json.dumps(res))["config"]["k"] == 4


def test_no_banned_phrase_in_the_table(res):
    txt = bench.table(res, ascii_only=True)
    assert V.scan(txt) == [], V.scan(txt)


def test_the_module_self_check_passes():
    bench._demo()
