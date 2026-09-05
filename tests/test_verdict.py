"""The vocabulary: statuses, exit codes, the catalogue, and what may never be printed.

These tests touch no array. That is the point of `verdict.py` importing only the standard
library: the honesty rules are enforceable without numpy, so they cannot quietly become
conditional on a numerical path.
"""

from __future__ import annotations

import pathlib

import pytest

import separatrix
from separatrix import verdict as V


def test_exit_codes_are_the_documented_five():
    assert V.EXIT[V.CERTIFIED] == 0
    assert V.EXIT[V.NOT_CERTIFIED] == 1
    assert V.EXIT[V.REFUSED] == 2
    assert V.EXIT_USAGE == 3
    assert V.EXIT[V.CERTIFIED_UPCAST] == 4
    # CERTIFIED_UPCAST is never 0: a CI job must not read a pass for a computation the
    # caller's own pipeline does not run
    assert V.EXIT[V.CERTIFIED_UPCAST] != 0


def test_every_catalogue_entry_has_an_exit_and_a_next_action():
    for reason in V.REASONS:
        assert reason in V.REASON_EXIT
        assert V.NEXT_ACTION[reason].strip()
        assert V.scan(V.NEXT_ACTION[reason]) == []
    # EXACT_TIE is the one entry that is a NOT CERTIFIED rather than a refusal to try
    assert V.REASON_EXIT[V.EXACT_TIE] == 1
    assert all(V.REASON_EXIT[r] == 2 for r in V.REASONS if r != V.EXACT_TIE)


def test_gram_cancellation_names_a_code_change_not_a_re_observation():
    """The only refusal in the catalogue that names something to change rather than redo."""
    a = V.NEXT_ACTION[V.GRAM_CANCELLATION]
    assert "kernel='direct'" in a
    assert "donot_use_mm_for_euclid_dist" in a
    assert "25 rows" in a


def test_range_unsafe_leads_with_the_damage():
    """A refusal that names a precondition gets disabled; one that names damage gets acted on."""
    a = V.NEXT_ACTION[V.RANGE_UNSAFE]
    assert "float32" in a and "upcast=True" in a


def test_a_refusal_must_carry_a_next_action():
    with pytest.raises(ValueError):
        V.Verdict(status=V.REFUSED, reason=V.BOUNDARY_UNDETERMINED, next_action="")


def test_a_refusal_must_name_a_catalogued_reason():
    with pytest.raises(ValueError):
        V.Verdict(status=V.REFUSED, reason="VIBES", next_action="x")
    with pytest.raises(ValueError):
        V.Verdict(status=V.REFUSED, next_action="x")
    with pytest.raises(ValueError):
        V.Refusal("VIBES")


def test_a_certificate_carries_no_reason():
    with pytest.raises(ValueError):
        V.Verdict(status=V.CERTIFIED, reason=V.BOUNDARY_UNDETERMINED)


def test_accumulator_assumption_is_printed_on_every_verdict():
    """P5 is an assumption, so it travels where a consumer can see it."""
    v = V.Verdict(status=V.CERTIFIED)
    assert v.accum_assumed == V.ACCUM_ASSUMED and v.accum_assumed
    assert "accumulator assumed" in str(v)
    with pytest.raises(ValueError):
        V.Verdict(status=V.CERTIFIED, accum_assumed="")


def test_no_banned_phrase_in_any_printed_field():
    for field, value in (
        ("detail", "this ranking should be fine"),
        ("detail", "the score is wrong"),
        ("next_action", "the result is probably right"),
        ("detail", "95% confidence"),
        ("kernel", "unreliable"),
    ):
        with pytest.raises(ValueError):
            V.Verdict(status=V.CERTIFIED, **{field: value})


def test_the_banned_scan_is_word_bounded():
    """RANGE_UNSAFE must survive \\bsafe\\b, or the catalogue cannot print its own codes."""
    assert V.scan(V.RANGE_UNSAFE) == []
    assert V.scan("an unsafe range") == []
    assert V.scan("unlikely") == []
    assert V.scan("this is safe") == ["safe"]
    assert V.scan("Probably") == ["Probably"]


# `verdict.py` is the one exemption, by name: it is where the banned words are DEFINED and
# where the scanner's own regression exercises them, so the literals have to appear.  It is
# not exempt from the rule -- every string it can actually print is checked by construction
# in `test_every_catalogue_entry_has_an_exit_and_a_next_action` and
# `test_no_banned_phrase_in_any_printed_field` above, which are stronger than a grep.
DEFINITION_SITE = "verdict.py"


def _package_files():
    root = pathlib.Path(separatrix.__file__).resolve().parent
    files = [p for p in sorted(root.glob("*.py")) if p.name != DEFINITION_SITE]
    repo = root.parent
    for extra in ("README.md", "RESULTS.md"):
        p = repo / extra
        if p.exists():
            files.append(p)
    return files


def test_the_definition_site_is_the_only_exemption_and_it_is_complete():
    src = (pathlib.Path(separatrix.__file__).resolve().parent / DEFINITION_SITE).read_text(
        encoding="utf-8"
    )
    for phrase in V.BANNED:
        assert f'"{phrase}"' in src, f"{phrase!r} is not in the BANNED literal any more"
    assert len(V.BANNED) >= 12


def test_no_banned_phrase_in_the_package_source():
    """A grep of the installed package plus README and RESULTS when they exist.

    Lines that must quote prior art honestly carry a trailing `# sx: quote` marker; those
    are skipped, and the marker count itself is capped so the exemption cannot grow into a
    hole.
    """
    hits = []
    markers = 0
    for path in _package_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if V.QUOTE_MARKER in line:
                markers += 1
                continue
            found = V.scan(line)
            if found:
                hits.append(f"{path.name}:{lineno}: {found} in {line.strip()!r}")
    assert hits == [], "\n".join(hits)
    assert markers <= V.MAX_QUOTE_MARKERS, f"{markers} quote markers, cap is {V.MAX_QUOTE_MARKERS}"


def test_verdict_str_is_readable_and_carries_the_counts():
    v = V.Verdict(
        status=V.REFUSED,
        reason=V.BOUNDARY_UNDETERMINED,
        next_action=V.NEXT_ACTION[V.BOUNDARY_UNDETERMINED],
        kernel="gram",
        bound="cheap",
        dtype_in="float32",
        n_queries=300,
        n_refused=19,
    )
    s = str(v)
    assert "REFUSED (BOUNDARY_UNDETERMINED)" in s
    assert "281/300 determined" in s
    assert "next:" in s
    assert v.exit_code == 2 and not v.certified


def test_frontier_deficit_is_gap_minus_width():
    f = V.Frontier(0, 1, 2, 0.9, 1.1, 1.05, 1.25, gap=0.15, width=0.4)
    assert f.deficit == pytest.approx(-0.25)
    assert not f.determined
    assert V.Frontier(0, 1, 2, 0.9, 1.1, 1.5, 1.7, gap=0.6, width=0.4).determined


BLOCKER = """
import sys

class Block:
    BANNED = ("torch", "faiss", "datasets", "sentence_transformers")

    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BANNED:
            raise ImportError("blocked: " + name)
        return None

sys.meta_path.insert(0, Block())

import numpy as np
import separatrix

assert separatrix.gamma(386, "float32") > 0
for banned in Block.BANNED:
    assert banned not in sys.modules, banned

rng = np.random.default_rng(0)
X = rng.normal(size=(120, 12)).astype(np.float32)
Q = X[:4] + np.float32(1e-2) * rng.normal(size=(4, 12)).astype(np.float32)
idx, v = separatrix.certified_topk(X, Q, k=4)
assert idx.shape == (4, 4) and v.status
for banned in Block.BANNED:
    assert banned not in sys.modules, banned
print("clean")
"""


def test_import_costs_nothing_optional():
    """`import separatrix` and the numpy certification path never reach torch or datasets.

    Run in a subprocess with a meta-path finder that raises on the optional imports.  The
    in-process version of this test was measured to be vacuous once any other test file
    imported torch: `sys.modules` is shared, so it passed by accident and then failed for a
    reason that had nothing to do with the package.
    """
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-c", BLOCKER], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "clean" in r.stdout


def test_public_surface_is_small_and_the_decisions_are_lazy():
    assert "certified_topk" in separatrix.__all__
    assert "topk_determined" in separatrix.__all__
    with pytest.raises(AttributeError):
        separatrix.no_such_name
