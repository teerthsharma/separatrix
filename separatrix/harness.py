"""The build gate: ``with separatrix.harness.gate(...) as g: ...``

The third surface, after the library and the CLI.  It answers one question a CI job can
act on -- *did the fraction of decisions this build could not determine go up?* -- and it
is deliberately the smallest thing that can answer it.

Two rules it does not break:

  * **``max_refused`` is required.**  A default budget is a number nobody chose, and the
    first time it goes red on a machine swap the gate is deleted.
  * **A digest mismatch refuses to compare; it does not fail.**  The refused fraction is a
    function of the arithmetic that produced it -- dtype, kernel, bound, rung, k, and the
    P4 canary -- so comparing across a change in any of them measures the wrong
    denominator, which is the same class of error this package exists to attack.

The gate **reads** ``.separatrix-gate.json`` and never writes it.  A gate that records its
own budget is vacuous: it passes by construction on the run that wrote it.

Scope is per-process, via a ``ContextVar``.  Under ``pytest-xdist`` that means **per
worker**, so a gate opened in one worker never sees a verdict produced in another; the
workaround is one line -- pass the ``Gate`` to the code that produces verdicts and call
``g.record(v)`` explicitly instead of relying on the ambient one.  This is stated on
``GateReport`` too, where a reader of the output can see it.

``d`` is not part of the digest because a ``Verdict`` does not carry it.  The shape belongs
in the fixture name (``"scifact-d384-fp32"``), which is compared as a plain string.

Self-check:  python -m separatrix.harness
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from contextvars import ContextVar
from dataclasses import dataclass

from .verdict import REFUSED, Verdict

GATE_FILE = ".separatrix-gate.json"
GATE_SCHEMA = "separatrix-gate/1"

# The fields of a Verdict that change what a refused fraction means.  Every one of them is
# a property of the arithmetic, not of the corpus: the corpus is named by `fixture`.
DIGEST_FIELDS = (
    "kernel",
    "bound",
    "per_pair",
    "k",
    "largest",
    "dtype_in",
    "dtype_used",
    "canary",
    "accum_assumed",
)

PASS = "pass"
FAIL = "fail"
NOT_COMPARED = "not compared"

# Per-process, and stated as such on every report.
_CURRENT: ContextVar = ContextVar("separatrix_gate", default=None)

SCOPE_NOTE = (
    "gate scope is per-process (a ContextVar); under pytest-xdist that is per worker. "
    "Pass the Gate explicitly and call g.record(v) to cross a process boundary."
)


def digest(v: Verdict) -> str:
    """A short hash of the arithmetic that produced a verdict.  Not of the data."""
    payload = "|".join(f"{f}={getattr(v, f)!r}" for f in DIGEST_FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def within_budget(observed: float, max_refused: float) -> bool:
    """The one comparison, so the CLI and the gate cannot disagree about one corpus."""
    return observed <= max_refused


def read_record(fixture: str, path=GATE_FILE):
    """The recorded entry for ``fixture``, or None.  Never writes, never creates."""
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (FileNotFoundError, IsADirectoryError, PermissionError, json.JSONDecodeError):
        return None
    rec = doc.get("fixtures", {}).get(fixture)
    return rec if isinstance(rec, dict) else None


@dataclass(frozen=True)
class GateReport:
    """What the gate saw, and whether it compared at all."""

    fixture: str
    n: int
    n_refused: int
    max_refused: float
    status: str
    detail: str
    digests: tuple = ()
    recorded: float = None
    recorded_digest: str = None
    scope: str = SCOPE_NOTE

    @property
    def observed(self) -> float:
        return self.n_refused / self.n if self.n else 0.0

    def __str__(self) -> str:
        obs = f"{self.n_refused}/{self.n}" + (f" ({self.observed:.4f})" if self.n else "")
        head = f"gate {self.fixture}: {self.status}  {obs}  budget {self.max_refused:.4f}"
        if self.recorded is not None:
            head += f"  recorded {self.recorded:.4f}"
        return head + ("\n  " + self.detail if self.detail else "")


class Gate:
    """Accumulates (n_queries, n_refused) over the verdicts recorded inside the block."""

    def __init__(self, max_refused, fixture: str, path=GATE_FILE) -> None:
        if not isinstance(max_refused, (int, float)) or isinstance(max_refused, bool):
            raise TypeError(
                f"max_refused is a fraction in [0, 1] and has no default; "
                f"got {max_refused!r}"
            )
        if not 0.0 <= float(max_refused) <= 1.0:
            raise ValueError(f"max_refused must lie in [0, 1]; got {max_refused!r}")
        if not fixture:
            raise ValueError(
                "fixture names the corpus and its shape (e.g. 'scifact-d384-fp32'); "
                "a recorded fraction with no name cannot be compared to anything"
            )
        self.max_refused = float(max_refused)
        self.fixture = fixture
        self.path = path
        self.n = 0
        self.n_refused = 0
        self._digests: list = []

    def record(self, v: Verdict) -> None:
        """Add one verdict.  A verdict carrying no per-row counts counts as one row."""
        if not isinstance(v, Verdict):
            raise TypeError(
                f"a gate records Verdicts, not {type(v).__name__}; call certified_topk "
                f"and hand it the second return value"
            )
        self.n += v.n_queries or 1
        self.n_refused += v.n_refused if v.n_queries else int(v.status == REFUSED)
        d = digest(v)
        if d not in self._digests:
            self._digests.append(d)

    def report(self) -> GateReport:
        rec = read_record(self.fixture, self.path)
        digests = tuple(self._digests)
        common = dict(
            fixture=self.fixture,
            n=self.n,
            n_refused=self.n_refused,
            max_refused=self.max_refused,
            digests=digests,
            recorded=rec.get("refused") if rec else None,
            recorded_digest=rec.get("digest") if rec else None,
        )
        recorded = common["recorded"]
        rec_digest = common["recorded_digest"]

        if self.n == 0:
            return GateReport(
                status=NOT_COMPARED,
                detail="no verdict was recorded inside the block. " + SCOPE_NOTE,
                **common,
            )
        if len(digests) > 1:
            return GateReport(
                status=NOT_COMPARED,
                detail=(
                    f"{len(digests)} different arithmetic configurations were recorded "
                    f"({', '.join(digests)}); a refused fraction pooled across them has "
                    f"no single denominator"
                ),
                **common,
            )
        observed = self.n_refused / self.n
        if rec_digest is not None and rec_digest != digests[0]:
            return GateReport(
                status=NOT_COMPARED,
                detail=(
                    f"{self.path} recorded fixture {self.fixture!r} under digest "
                    f"{rec_digest} and this run is {digests[0]}: dtype, kernel, bound, "
                    f"rung, k or the P4 canary changed. The gate refuses to compare "
                    f"rather than going red on an arithmetic change"
                ),
                **common,
            )
        if not within_budget(observed, self.max_refused):
            return GateReport(
                status=FAIL,
                detail=(
                    f"{self.n_refused} of {self.n} decisions were not determined "
                    f"({observed:.4f}), over the budget of {self.max_refused:.4f}"
                ),
                **common,
            )
        if recorded is not None and observed > recorded:
            return GateReport(
                status=FAIL,
                detail=(
                    f"the undetermined fraction rose from the recorded {recorded:.4f} to "
                    f"{observed:.4f} under the same digest {digests[0]}"
                ),
                **common,
            )
        return GateReport(
            status=PASS,
            detail=f"{self.n - self.n_refused} of {self.n} decisions were determined",
            **common,
        )


def report(v: Verdict) -> None:
    """Record ``v`` into the gate open in this process, if there is one; else a no-op.

    One line for a producer to call.  The ambient gate is optional by construction, so no
    code path in the library depends on a gate being open.
    """
    g = _CURRENT.get()
    if g is not None:
        g.record(v)


@contextlib.contextmanager
def gate(max_refused, fixture: str, path=GATE_FILE):
    """Open a gate.  Raises ``AssertionError`` on exit when the report says ``fail``.

    ``max_refused`` is required and is a fraction in [0, 1].  ``fixture`` names the corpus
    and its shape; it keys the recorded fraction in ``.separatrix-gate.json``, which this
    function reads and never writes.
    """
    g = Gate(max_refused, fixture, path)
    token = _CURRENT.set(g)
    try:
        yield g
    finally:
        _CURRENT.reset(token)
    rep = g.report()
    g.last_report = rep
    if rep.status == FAIL:
        raise AssertionError(str(rep))


def _demo() -> None:
    import tempfile

    from .verdict import BOUNDARY_UNDETERMINED, CERTIFIED, NEXT_ACTION

    def mk(n, refused, **kw):
        base = dict(
            kernel="gram",
            bound="cheap",
            k=10,
            dtype_in="float32",
            dtype_used="float32",
            canary="numpy/float32 clean",
        )
        base.update(kw)
        if refused:
            return Verdict(
                status=REFUSED,
                reason=BOUNDARY_UNDETERMINED,
                next_action=NEXT_ACTION[BOUNDARY_UNDETERMINED],
                n_queries=n,
                n_refused=refused,
                **base,
            )
        return Verdict(status=CERTIFIED, n_queries=n, n_refused=0, **base)

    # no default budget, and a fixture is required
    for bad in ({"max_refused": None}, {"max_refused": 1.5}, {"fixture": ""}):
        kw = dict(max_refused=0.1, fixture="f")
        kw.update(bad)
        try:
            Gate(**kw)
        except (ValueError, TypeError):
            pass
        else:  # pragma: no cover
            raise AssertionError(f"Gate accepted {bad}")

    # under budget passes; over budget raises out of the block
    with gate(0.10, "demo") as g:
        report(mk(300, 19))
    assert g.report().status == PASS
    assert abs(g.report().observed - 19 / 300) < 1e-12

    try:
        with gate(0.10, "demo") as g:
            report(mk(300, 36))
    except AssertionError as e:
        assert "over the budget" in str(e)
    else:  # pragma: no cover
        raise AssertionError("a 12 percent run passed a 10 percent budget")

    # two arithmetic configurations in one gate: no single denominator, so no comparison
    with gate(0.0, "demo") as g:
        g.record(mk(100, 0))
        g.record(mk(100, 50, kernel="direct"))
    assert g.report().status == NOT_COMPARED
    assert "denominator" in g.report().detail

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, GATE_FILE)
        v = mk(300, 19)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "schema": GATE_SCHEMA,
                    "fixtures": {"demo": {"digest": digest(v), "refused": 19 / 300}},
                },
                fh,
            )

        with gate(0.50, "demo", p) as g:  # same digest, same fraction
            g.record(v)
        assert g.report().status == PASS

        try:  # same digest, a rise -> fail even though the budget is loose
            with gate(0.50, "demo", p) as g:
                g.record(mk(300, 25))
        except AssertionError as e:
            assert "rose from" in str(e)
        else:  # pragma: no cover
            raise AssertionError("a rise under the budget was not caught")

        with gate(0.50, "demo", p) as g:  # a different dtype -> refuses to compare
            g.record(mk(300, 300, dtype_used="float16"))
        assert g.report().status == NOT_COMPARED
        assert "refuses to compare" in g.report().detail

        with open(p, encoding="utf-8") as fh:  # the gate never writes what it read
            assert json.load(fh)["fixtures"]["demo"]["refused"] == 19 / 300

    # nothing recorded is not a pass
    with gate(0.0, "demo") as g:
        pass
    assert g.report().status == NOT_COMPARED

    print("harness: ok")


if __name__ == "__main__":
    _demo()
