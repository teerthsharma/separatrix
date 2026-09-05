"""The vocabulary: statuses, exit codes, the refusal catalogue, Frontier, Verdict.

Imports only the standard library.  No numpy reaches this module, which is the point:
the honesty rules -- what may never be printed, what a refusal must carry -- are then
enforceable by a test that never touches an array.

The three things this file fixes for every other module:

  * a status is one of four strings and maps to one of five exit codes;
  * a refusal carries a typed code, the named object, and a next action -- never a
    bare boolean, never a score;
  * a printed field is scanned against BANNED in ``Verdict.__post_init__``, raising
    ``ValueError`` (never ``assert``, which is stripped under ``python -O``).

Usage errors -- a bare score array, k out of range, mixed dtypes -- are exceptions of
exit class 3 and are deliberately NOT in the refusal catalogue.  A Verdict is a
statement about the caller's data; a usage error is a statement about the caller's
code, and an agent consuming the JSON must be able to tell "go re-observe" from "go
fix your call".

Self-check:  python -m separatrix.verdict
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields

# -- statuses ---------------------------------------------------------------------------

CERTIFIED = "CERTIFIED"
CERTIFIED_UPCAST = "CERTIFIED_UPCAST"
NOT_CERTIFIED = "NOT CERTIFIED"
REFUSED = "REFUSED"

STATUSES = (CERTIFIED, CERTIFIED_UPCAST, NOT_CERTIFIED, REFUSED)

# Exit class 3 belongs to exceptions and has no status.  CERTIFIED_UPCAST is 4 and never
# 0, because a CI job must not read a pass for a computation the caller's pipeline does
# not run.
EXIT = {CERTIFIED: 0, NOT_CERTIFIED: 1, REFUSED: 2, CERTIFIED_UPCAST: 4}
EXIT_USAGE = 3

# -- the refusal catalogue --------------------------------------------------------------

BOUNDARY_UNDETERMINED = "BOUNDARY_UNDETERMINED"
GRAM_CANCELLATION = "GRAM_CANCELLATION"
EXACT_TIE = "EXACT_TIE"
RANGE_UNSAFE = "RANGE_UNSAFE"
NONFINITE_INPUT = "NONFINITE_INPUT"
BOUND_VACUOUS = "BOUND_VACUOUS"
REDUCED_PRECISION_ARITHMETIC = "REDUCED_PRECISION_ARITHMETIC"
ESCALATION_BUDGET = "ESCALATION_BUDGET"

REASONS = (
    BOUNDARY_UNDETERMINED,
    GRAM_CANCELLATION,
    EXACT_TIE,
    RANGE_UNSAFE,
    NONFINITE_INPUT,
    BOUND_VACUOUS,
    REDUCED_PRECISION_ARITHMETIC,
    ESCALATION_BUDGET,
)

# EXACT_TIE is the one entry that exits 1 rather than 2: the computation succeeded and
# proved the two exact scores equal.  No arithmetic anywhere resolves it, so it is a
# NOT CERTIFIED, not a refusal to try.
REASON_EXIT = {r: 2 for r in REASONS}
REASON_EXIT[EXACT_TIE] = 1

NEXT_ACTION = {
    BOUNDARY_UNDETERMINED: (
        "The two enclosures at the rank-k boundary overlap. Pass escalate=True to decide "
        "it exactly, or bound='tight', or per_pair=True, or recompute in a wider dtype."
    ),
    GRAM_CANCELLATION: (
        "The direct kernel separates this pair; the Gram identity does not. Pass "
        "kernel='direct', or torch.cdist(..., compute_mode='donot_use_mm_for_euclid_dist'), "
        "or scipy.spatial.distance.cdist for these queries. torch.cdist switches to the "
        "Gram identity above 25 rows, and this refusal names that switch as the cause."
    ),
    EXACT_TIE: (
        "The two scores are equal in exact arithmetic. Supply a tie-break rule you own; "
        "no precision removes this one."
    ),
    RANGE_UNSAFE: (
        "Recompute in float32 in your own pipeline, or pass upcast=True to certify the "
        "float32 ranking instead."
    ),
    NONFINITE_INPUT: (
        "Re-observe or drop the named row. separatrix will not impute a missing value."
    ),
    BOUND_VACUOUS: (
        "Recompute in a wider dtype, or reduce d. This is a stated domain limit of the "
        "a-priori bound, not a property of your data."
    ),
    REDUCED_PRECISION_ARITHMETIC: (
        "The arithmetic rounds coarser than the declared unit roundoff (TF32, bfloat16 "
        "inputs, AMX). Set the backend to full precision for this call, or pass the true "
        "work_dtype."
    ),
    ESCALATION_BUDGET: (
        "The frontier is wider than max_escalations. Raise the budget, or narrow k."
    ),
}

# -- what is never printed ----------------------------------------------------------------

# Enforced in Verdict.__post_init__ over every printed field, and by
# test_no_banned_phrase, which greps the installed package as well.
BANNED = (
    "probably",
    "likely",
    "confidence",
    "confident",
    "approximately correct",
    "should be fine",
    "no error detected",
    "verified correct",
    "guaranteed accurate",
    "is wrong",
    "unreliable",
    "safe",
)

# Word boundaries, so RANGE_UNSAFE and "unsafe" do not trip \bsafe\b, and "unlikely"
# does not trip \blikely\b.
_BANNED_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in BANNED) + r")\b", re.I)

# A line of prose that must quote prior art honestly (CADNA's "95 percent" estimate, for
# one) carries this trailing marker.  The grep test skips marked lines and asserts the
# marker count is at most MAX_QUOTE_MARKERS, so the exemption cannot grow into a hole.
QUOTE_MARKER = "# sx: quote"
MAX_QUOTE_MARKERS = 12


def scan(text: str) -> list[str]:
    """Every banned phrase occurring in ``text``. Empty list is the only passing result."""
    return [m.group(0) for m in _BANNED_RE.finditer(text)]


# -- exceptions ---------------------------------------------------------------------------


class Refusal(Exception):
    """A typed refusal raised by the numerical layer, carrying its next action.

    Raised, not returned, because it aborts the enclosure -- a precondition failure is a
    refusal with no verdict, never a verdict with a warning.  The public surface catches
    it and builds the REFUSED Verdict; nothing here decides how it is presented.
    """

    def __init__(self, reason: str, detail: str = "", next_action: str = "") -> None:
        if reason not in REASONS:
            raise ValueError(f"{reason!r} is not in the refusal catalogue: {REASONS}")
        self.reason = reason
        self.detail = detail
        self.next_action = next_action or NEXT_ACTION[reason]
        super().__init__(f"{reason}: {detail}" if detail else reason)

    @property
    def exit_code(self) -> int:
        return REASON_EXIT[self.reason]


# -- the frontier -------------------------------------------------------------------------


@dataclass(frozen=True)
class Frontier:
    """The pair whose enclosures decide the rank-k boundary, and by how much.

    ``inside`` is the member of the returned set with the largest upper bound; ``outside``
    is the non-member with the smallest lower bound.  Those two indices, and no others,
    are what the rule compares -- see ``decide.topk_determined``.

    ``deficit = gap - width``.  Positive is determined, non-positive is not.  There is no
    third scale and no normalisation: gap and width are both in score units.
    """

    row: int
    inside: int
    outside: int
    inside_lo: float
    inside_hi: float
    outside_lo: float
    outside_hi: float
    gap: float  # D[outside] - D[inside]
    width: float  # R[inside] + R[outside]

    @property
    def deficit(self) -> float:
        return self.gap - self.width

    @property
    def determined(self) -> bool:
        return self.deficit > 0.0

    def __str__(self) -> str:
        return (
            f"row {self.row}: in #{self.inside} [{self.inside_lo:.6e}, {self.inside_hi:.6e}]  "
            f"out #{self.outside} [{self.outside_lo:.6e}, {self.outside_hi:.6e}]  "
            f"gap {self.gap:.6e}  width {self.width:.6e}  deficit {self.deficit:.6e}"
        )


# -- the verdict --------------------------------------------------------------------------

# P5 is not testable by any probe tried (see enclose.P5_NOTE), so the accumulator width is
# a declared assumption.  It travels on every Verdict, where a consumer can see it, rather
# than in a docstring where no consumer can.
ACCUM_ASSUMED = "storage dtype"


@dataclass(frozen=True)
class Verdict:
    """One decision's status, and everything a caller needs to act on it.

    Never carries a probability, a confidence, a percentage or a score. Four statuses, an
    exact margin, and a next action.
    """

    status: str
    reason: str = ""
    detail: str = ""
    next_action: str = ""

    # what was computed
    kernel: str = ""
    bound: str = ""
    per_pair: bool = False
    k: int = 0
    largest: bool = False
    n_queries: int = 0
    n_refused: int = 0
    dtype_in: str = ""
    dtype_used: str = ""
    accum_assumed: str = ACCUM_ASSUMED
    canary: str = ""

    # escalation
    escalated: bool = False
    n_escalated: int = 0
    float_set_differed: bool = False

    frontiers: tuple[Frontier, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"{self.status!r} is not a status: {STATUSES}")
        if self.reason and self.reason not in REASONS:
            raise ValueError(f"{self.reason!r} is not in the refusal catalogue: {REASONS}")
        if self.status in (CERTIFIED, CERTIFIED_UPCAST) and self.reason:
            raise ValueError(f"a certificate carries no refusal reason, got {self.reason!r}")
        if self.status in (NOT_CERTIFIED, REFUSED) and not self.reason:
            raise ValueError(f"{self.status} must name a reason from the catalogue")
        if self.status == REFUSED and not self.next_action:
            raise ValueError("a refusal must name a next action")
        if not self.accum_assumed:
            raise ValueError("accum_assumed must travel on every verdict (P5 is an assumption)")
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, str):
                hits = scan(v)
                if hits:
                    raise ValueError(f"banned phrase {hits!r} in Verdict.{f.name}: {v!r}")

    @property
    def exit_code(self) -> int:
        return EXIT[self.status]

    @property
    def certified(self) -> bool:
        return self.status in (CERTIFIED, CERTIFIED_UPCAST)

    def __str__(self) -> str:
        head = self.status if not self.reason else f"{self.status} ({self.reason})"
        bits = [head]
        if self.n_queries:
            bits.append(f"{self.n_queries - self.n_refused}/{self.n_queries} determined")
        if self.kernel:
            bits.append(f"kernel={self.kernel} bound={self.bound}")
        if self.dtype_in:
            used = self.dtype_used or self.dtype_in
            bits.append(f"dtype {self.dtype_in} -> {used}")
        bits.append(f"accumulator assumed {self.accum_assumed}")
        out = "  ".join(bits)
        if self.detail:
            out += "\n  " + self.detail
        if self.next_action:
            out += "\n  next: " + self.next_action
        return out


def _demo() -> None:
    v = Verdict(status=CERTIFIED, kernel="gram", bound="cheap", k=10, n_queries=300)
    assert v.exit_code == 0 and v.certified and v.accum_assumed
    assert "accumulator assumed" in str(v)

    r = Verdict(
        status=REFUSED,
        reason=BOUNDARY_UNDETERMINED,
        next_action=NEXT_ACTION[BOUNDARY_UNDETERMINED],
        n_queries=300,
        n_refused=19,
    )
    assert r.exit_code == 2 and not r.certified

    t = Verdict(status=NOT_CERTIFIED, reason=EXACT_TIE, next_action=NEXT_ACTION[EXACT_TIE])
    assert t.exit_code == 1

    # every catalogue entry has a next action, and none of them is phrased as a judgement
    for reason in REASONS:
        assert NEXT_ACTION[reason]
        assert not scan(NEXT_ACTION[reason]), (reason, scan(NEXT_ACTION[reason]))

    # the banned scan is word-bounded: the refusal code itself must survive it
    assert scan(RANGE_UNSAFE) == [] and scan("unsafe range") == []
    assert scan("this is safe") == ["safe"]
    assert scan("unlikely") == [] and scan("likely fine") == ["likely"]

    # a banned phrase in any printed field is a ValueError, not a warning
    try:
        Verdict(status=CERTIFIED, detail="this ranking should be fine")
    except ValueError as e:
        assert "should be fine" in str(e)
    else:  # pragma: no cover
        raise AssertionError("a banned phrase was printed")

    # a refusal without a next action is not constructible
    try:
        Verdict(status=REFUSED, reason=BOUNDARY_UNDETERMINED, next_action="")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a refusal shipped with no next action")

    f = Frontier(0, 3, 7, 1.0, 1.2, 1.1, 1.4, gap=0.1, width=0.5)
    assert not f.determined and abs(f.deficit + 0.4) < 1e-12
    assert Frontier(0, 3, 7, 1.0, 1.2, 1.9, 2.1, gap=1.0, width=0.5).determined

    print("verdict: ok")


if __name__ == "__main__":
    _demo()
