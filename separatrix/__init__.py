"""separatrix -- which of your decisions were determined by your data, and which by rounding.

separatrix tells you which of your top-k, argmin or threshold decisions were determined by
your data, and which were decided by the rounding of the kernel you happened to call -- or
refuses, naming the boundary and the code change that would settle it.

What CERTIFIED means, in full
-----------------------------

Let X and Q be the arrays **as stored in memory** and let s be the value of the named score
formula in **exact real arithmetic** on those stored numbers.  separatrix returns an index
set T and radii R >= 0 with |D_i - s_i| <= R_i for every i, and says CERTIFIED only when

    max_{i in T} (D_i + R_i)  <  min_{j not in T} (D_j - R_j)

and then T is the top-k set of s, and of every vector in the box.  The corollary is the
only sentence worth quoting: **any other evaluation of the same formula on the same stored
bytes, whose own error is contained by its own certified box, returns the same T.**  Batch
size, BLAS backend, thread count, chunk size, reduction order, FMA, and `torch.cdist`'s
25-row switch to the Gram identity cannot change it.  Determinism across backends is a
theorem here, not a measurement over reruns.

What it never claims
--------------------

Not that the scores are correct, and nothing at all about how the numbers got into the
array: a 384-dimensional embedding out of a float16 forward pass carries ~1e-3 relative
error, three to four orders above the float32 rounding this certifies.  CERTIFIED means
*rounding did not choose this ranking*.  It says nothing about the model, the quantiser or
the sensor.  A refusal is not a finding: it means no certificate was formed, and only
`escalate=True` decides which way the boundary actually falls.

The mechanism is not new
------------------------

CGAL's `Filtered_predicate` and `Interval_nt` have shipped exactly this rule since ~2001 --
interval enclosure, disjoint certifies, overlap escalates to exact -- and Melquiond and
Pion formally certified the static a-priori filters, which are the same class of bound.
The bound itself is one page of Higham's *ASNA* chapter 3.  The difference here is the
adversary: a rank-k *set* boundary in 384 dimensions in Python against a predicate's sign
in 2 or 3 dimensions in C++.

Layout
------

    separatrix.verdict    statuses, exit codes, the refusal catalogue, Frontier, Verdict
    separatrix.enclose    unit roundoff, gamma, preconditions P1-P5, the two kernels' radii
    separatrix.decide     the rule, over (D, R) alone
    separatrix.exact      the scaled-integer oracle and the escalation rung

Each module has a runnable self-check: ``python -m separatrix.enclose``.
"""

from __future__ import annotations

from . import decide, enclose, exact, verdict
from .decide import rows_determined, topk_determined, topk_set, worst_corner
from .enclose import Enclosure, enclose_scores, eta, gamma, unit_roundoff
from .exact import escalate, escalate_row, exact_sq, exact_topk
from .verdict import (
    ACCUM_ASSUMED,
    BANNED,
    CERTIFIED,
    CERTIFIED_UPCAST,
    EXIT,
    NOT_CERTIFIED,
    REASONS,
    REFUSED,
    Frontier,
    Refusal,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # the three decisions (see __getattr__: they live in separatrix.api)
    "certified_topk",
    "certified_argmin",
    "certified_threshold",
    "gate",
    # the pure layers, exported because the tests and the users both need them
    "topk_determined",
    "topk_set",
    "worst_corner",
    "rows_determined",
    "enclose_scores",
    "Enclosure",
    "gamma",
    "eta",
    "unit_roundoff",
    "exact_sq",
    "exact_topk",
    "escalate",
    "escalate_row",
    # the vocabulary
    "Verdict",
    "Frontier",
    "Refusal",
    "CERTIFIED",
    "CERTIFIED_UPCAST",
    "NOT_CERTIFIED",
    "REFUSED",
    "REASONS",
    "EXIT",
    "BANNED",
    "ACCUM_ASSUMED",
    "decide",
    "enclose",
    "exact",
    "verdict",
]

# The three decisions and the gate live in `separatrix.api`, resolved on first use.  This
# import costs numpy and nothing else: torch, faiss and datasets are never imported here,
# and `test_import_costs_nothing_optional` blocks them with a meta-path finder to prove it.
_LAZY = {
    "certified_topk": "api",
    "certified_argmin": "api",
    "certified_threshold": "api",
    "gate": "api",
}


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{mod}", __name__), name)


def __dir__():
    return sorted(__all__)
