"""The torch arm: the 25-row switch, the keyword that turns it off, and batch 32 vs 64.

torch is an optional dependency and every test here skips cleanly without it.  What these
tests pin is the origin observation the whole package is about -- `torch.cdist` switches to
the cancellation-prone Gram identity above 25 rows -- as a *measurement on the installed
torch*, not as a claim repeated from a docstring.  The exact spelling of the keyword that
turns the switch off is also verified here, because a next action that names a keyword the
library does not have is worse than no next action.
"""

from __future__ import annotations

import numpy as np
import pytest

from separatrix.api import certified_topk
from separatrix.cli import near_duplicate_corpus
from separatrix.decide import topk_set

torch = pytest.importorskip("torch", reason="torch is optional; this arm is skipped")

DIRECT = "donot_use_mm_for_euclid_dist"


def _sets(D, k):
    return [frozenset(topk_set(np.asarray(D[i], dtype=np.float64), k).tolist())
            for i in range(D.shape[0])]


def test_the_compute_mode_keyword_exists_and_names_the_direct_kernel():
    """The next action on GRAM_CANCELLATION names this keyword. It must be the real one."""
    X = np.array([[1e6, 0.0], [1e6 + 1e-6, 0.0]], dtype=np.float64)
    t = torch.from_numpy(X)
    for mode in ("use_mm_for_euclid_dist_if_necessary", "use_mm_for_euclid_dist", DIRECT):
        torch.cdist(t[:1], t, compute_mode=mode)  # no TypeError: the spelling is right
    mm = torch.cdist(t[:1], t, compute_mode="use_mm_for_euclid_dist").numpy()
    direct = torch.cdist(t[:1], t, compute_mode=DIRECT).numpy()
    assert mm[0, 1] == 0.0, "the Gram identity cancels this pair to exactly zero"
    assert direct[0, 1] > 0.0, "the direct kernel separates it"


def test_the_switch_is_at_twenty_five_rows_and_is_measured_not_quoted():
    """Below the switch the two compute modes are bit-identical; above it they are not.

    This is the whole origin story as one assertion: the same stored bytes, the same call,
    and the answer changes because the row count crossed a threshold inside the library.
    """
    rng = np.random.default_rng(0)
    A = np.ascontiguousarray(rng.normal(size=(40, 8)).astype(np.float32))
    t = torch.from_numpy(A)

    def spread(r):
        mm = torch.cdist(t[:r], t[:r]).numpy()
        direct = torch.cdist(t[:r], t[:r], compute_mode=DIRECT).numpy()
        return float(np.max(np.abs(mm - direct)))

    assert spread(24) == 0.0 and spread(25) == 0.0, "at or below 25 rows there is no switch"
    assert spread(26) > 0.0, "above 25 rows torch.cdist switches to the Gram identity"


def test_a_certified_set_survives_torch_at_two_batch_sizes():
    """Frame 2, as an assertion: certified rows cannot move, refused rows may.

    Two batch sizes are two numerically distinct evaluations of one formula on one set of
    stored bytes.  The certificate's corollary says a CERTIFIED row cannot move between
    them; `torch.use_deterministic_algorithms` would make the two runs agree on a value
    that was never determined, which is a different and weaker property.
    """
    X, Q = near_duplicate_corpus(n=400, d=48, m=64, dups=150, jitter=1e-8)
    X = np.ascontiguousarray(X.astype(np.float32))
    Q = np.ascontiguousarray(Q.astype(np.float32))
    k = 10
    tX, tQ = torch.from_numpy(X), torch.from_numpy(Q)

    def batched(b):
        return np.concatenate([torch.cdist(tQ[s:s + b], tX).numpy()
                               for s in range(0, len(tQ), b)])

    s32, s64 = _sets(batched(32), k), _sets(batched(64), k)
    _, v = certified_topk(X, Q, k=k)
    refused = {f.row for f in v.frontiers}
    assert len(refused) == v.n_refused

    moved = [i for i in range(len(Q)) if s32[i] != s64[i]]
    assert not (set(moved) - refused), (
        f"rows {sorted(set(moved) - refused)} were certified and still moved between "
        f"torch batch 32 and batch 64"
    )


def test_torch_cdist_agrees_with_separatrix_on_every_certified_row():
    """torch's Gram path and torch's direct path against the numpy certificate."""
    X, Q = near_duplicate_corpus(n=300, d=32, m=48, dups=100, jitter=1e-8)
    X = np.ascontiguousarray(X.astype(np.float32))
    Q = np.ascontiguousarray(Q.astype(np.float32))
    k = 8
    tX, tQ = torch.from_numpy(X), torch.from_numpy(Q)
    idx, v = certified_topk(X, Q, k=k)
    refused = {f.row for f in v.frontiers}

    for name, D in (
        ("torch.cdist mm", torch.cdist(tQ, tX).numpy()),
        ("torch.cdist direct", torch.cdist(tQ, tX, compute_mode=DIRECT).numpy()),
    ):
        got = _sets(D, k)
        for i in range(len(Q)):
            if i in refused:
                continue
            assert got[i] == frozenset(idx[i].tolist()), (name, i)
