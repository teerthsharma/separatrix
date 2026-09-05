"""The rule, scored against exhibited witnesses and against the box's own extremum.

Nothing here computes a distance.  These tests are evidence that *given a correct
enclosure the verdict is exact*, and none of them is evidence that the enclosure is
correct -- that is `test_enclose.py`, and the separation is deliberate.
"""

from __future__ import annotations

import numpy as np
import pytest

from separatrix.decide import (
    broadcast_radius,
    naive_boundary_pair,
    rows_determined,
    topk_determined,
    topk_set,
    worst_corner,
)


def test_boundary_pair_rule_is_unsound_and_ours_is_not():
    """The false theorem, its witness, and the rule that survives it.

    A benchmark cannot find this: on all three prototype corpora the naive rule and the
    sound one returned identical refusal counts (19/19, 107/107, 20/20).  Only this
    counterexample and a heteroscedastic-norm corpus catch it.
    """
    D = np.array([0.0, 1.0, 2.0, 10.0])
    R = np.array([12.0, 0.0, 0.0, 0.0])

    # the naive rank-k / rank-(k+1) comparison certifies {0, 1}
    assert naive_boundary_pair(D, R, 2) is True
    assert set(topk_set(D, 2).tolist()) == {0, 1}

    # and the exhibited witness inside the box has a different top-2
    witness = np.array([11.0, 1.0, 2.0, 10.0])
    assert np.all(np.abs(witness - D) <= R)
    assert set(topk_set(witness, 2).tolist()) == {1, 2}

    # the sound rule refuses, and names the pair that actually blocks it
    f = topk_determined(D, R, 2)
    assert f is not None
    assert (f.inside, f.outside) == (0, 2)
    assert f.deficit <= 0


def test_the_two_rules_agree_when_radii_are_constant():
    """Nothing is given up: with constant R the sound rule degenerates to the naive one."""
    rng = np.random.default_rng(0)
    agree = 0
    for _ in range(3000):
        D = rng.standard_normal(10)
        R = np.full(10, float(rng.random()) * 0.3)
        k = int(rng.integers(1, 9))
        assert (topk_determined(D, R, k) is None) == naive_boundary_pair(D, R, k)
        agree += 1
    assert agree == 3000


def test_certified_survives_the_worst_corner():
    """Exhaustive over the box, not Monte Carlo over it.

    The corner (members at D+R, non-members at D-R) is the extremum of the whole box for
    this question, so a decision that survives it survives every point of the box.  This
    is a proof of the rule, not a sample of it.
    """
    rng = np.random.default_rng(1)
    certified = 0
    for _ in range(5000):
        n = int(rng.integers(4, 20))
        k = int(rng.integers(1, n - 1))
        D = rng.standard_normal(n) * float(rng.choice([1.0, 1e-6, 1e6]))
        R = rng.random(n) ** 2 * float(np.abs(D).max()) * 0.3
        largest = bool(rng.integers(2))
        if topk_determined(D, R, k, largest=largest) is not None:
            continue
        T = topk_set(D, k, largest=largest)
        corner = worst_corner(-D if largest else D, R, T)
        assert set(topk_set(corner, k).tolist()) == set(T.tolist())
        certified += 1
    assert certified > 500, certified


def test_largest_is_negate_and_reuse():
    """On HETEROSCEDASTIC radii -- constant radii would pass under the unsound rule too."""
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(4000):
        n = int(rng.integers(4, 14))
        k = int(rng.integers(1, n - 1))
        D = rng.standard_normal(n)
        R = rng.random(n) ** 3  # heteroscedastic on purpose
        a = topk_determined(D, R, k, largest=True)
        b = topk_determined(-D, R, k, largest=False)
        assert (a is None) == (b is None)
        if a is None:
            assert set(topk_set(D, k, largest=True).tolist()) == set(
                topk_set(-D, k).tolist()
            )
            checked += 1
    assert checked > 200, checked
    # the returned interval is reported in the caller's orientation, not the negated one
    D = np.array([5.0, 1.0, 0.9])
    R = np.array([0.0, 0.2, 0.2])
    f = topk_determined(D, R, 1, largest=True)
    assert f is None
    f2 = topk_determined(D, R, 2, largest=True)
    assert f2 is not None and f2.inside_lo <= f2.inside_hi


def test_order_is_not_claimed_unless_asked():
    """A box where the SET is fixed and two members swap inside it."""
    D = np.array([0.0, 0.01, 5.0])
    R = np.array([0.05, 0.05, 0.05])
    assert topk_determined(D, R, 2) is None
    f = topk_determined(D, R, 2, ordered=True)
    assert f is not None and {f.inside, f.outside} == {0, 1}


def test_ordered_refuses_more_often_than_the_set():
    """k boundaries instead of one; the multiplier is measured, not asserted as a footnote."""
    rng = np.random.default_rng(3)
    n, k, trials = 200, 10, 400
    set_ref = ordered_ref = 0
    for _ in range(trials):
        D = rng.standard_normal(n)
        R = np.full(n, 0.02)
        set_ref += topk_determined(D, R, k) is not None
        ordered_ref += topk_determined(D, R, k, ordered=True) is not None
    assert ordered_ref >= set_ref
    assert ordered_ref > 0, "the corpus is too easy for the ordered claim to differ"


def test_a_frontier_carries_the_pair_the_intervals_and_the_deficit():
    D = np.array([1.0, 1.2, 3.0])
    R = np.array([0.3, 0.3, 0.0])
    f = topk_determined(D, R, 1)
    assert f is not None
    assert (f.inside, f.outside) == (0, 1)
    assert f.inside_lo == pytest.approx(0.7) and f.inside_hi == pytest.approx(1.3)
    assert f.gap == pytest.approx(0.2) and f.width == pytest.approx(0.6)
    assert f.deficit == pytest.approx(-0.4)
    assert "deficit" in str(f)


def test_strictness_at_exactly_touching_enclosures():
    """Touching is not disjoint. The rule is strict, so equality does not certify."""
    D = np.array([0.0, 1.0])
    R = np.array([0.5, 0.5])
    assert topk_determined(D, R, 1) is not None  # hi = 0.5 == lo = 0.5
    assert topk_determined(D, R * 0.999, 1) is None


def test_zero_radius_certifies_whatever_the_floats_say():
    D = np.array([3.0, 1.0, 2.0, 9.0])
    assert topk_determined(D, 0.0, 2) is None
    assert set(topk_set(D, 2).tolist()) == {1, 2}


def test_k_out_of_range_is_a_usage_error():
    D = np.arange(5.0)
    for k in (0, -1, 5, 9):
        with pytest.raises(ValueError):
            topk_determined(D, 0.0, k)


def test_rows_determined_matches_the_row_loop():
    rng = np.random.default_rng(4)
    D = rng.standard_normal((7, 30))
    R = rng.random((7, 1)) * 0.2
    got = rows_determined(D, R, 5)
    assert len(got) == 7
    for i, f in enumerate(got):
        want = topk_determined(D[i], R[i], 5, row=i)
        assert (f is None) == (want is None)
        if f is not None:
            assert f.row == i and (f.inside, f.outside) == (want.inside, want.outside)


def test_scalar_radius_broadcasts():
    D = np.array([0.0, 1.0, 2.0])
    assert topk_determined(D, 0.1, 1) is None
    assert topk_determined(D, np.float64(0.1), 1) is None


def test_a_shape_mismatched_radius_is_this_librarys_own_error_not_a_bare_numpy_one():
    """A bring-your-own ``(D, R)`` caller can pass a shape that does not fit ``D``.  The
    message must name both shapes, never surface as the bare `ValueError` numpy's own
    `broadcast_to` raises three frames down with neither shape attributed to the caller.
    """
    D = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError) as e:
        topk_determined(D, np.array([0.1, 0.2]), 1)
    msg = str(e.value)
    assert "(2,)" in msg and "(3,)" in msg

    with pytest.raises(ValueError) as e:
        broadcast_radius(D, np.array([0.1, 0.2]))
    assert "(2,)" in str(e.value) and "(3,)" in str(e.value)

    # the passing case still returns a genuine broadcast view
    got = broadcast_radius(D, 0.5)
    assert got.shape == D.shape and (got == 0.5).all()
