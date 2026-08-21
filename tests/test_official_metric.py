"""Regression tests for organizer-style global PQ semantics."""

import numpy as np
import pytest

from filaseg.official_metric import OfficialPQAccumulator, evaluate_official_pq


def _mask(coords, shape=(8, 8)):
    out = np.zeros(shape, dtype=bool)
    for y, x in coords:
        out[y, x] = True
    return out


def test_official_pq_is_one_for_perfect_instances():
    a = _mask([(1, 1), (1, 2), (2, 1), (2, 2)])
    b = _mask([(5, 5), (5, 6), (6, 5), (6, 6)])
    result = evaluate_official_pq([([a, b], [a.copy(), b.copy()])])
    assert result.pq == pytest.approx(1.0)
    assert result.sq == pytest.approx(1.0)
    assert result.rq == pytest.approx(1.0)
    assert (result.tp, result.fp, result.fn) == (2, 0, 0)


def test_official_match_threshold_is_strictly_greater_than_half():
    truth = _mask([(1, 1), (1, 2)])
    pred = _mask([(1, 1)])  # intersection=1, union=2 => IoU exactly 0.5
    result = evaluate_official_pq([([truth], [pred])])
    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 1
    assert result.pq == pytest.approx(0.0)


def test_official_pq_is_formed_after_global_accumulation():
    truth = _mask([(1, 1), (1, 2)])
    result = evaluate_official_pq(
        [
            ([truth], [truth.copy()]),  # TP IoU=1
            ([truth], []),              # one FN
        ]
    )
    # Global PQ = 1 / (1 + 0.5*FN) = 2/3, not mean(per-record PQ)=0.5.
    assert result.pq == pytest.approx(2.0 / 3.0)
    assert (result.tp, result.fp, result.fn) == (1, 0, 1)
    assert result.n_records == 2


def test_official_semantics_allow_multiple_hit_pairs():
    # Independent GT masks are intentionally allowed to overlap. One prediction
    # can therefore produce two IoU>0.5 hit pairs under the organizer notebook
    # semantics; a one-to-one matcher would incorrectly reduce this to one TP.
    truth_a = _mask([(2, 2), (2, 3), (3, 2), (3, 3)])
    truth_b = truth_a.copy()
    pred = truth_a.copy()

    acc = OfficialPQAccumulator()
    acc.add([truth_a, truth_b], [pred])
    result = acc.result()

    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0
    assert result.pq == pytest.approx(1.0)


def test_official_empty_record_accounting():
    truth = _mask([(1, 1)])
    pred = _mask([(5, 5)])
    acc = OfficialPQAccumulator()
    acc.add([], [pred, pred.copy()])
    acc.add([truth, truth.copy(), truth.copy()], [])
    result = acc.result()
    assert result.tp == 0
    assert result.fp == 2
    assert result.fn == 3
    assert result.pq == pytest.approx(0.0)
