import numpy as np

from score_deft_oof_predictions import boundary, percentile_ranks, score_pair


def test_perfect_prediction_scores_one():
    target = np.asarray(
        [
            [0, 0, 3, 3],
            [0, 1, 1, 3],
            [2, 2, 1, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.uint8,
    )
    scores = score_pair(target.copy(), target)
    assert scores == {"miou": 1.0, "rare_recall": 1.0, "boundary_f1": 1.0}


def test_missing_rare_classes_is_hard():
    target = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
    prediction = np.asarray([[0, 0], [0, 3]], dtype=np.uint8)
    scores = score_pair(prediction, target)
    assert scores["miou"] < 1.0
    assert scores["rare_recall"] == 0.0
    assert 0.0 <= scores["boundary_f1"] <= 1.0


def test_boundary_marks_both_sides_of_transition():
    mask = np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.uint8)
    observed = boundary(mask)
    assert observed[:, 0].sum() == 0
    assert observed[:, 1].all()
    assert observed[:, 2].all()


def test_percentile_rank_increases_with_error():
    assert percentile_ranks([0.1, 0.3, 0.2]) == [0.0, 1.0, 0.5]
