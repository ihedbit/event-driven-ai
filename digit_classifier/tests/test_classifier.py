"""
test_classifier.py — Tests for classifier.py

Pure Python — no NEST required.

Tests the Winner-Take-All logic and ClassificationResult dataclass in isolation.
"""

import pytest
from classifier import get_prediction, classify_spike_counts, ClassificationResult


# ── get_prediction() ──────────────────────────────────────────────────────────

class TestGetPrediction:
    def test_clear_winner(self):
        counts = [0, 0, 0, 5, 0, 0, 0, 0, 0, 0]
        pred, conf = get_prediction(counts)
        assert pred == 3
        assert conf == 5

    def test_all_silent_returns_none(self):
        counts = [0] * 10
        pred, conf = get_prediction(counts)
        assert pred is None
        assert conf == 0

    def test_tie_returns_first_winner(self):
        # argmax returns the first occurrence in a tie
        counts = [0, 3, 0, 3, 0, 0, 0, 0, 0, 0]
        pred, conf = get_prediction(counts)
        assert pred == 1   # first maximum
        assert conf == 3

    def test_last_digit_wins(self):
        counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 7]
        pred, conf = get_prediction(counts)
        assert pred == 9
        assert conf == 7

    def test_first_digit_wins(self):
        counts = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred, conf = get_prediction(counts)
        assert pred == 0
        assert conf == 4

    def test_confidence_is_winner_spike_count(self):
        counts = [1, 2, 10, 3, 4, 5, 6, 7, 8, 9]
        pred, conf = get_prediction(counts)
        assert pred == 2
        assert conf == 10

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            get_prediction([1, 2, 3])   # only 3 elements

    def test_all_ones_returns_digit_0(self):
        counts = [1] * 10
        pred, conf = get_prediction(counts)
        assert pred == 0   # first of all equal

    def test_prediction_is_valid_digit(self):
        import random
        for _ in range(50):
            counts = [random.randint(0, 10) for _ in range(10)]
            pred, conf = get_prediction(counts)
            if max(counts) > 0:
                assert 0 <= pred <= 9
            else:
                assert pred is None


# ── ClassificationResult dataclass ───────────────────────────────────────────

class TestClassificationResult:
    def test_is_confident_when_exactly_one_fires(self):
        cr = classify_spike_counts([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        assert cr.is_confident is True

    def test_not_confident_when_multiple_fire(self):
        cr = classify_spike_counts([0, 2, 0, 1, 0, 0, 0, 0, 0, 0])
        assert cr.is_confident is False

    def test_all_silent_flag(self):
        cr = classify_spike_counts([0] * 10)
        assert cr.all_silent is True
        assert cr.predicted_digit is None

    def test_not_all_silent_when_something_fires(self):
        cr = classify_spike_counts([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        assert cr.all_silent is False

    def test_predicted_digit_matches_get_prediction(self):
        counts = [0, 0, 7, 0, 0, 0, 0, 0, 0, 0]
        cr = classify_spike_counts(counts)
        pred, conf = get_prediction(counts)
        assert cr.predicted_digit == pred
        assert cr.confidence == conf

    def test_spike_counts_stored_correctly(self):
        counts = list(range(10))
        cr = classify_spike_counts(counts)
        assert cr.spike_counts == counts

    def test_repr_runs_without_error(self):
        cr = classify_spike_counts([0, 0, 3, 0, 0, 0, 0, 0, 0, 0])
        repr_str = repr(cr)
        assert "ClassificationResult" in repr_str
        assert "2" in repr_str   # predicted digit


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_large_spike_counts(self):
        counts = [0] * 10
        counts[5] = 10_000
        pred, conf = get_prediction(counts)
        assert pred == 5
        assert conf == 10_000

    def test_single_spike_is_sufficient(self):
        counts = [0] * 10
        counts[9] = 1
        pred, conf = get_prediction(counts)
        assert pred == 9
        assert conf == 1
