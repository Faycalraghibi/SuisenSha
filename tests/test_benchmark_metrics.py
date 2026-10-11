"""Unit tests for extended PR-04 benchmark metrics and breakdowns."""

import math

import numpy as np

from pipeline.evaluation.metrics import (
    average_popularity_at_k,
    catalog_coverage_at_k,
    compute_item_popularity_tiers,
    compute_user_activity_quartiles,
    evaluate_by_item_popularity_tiers,
    evaluate_by_user_activity,
    gini_exposure,
    intra_list_diversity_at_k,
    mean_average_popularity_at_k,
    mean_intra_list_diversity_at_k,
    mean_novelty_at_k,
    mrr_at_k,
    novelty_at_k,
    precision_at_k,
    recall_at_k,
)


class TestMRR:
    def test_first_rank(self):
        assert mrr_at_k([10, 20, 30], target=10, k=3) == 1.0

    def test_second_rank(self):
        assert mrr_at_k([10, 20, 30], target=20, k=3) == 0.5

    def test_third_rank(self):
        assert abs(mrr_at_k([10, 20, 30], target=30, k=3) - 1.0 / 3.0) < 1e-6

    def test_miss(self):
        assert mrr_at_k([10, 20, 30], target=99, k=3) == 0.0

    def test_cutoff(self):
        assert mrr_at_k([10, 20, 30, 40], target=40, k=3) == 0.0

    def test_set_target(self):
        assert mrr_at_k([10, 20, 30], target={20, 30}, k=3) == 0.5


class TestRecallAndPrecisionExtended:
    def test_recall_single_int(self):
        assert recall_at_k([1, 2, 3], relevant=2, k=3) == 1.0
        assert recall_at_k([1, 2, 3], relevant=99, k=3) == 0.0

    def test_precision_single_int(self):
        assert precision_at_k([1, 2, 3], relevant=2, k=3) == 1.0 / 3.0
        assert precision_at_k([1, 2, 3], relevant=99, k=3) == 0.0
        assert precision_at_k([1, 2, 3], relevant=1, k=0) == 0.0


class TestCatalogCoverage:
    def test_full_coverage(self):
        recs = [[1, 2], [3, 4], [5]]
        assert catalog_coverage_at_k(recs, num_items=5, k=2) == 1.0

    def test_partial_coverage(self):
        recs = {1: [1, 2], 2: [2, 3]}
        assert catalog_coverage_at_k(recs, num_items=6, k=2) == 0.5

    def test_zero_items(self):
        assert catalog_coverage_at_k([[1, 2]], num_items=0, k=2) == 0.0


class TestIntraListDiversity:
    def test_short_list(self):
        assert intra_list_diversity_at_k([1], item_features={}, k=5) == 0.0

    def test_genre_diversity(self):
        features = {
            1: {"Action"},
            2: {"Comedy"},
            3: {"Action", "Comedy"},
        }
        # pair (1, 2): union={Action, Comedy}, overlap=0 -> dist=1.0
        # pair (1, 3): union={Action, Comedy}, overlap={Action} -> dist=1 - 1/2 = 0.5
        # pair (2, 3): union={Action, Comedy}, overlap={Comedy} -> dist=1 - 1/2 = 0.5
        # mean = (1.0 + 0.5 + 0.5) / 3 = 2.0 / 3 = 0.6667
        score = intra_list_diversity_at_k([1, 2, 3], item_features=features, k=3)
        assert abs(score - 2.0 / 3.0) < 1e-4

    def test_embedding_diversity(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        features = {1: v1, 2: v2}
        # orthogonal vectors: cosine=0.0 -> dist=1.0
        score = intra_list_diversity_at_k([1, 2], item_features=features, k=2)
        assert abs(score - 1.0) < 1e-4

    def test_mean_diversity(self):
        features = {1: {"Action"}, 2: {"Action"}}
        recs = {10: [1, 2]}
        assert mean_intra_list_diversity_at_k(recs, features, k=2) == 0.0


class TestNoveltyAndPopularity:
    def test_novelty_calculation(self):
        popularity = {1: 100, 2: 10}
        total_users = 100
        # Item 1: pop=100/100 -> p=1.0 -> -log2(1.0) = 0.0
        # Item 2: pop=10/100 -> p=0.1 -> -log2(0.1) = 3.3219
        score = novelty_at_k([1, 2], item_popularity=popularity, total_users=total_users, k=2)
        expected = (0.0 + (-math.log2(0.1))) / 2.0
        assert abs(score - expected) < 1e-4

    def test_mean_novelty_and_popularity(self):
        popularity = {1: 50, 2: 20}
        recs = {10: [1, 2]}
        assert average_popularity_at_k([1, 2], popularity, k=2) == 35.0
        assert mean_average_popularity_at_k(recs, popularity, k=2) == 35.0
        assert mean_novelty_at_k(recs, popularity, 100, k=2) > 0.0


class TestGiniExposure:
    def test_perfect_equality(self):
        # 4 items each recommended exactly once
        recs = [[1, 2], [3, 4]]
        # When all items have equal count, Gini index should be 0.0
        gini = gini_exposure(recs, num_items=4, k=2)
        assert abs(gini - 0.0) < 1e-5

    def test_high_inequality(self):
        # 10 items in catalog, but item 1 gets all recommendations
        recs = [[1], [1], [1], [1]]
        gini = gini_exposure(recs, num_items=10, k=1)
        # 9 items with 0, 1 item with 4: Gini should be close to (n-1)/n = 9/10 = 0.9
        assert abs(gini - 0.9) < 1e-5

    def test_empty_or_single_item(self):
        assert gini_exposure([], num_items=10, k=5) == 0.0
        assert gini_exposure([[1]], num_items=1, k=1) == 0.0


class TestBreakdowns:
    def test_user_activity_quartiles(self):
        interactions = {
            1: [1],
            2: [1, 2],
            3: [1, 2, 3],
            4: [1, 2, 3, 4],
        }
        quartiles = compute_user_activity_quartiles(interactions)
        assert quartiles["Q1"] == {1}
        assert quartiles["Q2"] == {2}
        assert quartiles["Q3"] == {3}
        assert quartiles["Q4"] == {4}

    def test_item_popularity_tiers(self):
        popularity = {1: 100, 2: 80, 3: 50, 4: 20, 5: 10, 6: 5, 7: 2, 8: 1, 9: 0, 10: 0}
        tiers = compute_item_popularity_tiers(popularity, num_items=10)
        assert len(tiers["Head"]) == 2  # top 20%
        assert len(tiers["Torso"]) == 3  # next 30%
        assert len(tiers["Tail"]) == 5  # bottom 50%
        assert 1 in tiers["Head"]
        assert 10 in tiers["Tail"]

    def test_evaluate_by_user_activity(self):
        recs = {1: [10, 20], 2: [30, 40]}
        targets = {1: 10, 2: 99}
        quartiles = {"Low": {1}, "High": {2}}
        res = evaluate_by_user_activity(recs, targets, quartiles, k_values=(2,))
        assert res["Low"]["HitRate@2"] == 1.0
        assert res["Low"]["Recall@2"] == 1.0
        assert res["Low"]["MRR@2"] == 1.0
        assert res["High"]["HitRate@2"] == 0.0

    def test_evaluate_by_item_popularity_tiers(self):
        recs = {1: [10, 20], 2: [10, 30]}
        targets = {1: 10, 2: 30}
        item_tiers = {"Head": {10}, "Tail": {20, 30}}
        res = evaluate_by_item_popularity_tiers(recs, targets, item_tiers, k_values=(2,))
        assert res["Head"]["HitRate@2"] == 1.0
        assert res["Tail"]["HitRate@2"] == 1.0
        # 10 appears twice out of 4 recommendations -> 50% exposure
        assert res["Head"]["exposure_share@10"] == 0.5
