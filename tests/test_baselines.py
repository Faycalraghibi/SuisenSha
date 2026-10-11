from __future__ import annotations

from pipeline.models.base import Recommender
from pipeline.models.baselines import (
    BPRMFRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
)


def _get_train_data():
    return {
        1: [1, 2, 3, 4],
        2: [2, 3, 5],
        3: [1, 3, 4, 5],
        4: [2, 4],
    }


def test_random_recommender():
    train = _get_train_data()
    rec = RandomRecommender(seed=42)
    assert isinstance(rec, Recommender)
    rec.fit(train)

    recs = rec.recommend(history=[1, 2], k=2)
    assert len(recs) <= 2
    # Ensure items in history are not recommended
    assert 1 not in recs
    assert 2 not in recs


def test_popularity_recommender():
    train = _get_train_data()
    rec = PopularityRecommender()
    assert isinstance(rec, Recommender)
    rec.fit(train)

    # Item counts: 2 occurs 3 times (users 1, 2, 4);
    # 3 occurs 3 times (1, 2, 3); 4 occurs 3 times (1, 3, 4)
    # Item 1 occurs 2 times (1, 3), Item 5 occurs 2 times (2, 3)
    recs = rec.recommend(history=[2, 3], k=2)
    assert len(recs) == 2
    assert 2 not in recs
    assert 3 not in recs
    assert recs[0] == 4  # Next most popular


def test_item_knn_recommender():
    train = _get_train_data()
    rec = ItemKNNRecommender()
    assert isinstance(rec, Recommender)
    rec.fit(train)

    recs = rec.recommend(history=[1], k=2)
    assert len(recs) == 2
    assert 1 not in recs


def test_bprmf_recommender():
    train = _get_train_data()
    rec = BPRMFRecommender(embedding_dim=8, epochs=3, batch_size=4)
    assert isinstance(rec, Recommender)
    rec.fit(train)

    recs = rec.recommend(history=[1, 2], k=2)
    assert len(recs) == 2
    assert 1 not in recs
    assert 2 not in recs


def test_all_baselines_beat_random():
    """Verify that Popularity, ItemKNN, and BPR-MF all beat Random recommender."""
    train_data = {}
    test_data = {}
    for u in range(1, 31):
        train_data[u] = [1, 2, 10 + (u % 10)]
        test_data[u] = 3

    for u in range(1, 25):
        train_data[u].append(3)

    k = 3
    models = {
        "random": RandomRecommender(seed=42),
        "popularity": PopularityRecommender(),
        "item_knn": ItemKNNRecommender(k_neighbors=5),
        "bpr_mf": BPRMFRecommender(embedding_dim=8, epochs=15, lr=0.05, batch_size=16, seed=42),
    }

    hit_rates = {}
    for name, model in models.items():
        model.fit(train_data)
        hits = 0
        for u, target in test_data.items():
            recs = model.recommend(history=[1, 2], k=k)
            if target in recs:
                hits += 1
        hit_rates[name] = hits / len(test_data)

    assert hit_rates["popularity"] > hit_rates["random"]
    assert hit_rates["item_knn"] > hit_rates["random"]
    assert hit_rates["bpr_mf"] > hit_rates["random"]
