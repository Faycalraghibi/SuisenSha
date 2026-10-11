"""Locust load test scenarios for SuisenSha recommendation endpoints."""

from __future__ import annotations

import random

from locust import HttpUser, between, task


class RecommenderLoadUser(HttpUser):
    """Simulates realistic traffic across recommendation, feedback, and metrics endpoints."""

    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        """Authenticate or initialize user session parameters."""
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @task(5)
    def test_sasrec_recommendations(self) -> None:
        """Sequential SASRec recommendations with varying history lengths."""
        seq_len = random.randint(2, 10)
        items = random.sample(range(1, 1000), seq_len)
        payload = {
            "item_ids": items,
            "k": 10,
        }
        with self.client.post(
            "/recommend/sasrec",
            json=payload,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"SASRec recommendation failed: {response.status_code}")

    @task(3)
    def test_embedding_recommendations(self) -> None:
        """Vector ANN similarity recommendations."""
        payload = {
            "item_id": random.randint(1, 500),
            "k": 10,
        }
        with self.client.post(
            "/recommend/embedding",
            json=payload,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Embedding recommendation failed: {response.status_code}")

    @task(2)
    def test_record_feedback_event(self) -> None:
        """User feedback event logging."""
        payload = {
            "user_id": f"u_{random.randint(1, 100)}",
            "item_id": random.randint(1, 1000),
            "event_type": random.choice(["click", "impression", "purchase"]),
            "recommendation_id": f"rec_{random.randint(1, 5000)}",
        }
        with self.client.post(
            "/events",
            json=payload,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code in (200, 201):
                response.success()
            else:
                response.failure(f"Event logging failed: {response.status_code}")

    @task(1)
    def test_metrics_scrape(self) -> None:
        """Prometheus metrics scraping endpoint."""
        with self.client.get(
            "/metrics",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics scrape failed: {response.status_code}")
