from fastapi.testclient import TestClient

from aiml.backend.api import app


client = TestClient(app)
PRODUCTION_ORIGIN = "https://synroute.vercel.app"


def test_production_origin_receives_cors_header():
    response = client.get("/api/health", headers={"Origin": PRODUCTION_ORIGIN})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == PRODUCTION_ORIGIN
    assert "access-control-allow-credentials" not in response.headers


def test_production_origin_preflight_allows_post_content_type():
    response = client.options(
        "/api/predict",
        headers={
            "Origin": PRODUCTION_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == PRODUCTION_ORIGIN
    allowed_methods = response.headers["access-control-allow-methods"]
    assert "POST" in allowed_methods
    assert "OPTIONS" in allowed_methods
    assert "content-type" in response.headers["access-control-allow-headers"].lower()


def test_overview_uses_precomputed_metadata_and_preserves_contract():
    response = client.get("/api/overview")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "project", "target", "labels", "locations", "train_rows", "test_rows",
        "feature_count", "split", "observed_time_range", "best_verified", "experiments",
        "observation_trend", "federated_available", "federated_rounds",
        "federated_partitions", "fedavg_summary", "saved_fedavg_runs",
        "communication_measured",
    }
    assert payload["communication_measured"] is False
    assert payload["locations"] == 4095
    assert payload["train_rows"] == 52735
    assert payload["test_rows"] == 13705


def test_locations_returns_training_vocabulary_without_loading_pipeline():
    response = client.get("/api/locations")

    assert response.status_code == 200
    locations = response.json()["locations"]
    assert len(locations) == 4095
    assert locations == sorted(locations)
    assert all(location.isdigit() for location in locations)


def test_prediction_uses_cached_pretrained_model():
    locations = client.get("/api/locations").json()["locations"]
    response = client.post("/api/predict", json={"location": locations[0], "time": "08:30"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["location"] == locations[0]
    assert payload["time"] == "08:30:00"
    assert payload["model"] == "centralized CongestionMLP (seed 42)"
    assert 0 <= payload["category_index"] <= 3
    assert payload["category"] in {"Very smooth", "Smooth", "Mild congestion", "Heavy congestion"}
