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
