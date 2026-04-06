from unittest.mock import AsyncMock, MagicMock

import aiohttp
import polars as pl
import pytest
from fastapi.testclient import TestClient

import app as appmod


class MockDataset:
    def __init__(self):
        self.seasonal = pl.DataFrame({"id": [1], "title": ["Test Anime"]})
        self.recommendations = pl.DataFrame(
            {"id": [1], "title": ["Test Anime"], "cover_image": ["img.jpg"]}
        )
        self.all_features = {"Action"}
        self.nsfw_tags = set()


class MockProfile:
    def __init__(self):
        self.watchlist = pl.DataFrame({"id": [1], "title": ["Test"], "cover_image": ["img.jpg"]})
        self.user = "Test"
        self.characteristics = None
        self.user_profile = MagicMock()
        self.user_profile.genre_correlations = None


@pytest.fixture
def client(monkeypatch):
    """Test client with mocked recommender and profiler."""
    mock_recommender = MagicMock()
    mock_recommender.recommend_seasonal_anime = AsyncMock(return_value=MockDataset())
    mock_recommender.get_categories.return_value = []

    mock_profiler = MagicMock()
    mock_profiler.analyse = AsyncMock(return_value=(MockProfile(), [], None))
    mock_profiler.provider.get_genres.return_value = set()

    # Characteristics is computed in /profile route — mock it to avoid needing full data
    monkeypatch.setattr(
        "app.Characteristics",
        lambda watchlist, genres: MagicMock(genre_variance=0.5),
    )

    monkeypatch.setattr(
        "app.recommenders", {"anilist": mock_recommender, "mixed": mock_recommender}
    )
    monkeypatch.setattr("app.profilers", {"anilist": mock_profiler, "mixed": mock_profiler})

    return TestClient(appmod.app, raise_server_exceptions=False)


def _make_client_response_error(status_code):
    return aiohttp.ClientResponseError(
        request_info=aiohttp.RequestInfo(
            url="http://test", method="POST", headers={}, real_url="http://test"
        ),
        history=(),
        status=status_code,
    )


# --- Validation errors ---


def test_seasonal_requires_year(client):
    response = client.get("/seasonal")
    assert response.status_code == 400
    assert "year" in response.json()["error"]


def test_recommend_requires_user(client):
    response = client.get("/recommend?year=2025")
    assert response.status_code == 400
    assert "user" in response.json()["error"]


def test_recommend_requires_year(client):
    response = client.get("/recommend?user=Test")
    assert response.status_code == 400
    assert "year" in response.json()["error"]


def test_analyse_requires_user(client):
    response = client.get("/analyse")
    assert response.status_code == 400
    assert "user" in response.json()["error"]


def test_profile_requires_user(client):
    response = client.get("/profile")
    assert response.status_code == 400
    assert "user" in response.json()["error"]


# --- Successful responses ---


def test_seasonal_returns_200(client):
    response = client.get("/seasonal?year=2025")
    assert response.status_code == 200


def test_recommend_returns_200(client):
    response = client.get("/recommend?user=Test&year=2025")
    assert response.status_code == 200


def test_analyse_returns_200(client):
    response = client.get("/analyse?user=Test")
    assert response.status_code == 200


def test_profile_returns_200(client):
    response = client.get("/profile?user=Test")
    assert response.status_code == 200


# --- User not found (API 404) ---


def test_recommend_returns_404_for_unknown_user(client):
    appmod.recommenders["anilist"].recommend_seasonal_anime = AsyncMock(
        side_effect=_make_client_response_error(404)
    )
    response = client.get("/recommend?user=Nobody&year=2025")
    assert response.status_code == 404
    assert "Nobody" in response.json()["error"]


def test_analyse_returns_404_for_unknown_user(client):
    appmod.profilers["anilist"].analyse = AsyncMock(side_effect=_make_client_response_error(404))
    response = client.get("/analyse?user=Nobody")
    assert response.status_code == 404
    assert "Nobody" in response.json()["error"]


# --- Upstream API failure (non-404) ---


def test_recommend_returns_502_on_api_error(client):
    appmod.recommenders["anilist"].recommend_seasonal_anime = AsyncMock(
        side_effect=_make_client_response_error(500)
    )
    response = client.get("/recommend?user=Test&year=2025")
    assert response.status_code == 502


def test_recommend_returns_502_on_network_error(client):
    appmod.recommenders["anilist"].recommend_seasonal_anime = AsyncMock(
        side_effect=aiohttp.ClientError("connection refused")
    )
    response = client.get("/recommend?user=Test&year=2025")
    assert response.status_code == 502


# --- RuntimeError (empty watchlist) ---


def test_recommend_returns_404_on_runtime_error(client):
    appmod.recommenders["anilist"].recommend_seasonal_anime = AsyncMock(
        side_effect=RuntimeError("Trying to recommend anime without proper data.")
    )
    response = client.get("/recommend?user=Ghost&year=2025")
    assert response.status_code == 404
    assert "Ghost" in response.json()["error"]
