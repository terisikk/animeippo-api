import json

import polars as pl

from animeippo.profiling import characteristics, model
from animeippo.view import views
from tests import test_data


def test_web_view_can_render_seasonal_data():
    df = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    assert (
        json.loads(views.recommendations_web_view(df))["data"]["shows"][0]["title"]
        == df[0]["title"].item()
    )


def test_web_view_can_render_profile_data():
    df = pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

    uprofile = model.UserProfile("Janiskeisari", df)

    assert (
        json.loads(views.profile_cluster_web_view(uprofile.watchlist, []))["data"]["shows"][0][
            "title"
        ]
        == df[0]["title"].item()
    )


def test_web_view_can_render_profile_characteristics():
    df = pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

    uprofile = model.UserProfile("Janiskeisari", df)
    uprofile.characteristics = characteristics.Characteristics(uprofile.watchlist, ["Action"])

    assert (
        json.loads(views.profile_characteristics_web_view(uprofile))["data"]["user"]
        == "Janiskeisari"
    )


def test_console_view_prints_stuff(capfd):
    df = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)
    df = df.with_columns(recommend_score=[8.5, 7.0, 9.0])

    views.console_view(df)

    out, _ = capfd.readouterr()

    assert df.item(0, "title") in out


def test_recommendations_returns_categories_even_when_no_recommendations():
    assert json.loads(views.recommendations_web_view(None, ["Action"]))["data"]["categories"] == [
        "Action"
    ]


def test_recommendations_web_view_includes_scorer_columns_in_debug_mode():
    df = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)
    # Add some scorer columns
    df = df.with_columns(
        recommend_score=pl.lit(0.8),
        directscore=pl.lit(0.7),
        featurecorrelationscore=pl.lit(0.6),
    )

    result = json.loads(views.recommendations_web_view(df, debug=True))
    shows = result["data"]["shows"]

    # Verify scorer columns are included in debug mode
    assert "recommend_score" in shows[0]
    assert "directscore" in shows[0]
    assert "featurecorrelationscore" in shows[0]


def test_recommendations_web_view_excludes_scorer_columns_in_normal_mode():
    df = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)
    # Add some scorer columns
    df = df.with_columns(
        recommend_score=pl.lit(0.8),
        directscore=pl.lit(0.7),
    )

    result = json.loads(views.recommendations_web_view(df, debug=False))
    shows = result["data"]["shows"]

    # Verify scorer columns are NOT included in normal mode
    assert "recommend_score" not in shows[0]
    assert "directscore" not in shows[0]
