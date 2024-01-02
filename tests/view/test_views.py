import polars as pl
import json

from animeippo.view import views

from animeippo.recommendation import profile

from tests import test_data


def test_web_view_can_render_seasonal_data():
    df = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    assert (
        json.loads(views.recommendations_web_view(df))["data"]["shows"][0]["title"]
        == df[0]["title"].item()
    )


def test_web_view_can_render_profile_data():
    df = pl.DataFrame(test_data.FORMATTED_MAL_USER_LIST)

    uprofile = profile.UserProfile("Janiskeisari", df)

    assert (
        json.loads(views.profile_web_view(uprofile, []))["data"]["watchlist"][0]["title"]
        == df[0]["title"].item()
    )


def test_console_view_prints_stuff(capfd):
    df = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    views.console_view(df)

    out, _ = capfd.readouterr()

    assert df.item(0, "title") in out
