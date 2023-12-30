import polars as pl
import json

from animeippo.view import views

from tests import test_data


def test_web_view_can_render_seasonal_data():
    df = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    assert (
        json.loads(views.recommendations_web_view(df))["data"]["shows"][0]["title"]
        == df[0]["title"].item()
    )


def test_web_view_adds_id():
    df = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)
    df = df.drop(["id"], axis=1)

    assert json.loads(views.recommendations_web_view(df))["data"]["shows"][0]["id"] is not None


def test_console_view_prints_stuff(capfd):
    df = pl.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    views.console_view(df)

    out, _ = capfd.readouterr()

    assert df.loc[0]["title"] in out
