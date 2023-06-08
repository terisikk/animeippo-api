import pandas as pd
import json

from animeippo.view import views

from tests import test_data


def test_web_view_can_render_seasonal_data():
    df = pd.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    assert json.loads(views.web_view(df))["data"]["shows"][0]["title"] == df.loc[0]["title"]


def test_web_view_adds_id():
    df = pd.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)
    df = df.drop(["id"], axis=1)

    assert json.loads(views.web_view(df))["data"]["shows"][0]["id"] is not None


def test_console_view_prints_stuff(capfd):
    df = pd.DataFrame(test_data.FORMATTED_MAL_SEASONAL_LIST)

    views.console_view(df)

    out, _ = capfd.readouterr()

    assert df.loc[0]["title"] in out
