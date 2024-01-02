import json
import polars as pl


def recommendations_web_view(dataframe, categories=None):
    fields = set(
        ["id", "title", "cover_image", "cluster", "genres", "status", "user_status", "start_season"]
    )

    filtered_fields = list(set(dataframe.columns).intersection(fields))

    df_json = dataframe.select(filtered_fields).to_dicts()

    return json.dumps(
        {
            "data": {
                "shows": df_json,
                "categories": categories,
            }
        }
    )


def profile_web_view(user_profile, categories):
    dataframe = user_profile.watchlist

    fields = set(["id", "title", "cover_image", "status", "user_status"])
    filtered_fields = list(set(dataframe.columns).intersection(fields))

    df_json = dataframe[filtered_fields].to_dicts()

    return json.dumps(
        {
            "data": {
                "username": user_profile.user,
                "watchlist": df_json,
                "categories": categories,
            }
        }
    )


def console_view(dataframe):
    with pl.Config(tbl_rows=40):
        print(dataframe.head(25).select(["title", "genres"]))

    # For debug purposes
    # dataframe.sort("id").write_excel()
