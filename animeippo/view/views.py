import json
import polars as pl


def recommendations_web_view(dataframe, categories=None):
    fields = set(
        ["id", "title", "cover_image", "cluster", "genres", "status", "user_status", "start_season"]
    )

    filtered_fields = list(set(dataframe.columns).intersection(fields))

    # dataframe["genres"] = dataframe["genres"].apply(list)

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

    if "id" not in dataframe.columns:
        dataframe["id"] = dataframe.index
    else:
        dataframe["id"] = dataframe.index.to_series()

    fields = set(["id", "title", "cover_image", "status", "user_status"])
    filtered_fields = list(set(dataframe.columns.tolist()).intersection(fields))

    df_json = dataframe[filtered_fields].fillna("").to_dict(orient="records")

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
        print(dataframe.head(25).select(["title", "ranks", "genres"]))

    dataframe.sort("id").write_excel()
