import json
import polars as pl


def recommendations_web_view(dataframe, categories=None):
    if dataframe is None:
        return json.dumps(
            {
                "data": {
                    "categories": categories,
                }
            }
        )

    fields = ["id", "title", "cover_image", "genres", "status", "season_year", "season"]
    df_json = dataframe.select(fields).to_dicts()

    return json.dumps(
        {
            "data": {
                "shows": df_json,
                "categories": categories,
            }
        }
    )


def profile_web_view(watchlist, categories):
    fields = set(["id", "title", "cover_image", "genres", "user_status"])
    filtered_fields = list(set(watchlist.columns).intersection(fields))

    df_json = watchlist.select(filtered_fields).to_dicts()

    return json.dumps(
        {
            "data": {
                "shows": df_json,
                "categories": categories,
            }
        }
    )


def console_view(dataframe):
    with pl.Config(tbl_rows=40):
        print(dataframe.head(25).select(["title", "genres"]))

    # For debug purposes
    # dataframe.sort("id").write_excel()
