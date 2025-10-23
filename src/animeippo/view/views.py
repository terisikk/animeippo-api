import json

import polars as pl


def recommendations_web_view(dataframe, categories=None, tags_and_genres=None, debug=False):
    tags_and_genres = tags_and_genres or []

    if dataframe is None:
        return json.dumps(
            {
                "data": {
                    "categories": categories,
                }
            }
        )

    # Base fields to always include
    fields = ["id", "title", "cover_image", "genres", "tags", "status", "season_year", "season"]

    # In debug mode, include all scorer columns and final scores
    if debug:
        # Common scorer names
        scorer_fields = [
            "recommend_score",
            "directscore",
            "featurecorrelationscore",
            "clusterscore",
            "continuationscore",
            "adaptationscore",
            "popularityscore",
            "genreaverage",
            "formatscore",
            "studiocorrelationscore",
            "directorcorrelationscore",
        ]
        fields.extend(scorer_fields)

    filtered_fields = list(set(dataframe.columns).intersection(fields))
    df_json = dataframe.select(filtered_fields).to_dicts()

    return json.dumps(
        {"data": {"shows": df_json, "categories": categories, "tags": sorted(tags_and_genres)}}
    )


def profile_cluster_web_view(watchlist, categories):
    fields = {"id", "title", "cover_image", "genres", "user_status"}
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


def profile_characteristics_web_view(profile):
    return json.dumps(
        {
            "data": {
                "user": profile.user,
                "characteristics": {
                    "Variance": profile.characteristics.genre_variance,
                },
            }
        }
    )


def console_view(dataframe):
    with pl.Config(tbl_rows=40):
        print(dataframe.sort("recommend_score", descending=True).head(25).select(["title"]))

    # For debug purposes
    # dataframe.sort("id").write_excel()
