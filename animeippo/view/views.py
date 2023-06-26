import json


def web_view(dataframe, categories=None):
    if "id" not in dataframe.columns:
        dataframe["id"] = dataframe.index

    fields = set(["id", "title", "coverImage", "cluster", "genres", "status", "start_season"])

    filtered_fields = list(set(dataframe.columns.tolist()).intersection(fields))

    dataframe["genres"] = dataframe["genres"].apply(list)
    df_json = dataframe[filtered_fields].to_dict(orient="records")

    return json.dumps(
        {
            "data": {
                "shows": df_json,
                "categories": categories,
            }
        }
    )


def console_view(dataframe):
    dataframe = dataframe.reset_index()
    print(dataframe.reset_index().loc[0:25][["title", "features", "cluster"]])
