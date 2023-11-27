import json


def web_view(dataframe, categories=None):
    if "id" not in dataframe.columns:
        dataframe["id"] = dataframe.index
    else:
        dataframe["id"] = dataframe["id"].fillna(dataframe.index.to_series())

    fields = set(["id", "title", "coverImage", "cluster", "genres", "status", "user_status", "start_season"])

    filtered_fields = list(set(dataframe.columns.tolist()).intersection(fields))

    dataframe["genres"] = dataframe["genres"].apply(list)

    df_json = dataframe[filtered_fields].fillna("").to_dict(orient="records")

    return json.dumps(
        {
            "data": {
                "shows": df_json,
                "categories": categories,
            }
        }
    )


def console_view(dataframe):
    # dataframe = dataframe.reset_index()
    print(dataframe.iloc[0:25][["title", "genres"]])
