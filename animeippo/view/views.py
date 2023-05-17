def web_view(dataframe):
    if "id" not in dataframe.columns:
        dataframe["id"] = dataframe.index

    return dataframe[["id", "title", "coverImage"]].to_json(orient="records")


def console_view(dataframe):
    print(dataframe.reset_index().loc[0:25][["title", "genres"]])
