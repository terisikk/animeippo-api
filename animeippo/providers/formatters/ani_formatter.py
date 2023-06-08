import numpy as np
import pandas as pd

from . import util


def transform_to_animeippo_format(data, feature_names=None, normalize_level=1):
    if len(data.get("data", [])) == 0:
        return pd.DataFrame()

    df = pd.json_normalize(data["data"], max_level=normalize_level)

    column_mapping = util.get_column_name_mappers(df.columns)
    column_mapping["relations"] = "related_anime"

    df = df.rename(columns=column_mapping)

    df = util.format_with_formatters(df, formatters)

    df["start_season"] = df.apply(format_season, axis=1)

    df["features"] = df.apply(util.get_features, args=(feature_names,), axis=1)

    dropped = ["seasonYear", "season"]
    df = df.drop(dropped, errors="ignore", axis=1)

    if "id" in df.columns:
        df = df.set_index("id")

    return df


@util.default_if_error([])
def filter_related_anime(field):
    meaningful_relations = ["PARENT", "PREQUEL"]

    relations = []

    for item in field["edges"]:
        relationType = item.get("relationType", "")
        node = item.get("node", {})

        if relationType in meaningful_relations:
            relations.append(node.get("id", None))

    return relations


@util.default_if_error("?/?")
def format_season(row):
    year = row.get("seasonYear", 0)
    season = str(row.get("season", "?"))

    if year == 0 or np.isnan(year):
        year = "?"
    else:
        year = str(int(year))

    return f"{year}/{season.lower()}"


@util.default_if_error(None)
def split_studios(field):
    return [
        studio["node"].get("name", None)
        for studio in field["edges"]
        if studio["node"].get("isAnimationStudio", False)
    ]


@util.default_if_error(None)
def get_image_url(field):
    return field.get("large", None)


@util.default_if_error(None)
def get_title(field):
    return field.get("romaji", None)


@util.default_if_error([])
def get_tags(tags):
    return [tag["name"] for tag in tags]


formatters = {
    "related_anime": filter_related_anime,
    "status": str.lower,
    "studios": split_studios,
    "score": lambda score: score if score != 0 else np.nan,
    "coverImage": get_image_url,
    "title": get_title,
    "tags": get_tags,
}
