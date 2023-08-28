import pandas as pd
import datetime
import numpy as np

from . import util
from animeippo.providers.formatters.schema import DefaultMapper, SingleMapper, MultiMapper


def transform_seasonal_data(data, feature_names):
    original = pd.json_normalize(data["data"], "media", record_prefix="media.")

    keys = [
        "id",
        "title",
        "genres",
        "coverImage",
        "mean_score",
        "popularity",
        "status",
        "relations",
        "score",
        "source",
        "tags",
        "ranks",
        "studios",
        "start_season",
    ]

    return transform_to_animeippo_format(original, feature_names, keys)


def transform_watchlist_data(data, feature_names):
    original = pd.json_normalize(data["data"])

    keys = [
        "id",
        "title",
        "genres",
        "coverImage",
        "user_status",
        "mean_score",
        "score",
        "source",
        "tags",
        "ranks",
        "studios",
        "user_complete_date",
        "start_season",
    ]

    return transform_to_animeippo_format(original, feature_names, keys)


def transform_to_animeippo_format(original, feature_names, keys):
    df = pd.DataFrame(columns=keys)

    if len(original) == 0:
        return df

    df = run_mappers(df, original, ANILIST_MAPPING)

    df["features"] = df.apply(util.get_features, args=(feature_names,), axis=1)

    if "id" in df.columns:
        df = df.set_index("id")

    return remove_duplicates(df).infer_objects()


def remove_duplicates(dataframe):
    return dataframe[~dataframe.index.duplicated(keep="first")]


def run_mappers(dataframe, original, mapping):
    for key, mapper in mapping.items():
        if key in dataframe.columns:
            dataframe[key] = mapper.map(original)

    return dataframe


def filter_related_anime(field):
    meaningful_relations = ["PARENT", "PREQUEL"]

    relations = []

    for item in field:
        relationType = item.get("relationType", "")
        node = item.get("node", {})

        if relationType in meaningful_relations:
            relations.append(node.get("id", None))

    return relations


def get_tags(tags):
    return [tag["name"] for tag in tags]


def get_user_complete_date(year, month, day):
    if pd.isna(year) or pd.isna(month) or pd.isna(day):
        return pd.NA

    return datetime.date(int(year), int(month), int(day))


@util.default_if_error(pd.NA)
def get_season(year, season):
    if year == 0 or pd.isna(year):
        year = "?"
    else:
        year = str(int(year))

    if pd.isna(season):
        season = "?"

    return f"{year}/{str(season).lower()}"


def get_score(score):
    # np.nan is a float, pd.NA is not, causes problems
    return score if score != 0 else np.nan


def get_ranks(items):
    return {item["name"]: item["rank"] / 100 for item in items}


def get_studios(studios):
    return [
        studio["node"]["name"]
        for studio in studios
        if studio["node"].get("isAnimationStudio", False)
    ]


get_user_complete_date

# fmt: off

ANILIST_MAPPING = {
    "id":           DefaultMapper("media.id"),
    "title":        DefaultMapper("media.title.romaji"),
    "genres":       DefaultMapper("media.genres"),
    "coverImage":   DefaultMapper("media.coverImage.large"),
    "mean_score":   DefaultMapper("media.meanScore"),
    "popularity":   DefaultMapper("media.popularity"),
    "user_status":  SingleMapper("status", str.lower),
    "status":       SingleMapper("media.status", str.lower),
    "score":        SingleMapper("score", get_score),
    "source":       SingleMapper("media.source", str.lower),
    "tags":         SingleMapper("media.tags", get_tags),
    "relations":    SingleMapper("media.relations.edges", filter_related_anime),
    "ranks":        SingleMapper("media.tags", get_ranks),
    "studios":      SingleMapper("media.studios.edges", get_studios),
    "user_complete_date": 
                    MultiMapper(
                        lambda row: get_user_complete_date(
                            row["completedAt.year"], 
                            row["completedAt.month"], 
                            row["completedAt.day"]
                        ),
                    pd.NaT,
    ),
    "start_season": MultiMapper(
        lambda row: get_season(row["media.seasonYear"], row["media.season"]),
    ),
}
# fmt: on
