import pandas as pd
import datetime
import numpy as np

from . import util
from animeippo.providers.formatters.schema import DefaultMapper, SingleMapper, MultiMapper


def transform_seasonal_data(data, feature_names):
    original = pd.json_normalize(data["data"], "media", record_prefix="media.")

    keys = [
        "id",
        "idMal",
        "title",
        "format",
        "genres",
        "coverImage",
        "mean_score",
        "popularity",
        "status",
        "continuation_to",
        "adaptation_of",
        "score",
        "duration",
        "episodes",
        "source",
        "tags",
        "nsfw_tags",
        "ranks",
        "studios",
        "start_season",
    ]

    return transform_to_animeippo_format(original, feature_names, keys)


def transform_watchlist_data(data, feature_names):
    original = pd.json_normalize(data["data"])

    keys = [
        "id",
        "idMal",
        "title",
        "format",
        "genres",
        "coverImage",
        "user_status",
        "mean_score",
        "score",
        "source",
        "tags",
        "ranks",
        "nsfw_tags",
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
        df = df.drop_duplicates(subset="id")
        df = df.set_index("id")

    return df.infer_objects()


def run_mappers(dataframe, original, mapping):
    for key, mapper in mapping.items():
        if key in dataframe.columns:
            dataframe[key] = mapper.map(original)

    return dataframe


def filter_relations(field, meaningful_relations):
    relations = []

    for item in field:
        relationType = item.get("relationType", "")
        node = item.get("node", {})
        id = node.get("id", None)

        if relationType in meaningful_relations and id is not None:
            relations.append(id)

    return relations


def get_continuation(field):
    meaningful_relations = ["PARENT", "PREQUEL"]

    return filter_relations(field, meaningful_relations)


def get_adaptation(field):
    meaningful_relations = ["ADAPTATION"]

    return filter_relations(field, meaningful_relations)


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


def get_nsfw_tags(items):
    return [item["name"] for item in items if item["isAdult"] is True]


def get_studios(studios):
    return set([studio["node"]["name"] for studio in studios if studio["node"].get("isAnimationStudio", False)])


# fmt: off

ANILIST_MAPPING = {
    "id":           DefaultMapper("media.id"),
    "idMal":        DefaultMapper("media.idMal"),
    "title":        DefaultMapper("media.title.romaji"),
    "format":       DefaultMapper("media.format"),
    "genres":       DefaultMapper("media.genres"),
    "coverImage":   DefaultMapper("media.coverImage.large"),
    "mean_score":   DefaultMapper("media.meanScore"),
    "popularity":   DefaultMapper("media.popularity"),
    "duration":     DefaultMapper("media.duration"),
    "episodes":     DefaultMapper("media.episodes"),
    "user_status":  SingleMapper("status", str.lower),
    "status":       SingleMapper("media.status", str.lower),
    "score":        SingleMapper("score", get_score),
    "source":       SingleMapper("media.source", str.lower),
    "tags":         SingleMapper("media.tags", get_tags),
    "continuation_to":    
                    SingleMapper("media.relations.edges", get_continuation),
    "adaptation_of":
                    SingleMapper("media.relations.edges", get_adaptation),
    "ranks":        SingleMapper("media.tags", get_ranks),
    "nsfw_tags":    SingleMapper("media.tags", get_nsfw_tags),
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
