import pandas as pd
import datetime

from . import util
from animeippo.providers.formatters.schema import DefaultMapper, SingleMapper, MultiMapper


def transform_to_animeippo_format(data, feature_names, mapping):
    df = pd.DataFrame(columns=mapping.keys())

    if len(data.get("data", [])) == 0:
        return df

    original = pd.json_normalize(data["data"])

    df = run_mappers(df, original, mapping)

    df["features"] = df.apply(util.get_features, args=(feature_names,), axis=1)

    return df.set_index("id")


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
    return datetime.date(int(year), int(month), int(day))


def format_season(year, season):
    if year == 0 or pd.isna(year):
        year = "?"
    else:
        year = str(int(year))

    if pd.isna(season):
        season = "?"

    return f"{year}/{str(season).lower()}"


def get_score(score):
    return score if score != 0 else pd.NA


# fmt: off

WATCHLIST_MAPPING = {
    "id":           DefaultMapper("media.id"),
    "title":        DefaultMapper("media.title.romaji"),
    "genres":       DefaultMapper("media.genres"),
    "coverImage":   DefaultMapper("media.coverImage.large"),
    "status":       SingleMapper("status", str.lower),
    "score":        SingleMapper("score", get_score),
    "source":       SingleMapper("media.source", str.lower),
    "tags":         SingleMapper("media.tags", get_tags),
    "ranks":        SingleMapper("media.tags",
                                 lambda items: 
                                 {item["name",]: item["rank"] / 100 for item in items}
    ),
    "studios":      SingleMapper("media.studios.edges",
                                lambda studios: [
                                    studio["node"]["name"]
                                    for studio in studios
                                    if studio["node"].get("isAnimationStudio", False)
                                ],
    ),
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
        lambda row: format_season(row["media.seasonYear"], row["media.season"]),
    ),
}

SEASONAL_MAPPING = {
    "id":           DefaultMapper("id"),
    "title":        DefaultMapper("title.romaji"),
    "genres":       DefaultMapper("genres"),
    "coverImage":   DefaultMapper("coverImage.large"),
    "mean_score":   DefaultMapper("meanScore"),
    "popularity":   DefaultMapper("popularity"),
    "status":       SingleMapper("status", str.lower),
    "relations":    SingleMapper("relations.edges", filter_related_anime),
    "source":       SingleMapper("source", str.lower),
    "tags":         SingleMapper("tags", get_tags),
    "ranks":        SingleMapper("tags",
                                 lambda items: 
                                 {item["name"]: item["rank"] / 100 for item in items}
    ),
    "studios":      SingleMapper("studios.edges",
                                lambda studios: [
                                    studio["node"]["name"]
                                    for studio in studios
                                    if studio["node"].get("isAnimationStudio", False)
                                ],
    ),
    "start_season": MultiMapper(
        lambda row: format_season(row["seasonYear"], row["season"]),
    ),
}
# fmt: on
