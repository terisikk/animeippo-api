import numpy as np
import pandas as pd

from . import util


def transform_to_animeippo_format(data):
    if len(data.get("data", [])) == 0:
        return pd.DataFrame()

    df = pd.json_normalize(data["data"], max_level=1)

    column_mapping = util.get_column_name_mappers(df.columns)
    column_mapping["node.num_list_users"] = "popularity"

    df = df.rename(columns=column_mapping)

    df = util.format_with_formatters(df, formatters)

    if "relation_type" in df.columns:
        df = filter_related_anime(df)

    dropped = [
        "num_episodes_watched",
        "is_rewatching",
        "updated_at",
        "start_date",
        "finish_date",
    ]
    df = df.drop(dropped, errors="ignore", axis=1)

    if "id" in df.columns:
        df = df.set_index("id")

    return df


@util.default_if_error([])
def split_id_name_field(field):
    names = []

    for item in field:
        names.append(item.get("name", np.nan))

    return names


@util.default_if_error("?/?")
def split_season(season_field):
    season_ret = np.nan

    year = season_field.get("year", "?")
    season = season_field.get("season", "?")

    season_ret = f"{year}/{season}"

    return season_ret


def filter_related_anime(df):
    meaningful_relations = ["parent_story", "prequel"]
    return df[df["relation_type"].isin(meaningful_relations)]


formatters = {
    "genres": split_id_name_field,
    "studios": split_id_name_field,
    "start_season": split_season,
    "score": lambda score: score if score != 0 else np.nan,
}
