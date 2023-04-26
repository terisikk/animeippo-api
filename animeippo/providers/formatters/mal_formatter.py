import numpy as np
import pandas as pd


def get_user_score(score):
    if score == 0:
        score = np.nan

    return score


def split_id_name_field(field):
    names = []

    if field:
        try:
            for item in field:
                if isinstance(item, dict):
                    names.append(item.get("name", np.nan))
        except TypeError as e:
            print(f"Could not extract items from {field}: {e}")

    return names


def split_season(season_field):
    season_ret = np.nan

    if season_field and isinstance(season_field, dict):
        year = season_field.get("year", "?")
        season = season_field.get("season", "?")

        season_ret = f"{year}/{season}"

    return season_ret


def filter_related_anime(df):
    meaningful_relations = ["parent_story", "prequel"]
    return df[df["relation_type"].isin(meaningful_relations)]


def transform_to_animeippo_format(data):
    df = pd.DataFrame()

    if len(data.get("data", [])) > 0:
        df = pd.json_normalize(data["data"], max_level=1)

        if "relation_type" in df.columns:
            df = filter_related_anime(df)

        for key, formatter in formatters.items():
            if key in df.columns:
                df[key] = df[key].apply(formatter)

        column_mapping = {}

        for key in df.columns:
            if "." in key:
                new_key = key.split(".")[-1]
                column_mapping[key] = new_key

        df = df.rename(columns=column_mapping)

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


formatters = {
    "node.genres": split_id_name_field,
    "node.studios": split_id_name_field,
    "node.start_season": split_season,
    "list_status.score": get_user_score,
}
