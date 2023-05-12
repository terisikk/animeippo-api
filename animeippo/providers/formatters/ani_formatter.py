import numpy as np
import pandas as pd


def transform_to_animeippo_format(data):
    df = pd.DataFrame()

    if len(data.get("data", [])) > 0:
        df = pd.json_normalize(data["data"], max_level=2)

        column_mapping = {}

        if "seasonYear" in df.columns:
            df["start_season"] = df.apply(
                lambda row: str(row["seasonYear"]) + "/" + row["season"].lower(), axis=1
            )

        for key in df.columns:
            new_key = key

            if "media." in new_key:
                new_key = new_key.split("media.")[-1]

            if "." in new_key:
                new_key = new_key.split(".")[0]

            column_mapping[key] = new_key

        column_mapping["relations.edges"] = "related_anime"
        df = df.rename(columns=column_mapping)

        for key, formatter in formatters.items():
            if key in df.columns:
                df[key] = df[key].apply(formatter)

        if "id" in df.columns:
            df = df.set_index("id")

        dropped = ["seasonYear", "season"]

        df = df.drop(dropped, errors="ignore", axis=1)

    return df


def filter_related_anime(field):
    meaningful_relations = ["PARENT", "PREQUEL"]

    relations = []

    for item in field:
        relationType = item.get("relationType", None)

        if relationType in meaningful_relations:
            id = item.get("node", None)["id"]

            if id:
                relations.append(id)

    return relations


def split_studios(field):
    studios = []

    for studio in field:
        if studio:
            id = studio.get("id", None)
            if id:
                studios.append(id)

    return studios


def get_user_score(score):
    if score == 0:
        score = np.nan

    return score


formatters = {
    "related_anime": filter_related_anime,
    "status": str.lower,
    "studios": split_studios,
    "score": get_user_score,
}
