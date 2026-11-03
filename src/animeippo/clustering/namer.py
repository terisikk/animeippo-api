from typing import ClassVar

import polars as pl

from animeippo.analysis import statistics
from animeippo.providers.anilist.data import TAG_BY_NAME


class ClusterNamer:
    """Generates natural language names for anime clusters from their feature lists."""

    # Adjectival forms for terms that sound better as modifiers
    ADJECTIVE_FORM: ClassVar[dict[str, str]] = {
        # Genres with adjectival forms
        "Comedy": "Comedic",
        "Magic": "Magical",
        "Mystery": "Mysterious",
        "Romance": "Romantic",
        "Tragedy": "Tragic",
        # Plurals to singular (better as modifiers)
        "Aliens": "Alien",
        "Angels": "Angel",
        "Animals": "Animal",
        "Assassins": "Assassin",
        "Cars": "Car",
        "Cowboys": "Cowboy",
        "Curses": "Curse",
        "Delinquents": "Delinquent",
        "Demons": "Demon",
        "Dinosaurs": "Dinosaur",
        "Dragons": "Dragon",
        "Firefighters": "Firefighter",
        "Gangs": "Gang",
        "Gods": "God",
        "Guns": "Gun",
        "Maids": "Maid",
        "Mopeds": "Moped",
        "Motorcycles": "Motorcycle",
        "Pirates": "Pirate",
        "Robots": "Robot",
        "Ships": "Ship",
        "Tanks": "Tank",
        "Trains": "Train",
        "Triads": "Triad",
        "Twins": "Twin",
        "Vikings": "Viking",
        "Witches": "Witch",
        # Nouns with better adjectival forms
        "Anachronism": "Anachronistic",
        "Agriculture": "Agricultural",
        "Anthropomorphism": "Anthropomorphic",
        "Cannibalism": "Cannibal",
        "Crime": "Criminal",
        "Mythology": "Mythological",
        "Philosophy": "Philosophical",
        "Politics": "Political",
        "Terrorism": "Terrorist",
        "Drugs": "Drug-Fueled",
        "Super Power": "Super Powered",
        "Religion": "Religious",
        # Long tag names shortened for readability
        "Cute Boys Doing Cute Things": "Cute Boys",
        "Cute Girls Doing Cute Things": "Cute Girls",
        # Cast composition tags shortened to modifiers
        "Primarily Male Cast": "Male-Led",
        "Primarily Female Cast": "Female-Led",
        "Primarily Adult Cast": "Adult",
        "Primarily Child Cast": "Children's",
        "Primarily Teen Cast": "Teen",
        "Primarily Animal Cast": "Animal",
    }

    # Modifier priority: lower number = placed first (modifier position)
    # Higher number = core/noun position
    MODIFIER_PRIORITY = pl.DataFrame(
        [
            ("Technical", 1),
            ("Setting", 2),
            ("Cast", 3),
            ("Theme", 4),
            ("Genre", 5),
            ("Demographic", 6),
        ],
        schema=["category_short", "priority"],
        orient="row",
    )

    # Features that only work as adjectives and can't anchor a name as a noun
    ADJECTIVE_ONLY: ClassVar[set[str]] = {
        "Autobiographical",
        "Biographical",
        "Coastal",
        "Dystopian",
        "Educational",
        "Environmental",
        "Foreign",
        "Historical",
        "Medieval",
        "Post-Apocalyptic",
        "Primarily Adult Cast",
        "Primarily Animal Cast",
        "Primarily Child Cast",
        "Primarily Female Cast",
        "Primarily Male Cast",
        "Primarily Teen Cast",
        "Psychological",
        "Rural",
        "Supernatural",
        "Urban",
    }

    # Noun substitutes for adjective-only features when they'd otherwise be the core
    PREFERRED_NOUN: ClassVar[dict[str, str]] = {
        "Coastal": "Seaside",
        "Dystopian": "Dystopia",
        "Educational": "Academy",
        "Environmental": "Nature",
        "Historical": "Period Piece",
        "Medieval": "Medieval Era",
        "Post-Apocalyptic": "Wasteland",
        "Rural": "Countryside",
        "Supernatural": "Occult",
        "Urban": "City",
    }

    # Fallback nouns per tag category when no preferred noun is available
    CATEGORY_NOUNS: ClassVar[dict[str, str]] = {
        "Cast-Main Cast": "People",
        "Setting-Scene": "World",
        "Setting-Time": "Era",
        "Setting-Universe": "World",
        "Theme-Other": "Stories",
        "Theme-Drama": "Drama",
        "Theme-Action": "Action",
        "Theme-Fantasy": "Fantasy",
        "Theme-Romance": "Romance",
        "Theme-Sci-Fi": "Sci-Fi",
    }

    def __init__(self, genres):
        self.genres = genres

        self.tag_df = pl.from_dicts(list(TAG_BY_NAME.values()))

        self.genre_df = pl.DataFrame(
            list(self.genres.keys()),
            schema=["name"],
        )

        self.classified_features = self.fit()

    def fit(self):
        return (
            pl.concat(
                [self.tag_df, self.genre_df.with_columns(category=pl.lit("Genre"))], how="diagonal"
            )
            .with_columns(
                pl.col("category").str.split("-").list.first().alias("category_short"),
            )
            .join(self.MODIFIER_PRIORITY, on="category_short", how="left")
            .with_columns(pl.col("priority").fill_null(99))
            .with_columns(adjective_only=pl.col("name").is_in(self.ADJECTIVE_ONLY))
        )

    def name_clusters_from_data(self, dataframe, feature_column, cluster_column):
        """Extract descriptive features via TF-IDF and generate deduplicated cluster names."""
        cluster_features = statistics.get_descriptive_features(
            dataframe,
            feature_column,
            cluster_column,
            n_features=5,
            boost_features=self.genres,
        )

        return self.name_clusters(cluster_features)

    def name_clusters(self, cluster_features):
        """
        Generate interesting names for all clusters.
        """
        cluster_features = (
            cluster_features.explode("description")
            .select(
                pl.col("cluster"),
                pl.col("description").cast(pl.String),
            )
            .join(
                self.classified_features,
                left_on="description",
                right_on="name",
                how="left",
                maintain_order="left",
            )
        )

        cluster_features = self.drop_substring_features(cluster_features)
        top_features = self.select_top_2_features(cluster_features)

        return (
            top_features.group_by("cluster", maintain_order=True)
            .agg(
                pl.col("description").first().alias("modifier"),
                pl.col("description").last().alias("core"),
                pl.col("adjective_only").all(),
                pl.col("category"),
            )
            .with_columns(
                name=pl.when((pl.col("modifier") != pl.col("core")) & pl.col("core").is_not_null())
                .then(
                    pl.struct("modifier", "core", "adjective_only", "category").map_elements(
                        self.conflict_resolver,
                        return_dtype=pl.String,
                    )
                )
                .otherwise(pl.col("modifier"))
            )
            .select(
                pl.col("cluster"),
                pl.col("name"),
            )
        )

    def select_top_2_features(self, cluster_features):
        first_features = cluster_features.filter(pl.col("row_idx") == 0)

        second_features = (
            cluster_features.filter(pl.col("row_idx") > 0)
            .join(
                first_features.select("cluster", "category_short"),
                on="cluster",
                how="left",
                maintain_order="left",
            )
            # First take from a different category.
            # If not, fall back to the second item of the first category
            .sort(
                [
                    pl.col("cluster"),
                    pl.col("category_short") == pl.col("category_short_right"),
                    pl.col("row_idx"),
                ]
            )
            .unique(
                subset=["cluster"],
                keep="first",
                maintain_order=True,
            )
            .select(pl.exclude("category_short_right"))
        )

        return pl.concat([first_features, second_features], how="vertical").sort(
            ["cluster", "adjective_only", "priority"],
            descending=[False, True, False],
            maintain_order=True,
        )

    def drop_substring_features(self, cluster_features):
        """
        Drop features that are just a substring of the other, more specific feature,
        like Space - Space opera.
        """
        substring_features = (
            cluster_features.select(
                "cluster",
                "description",
                "category_short",
            )
            .join(cluster_features, on="cluster", suffix="_other")
            .filter(
                (pl.col("description") != pl.col("description_other"))
                & (pl.col("description_other").str.contains(r"\b" + pl.col("description") + r"\b"))
            )
        )

        return cluster_features.join(
            substring_features,
            on=["cluster", "description"],
            how="anti",
            maintain_order="left",
        ).with_columns(
            row_idx=pl.int_range(0, pl.len()).over("cluster"),
        )

    def conflict_resolver(self, row):
        """Apply conflict resolution rules in order to produce the final name string."""
        modifier = row["modifier"]
        core = row["core"]

        # Both adjective-only: resolve with preferred noun or fallback
        if row["adjective_only"]:
            core_category = row.get("category", [None])[-1]
            modifier, core = self.resolve_adjective_pair(modifier, core, core_category)
            return f"{modifier} {core}"

        modifier = self.to_modifier_form(modifier)
        return f"{modifier} {core}"

    def resolve_adjective_pair(self, modifier_name, core_name, core_category):
        """Resolve both-adjective-only pairs by finding a suitable noun."""
        if core_name in self.PREFERRED_NOUN:
            return self.to_modifier_form(modifier_name), self.PREFERRED_NOUN[core_name]

        if modifier_name in self.PREFERRED_NOUN:
            return self.to_modifier_form(core_name), self.PREFERRED_NOUN[modifier_name]

        if core_category in self.CATEGORY_NOUNS:
            return self.to_modifier_form(modifier_name), self.CATEGORY_NOUNS[core_category]

        return self.to_modifier_form(modifier_name), f"{core_name}"

    def to_modifier_form(self, feature_name):
        """Convert feature to its adjectival form when used as modifier."""
        return self.ADJECTIVE_FORM.get(feature_name, feature_name)
