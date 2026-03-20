import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import polars as pl

from animeippo.analysis import encoding, statistics
from animeippo.clustering import model
from animeippo.profiling import characteristics
from animeippo.profiling.model import UserProfile


class ProfileAnalyser:
    """Clusters a user watchlist titles to clusters of similar anime."""

    def __init__(self, provider):
        self.provider = provider
        self.encoder = encoding.WeightedCategoricalEncoder()
        self.clusterer = model.AnimeClustering(
            distance_metric="cosine", distance_threshold=0.78, linkage="complete"
        )

    def async_get_profile(self, user):
        # If we run from jupyter, loop is already running and we need
        # to act differently. If the loop is not running,
        # we break into "normal path" with RuntimeError
        try:
            asyncio.get_running_loop()

            with ThreadPoolExecutor(1) as pool:
                return pool.submit(lambda: asyncio.run(self.databuilder(user))).result()
        except RuntimeError:
            return asyncio.run(self.databuilder(user))

    async def databuilder(self, user):
        user_watchlist = await self.provider.get_user_anime_list(user)
        user_profile = UserProfile(user, user_watchlist)

        all_features = user_profile.watchlist.explode("features")["features"].unique().drop_nulls()

        print(all_features)

        self.encoder.fit(all_features)
        user_profile.watchlist = user_profile.watchlist.with_columns(
            encoded=self.encoder.encode(user_profile.watchlist)
        )

        user_profile.watchlist = user_profile.watchlist.with_columns(
            cluster=self.clusterer.cluster_by_features(user_profile.watchlist)
        )

        return user_profile

    def analyse(self, user):
        self.profile = self.async_get_profile(user)

        self.profile.characteristics = characteristics.Characteristics(
            self.profile.watchlist, self.provider.get_genres()
        )

        return self.get_cluster_categories(self.profile)

    def get_categories(self, profile):
        categories = []
        top_genre_items = []
        top_tag_items = []

        if (
            "genres" in profile.watchlist.columns
            and profile.user_profile.genre_correlations is not None
        ):
            top_5_genres = profile.user_profile.genre_correlations[0:5]["name"].to_list()

            for genre in top_5_genres:
                gdf = profile.watchlist_explode_cached("genres")
                filtered = gdf.filter(pl.col("genres") == genre).sort("score", descending=True)

                if len(filtered) > 0:
                    top_genre_items.append(filtered.item(0, "id"))

            categories.append({"name": ", ".join(top_5_genres), "items": top_genre_items})

        if "tags" in profile.watchlist.columns:
            tag_correlations = statistics.weight_categoricals_correlation(
                profile.watchlist.explode("tags"), "tags"
            ).sort("weight", descending=True)

            top_5_tags = tag_correlations[0:5]["name"].to_list()

            for tag in top_5_tags:
                gdf = profile.watchlist.filter(
                    ~pl.col("id").is_in(pl.Series(top_genre_items.extend(top_tag_items)).implode())
                ).explode("tags")
                filtered = gdf.filter(pl.col("tags") == tag).sort("score", descending=True)

                if len(filtered) > 0:
                    top_tag_items.append(str(filtered.item(0, "id")))

            categories.append({"name": ", ".join(top_5_tags), "items": top_tag_items})

        return categories

    # Adjectival forms for terms that sound better as modifiers
    ADJECTIVE_FORM: ClassVar[dict[str, str]] = {
        # Genres with adjectival forms
        "Romance": "Romantic",
        "Tragedy": "Tragic",
        "Comedy": "Comedic",
        "Mystery": "Mysterious",
        "Horror": "Horrific",
        "Magic": "Magical",
        # Plurals to singular (better as modifiers)
        "Aliens": "Alien",
        "Angels": "Angel",
        "Animals": "Animal",
        "Assassins": "Assassin",
        "Cowboys": "Cowboy",
        "Delinquents": "Delinquent",
        "Demons": "Demon",
        "Dinosaurs": "Dinosaur",
        "Dragons": "Dragon",
        "Firefighters": "Firefighter",
        "Gangs": "Gang",
        "Gods": "God",
        "Guns": "Gun",
        "Kids": "Kid",
        "Maids": "Maid",
        "Pirates": "Pirate",
        "Robots": "Robot",
        "Tanks": "Tank",
        "Trains": "Train",
        "Twins": "Twin",
        "Vikings": "Viking",
        "Witches": "Witch",
        # Nouns with better adjectival forms
        "Crime": "Criminal",
        "Mythology": "Mythological",
        "Politics": "Political",
    }

    # Modifier priority: lower number = placed first (modifier position)
    # Higher number = core/noun position
    MODIFIER_PRIORITY: ClassVar[dict[str, int]] = {
        "Demographic": 1,
        "Technical": 2,
        "Setting": 3,
        "Cast": 4,
        "Theme": 5,
        "Genre": 6,
    }

    def _get_feature_category(self, feature):
        """Determine the category of a feature (Genre, Theme, Setting, Cast, etc.)."""
        tag_lookup = self.provider.get_tag_lookup()
        all_genres = self.provider.get_genres()
        tag_by_name = {tag_info["name"]: tag_info for tag_info in tag_lookup.values()}

        if feature in all_genres:
            return "Genre", feature

        if feature not in tag_by_name:
            return None, feature

        tag_info = tag_by_name[feature]
        category_str = tag_info.get("category", "")

        # Map tag category strings to simplified category names
        category_map = {
            "Theme-": "Theme",
            "Setting-": "Setting",
            "Cast-": "Cast",
        }

        for prefix, category_name in category_map.items():
            if category_str.startswith(prefix):
                return category_name, feature

        if category_str in ("Technical", "Demographic"):
            return category_str, feature

        return None, feature

    def _as_modifier(self, feature_name):
        """Convert feature to its adjectival form when used as modifier."""
        return self.ADJECTIVE_FORM.get(feature_name, feature_name)

    def _generate_natural_cluster_name(self, features):
        """Generate natural language cluster name using priority-based ordering.

        Ordering principle: [Modifier] [Core]
        - Core priority (rightmost/noun): Genre > Theme > Cast-Traits
        - Modifier priority (leftmost/adjective): Setting > Demographic > Technical > Cast > Theme

        Examples:
        - "Shounen Action" (Demographic + Genre)
        - "Historical Drama" (Setting + Genre)
        - "School Romance" (Setting + Genre)
        - "Vampire Fantasy" (Cast + Genre)
        - "Isekai Cultivation" (Theme + Theme)
        """
        if not features:
            return ""

        # Categorize each feature
        categorized_features = []
        for feature in features[:2]:  # Only use first 2 descriptive features
            category, name = self._get_feature_category(feature)
            if category:
                priority = self.MODIFIER_PRIORITY.get(category, 99)
                categorized_features.append((priority, category, name))

        if not categorized_features:
            return " ".join(features[:2])

        # Sort by priority (lower = modifier, higher = core)
        categorized_features.sort(key=lambda x: x[0])

        # Build name: apply adjectival transform to modifier position
        if len(categorized_features) == 1:
            name = categorized_features[0][2]
        else:
            modifier = self._as_modifier(categorized_features[0][2])
            core = categorized_features[1][2]
            name = f"{modifier} {core}"

        # Capitalize first letter only, preserving rest (e.g., "CGI" stays "CGI")
        return name[0].upper() + name[1:] if name else ""

    def get_cluster_categories(self, profile):
        target = profile.watchlist

        gdf = target.explode("features")

        gdf = gdf.filter(~pl.col("features").is_in(self.provider.get_nsfw_tags()))

        descriptions = statistics.get_descriptive_features(gdf, "features", "cluster", 2).select(
            pl.col("cluster"), pl.concat_list(pl.exclude("cluster")).alias("description")
        )

        cluster_stats = target.group_by("cluster").agg(
            [
                pl.col("id").count().alias("count"),
                pl.col("score").mean().alias("mean_score"),
                (pl.col("user_status") == "completed").sum().alias("completed_count"),
            ]
        )

        cluster_stats = cluster_stats.with_columns(
            completion_rate=(pl.col("completed_count") / pl.col("count") * 100).round(1)
        ).with_columns(mean_score=pl.col("mean_score").round(1))

        clustergroups = target.sort("title").group_by(["cluster"])

        return [
            {
                "name": self._generate_natural_cluster_name(
                    list(
                        descriptions.filter(pl.col("cluster") == str(key[0]))["description"].item()
                    )
                ),
                "items": value["id"].to_list(),
                "stats": cluster_stats.filter(pl.col("cluster") == key[0])
                .select(["count", "mean_score", "completion_rate"])
                .to_dicts()[0],
            }
            for key, value in clustergroups
        ]
