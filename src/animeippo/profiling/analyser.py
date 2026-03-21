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
        # More plurals to singular
        "Cars": "Car",
        "Curses": "Curse",
        "Drugs": "Drug",
        "Mopeds": "Moped",
        "Motorcycles": "Motorcycle",
        "Ships": "Ship",
        "Triads": "Triad",
        # Nouns with better adjectival forms
        "Agriculture": "Agricultural",
        "Anthropomorphism": "Anthropomorphic",
        "Cannibalism": "Cannibal",
        "Crime": "Criminal",
        "Mythology": "Mythological",
        "Politics": "Political",
        "Terrorism": "Terrorist",
    }

    # Modifier priority: lower number = placed first (modifier position)
    # Higher number = core/noun position
    MODIFIER_PRIORITY: ClassVar[dict[str, int]] = {
        "Technical": 1,
        "Setting": 2,
        "Cast": 3,
        "Theme": 4,
        "Genre": 5,
        "Demographic": 6,
    }

    # Features that only work as adjectives and can't anchor a name as a noun
    ADJECTIVE_ONLY: ClassVar[set[str]] = {
        "Rural",
        "Urban",
        "Coastal",
        "Foreign",
        "Historical",
        "Dystopian",
        "Medieval",
        "Environmental",
        "Educational",
        "Autobiographical",
        "Biographical",
        "Post-Apocalyptic",
        "Supernatural",
        "Psychological",
    }

    # Noun substitutes for adjective-only features when they'd otherwise be the core
    PREFERRED_NOUN: ClassVar[dict[str, str]] = {
        "Historical": "Period",
        "Dystopian": "Dystopia",
        "Medieval": "Fantasy",
        "Post-Apocalyptic": "Wasteland",
        "Environmental": "Nature",
        "Rural": "Countryside",
        "Urban": "City",
        "Coastal": "Seaside",
        "Educational": "Academy",
        "Supernatural": "Occult",
    }

    # Fallback nouns per tag category when no preferred noun is available
    CATEGORY_NOUNS: ClassVar[dict[str, str]] = {
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

    def _select_diverse_features(self, features):
        """Pick top 2 features from different categories, avoiding all-adjective pairs."""
        categorized = []
        for feature in features:
            category, name = self._get_feature_category(feature)
            if category:
                priority = self.MODIFIER_PRIORITY.get(category, 99)
                categorized.append((priority, category, name))

        if len(categorized) <= 1:
            return categorized

        # Take the best feature, then the best remaining from a different category
        best = categorized[0]
        second = None
        for candidate in categorized[1:]:
            if candidate[1] != best[1]:
                second = candidate
                break

        if second is None:
            second = categorized[1]

        pair = sorted([best, second], key=lambda x: x[0])

        # If both are adjective-only, try to swap one for a noun-capable candidate
        if pair[0][2] in self.ADJECTIVE_ONLY and pair[1][2] in self.ADJECTIVE_ONLY:
            for candidate in categorized:
                if candidate[2] not in self.ADJECTIVE_ONLY and candidate not in pair:
                    pair[1] = candidate
                    pair.sort(key=lambda x: x[0])
                    break

        return pair

    def _get_tag_category_string(self, feature):
        """Get the raw category string for a tag (e.g. 'Setting-Scene', 'Theme-Other')."""
        tag_lookup = self.provider.get_tag_lookup()
        tag_by_name = {tag_info["name"]: tag_info for tag_info in tag_lookup.values()}
        if feature in tag_by_name:
            return tag_by_name[feature].get("category", "")
        return ""

    def _resolve_adjective_core(self, modifier_name, core_name):
        """Resolve both-adjective-only pairs by finding a suitable noun."""
        # Try preferred noun for the core
        if core_name in self.PREFERRED_NOUN:
            return self._as_modifier(modifier_name), self.PREFERRED_NOUN[core_name]

        # Try preferred noun for the modifier (swap roles)
        if modifier_name in self.PREFERRED_NOUN:
            return self._as_modifier(core_name), self.PREFERRED_NOUN[modifier_name]

        # Try category noun for the core
        core_category = self._get_tag_category_string(core_name)
        if core_category in self.CATEGORY_NOUNS:
            return self._as_modifier(modifier_name), self.CATEGORY_NOUNS[core_category]

        # Last resort
        return self._as_modifier(modifier_name), f"{core_name} Anime"

    def _generate_natural_cluster_name(self, features):
        """Generate natural language cluster name using priority-based ordering.

        Ordering principle: [Modifier] [Core]
        - Core priority (rightmost/noun): Demographic > Genre > Theme > Cast
        - Modifier priority (leftmost/adjective): Technical > Setting > Cast > Theme > Genre

        Examples:
        - "Action Shounen" (Genre + Demographic)
        - "Historical Drama" (Setting + Genre)
        - "School Romance" (Setting + Genre)
        - "Vampire Fantasy" (Cast + Genre)
        - "Isekai Cultivation" (Theme + Theme)
        """
        if not features:
            return ""

        categorized_features = self._select_diverse_features(features)

        if not categorized_features:
            return " ".join(features[:2])

        if len(categorized_features) == 1:
            name = categorized_features[0][2]
        else:
            modifier_name = categorized_features[0][2]
            core_name = categorized_features[1][2]
            core_cat = categorized_features[1][1]

            # Swap if core is adjective-only but modifier is noun-capable
            if core_name in self.ADJECTIVE_ONLY and modifier_name not in self.ADJECTIVE_ONLY:
                modifier_name, core_name = core_name, modifier_name
                core_cat = categorized_features[0][1]

            # Check substring overlap — keep the longer/more specific one
            mod_lower = modifier_name.lower()
            core_lower = core_name.lower()
            if mod_lower in core_lower or core_lower in mod_lower:
                name = max(modifier_name, core_name, key=len)
            elif modifier_name in self.ADJECTIVE_ONLY and core_name in self.ADJECTIVE_ONLY:
                modifier, core = self._resolve_adjective_core(modifier_name, core_name)
                name = f"{modifier} {core}"
            elif core_cat == "Demographic":
                name = f"{modifier_name} {core_name}"
            else:
                modifier = self._as_modifier(modifier_name)
                name = f"{modifier} {core_name}"

        # Capitalize first letter only, preserving rest (e.g., "CGI" stays "CGI")
        return name[0].upper() + name[1:] if name else ""

    def _deprioritize_shared_features(self, features, shared):
        """Move shared features to the end so more distinctive ones get picked."""
        return [f for f in features if f not in shared] + [f for f in features if f in shared]

    def _deduplicate_cluster_names(self, cluster_features):
        """Re-generate names for clusters that collide by penalizing shared features.

        Takes a dict of {cluster_id: [features]}, returns {cluster_id: name}.
        """
        # First pass: generate all names
        names = {
            cid: self._generate_natural_cluster_name(feats)
            for cid, feats in cluster_features.items()
        }

        # Group clusters by name to find collisions
        name_to_clusters = {}
        for cid, name in names.items():
            name_to_clusters.setdefault(name, []).append(cid)

        # Re-generate colliding names with shared features deprioritized
        for _name, cluster_ids in name_to_clusters.items():
            if len(cluster_ids) <= 1:
                continue

            for cid in cluster_ids:
                features = cluster_features[cid]
                sibling_features = {
                    f for other in cluster_ids if other != cid for f in cluster_features[other]
                }
                shared = set(features) & sibling_features
                reordered = self._deprioritize_shared_features(features, shared)
                names[cid] = self._generate_natural_cluster_name(reordered)

        return names

    def get_cluster_categories(self, profile):
        target = profile.watchlist

        gdf = target.explode("features")

        gdf = gdf.filter(~pl.col("features").is_in(self.provider.get_nsfw_tags()))

        descriptions = statistics.get_descriptive_features(
            gdf,
            "features",
            "cluster",
            n_features=5,
            boost_features=set(self.provider.get_genres()),
        ).select(pl.col("cluster"), pl.concat_list(pl.exclude("cluster")).alias("description"))

        cluster_features = {
            row["cluster"]: list(row["description"]) for row in descriptions.iter_rows(named=True)
        }
        cluster_names = self._deduplicate_cluster_names(cluster_features)

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
                "name": cluster_names.get(str(key[0]), ""),
                "items": value["id"].to_list(),
                "stats": cluster_stats.filter(pl.col("cluster") == key[0])
                .select(["count", "mean_score", "completion_rate"])
                .to_dicts()[0],
            }
            for key, value in clustergroups
        ]
