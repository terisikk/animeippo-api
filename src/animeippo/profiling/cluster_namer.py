from typing import ClassVar

from animeippo.analysis import statistics


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
        "Drugs": "Drug",
        "Firefighters": "Firefighter",
        "Gangs": "Gang",
        "Gods": "God",
        "Guns": "Gun",
        "Kids": "Kid",
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
        "Historical": "Period",
        "Medieval": "Medieval Era",
        "Post-Apocalyptic": "Wasteland",
        "Rural": "Countryside",
        "Supernatural": "Occult",
        "Urban": "City",
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

    def __init__(self, tag_lookup, genres):
        self.tag_lookup = tag_lookup
        self.genres = genres
        self.tag_by_name = {tag_info["name"]: tag_info for tag_info in tag_lookup.values()}

    def classify_feature(self, feature):
        """Determine the category of a feature (Genre, Theme, Setting, Cast, etc.)."""
        if feature in self.genres:
            return "Genre", feature

        if feature not in self.tag_by_name:
            return None, feature

        tag_info = self.tag_by_name[feature]
        category_str = tag_info.get("category", "")

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

    def select_diverse_features(self, features):
        """Pick top 2 features from different categories, avoiding all-adjective pairs."""
        categorized = []
        for feature in features:
            category, name = self.classify_feature(feature)
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

    def assign_roles(self, categorized_features):
        """Assign modifier and core roles based on priority, swapping if core is adjective-only."""
        modifier_name = categorized_features[0][2]
        core_name = categorized_features[1][2]
        core_cat = categorized_features[1][1]

        if core_name in self.ADJECTIVE_ONLY and modifier_name not in self.ADJECTIVE_ONLY:
            modifier_name, core_name = core_name, modifier_name
            core_cat = categorized_features[0][1]

        return modifier_name, core_name, core_cat

    def to_modifier_form(self, feature_name):
        """Convert feature to its adjectival form when used as modifier."""
        return self.ADJECTIVE_FORM.get(feature_name, feature_name)

    def get_tag_category_string(self, feature):
        """Get the raw category string for a tag (e.g. 'Setting-Scene', 'Theme-Other')."""
        if feature in self.tag_by_name:
            return self.tag_by_name[feature].get("category", "")
        return ""

    def resolve_adjective_pair(self, modifier_name, core_name):
        """Resolve both-adjective-only pairs by finding a suitable noun."""
        if core_name in self.PREFERRED_NOUN:
            return self.to_modifier_form(modifier_name), self.PREFERRED_NOUN[core_name]

        if modifier_name in self.PREFERRED_NOUN:
            return self.to_modifier_form(core_name), self.PREFERRED_NOUN[modifier_name]

        core_category = self.get_tag_category_string(core_name)
        if core_category in self.CATEGORY_NOUNS:
            return self.to_modifier_form(modifier_name), self.CATEGORY_NOUNS[core_category]

        return self.to_modifier_form(modifier_name), f"{core_name} Anime"

    def resolve_conflicts(self, modifier_name, core_name, core_cat):
        """Apply conflict resolution rules in order to produce the final name string."""
        mod_lower = modifier_name.lower()
        core_lower = core_name.lower()

        # Substring overlap — keep the longer/more specific one
        if mod_lower in core_lower or core_lower in mod_lower:
            return max(modifier_name, core_name, key=len)

        # Both adjective-only — resolve with preferred noun or fallback
        if modifier_name in self.ADJECTIVE_ONLY and core_name in self.ADJECTIVE_ONLY:
            modifier, core = self.resolve_adjective_pair(modifier_name, core_name)
            return f"{modifier} {core}"

        # Demographics read better as suffixes without adjectival transform
        if core_cat == "Demographic":
            return f"{modifier_name} {core_name}"

        # Standard case: apply adjectival transform to modifier
        modifier = self.to_modifier_form(modifier_name)
        return f"{modifier} {core_name}"

    def format_name(self, name):
        """Capitalize first letter only, preserving rest (e.g., 'CGI' stays 'CGI')."""
        return name[0].upper() + name[1:] if name else ""

    def name_single_cluster(self, features):
        """Generate a natural language name for a single cluster from its feature list."""
        if not features:
            return ""

        categorized_features = self.select_diverse_features(features)

        if not categorized_features:
            return " ".join(features[:2])

        if len(categorized_features) == 1:
            name = categorized_features[0][2]
        else:
            modifier_name, core_name, core_cat = self.assign_roles(categorized_features)
            name = self.resolve_conflicts(modifier_name, core_name, core_cat)

        return self.format_name(name)

    def deprioritize_shared(self, features, shared):
        """Move shared features to the end so more distinctive ones get picked."""
        return [f for f in features if f not in shared] + [f for f in features if f in shared]

    def resolve_duplicates(self, names, cluster_features):
        """Re-generate colliding names with shared features deprioritized."""
        name_to_clusters = {}
        for cid, name in names.items():
            name_to_clusters.setdefault(name, []).append(cid)

        for _name, cluster_ids in name_to_clusters.items():
            if len(cluster_ids) <= 1:
                continue

            for cid in cluster_ids:
                features = cluster_features[cid]
                sibling_features = {
                    f for other in cluster_ids if other != cid for f in cluster_features[other]
                }
                shared = set(features) & sibling_features
                reordered = self.deprioritize_shared(features, shared)
                names[cid] = self.name_single_cluster(reordered)

        return names

    def name_clusters(self, cluster_features):
        """Generate deduplicated names for all clusters.

        Takes a dict of {cluster_id: [features]}, returns {cluster_id: name}.
        """
        names = {cid: self.name_single_cluster(feats) for cid, feats in cluster_features.items()}

        return self.resolve_duplicates(names, cluster_features)

    def name_clusters_from_data(self, dataframe, feature_column, cluster_column):
        """Extract descriptive features via TF-IDF and generate deduplicated cluster names."""
        descriptions = statistics.get_descriptive_features(
            dataframe,
            feature_column,
            cluster_column,
            n_features=5,
            boost_features=self.genres,
        )

        cluster_features = {
            row["cluster"]: row["description"] for row in descriptions.iter_rows(named=True)
        }

        return self.name_clusters(cluster_features)
