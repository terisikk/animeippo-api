import polars as pl

from animeippo.analysis import statistics


class UserProfile:
    def __init__(self, user, watchlist, mangalist=None):
        self.user = user
        self.watchlist = watchlist
        self.mangalist = mangalist
        self.last_liked = None
        self.genre_correlations = None
        self.director_correlations = None
        self.studio_correlations = None
        self.favourite_source = None

        if self.watchlist is not None and "score" in self.watchlist.columns:
            self.fit()

    def fit(self):
        self.genre_correlations = self.get_genre_correlations()

        self.director_correlations = self.get_director_correlations()
        self.studio_correlations = self.get_studio_correlations()
        self.last_liked = self.get_last_liked()
        self.favourite_source = self.get_favourite_source()

    def get_last_liked(self):
        if "user_complete_date" not in self.watchlist.columns:
            return None

        mask = (
            pl.col("score").ge(pl.col("score").mean()) & pl.col("user_complete_date").is_not_null()
        )

        return self.watchlist.filter(mask).sort("user_complete_date", descending=True).head(10)

    def get_genre_correlations(self):
        if "genres" not in self.watchlist.columns:
            return None

        gdf = self.watchlist.explode("genres")

        return statistics.weight_categoricals_correlation(gdf, "genres").sort(
            "weight", descending=True
        )

    def get_studio_correlations(self):
        if "studios" not in self.watchlist.columns:
            return None

        gdf = self.watchlist.explode("studios")

        return statistics.weight_categoricals_correlation(gdf, "studios").sort(
            "weight", descending=True
        )

    def get_director_correlations(self):
        if "directors" not in self.watchlist.columns:
            return None

        gdf = self.watchlist.explode("directors")

        return statistics.weight_categoricals_correlation(gdf, "directors").sort(
            "weight", descending=True
        )

    def get_favourite_source(self):
        if "source" not in self.watchlist.columns:
            return None

        favourite_source = (
            self.watchlist.group_by("source")
            .agg(pl.col("score").mean() * pl.col("source").count().sqrt())
            .drop_nulls("score")
            .sort("score", descending=True)
            .select(pl.first("source"))
            .item()
        )

        if favourite_source is None:
            favourite_source = "Manga"

        return favourite_source.lower()
