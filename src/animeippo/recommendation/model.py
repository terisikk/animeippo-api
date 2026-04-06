import polars as pl

from ..analysis import similarity
from ..providers.util import filter_continuation
from ..recommendation import cluster_naming


class RecommendationModel:
    """Collection of dataframes and other data related to
    the recommendation system."""

    def __init__(self, user_profile, seasonal, features=None):
        self.user_profile = user_profile
        self.watchlist = user_profile.watchlist if user_profile is not None else None
        self.seasonal = seasonal
        self.mangalist = user_profile.mangalist if user_profile is not None else None
        self.recommendations = None

        self.all_features = features
        self.nsfw_tags = []
        self.similarity_matrix = None
        self._cluster_names = None
        self._cluster_rankings = None
        self._explode_cache = {}
        self._sim_matrix_cache = {}

    def validate(self):
        is_missing_seasonal = self.seasonal is None
        is_missing_watchlist = self.watchlist is None

        if is_missing_seasonal or is_missing_watchlist:
            error_desc = (
                f"Watchlist invalid?: {is_missing_watchlist}. "
                + f"Seasonal invalid?: {is_missing_seasonal}"
            )

            raise RuntimeError("Trying to recommend anime without proper data. " + error_desc)

    def encode(self, encoder):
        self.all_features = self.extract_features()

        encoder.fit(self.all_features)

        self.watchlist = self.watchlist.with_columns(encoded=encoder.encode(self.watchlist))
        self.seasonal = self.seasonal.with_columns(encoded=encoder.encode(self.seasonal))

    def fill_user_status_data_from_watchlist(self):
        self.seasonal = self.seasonal.join(
            self.watchlist.select(["id", "user_status"]), on="id", how="left"
        )

    def filter_continuation(self):
        if self.seasonal is not None and self.watchlist is not None:
            previously_watched = self.watchlist.filter(
                pl.col("user_status").is_in(["COMPLETED", "CURRENT", "PAUSED"])
            )["id"].to_list()

            self.seasonal = filter_continuation(self.seasonal, previously_watched)

    def build_relation_context(self):
        """Tag seasonal items that are summaries/compilations of watched anime.

        Uses the reverse lookup: if a watchlist item lists a seasonal item
        as SUMMARY/COMPILATION in its franchise_relations, that seasonal item
        is a recap rather than a real sequel.
        """
        if (
            self.watchlist is None
            or self.seasonal is None
            or "franchise_relations" not in self.watchlist.columns
        ):
            self.seasonal = self.seasonal.with_columns(is_summary=pl.lit(False))
            return

        summary_ids = (
            self.watchlist.explode("franchise_relations")
            .unnest("franchise_relations")
            .filter(pl.col("relation_type").is_in(["SUMMARY", "COMPILATION"]))
            .select("related_id")
            .unique()
            .to_series()
        )

        self.seasonal = self.seasonal.with_columns(
            is_summary=pl.col("id").is_in(summary_ids.to_list())
        )

    def fit(self, encoder, clustering_model):
        self.validate()

        self.fill_user_status_data_from_watchlist()
        self.filter_continuation()
        self.build_relation_context()

        self.encode(encoder)

        self.watchlist = self.watchlist.with_columns(
            cluster=clustering_model.cluster_by_features(self.watchlist)
        )

        self.similarity_matrix = similarity.categorical_similarity(
            self.watchlist["encoded"],
            self.seasonal["encoded"],
            clustering_model.distance_metric,
            self.seasonal["id"].cast(pl.Utf8),
        ).with_columns(id=self.watchlist["id"])
        # Categories could use unfiltered watchlist, but scoring needs to filter it

        # Rechunk to maximize performance, not sure if it has any real effect
        self.seasonal = self.seasonal.rechunk()
        self.watchlist = self.watchlist.rechunk()

    def extract_features(self):
        seasonal_cats = set(self.seasonal["features"].explode().cat.get_categories().to_list())
        watchlist_cats = set(self.watchlist["features"].explode().cat.get_categories().to_list())
        return seasonal_cats | watchlist_cats

    def watchlist_explode_cached(self, column):
        key = ("watchlist", column)
        if key not in self._explode_cache:
            self._explode_cache[key] = self.watchlist.explode(column)
        return self._explode_cache[key]

    def recommendations_explode_cached(self, column):
        key = ("recommendations", column)
        if key not in self._explode_cache:
            self._explode_cache[key] = self.recommendations.explode(column)
        return self._explode_cache[key]

    def seasonal_explode_cached(self, column):
        key = ("seasonal", column)
        if key not in self._explode_cache:
            self._explode_cache[key] = self.seasonal.explode(column)
        return self._explode_cache[key]

    def get_similarity_matrix(self, filtered=False, transposed=False):
        key = (filtered, transposed)
        if key not in self._sim_matrix_cache:
            ret = self.similarity_matrix

            if filtered:
                ret = ret.filter(~pl.col("id").is_in(self.seasonal["id"].implode()))

            if transposed:
                column_names = ret["id"].cast(pl.Utf8).to_list()
                ret = ret.select(pl.exclude("id")).transpose(
                    include_header=True, header_name="id", column_names=column_names
                )

            self._sim_matrix_cache[key] = ret
        return self._sim_matrix_cache[key]

    def get_cluster_names(self, tag_lookup, genres):
        if self._cluster_names is None:
            self._cluster_names = cluster_naming.name_all_clusters(
                self.watchlist, tag_lookup, genres
            )
        return self._cluster_names

    def get_cluster_rankings(self):
        if self._cluster_rankings is None:
            self._cluster_rankings = cluster_naming.rank_clusters(
                self.watchlist, self.recommendations
            )
        return self._cluster_rankings
