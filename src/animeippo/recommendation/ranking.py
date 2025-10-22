import polars as pl

from animeippo.recommendation import categories


class RankingOrchestrator:
    """Maintains global state for cross-category ranking adjustments."""

    DISCOURAGE_MAX = 0.75

    diversity_adjustment = None

    # FIXME: Maybe use a better syntax than type checking here?
    diversity_adjusted_categories = [categories.GenreCategory]  # noqa RUF012

    def __init__(self, categorizers_with_limits):
        """Initialize with list of (category, top_n) tuples.

        Args:
            categorizers_with_limits: List of (category, top_n) tuples where
                top_n is the maximum number of items to return (None for no limit)
        """
        self.categorizers_with_limits = categorizers_with_limits

    def render(self, data):
        """Render categories with their configured limits.

        Args:
            data: RecommendationModel with recommendations dataframe

        Returns:
            List of {"name": str, "items": [ids]} dicts
        """
        rendered_categories = []
        recommendations_df = data.recommendations

        self.diversity_adjustment = pl.DataFrame(
            {
                "id": recommendations_df["id"],
                "diversity_adjustment": 0,
            }
        )

        for category, top_n in self.categorizers_with_limits:
            mask, sorting_info = category.categorize(data)

            if mask is False:
                continue

            filtered_sorted = recommendations_df.filter(mask).sort(**sorting_info)

            if type(category) in self.diversity_adjusted_categories:
                item_ids = self.adjust_by_diversity(filtered_sorted, top_n=top_n)
            else:
                item_ids = filtered_sorted["id"][0:top_n].to_list()

            rendered_categories.append({"name": category.description, "items": item_ids})

        return rendered_categories

    def adjust_by_diversity(self, recommendations_df, top_n):
        """
        Calculate adjusted scores based on diversity_adjustment in dataframe.
        """

        df = (
            recommendations_df.join(self.diversity_adjustment, on="id", how="left")
            .select(
                pl.col("id"),
                (pl.col("recommend_score") - pl.col("diversity_adjustment")),
            )
            .sort(by=["recommend_score"], descending=[True])
        )

        if top_n is not None:
            df = df[0:top_n]

        ids = df["id"].to_list()

        self.diversity_adjustment = self.diversity_adjustment.with_columns(
            diversity_adjustment=pl.when(pl.col("id").is_in(ids))
            .then(pl.col("diversity_adjustment") + 0.25)
            .otherwise(pl.col("diversity_adjustment") + 0)
            .clip(upper_bound=self.DISCOURAGE_MAX)
        )

        return ids
