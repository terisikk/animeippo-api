import polars as pl

from animeippo.recommendation import categories


class RankingOrchestrator:
    """Maintains global state for cross-category ranking adjustments."""

    DISCOURAGE_MAX = 0.75

    diversity_adjustment = None

    # FIXME: Maybe use a better syntax than type checking here?
    diversity_adjusted_categories = [categories.GenreCategory]  # noqa RUF012

    def render(self, data, categories):
        """Render the recommendations dataframe with adjusted scores."""

        rendered_categories = []

        recommendations_df = data.recommendations

        self.diversity_adjustment = pl.DataFrame(
            {
                "id": recommendations_df["id"],
                "diversity_adjustment": 0,
            }
        )

        for category in categories:
            mask, sorting_info = category.categorize(data)

            # Skip categories that returned False (no results)
            if mask is False:
                continue

            if type(category) in self.diversity_adjusted_categories:
                item_ids = self.adjust_by_diversity(
                    recommendations_df.filter(mask).sort(**sorting_info), top_n=None
                )
            else:
                item_ids = (
                    recommendations_df.filter(mask).sort(**sorting_info)["id"][0:25].to_list()
                )

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
