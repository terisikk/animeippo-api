import polars as pl


class RankingOrchestrator:
    """Maintains global state for cross-category ranking adjustments."""

    DISCOURAGE_MAX = 0.75

    diversity_adjustment = None

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
        self.recommendations_df = recommendations_df

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
            item_ids = category.get_items(filtered_sorted, top_n)

            if category.needs_diversity:
                item_ids = self.adjust_by_diversity(item_ids, top_n=top_n)

            rendered_categories.append({"name": category.description, "items": item_ids})

        return rendered_categories

    def adjust_by_diversity(self, candidate_ids, top_n):
        """Select top_n from candidates, penalizing items shown in other genre lanes."""
        if not candidate_ids:
            return []

        candidates = pl.DataFrame({"id": pl.Series(candidate_ids, dtype=pl.UInt32)})

        selected = (
            candidates.join(self.diversity_adjustment, on="id", how="left")
            .join(self.recommendations_df.select("id", "discovery_score"), on="id", how="left")
            .with_columns(
                adjusted_score=pl.col("discovery_score") - pl.col("diversity_adjustment"),
            )
            .sort(by="adjusted_score", descending=True)[0:top_n]["id"]
            .to_list()
        )

        self.diversity_adjustment = self.diversity_adjustment.with_columns(
            diversity_adjustment=pl.when(pl.col("id").is_in(selected))
            .then(pl.col("diversity_adjustment") + 0.25)
            .otherwise(pl.col("diversity_adjustment"))
            .clip(upper_bound=self.DISCOURAGE_MAX)
        )

        return selected
