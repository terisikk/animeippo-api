import polars as pl


class RankingOrchestrator:
    """Selects category layout by data volume and renders categories."""

    DISCOURAGE_MAX = 0.75

    MINIMAL_THRESHOLD = 20
    FULL_THRESHOLD = 100

    def __init__(self, layouts):
        """Initialize with layouts dict.

        Args:
            layouts: Dict of {"minimal": [...], "standard": [...], "full": [...]}
                where each value is a list of (category, top_n) tuples.
                Also accepts a plain list for backward compatibility.
        """
        if isinstance(layouts, list):
            self.layouts = {"full": layouts}
        else:
            self.layouts = layouts

    def select_layout(self, item_count):
        if item_count < self.MINIMAL_THRESHOLD and "minimal" in self.layouts:
            return self.layouts["minimal"]
        if item_count < self.FULL_THRESHOLD and "standard" in self.layouts:
            return self.layouts["standard"]
        return self.layouts["full"]

    def render(self, data):
        """Render categories based on data volume.

        Returns:
            List of {"name": str, "items": [ids]} dicts
        """
        rendered_categories = []
        recommendations_df = data.recommendations

        layout = self.select_layout(len(recommendations_df))

        diversity_adjustment = pl.DataFrame(
            {
                "id": recommendations_df["id"],
                "diversity_adjustment": 0,
            }
        )

        for category, top_n in layout:
            mask, sorting_info = category.categorize(data)

            if mask is False:
                continue

            filtered_sorted = recommendations_df.filter(mask).sort(**sorting_info)

            if len(filtered_sorted) < category.min_items:
                continue

            item_ids = category.get_items(filtered_sorted, top_n)

            if category.needs_diversity:
                item_ids, diversity_adjustment = _adjust_by_diversity(
                    item_ids,
                    recommendations_df,
                    diversity_adjustment,
                    top_n=top_n,
                    discourage_max=self.DISCOURAGE_MAX,
                )

            rendered_categories.append({"name": category.description, "items": item_ids})

        return rendered_categories


def _adjust_by_diversity(
    candidate_ids, recommendations_df, diversity_adjustment, *, top_n, discourage_max
):
    """Select top_n from candidates, penalizing items shown in other genre lanes.

    Returns (selected_ids, updated_diversity_adjustment).
    """
    if not candidate_ids:
        return [], diversity_adjustment

    candidates = pl.DataFrame({"id": pl.Series(candidate_ids, dtype=pl.UInt32)})

    selected = (
        candidates.join(diversity_adjustment, on="id", how="left")
        .join(recommendations_df.select("id", "discovery_score"), on="id", how="left")
        .with_columns(
            adjusted_score=pl.col("discovery_score") - pl.col("diversity_adjustment"),
        )
        .sort(by="adjusted_score", descending=True)[0:top_n]["id"]
        .to_list()
    )

    diversity_adjustment = diversity_adjustment.with_columns(
        diversity_adjustment=pl.when(pl.col("id").is_in(selected))
        .then(pl.col("diversity_adjustment") + 0.25)
        .otherwise(pl.col("diversity_adjustment"))
        .clip(upper_bound=discourage_max)
    )

    return selected, diversity_adjustment
