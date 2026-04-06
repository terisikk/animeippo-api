from typing import ClassVar

import numpy as np
import polars as pl
import sklearn.cluster as skcluster
from sklearn.metrics import pairwise_distances

from ..analysis import similarity


class AnimeClustering:
    """Agglomerative clustering for anime feature vectors.

    Uses cosine distance with average linkage. Cosine is the right metric because
    feature vectors are sparse and high-dimensional (~370 dims) with weighted tag
    ranks — cosine measures similarity by angle, ignoring magnitude differences
    between anime with many vs few tags. Euclidean would penalize magnitude;
    Jaccard would discard the tag rank weights.

    Average linkage is used over complete linkage because complete linkage requires
    ALL pairwise distances within a cluster to be below threshold — one outlier pair
    (e.g. a franchise sequel with slightly different tags) blocks the entire merge.
    Average linkage uses mean pairwise distance, tolerating individual outliers.
    Single linkage was too aggressive (chaining effect merging dissimilar anime).

    HDBSCAN was evaluated but doesn't suit this data — the cosine distance
    distribution is very flat (median ~0.86, range 0.77-0.93) with no clear
    density structure. HDBSCAN classifies 30-50% of items as noise regardless
    of parameter tuning.

    Agglomerative clustering is one of the few sklearn methods that (1) doesn't
    need the number of clusters specified upfront, (2) supports precomputed
    distance matrices for franchise reduction, (3) assigns every item to a
    cluster, and (4) provides a distance threshold that's easy to reason about.

    Franchise reduction modifies the precomputed distance matrix to pull related
    anime closer before clustering. Small cluster merging reassigns clusters
    below min_cluster_size to their nearest larger cluster as a post-clustering
    cleanup.
    """

    DIRECT_SEQUEL_TYPES: ClassVar[set[str]] = {
        "SEQUEL",
        "PREQUEL",
        "SUMMARY",
        "COMPILATION",
        "ALTERNATIVE",
    }

    def __init__(  # noqa: PLR0913
        self,
        distance_metric="cosine",
        distance_threshold=0.85,
        linkage="average",
        n_clusters=None,
        min_cluster_size=1,
        franchise_reduction=False,
        direct_factor=0.4,
        related_factor=0.6,
        **kwargs,
    ):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.min_cluster_size = min_cluster_size
        self.franchise_reduction = franchise_reduction
        self.relation_tiers = {"direct": direct_factor, "related": related_factor}
        self.is_fit = False
        self.clustered_series = None
        self.distance_metric = distance_metric

    def cluster_by_features(self, dataframe):
        series = dataframe["encoded"].struct.unnest().fill_null(0).to_numpy()
        mask = self.get_valid_mask(series)

        dist_matrix = self.build_distance_matrix(series[mask], dataframe, mask)
        clusters = self.fit_clusters(dist_matrix, len(series), mask)
        self.postprocess_clusters(clusters, dist_matrix, mask)

        self.is_fit = True
        self.clustered_series = dataframe.with_columns(cluster=clusters)

        return clusters

    def get_valid_mask(self, series):
        """Cosine is undefined for zero-vectors; exclude them."""
        return series.sum(axis=1) > 0

    def build_distance_matrix(self, series, dataframe, mask):
        dist_matrix = pairwise_distances(series, metric=self.distance_metric)
        if self.franchise_reduction:
            relation_pairs = self.get_relation_pairs(dataframe)
            self.apply_franchise_reduction(dist_matrix, mask, relation_pairs)
        return dist_matrix

    def fit_clusters(self, dist_matrix, n_items, mask):
        clusters = np.full(n_items, -1)
        model = skcluster.AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric="precomputed",
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
        )
        clusters[mask] = model.fit_predict(dist_matrix)
        self.model = model
        return clusters

    def postprocess_clusters(self, clusters, dist_matrix, mask):
        if self.min_cluster_size > 1:
            self.merge_small_clusters(clusters, dist_matrix, mask)

    def merge_small_clusters(self, clusters, dist_matrix, mask):
        """Merge entire small clusters into their nearest larger cluster as a group."""
        masked_labels = clusters[mask]
        unique, counts = np.unique(masked_labels, return_counts=True)
        count_map = dict(zip(unique, counts, strict=True))

        small_labels = [lab for lab, cnt in count_map.items() if cnt < self.min_cluster_size]
        large_labels = [lab for lab, cnt in count_map.items() if cnt >= self.min_cluster_size]

        if not small_labels or not large_labels:
            return

        n_points = len(masked_labels)
        large_arr = np.array(large_labels)

        # Build normalized mask matrices: one row per cluster, normalized by size
        small_masks = np.zeros((len(small_labels), n_points))
        large_masks = np.zeros((len(large_labels), n_points))

        for i, lab in enumerate(small_labels):
            idx = masked_labels == lab
            small_masks[i, idx] = 1.0 / idx.sum()

        for i, lab in enumerate(large_labels):
            idx = masked_labels == lab
            large_masks[i, idx] = 1.0 / idx.sum()

        # (n_small, n_large) mean distance matrix in two matmuls
        mean_dists = small_masks @ dist_matrix @ large_masks.T
        nearest = large_arr[mean_dists.argmin(axis=1)]

        for i, small_label in enumerate(small_labels):
            masked_labels[masked_labels == small_label] = nearest[i]

        clusters[mask] = masked_labels

    def get_relation_pairs(self, dataframe):
        """Build dict of {(i, j): tier} from franchise and relation data.

        All franchise members get the "related" tier baseline (transitive closure).
        Direct sequel/prequel relations upgrade specific pairs to "direct" tier.
        """
        if "franchise" not in dataframe.columns:
            return {}

        pairs = self.get_franchise_pairs(dataframe)

        if "franchise_relations" in dataframe.columns:
            self.upgrade_direct_pairs(dataframe, pairs)

        return pairs

    def get_franchise_pairs(self, dataframe):
        """Build "related" tier pairs from franchise groups (transitive closure)."""
        franchise_groups = {}
        for idx, franchise_list in enumerate(dataframe["franchise"].to_list()):
            if franchise_list:
                franchise_groups.setdefault(franchise_list[0], []).append(idx)

        pairs = {}
        for members in franchise_groups.values():
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    pairs[(a, b)] = "related"

        return pairs

    def upgrade_direct_pairs(self, dataframe, pairs):
        """Upgrade pairs to "direct" tier based on typed relation edges."""
        id_to_idx = {aid: idx for idx, aid in enumerate(dataframe["id"].to_list())}

        for idx_a, relations in enumerate(dataframe["franchise_relations"].to_list()):
            if not relations:
                continue
            for rel in relations:
                idx_b = id_to_idx.get(rel["related_id"])
                if idx_b is not None and idx_a != idx_b:
                    if rel["relation_type"] in self.DIRECT_SEQUEL_TYPES:
                        pair = (min(idx_a, idx_b), max(idx_a, idx_b))
                        pairs[pair] = "direct"

    def apply_franchise_reduction(self, dist_matrix, mask, relation_pairs):
        """Reduce distances between related anime using tiered factors."""
        masked_indices = np.where(mask)[0]
        original_to_masked = {orig: masked for masked, orig in enumerate(masked_indices)}

        for (orig_i, orig_j), tier in relation_pairs.items():
            mi = original_to_masked.get(orig_i)
            mj = original_to_masked.get(orig_j)

            if mi is None or mj is None:
                continue

            factor = self.relation_tiers[tier]
            dist_matrix[mi, mj] *= factor
            dist_matrix[mj, mi] *= factor

    def predict(self, series, similarities=None):
        """Assign each new item to the cluster with highest average similarity.

        Returns a DataFrame with 'cluster' and 'similarity' columns.
        """
        if not self.is_fit:
            raise RuntimeError("Cluster is not fitted yet. Please call cluster_by_features first.")

        if similarities is None:
            similarities = similarity.categorical_similarity(
                self.clustered_series["encoded"],
                series,
                metric=self.distance_metric,
            ).with_columns(id=self.clustered_series["id"])

        sim_cols = [c for c in similarities.columns if c != "id"]

        return (
            similarities.with_columns(pl.col(sim_cols).fill_nan(0.0))
            .join(self.clustered_series.select("id", "cluster"), on="id")
            .filter(pl.col("cluster") >= 0)
            .group_by("cluster", maintain_order=True)
            .agg(pl.col(sim_cols).mean())
            .unpivot(index="cluster", variable_name="item", value_name="avg_sim")
            .sort("avg_sim", descending=True)
            .group_by("item", maintain_order=True)
            .agg(
                pl.col("cluster").first(),
                pl.col("avg_sim").first().alias("similarity"),
            )
            .sort("item")
            .select("cluster", "similarity")
        )
