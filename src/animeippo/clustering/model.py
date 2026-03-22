from typing import ClassVar

import numpy as np
import sklearn.cluster as skcluster
from sklearn.metrics import pairwise_distances

from animeippo.analysis import statistics

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
    anime closer before clustering. Singleton merging reassigns single-item
    clusters to their nearest multi-member cluster as a post-clustering cleanup.
    """

    def __init__(  # noqa: PLR0913
        self,
        distance_metric="jaccard",
        distance_threshold=0.85,
        linkage="average",
        n_clusters=None,
        min_cluster_size=1,
        franchise_reduction=False,
        **kwargs,
    ):
        self.model = skcluster.AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=distance_metric,
            distance_threshold=distance_threshold,
            linkage=linkage,
            **kwargs,
        )

        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.min_cluster_size = min_cluster_size
        self.franchise_reduction = franchise_reduction
        self.is_fit = False
        self.clustered_series = None
        self.distance_metric = distance_metric

    # Tiered distance reduction: direct sequels get stronger boost than loose relations
    DIRECT_SEQUEL_TYPES: ClassVar[set[str]] = {
        "SEQUEL",
        "PREQUEL",
        "SUMMARY",
        "COMPILATION",
        "ALTERNATIVE",
    }

    RELATION_TIERS: ClassVar[dict] = {
        "direct": 0.4,
        "related": 0.6,
    }

    def cluster_by_features(self, dataframe):
        series = dataframe["encoded"].struct.unnest().fill_null(0).to_numpy()

        relation_pairs = self.get_relation_pairs(dataframe) if self.franchise_reduction else {}

        if self.distance_metric == "cosine":
            # Cosine is undefined for zero-vectors
            clusters = np.full(len(series), -1)
            mask = series.sum(axis=1) > 0

            dist_matrix = pairwise_distances(series[mask], metric="cosine")
            self.apply_franchise_reduction(dist_matrix, mask, relation_pairs)

            precomputed_model = skcluster.AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric="precomputed",
                distance_threshold=self.distance_threshold,
                linkage=self.linkage,
            )
            clusters[mask] = precomputed_model.fit_predict(dist_matrix)
            if self.min_cluster_size > 1:
                self.merge_small_clusters(clusters, dist_matrix, mask)
            self.model = precomputed_model
        else:
            clusters = self.model.fit_predict(series)

        if clusters is not None:
            self.is_fit = True
            self.clustered_series = dataframe.with_columns(cluster=clusters)

        return clusters

    def merge_small_clusters(self, clusters, dist_matrix, mask):
        """Merge entire small clusters into their nearest larger cluster as a group."""
        masked_labels = clusters[mask]
        unique, counts = np.unique(masked_labels, return_counts=True)
        count_map = dict(zip(unique, counts, strict=True))

        small_labels = {lab for lab, cnt in count_map.items() if cnt < self.min_cluster_size}
        large_labels = {lab for lab, cnt in count_map.items() if cnt >= self.min_cluster_size}

        if not small_labels or not large_labels:
            return

        for small_label in small_labels:
            small_indices = np.where(masked_labels == small_label)[0]

            best_label = None
            best_dist = np.inf
            for target_label in large_labels:
                target_indices = np.where(masked_labels == target_label)[0]
                avg_dist = dist_matrix[np.ix_(small_indices, target_indices)].mean()
                if avg_dist < best_dist:
                    best_dist = avg_dist
                    best_label = target_label

            masked_labels[small_indices] = best_label

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

            factor = self.RELATION_TIERS[tier]
            dist_matrix[mi, mj] *= factor
            dist_matrix[mj, mi] *= factor

    def predict(self, series, similarities=None):
        if not self.is_fit:
            raise RuntimeError("Cluster is not fitted yet. Please call cluster_by_features first.")

        if similarities is None:
            similarities = similarity.categorical_similarity(
                self.clustered_series["encoded"],
                series,
                metric=self.distance_metric,
            ).with_columns(id=self.clustered_series["id"])

        idymax = statistics.idymax(similarities)

        return idymax.join(
            self.clustered_series.select("id", "cluster"),
            left_on="idymax",
            right_on="id",
        )["cluster"]
