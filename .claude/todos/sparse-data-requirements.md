# Sparse data handling requirements

## Problem

Shows with very few features (e.g., 2 tags + 2 genres) produce inflated similarity scores because there aren't enough dimensions to disagree on. A show whose entire feature vector is Adventure+Fantasy scores 0.8+ similarity against many watched items — not because it's genuinely similar, but because there's no evidence of dissimilarity. The current `MIN_FEATURE_THRESHOLD = 4` gives full confidence to these sparse shows, allowing them to rank artificially high.

## Fix 1: Bayesian shrinkage on similarity values

Before the DirectSimilarityScorer uses similarity values for scoring, shrink each pairwise similarity toward the global mean:

```
prior = global_mean_similarity  # mean of all pairwise values in the similarity matrix
shrunk_sim = (shared_feature_count * raw_sim + k * prior) / (shared_feature_count + k)
```

`shared_feature_count` is the number of features both the candidate and the watched item have in common (the intersection), not either show's total feature count. Compute per pair when building the match table.

`k` controls shrinkage strength (configurable, suggested default 8-10). With k=8, a pair sharing 4 features is pulled ~67% toward the prior. A pair sharing 30 features is pulled ~21% toward the prior.

The prior (global mean similarity) can be computed once from the similarity matrix at the start of scoring.

### Scope

Apply shrinkage only inside DirectSimilarityScorer. Do not modify the shared similarity matrix. Clustering and ClusterSimilarityScorer need raw unmodified similarities to group shows correctly — shrinking sparse shows' similarity toward the mean would make them equidistant from everything and unplaceable.

## Fix 2: Smooth confidence curve

Replace the linear feature confidence formula used across all scorers:

```
# Old: hard threshold, full confidence at MIN_FEATURE_THRESHOLD
feature_conf = min(candidate_feature_count / MIN_FEATURE_THRESHOLD, 1.0)

# New: smooth asymptotic curve, never fully saturates
feature_conf = candidate_feature_count / (candidate_feature_count + k)
```

Use the same `k` value as the shrinkage (suggested default 8). This produces:

- 4 features → 0.33
- 8 features → 0.50
- 12 features → 0.60
- 16 features → 0.67
- 30 features → 0.79

Sparse shows still contribute to the blend — they aren't zeroed out. But they contribute proportionally less, and the confidence-weighted blending redistributes their lost weight to scorers with more data.

### Scope

Apply the smooth curve everywhere `feature_confidence` is currently used: DirectSimilarityScorer, FeatureCorrelationScorer, and ClusterSimilarityScorer. This is a change to the shared `feature_confidence` method in the base scorer class, not per-scorer logic.

## Fix 3: Remove MIN_FEATURE_THRESHOLD constant

The smooth curve replaces the threshold entirely. There is no longer a point at which confidence jumps to 1.0. Remove `MIN_FEATURE_THRESHOLD` from the codebase to avoid confusion about which formula is in use.

## Interaction with existing confidence logic

Each scorer's confidence calculation multiplies `feature_conf` with other scorer-specific factors. Those factors are unchanged:

- DirectSimilarityScorer: `feature_conf * match_conf`
- FeatureCorrelationScorer: `feature_conf * history_conf * contested_penalty`
- ClusterSimilarityScorer: `feature_conf * cluster_cohesion`

Only the `feature_conf` component changes. Everything else stays as-is.

## Expected behavior after changes

A show with 4 features (like Neko to Ryuu) would see:

- DirectSimilarityScorer: raw similarity shrunk from ~0.8 toward ~0.5, feature confidence at 0.33 instead of 1.0. Effective contribution to blend reduced significantly.
- FeatureCorrelationScorer: feature confidence at 0.33, reducing its effective weight. Other scorers with more data absorb the slack.
- The show still appears in recommendations but at an honest position reflecting the limited evidence, not inflated to the top.

A show with 25 features would see:

- DirectSimilarityScorer: similarity barely affected by shrinkage, feature confidence at 0.76. Near-full contribution.
- Negligible change from current behavior.