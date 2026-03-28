# Lane blending architecture requirements

## Overview

Restructure the recommendation pipeline to separate continuation scoring from the discovery scoring blend, introduce per-scorer confidence values, and compose lane outputs by interleaving independently ranked pools rather than blending all signals into a single score.

## Scorer architecture changes

Each scorer returns a `(score, confidence)` pair instead of a bare score. Score represents the signal strength (0–1). Confidence represents data quality (0–1), meaning "how much data did I have to work with," not "how high is my score." A scorer that confidently finds zero match returns `(0.0, 1.0)`. A scorer working with sparse data returns `(whatever, 0.3)`.

Continuation scorer is excluded from the discovery blend. All other scorers participate in the discovery blend: direct similarity, feature correlation, cluster similarity, studio correlation, popularity, and adaptation.

## Per-scorer confidence calculations

Each scorer computes confidence based on the data quality available to it. Confidence reflects "how much data did I have to work with," not score magnitude. The following are proposed calculation methods.

### Direct similarity

Confidence is based on how many features the candidate has available for comparison. A candidate with very few tags or metadata fields cannot produce a meaningful similarity score even if the few features it has happen to match.

```
confidence = min(candidate_feature_count / MIN_FEATURE_THRESHOLD, 1.0)
```

`MIN_FEATURE_THRESHOLD` is the feature count at which the scorer considers data sufficient (configurable, inherits from the current feature coverage threshold).

### Feature correlation

Confidence depends on two factors: the candidate's feature richness (same as direct similarity) and how much watch history the user has. Both must be adequate for the correlation to be meaningful.

```
feature_confidence = min(candidate_feature_count / MIN_FEATURE_THRESHOLD, 1.0)
history_confidence = min(user_watch_history_count / MIN_HISTORY_THRESHOLD, 1.0)
confidence = feature_confidence * history_confidence
```

`MIN_HISTORY_THRESHOLD` is the watch history size at which correlations stabilize (configurable, suggested default ~20).

### Cluster similarity

Confidence depends on how well-defined the candidate's cluster is. A tight cluster with many similar members produces a strong neighborhood signal. A loose or catch-all cluster is noise. Also inherits feature sparsity since clustering depends on the same underlying features.

```
cluster_confidence = min(cluster_cohesion / COHESION_THRESHOLD, 1.0)
feature_confidence = min(candidate_feature_count / MIN_FEATURE_THRESHOLD, 1.0)
confidence = cluster_confidence * feature_confidence
```

Cluster cohesion can be measured as the average intra-cluster similarity, or inversely as the cluster's silhouette width. The exact metric depends on the clustering implementation.

### Studio correlation

Confidence is based on how many shows from this studio appear in the user's watch history. A studio the user has seen ten times gives a reliable signal. A studio they've never encountered gives zero confidence.

```
confidence = min(user_shows_from_studio / MIN_STUDIO_HISTORY, 1.0)
```

`MIN_STUDIO_HISTORY` is the number of shows from the same studio needed for the signal to stabilize (configurable, suggested default ~3). If the user has zero history with this studio, confidence is 0 and the scorer's weight redistributes to other scorers automatically.

### Popularity

Score is the community mean score, normalized to the effective rating range. Confidence is based on how many members have the show on their list.

Member count answers "how much do we trust the community's opinion." Mean score answers "what is that opinion." This avoids popularity bias — being widely watched is not inherently rewarded, but the system trusts the quality signal more for shows with large audiences.

```
score = (mean_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
```

`MIN_SCORE` and `MAX_SCORE` define the effective range (practically ~5.0–9.0 on MAL, since almost nothing sits below 5). This stretches the useful range to 0–1.

```
confidence = min(member_count / MIN_MEMBER_COUNT, 1.0)
```

`MIN_MEMBER_COUNT` is the member count at which the community score is considered stable (configurable, suggested default ~1000). Consider applying logarithmic scaling since the difference between 500 and 5,000 members matters more than the difference between 50,000 and 55,000.

### Adaptation

Confidence is based on whether the user has a manga rating for the source material and how direct the adaptation relationship is.

```
has_rating_confidence = 1.0 if user has rated the manga, else 0.5 (reading but unrated)
directness_confidence = 1.0 for direct adaptation, lower for loose adaptations
confidence = has_rating_confidence * directness_confidence
```

If the user has no relationship with the source manga at all, the adaptation scorer should return `(0.0, 0.0)` — no signal and no confidence.

### Continuation (separate from discovery blend)

Continuation strength already incorporates confidence-like factors (predecessor rating, completion status, airing status), but it also needs an explicit confidence value for the lane composition routing threshold.

```
rating_confidence = predecessor_rating / 10.0 (normalized user rating)
completion_confidence = 1.0 if finished, 0.7 if watching, 0.3 if paused, 0.1 if dropped
airing_confidence = 1.0 if confirmed/airing, 0.7 if announced, 0.3 if speculative
confidence = rating_confidence * completion_confidence * airing_confidence
```

This confidence value determines whether the continuation routes as strong (pinned) or weak (interleaved) in lane composition.

## Discovery scoring

The blending layer computes `discovery_score` using confidence-weighted combination. For each scorer, compute `effective_weight = base_weight * confidence`. Normalize all effective weights to sum to 1.0, then compute the weighted sum of scores.

Base weights:

| Scorer | Base weight |
|---|---|
| Direct similarity | 0.30 |
| Feature correlation | 0.25 |
| Cluster similarity | 0.20 |
| Studio correlation | 0.10 |
| Popularity | 0.10 |
| Adaptation | 0.05 |

Remove the existing `apply_feature_coverage` post-processing step. Its job is now handled by individual scorers reporting low confidence when feature data is sparse.

## Continuation scoring

Continuation runs as a separate pass, independent of the discovery blend. It produces a continuation strength value per candidate based on:

- User ratings of predecessor entries.
- Completion status of predecessors (finished vs. dropped vs. paused).
- Airing status of the candidate (confirmed and airing vs. announced vs. speculative).

A candidate qualifies as a continuation if it has a predecessor in the user's watch history. The continuation strength determines routing, not ranking position within a blended score.

## Lane composition

Multiple lanes exist. Each lane receives a temporal or categorical filter that determines which candidates are eligible. The backend composes each lane as an ordered list with metadata and sends it to the frontend for presentation. The frontend renders but does not reorder or compose.

The following lanes are affected by this change. All other existing lanes remain as-is unless otherwise specified.

### Top new picks

Output only the discovery pool, ranked by `discovery_score`. No continuations appear.

### Top simulcasts and top from next season

Compose the lane by merging two independently ranked pools.

**Pool 1 — Continuations**, ranked by continuation strength. Split into:

- Strong continuations: confidence >= threshold (configurable, default 0.7).
- Weak continuations: confidence < threshold.

**Pool 2 — Discoveries**, ranked by `discovery_score`. Excludes any candidates routed to strong continuations.

**Composition order:**

1. Strong continuations pin to the top of the lane in their ranked order.
2. Discoveries fill remaining slots, with weak continuations interleaved at regular intervals (configurable, default every 5th slot).
3. Total lane output is capped at a configurable maximum (default 30).

Candidates that qualify as strong continuations do not appear in the discovery pool within the same lane. Weak continuations may appear in both pools — their interleaved position is their only appearance, they are not also scored as discoveries.

### Browse

Covers all anime from the current year. Uses the same two-pool composition as simulcasts and next season, with adjustments for scale.

Because browse spans the full year (several hundred candidates), pinning all strong continuations to the top would create an oversized block of sequels before any discovery appears. To address this:

1. Cap the pinned strong continuation block (configurable, default 5).
2. Remaining strong continuations interleave into the discovery list at a denser interval than weak continuations (configurable, default every 3rd slot for overflow strong, every 5th slot for weak).
3. Discoveries fill remaining slots, ranked by `discovery_score`.
4. No hard cap on total output — browse shows the full eligible set.

Browse is composed by the backend and sent as an ordered list, replacing the current approach of sending the full list and letting the frontend sort by `recommend_score`.

## Adaptation routing

Adaptation is not a continuation. It captures "user read the source manga." It stays in the discovery blend at base weight 0.05 with its own confidence value based on how direct the adaptation relationship is and whether the user has a manga rating. It does not participate in continuation routing.

## Observable outputs

Each candidate in the final lane output should carry:

- Its `discovery_score` (if scored for discovery).
- Its continuation strength (if scored for continuation).
- An overall confidence value (weighted average of individual scorer confidences, for discovery candidates).
- Which pool it came from (continuation vs. discovery).

This metadata supports UI features like showing match strength or explaining why a show was recommended.

## Removed components

- Genre average scorer: removed, redundant with feature correlation.
- `apply_feature_coverage` post-processing: removed, replaced by per-scorer confidence.
- `recommend_score`: renamed to `discovery_score`.