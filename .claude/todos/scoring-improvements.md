# Similarity scorer improvement requirements

## Design principle

The three feature-based scorers should operate at three distinct levels of granularity, each capturing signals the others miss:

- **Feature correlation** operates at the profile level — aggregate taste patterns across the full watch history.
- **Direct similarity** operates at the item level — specific structural resemblance to individual watched shows.
- **Cluster similarity** operates at the neighborhood level — broad affinity with regions of taste space.

Both similarity scorers currently collapse to "find the single best match" which makes them behave too similarly to each other and undermines the granularity distinction. The improvements below restore their intended complementary roles.

## Direct similarity scorer

### Current problems

**Single best match.** The scorer uses argmax to find the one most similar watched item and bases the entire score on that pairing. A candidate connected to many watched items at moderate similarity scores lower than a candidate with one strong match. This makes the scorer brittle and throws away breadth-of-match information.

**Score multiplication overweights user rating.** `similarity * user_rating` gives user rating equal influence to structural similarity. A 0.9 similarity to a 6-rated show barely beats a 0.5 similarity to a 10-rated show. Similarity should be the primary signal with rating as a modifier, not an equal partner.

**No negative signal from dropped shows.** A candidate highly similar to a dropped show contributes a positive signal (or neutral if unrated, since nulls are filled with the mean). High similarity to a dropped show should be repulsive.

**NaN filling masks missing data.** Candidates with zero feature overlap against all watched items get NaN, which is filled with the column mean. This fabricates a moderate similarity score where there is genuinely no signal. These candidates should get a low score with low confidence.

### Requirements

**Aggregate top-K matches instead of argmax.** For each candidate, find the K most similar items in the user's watch history (suggested default K=5). Weight each by similarity with decay for lower-ranked matches, e.g.:

```
match_scores = [sim_1 * rating_mod_1, sim_2 * rating_mod_2, ...]
weights = [1.0, 0.6, 0.36, 0.22, 0.13]  # geometric decay
score = weighted_average(match_scores, weights)
```

This captures "broadly similar to things you like" rather than "very similar to one specific thing." The decay ensures the best match still dominates but doesn't monopolize.

**Reduce user rating influence.** Replace direct multiplication with a bounded modifier:

```
rating_modifier = 0.5 + 0.5 * (normalized_rating)
match_score = similarity * rating_modifier
```

This keeps similarity as the dominant factor. A 0-rated show halves the similarity contribution; a 10-rated show gives full similarity. The range of the modifier is 0.5–1.0 rather than 0.0–1.0, preventing highly similar shows from being zeroed out by low ratings.

**Incorporate negative signal from dropped shows.** When a top-K match is a dropped or paused show, invert its contribution:

```
if status in [DROPPED, PAUSED]:
    match_score = -similarity * drop_penalty  # e.g. drop_penalty = 0.5
```

The negative contribution reduces the aggregated score. A candidate that's similar to both loved and dropped shows ends up in contested territory rather than scoring high. The drop_penalty being less than 1.0 ensures drops don't completely dominate — one dropped similar show shouldn't cancel out three loved similar shows.

**Handle zero-overlap candidates honestly.** Candidates with no feature overlap against any watched item should receive `(0.0, 0.0)` — zero score, zero confidence. Do not fill NaNs with the mean. The confidence-weighted blending will redistribute this scorer's weight to other scorers automatically.

**Include match quality in confidence.** Confidence should combine feature sparsity with the strength of the best matches found:

```
feature_conf = min(candidate_feature_count / MIN_FEATURE_THRESHOLD, 1.0)
match_conf = min(best_similarity / MIN_SIMILARITY_THRESHOLD, 1.0)
confidence = feature_conf * match_conf
```

A candidate with plenty of features but no good matches in the watch history produces low confidence, correctly signaling that the scorer doesn't have much to say about this candidate.

## Cluster similarity scorer

### Current problems

**Single best cluster.** Same structural issue as direct similarity — the scorer finds the best-matching cluster via `.max()` and ignores all others. A candidate that moderately fits three clusters the user enjoys scores lower than one that strongly fits one cluster. This defeats the purpose of cluster similarity as an exploration signal.

**Score multiplication conflates three things.** The formula `mean_similarity * mean_score * sqrt(cluster_size)` mixes structural similarity, user enjoyment, and cluster scale in a way that's hard to interpret or tune. The sqrt of cluster size is meant to boost larger clusters, but it also means a large cluster of 6-rated shows can outscore a small cluster of 9-rated shows purely on size.

**No distinction between cluster quality.** A tight, well-defined cluster and a loose catch-all cluster contribute equally. The scorer should trust tight clusters more than diffuse ones, and this should feed into confidence.

### Requirements

**Aggregate across top-N clusters instead of max.** For each candidate, compute similarity to the top N clusters from the user's watch history (suggested default N=3). Weight with decay:

```
cluster_scores = [cluster_sim_1, cluster_sim_2, cluster_sim_3]
weights = [1.0, 0.5, 0.25]
score = weighted_average(cluster_scores, weights)
```

This captures "broadly fits your taste landscape" rather than "fits one specific pocket." The exploration value of cluster similarity comes from this breadth — a candidate that moderately overlaps with multiple enjoyed clusters is a better discovery candidate than one that strongly overlaps with just one.

**Separate similarity from user enjoyment.** Compute cluster affinity as two distinct factors:

```
cluster_similarity = mean similarity of candidate to cluster members
cluster_enjoyment = mean user rating of cluster members (normalized)
cluster_score = cluster_similarity * (0.5 + 0.5 * cluster_enjoyment)
```

Same bounded modifier pattern as direct similarity. This prevents the user rating from dominating — cluster similarity is about neighborhood membership, not about whether the user loved every specific show in the neighborhood.

**Remove cluster size boost.** The sqrt(cluster_size) factor rewards large clusters at the expense of small ones. A small cluster of 3 closely related shows the user all rated highly is a stronger signal than a large cluster of 15 loosely related shows with mixed ratings. If large clusters should be trusted more, express that through confidence, not score.

**Use cluster cohesion in confidence.** Confidence should reflect how well-defined the cluster is:

```
cluster_conf = mean intra-cluster similarity (or silhouette score)
feature_conf = min(candidate_feature_count / MIN_FEATURE_THRESHOLD, 1.0)
confidence = cluster_conf * feature_conf
```

A tight cluster produces high confidence — "this neighborhood is well-defined and the candidate either belongs in it or doesn't." A loose catch-all cluster produces low confidence — "this neighborhood is vague, so membership doesn't mean much."

## How the three scorers complement each other after changes

**Feature correlation** finds shows matching aggregate taste patterns. It's strong when the user has a long watch history and the candidate has rich metadata. It's weak for new users or sparse candidates. It doesn't care about individual shows — it sees patterns.

**Direct similarity (with top-K)** finds shows that closely resemble specific watched items. It's strong when the candidate has a clear structural analog in the watch history. It's weak when the candidate is novel — different from anything watched. It captures "more of the same" precisely.

**Cluster similarity (with top-N)** finds shows in the same taste neighborhoods. It's strong when the candidate broadly fits regions the user has explored, even without a single strong match. It's weak when the user's clusters are loose or the candidate is a true outlier. It captures "same vibe, different show."

The key behavior: a candidate that all three agree on is a very safe recommendation. A candidate that cluster similarity likes but direct similarity doesn't is a stretch recommendation — same neighborhood but nothing quite like it. A candidate that direct similarity likes but cluster similarity doesn't is a niche match — similar to one specific show but outside the user's usual territory. These distinctions produce meaningful variety in the recommendation output.