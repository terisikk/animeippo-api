# Collaborative recommendation scorer requirements

## Overview

A new scorer that uses community recommendation links (user-submitted "if you liked X, try Y" data with positive/negative thumbs) as a collaborative filtering signal. This is the only scorer in the discovery blend that captures taste relationships the feature space cannot describe.

## Data source

Each show has a list of community-recommended related shows, each with a thumb count (positive or negative). For example, show A might have recommendations: B(10), C(8), D(3), E(-1). The thumbs represent community agreement on the relationship — positive means "fans of A tend to enjoy this," negative means "despite apparent similarity, these don't actually go together."

Anilist graphql docs: https://docs.anilist.co/guide/graphql/

Possible query structure that can be incorporated to the existing queries:
  Media {
    recommendations {
      edges {
        node {
          media {
            id
          }
          rating
        }
      }
    }
  }


## Signal computation

For each seasonal candidate, find all shows in the user's watch history that have a community recommendation link pointing to the candidate. Each link contributes a signal based on three factors:

### Thumb normalization

Normalize thumbs relative to the source show's total recommendation thumb count. This prevents popular shows with thousands of thumbs from dominating over niche shows with single-digit thumbs.

```
normalized_thumbs = link.thumbs / source_show.total_recommendation_thumbs
```

### User rating modifier

The user's rating of the source show modulates how much to trust the community link. A link from a 9/10 show means the community is comparing against an experience the user strongly connected with. A link from a 5/10 show means the comparison is based on an experience the user didn't share.

```
rating_modifier = 0.5 + 0.5 * (user_rating / 10.0)
```

Unrated but completed shows fall back to the user's mean score.

### Watch status modifier

The user's relationship with the source show gates and modulates the signal.

| Status | Modifier | Rationale |
|---|---|---|
| Completed | 1.0 | Full experience, community comparison fully applies |
| Current | 0.8 | Partial experience, comparison mostly applies |
| Paused | 0.3 | Ambiguous — user may return or may have lost interest, weak positive signal |
| Dropped | -0.5 | User rejected the source, community link becomes weak evidence against |
| Not watched | excluded | Link is irrelevant if the user hasn't seen the source |

### Per-link signal

```
if status == DROPPED:
    link_signal = normalized_thumbs * completion_modifier
else:
    link_signal = normalized_thumbs * rating_modifier * completion_modifier
```

For dropped shows, the rating modifier is not applied because the drop itself is the primary signal — whatever rating was given (if any) is secondary to the decision to abandon the show.

### Aggregation

Sum all link signals for each candidate. The sum rather than mean is intentional — a candidate linked to many watched shows has more evidence supporting it than one linked to a single show, even if that single link is strong.

## Negative thumbs

Negative community thumbs are preserved and contribute negatively. A link with -5 thumbs from a completed show the user rated 8/10 produces a negative signal, meaning "the community actively thinks fans of that show would not enjoy this candidate." This is valuable corrective information that no content-based scorer can provide.

## Confidence

Confidence is based on link count — thumb strength already modulates the signal itself via normalized thumbs, so including it in confidence would double-penalize and suppress the scorer's contribution to the discovery blend.

```
confidence = min(link_count / MIN_LINK_COUNT, 1.0)
```

`MIN_LINK_COUNT` is the number of links needed for full confidence (configurable, default 3).

A candidate with no links to the user's watch history returns `(0.0, 0.0)` — no signal, no confidence. The weight redistributes to other scorers automatically.

## Edge cases

**New seasonal shows with no recommendation data.** These are common for upcoming and recently airing shows. The scorer returns `(0.0, 0.0)` and contributes nothing. This is correct — the collaborative signal doesn't exist yet and shouldn't be fabricated.

**Source shows with very few total recommendations.** A source show with only 3 total recommendations makes each link's normalized thumbs very large relative to source shows with hundreds of recommendations. This is acceptable — a niche show's recommendations are typically higher signal because the recommending community is more invested. If this causes problems in practice, consider flooring total_recommendation_thumbs at a minimum value.

**Circular recommendations.** A candidate could have links from multiple source shows that all point to each other. This doesn't cause problems because the scorer only looks at links pointing to seasonal candidates from watched shows — it doesn't traverse the recommendation graph.

## Position in the discovery blend

This scorer is a genuinely independent data source — community taste relationships rather than content features. Suggested base weight: 0.15, redistributed from direct similarity (0.25, down from 0.30) and cluster similarity (0.15, down from 0.20) which are the most redundant pair.

Revised discovery blend weights:

| Scorer | Base weight |
|---|---|
| Direct similarity | 0.25 |
| Feature correlation | 0.25 |
| Cluster similarity | 0.15 |
| Collaborative recommendation | 0.15 |
| Studio correlation | 0.10 |
| Popularity | 0.05 |
| Adaptation | 0.05 |

Popularity is reduced to 0.05 because the collaborative scorer partially subsumes its role — community recommendations are themselves a popularity-adjacent signal, but a more targeted one.

## Score normalization

Apply rank normalization before entering the discovery blend, consistent with other scorers. Raw scores are not on a natural 0–1 scale since they are sums of variable numbers of link signals.