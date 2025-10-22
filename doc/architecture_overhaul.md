Here’s a clean, layered plan that separates **signals (scorers)** from **policy (post-processing)** and gives each surface (category) its own knobs.

# 1) Goals

* Reduce single genre oversaturation without wrecking relevance.
* Keep scorers stable, policy tunable per surface.

# 2) Naming

* **`raw_score`**: linear blend of scorers (normalised per-scorer).
* **`adjusted_score`**: `raw_score` after policy/post-processing.
* **`ScorePostProcessor`**: wrapper that applies adjusters (per-surface profile).

# 3) Scorer layer (signals only)

**Normalisation**

* Rank-normalise to [0,1] all scorers **except**:

  * Continuation/Adaptation: fixed [0,1], no global rank.
  * Format: returns **penalty** in [0,1] (no rank).

**Weights (start)**

* `DirectSimilarity` **0.25**
* `FeatureCorrelation` **0.15** (with debias + mean baseline + empty-feature floor)
* `ClusterSimilarity` **0.15**
* `Continuation` **0.15**
* `Adaptation` **0.10**
* `Popularity` **0.10**
* `GenreAverage (residualised or low)** **0.05**
* `FormatPenalty` **−0.12** (keep light here; move heavy control to policy)
* `StudioCorrelation` **0.02**
* `DirectorCorrelation` **0.02**

**Orthogonality guard (optional, automatic)**

* Compute pairwise ρ (Spearman) over season; if ρ>0.8, down-weight the weaker (by AUC/NDCG) by 50%.
* Apply once per run; log the decision.

# 4) Post-processing (policy) — the adjuster pipeline

Implemented as a list of deterministic transforms:

1. **HardFilterAdjuster**

   * Drop `format ∈ {CM, AD}` (unless explicitly requested surface).
   * Optional: drop items with `episodes<2` and `duration<10` unless strong continuation/adaptation (≥0.75).

2. **FormatPolicyAdjuster** *(multiplicative dampener)*

   * Factor ∈ [0.5,1.0] based on shortness; exponent α per surface.
   * Exempt movies (`format==MOVIE` or `episodes==1 & duration≥50`).

3. **GenreSaturationAdjuster**

   * Soft penalty by current top-K composition.
   * `adjusted = score − λ * freq(primary_genre)` within the candidate set as you build the list.
   * Or per-genre caps (hard): e.g. ≤30% Adventure+Fantasy in top-20.

4. **Diversity/MMRAdjuster**

   * Greedy selection: `score' = score − μ * max_sim(selected)` where sim = Jaccard over genres/tags (or feature cosine).
   * μ per surface.

5. **SeenBeforeAdjuster** *(your existing discourager)*

   * Reduce scores for items already shown in higher-ranked categories/pages in the current session.

6. **ExplorationAdjuster** *(very small)*

   * Add a tiny random/novelty bonus (e.g. +0.01–0.02) to break ties and surface long-tail.

Output → **`adjusted_score`**.

# 5) Surface profiles (per category)

Define profiles with adjuster parameters; same core pipeline, different knobs:

* **YourTopPicks**

  * HardFilter: on
  * FormatPolicy: α=1.8
  * GenreSaturation: λ=0.25, cap Adventure+Fantasy ≤30% of top-20
  * MMR: μ=0.25
  * Exploration: 0.01

* **Simulcasts**

  * HardFilter: keep (ads off)
  * FormatPolicy: α=1.2
  * GenreSaturation: λ=0.10 (lighter)
  * MMR: μ=0.15
  * Exploration: 0.01

* **MostPopular**

  * HardFilter: on
  * FormatPolicy: α=1.0
  * GenreSaturation: off
  * MMR: μ=0.1
  * Exploration: 0.0

* **BecauseYouLiked (n)**

  * HardFilter: on
  * FormatPolicy: α=1.5
  * GenreSaturation: λ=0.2
  * MMR: μ=0.2
  * Exploration: 0.015

* **Genre buckets (0..6)**

  * HardFilter: on
  * FormatPolicy: α=1.2
  * GenreSaturation: local cap per bucket (e.g. within bucket, ensure sub-genres rotate)
  * MMR: μ=0.15
  * Exploration: 0.01

# 6) Category list (minimal edits)

Keep your category order; wrap those that need diversity:

```
MostPopular,
Simulcasts,
ContinueWatching,
YourTopPicks (profile: TopPicks),
TopUpcoming,
DiversityAdjuster(GenreCategory(0), profile: Genre),
Adaptation,
DiversityAdjuster(GenreCategory(1), profile: Genre),
Planning,
DiversityAdjuster(GenreCategory(2), profile: Genre),
Source,
DiversityAdjuster(GenreCategory(3), profile: Genre),
Studio,
DiversityAdjuster(GenreCategory(4), profile: Genre),
BecauseYouLiked(0, jaccard) -> profile: BYL,
DiversityAdjuster(GenreCategory(5), profile: Genre),
BecauseYouLiked(1, jaccard) -> profile: BYL,
DiversityAdjuster(GenreCategory(6), profile: Genre),
BecauseYouLiked(2, jaccard) -> profile: BYL
```

# 7) Config shape (suggestion)

YAML/JSON so you can tune without code:

```yaml
scorers:
  direct:         { weight: 0.25, norm: rank }
  featurecorr:    { weight: 0.15, norm: rank, beta: 0.7, lambda: 0.25, gamma: 0.5 }
  cluster:        { weight: 0.15, norm: rank }
  continuation:   { weight: 0.15, norm: fixed01 }
  adaptation:     { weight: 0.10, norm: fixed01 }
  popularity:     { weight: 0.10, norm: rank }
  genreavg:       { weight: 0.05, norm: rank, residual: true }
  format_penalty: { weight: -0.12, norm: none }   # penalty 0..1
  studio:         { weight: 0.02, norm: rank }
  director:       { weight: 0.02, norm: rank }

post_profiles:
  TopPicks:    { hard_filter: true,  format_alpha: 1.8, genre_lambda: 0.25, genre_cap: 0.30, mmr_mu: 0.25, explore: 0.01 }
  Simulcasts:  { hard_filter: true,  format_alpha: 1.2, genre_lambda: 0.10, genre_cap: null, mmr_mu: 0.15, explore: 0.01 }
  MostPopular: { hard_filter: true,  format_alpha: 1.0, genre_lambda: 0.00, genre_cap: null, mmr_mu: 0.10, explore: 0.00 }
  BYL:         { hard_filter: true,  format_alpha: 1.5, genre_lambda: 0.20, genre_cap: 0.40, mmr_mu: 0.20, explore: 0.015 }
  Genre:       { hard_filter: true,  format_alpha: 1.2, genre_lambda: 0.15, genre_cap: 0.50, mmr_mu: 0.15, explore: 0.01 }
```

# 8) Evaluation & guardrails

* **Per-surface metrics**: NDCG@K, Genre entropy@K, Novelty (avg popularity rank), Coverage.
* **Redundancy**: Top-N overlap between scorers; auto down-weight if ρ>0.8.
* **Saturation checks**: assert genre caps hold in TopPicks.
* **Explainers**: attach rationale snippets (e.g., “demoted due to short format”, “boosted for novelty”).

# 9) Migration steps

1. Keep current scorers; **lighten Format weight to −0.12**.
2. Introduce `ScorePostProcessor` with profiles above; apply to each category.
3. Switch `FeatureCorrelation` to debias + mean-baseline (already done).
4. Add orthogonality guard.
5. Tune per-surface knobs (small grid) and lock defaults.

This gives you a robust core signal, plus surface-level control to **avoid isekai floods** and **sink miniseries** without entangling core maths.


