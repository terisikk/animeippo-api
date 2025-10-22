[DataLayer]
  ├─ UserProfile
  ├─ Catalogue (seasonal, metadata)
  └─ Similarities/Embeddings
        │
        ▼
[Scoring]
  ├─ DirectSimilarityScorer
  ├─ FeatureCorrelationScorer
  ├─ ClusterSimilarityScorer
  ├─ ContinuationScorer
  ├─ AdaptationScorer
  ├─ PopularityScorer
  ├─ GenreAverageScorer
  ├─ StudioCorrelationScorer
  ├─ DirectorCorrelationScorer
  └─ FormatPenaltyScorer (penalty ∈[0,1])
        │  (each returns normalised Series; no cross-calls)
        ▼
[Engine]
  ├─ pulls scorer outputs
  ├─ applies weights (incl. negative for format penalty)
  └─ emits immutable raw_score per item (and parts for explain)
        │
        ├─────────────────────────────────────────┐
        │                                         │
        ▼                                         ▼
[CategoryBuilder]                             [CategoryBuilder]  ... (one per category)
  ├─ pulls raw_score + metadata
  ├─ filters to its candidate pool
  └─ sorts by raw_score (desc)
        │
        ▼
[LocalPostProcessor]  (profile per category)
  ├─ HardFilterAdjuster      (CM/AD, movie guard)
  ├─ FormatDampener          (α per category)
  ├─ GenreSaturation         (soft caps/penalties within category)
  ├─ MMR/Diversity           (μ per category)
  ├─ Exploration/Novelty     (tiny ε)
  └─ SeenBefore (in-page)    (light touch)
        │
        ▼
CategoryResult (name, local_adjusted list)
        │
        ▼
[RankingOrchestrator]   (global, stateful, minimal)
  ├─ state: exposed_ids, genre/format tallies, logs
  ├─ for categories in order:
  │    ├─ take local_adjusted
  │    ├─ drop exposed_ids (dedupe)
  │    ├─ apply soft page budgets (tiny penalties; hero rows only)
  │    └─ pick top-K → update state
  └─ emits PageLayout {category_name: picks}
        │
        ▼
[Renderer / API]
  ├─ displays lists
  └─ attaches explanations (from parts + post-process reasons)
