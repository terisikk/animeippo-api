---
name: verify
description: Fetch real recommendations for a user and inspect results against live AniList data. Use when debugging scoring, categories, or data pipeline issues.
---

# Verify against real data

Write and run an inline Python script to fetch recommendations with cache and inspect results. Requires redis running and `conf/prod.env` with AniList credentials.

## Base script

```python
import asyncio
import dotenv

dotenv.load_dotenv("conf/prod.env")

from animeippo.recommendation import recommender_builder

async def main():
    async with recommender_builder.build_recommender("anilist") as recommender:
        dataset = await recommender.databuilder("2025", None, "Janiskeisari")
        dataset.recommendations = recommender.engine.fit_predict(dataset)
        categories = recommender.get_categories(dataset)

        recs = dataset.recommendations
        wl = dataset.watchlist

        # --- inspect here ---

        return dataset

dataset = asyncio.run(main())
```

## Adapt based on what the user wants to verify

- Change year/season/username as needed (season `None` = full year)
- To inspect scores: `recs.select("title", "discovery_score", "collaborativescore", "cluster").head(20)`
- To inspect a category:
  ```python
  from animeippo.recommendation import categories as cats
  mask, sorting = cats.YourTopPicksCategory().categorize(dataset)
  recs.filter(mask).sort(**sorting).select("title", "discovery_score").head(10)
  ```
- To inspect watchlist clusters: `wl.select("title", "cluster", "score").sort("cluster")`
- To check the similarity matrix: `dataset.get_similarity_matrix(filtered=True)`
- Always use redis cache unless specifically testing provider network code
- In Jupyter notebooks, `await` works directly at top level instead of `asyncio.run()`
