from profilehooks import profile as pr


async def get_recs_async():
    year = "2025"
    season = None
    user = "Janiskeisari"

    async with recommender_builder.build_recommender("anilist") as recommender:
        dataset = await recommender.databuilder(year, season, user)

        if user:
            dataset.recommendations = recommender.engine.fit_predict(dataset)
        else:
            dataset.recommendations = dataset.seasonal.sort("popularity", descending=True)

        recommender.get_categories(dataset)
        return dataset.recommendations


@pr(filename=".profiling/cprofile.pstats")
def get_recs():
    import asyncio

    return asyncio.run(get_recs_async())


if __name__ == "__main__":
    import dotenv

    from animeippo.recommendation import recommender_builder
    from animeippo.view import views

    dotenv.load_dotenv("conf/prod.env")

    recommendations = get_recs()

    views.console_view(recommendations)
