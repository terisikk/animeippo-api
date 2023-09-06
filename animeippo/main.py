# @profile(filename=".profiling/cprofile.pstats")
def get_recs():
    year = "2024"
    season = None
    user = "Janiskeisari"

    recommender = recommender_builder.create_builder(os.environ.get("DEFAULT_PROVIDER")).build()
    dataset = recommender.recommend_seasonal_anime(year, season, user)

    return dataset.recommendations


if __name__ == "__main__":
    import dotenv
    import os

    from animeippo.view import views
    from animeippo.recommendation import recommender_builder

    dotenv.load_dotenv("conf/prod.env")

    recommendations = get_recs()

    views.console_view(recommendations)
