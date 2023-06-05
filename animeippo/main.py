from profilehooks import profile


@profile(filename=".profiling/cprofile.pstats")
def get_recs():
    year = "2023"
    season = "spring"
    user = "Janiskeisari"

    recommender = builder.create_builder(os.environ.get("DEFAULT_PROVIDER")).build()
    recommendations = recommender.recommend_seasonal_anime(year, season, user)
    return recommendations


if __name__ == "__main__":
    import dotenv
    import os

    from animeippo.view import views
    from animeippo.recommendation import builder

    dotenv.load_dotenv("conf/prod.env")

    recommendations = get_recs()

    views.console_view(recommendations)
