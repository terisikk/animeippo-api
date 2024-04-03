from profilehooks import profile as pr


@pr(filename=".profiling/cprofile.pstats")
def get_recs():
    year = "2019"
    season = None
    user = "Janiskeisari"

    recommender = recommender_builder.build_recommender("anilist")
    # recommender = recommender_builder.create_builder(os.environ.get("DEFAULT_PROVIDER"))
    dataset = recommender.recommend_seasonal_anime(year, season, user)

    recommender.get_categories(dataset)

    return dataset.recommendations


if __name__ == "__main__":
    import dotenv

    from animeippo.recommendation import recommender_builder
    from animeippo.view import views

    dotenv.load_dotenv("conf/prod.env")

    recommendations = get_recs()

    views.console_view(recommendations)
