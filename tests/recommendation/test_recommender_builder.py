from animeippo import providers
from animeippo.recommendation import recommender_builder


class CacheStub:
    def is_available(self):
        return self.is_available


def test_builder_creation_returns_correct_builders():
    assert (
        recommender_builder.build_recommender("anilist").provider.__class__
        == providers.anilist.AniListProvider
    )
    assert (
        recommender_builder.build_recommender("myanimelist").provider.__class__
        == providers.myanimelist.MyAnimeListProvider
    )

    assert (
        recommender_builder.build_recommender("faulty-provider").provider.__class__
        == providers.mixed.MixedProvider
    )


def test_builder_passes_with_or_without_cache(mocker):
    assert (
        recommender_builder.build_recommender("anilist").provider.__class__
        == providers.anilist.AniListProvider
    )

    mocker.patch("animeippo.cache.RedisCache", CacheStub)

    assert (
        recommender_builder.build_recommender("anilist").provider.__class__
        == providers.anilist.AniListProvider
    )
