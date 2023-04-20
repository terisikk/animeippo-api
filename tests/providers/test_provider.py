import pytest

from animeippo.providers import provider


def test_new_provider_can_be_instantiated():
    class ConcreteAnimeProvider(provider.AbstractAnimeProvider):
        def __init__(self):
            pass

        def get_user_anime_list(self, user_id):
            super().get_user_anime_list(user_id)

        def get_seasonal_anime_list(self, year, season):
            super().get_seasonal_anime_list(year, season)

        def transform_to_animeippo_format(self, anime_list):
            super().transform_to_animeippo_format(anime_list)

        def get_genre_tags(self):
            super().get_genre_tags()

        def get_related_anime(self, id):
            super().get_related_anime(id)

    actual = ConcreteAnimeProvider()
    actual.get_user_anime_list(None)
    actual.get_seasonal_anime_list(None, None)
    actual.transform_to_animeippo_format(None)
    actual.get_genre_tags()
    actual.get_related_anime(None)

    assert issubclass(actual.__class__, provider.AbstractAnimeProvider)


def test_new_provider_subclassing_fails_with_missing_methods():
    with pytest.raises(TypeError):

        class ConcreteAnimeProvider(provider.AbstractAnimeProvider):
            def __init__(self):
                pass

        ConcreteAnimeProvider()
