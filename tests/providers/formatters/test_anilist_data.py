import animeippo.providers.anilist_data as data


def test_data_is_unique():
    assert len(data.ALL_FEATURES) == len(set(data.ALL_FEATURES))