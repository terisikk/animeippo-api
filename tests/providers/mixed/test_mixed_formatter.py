from animeippo.providers.mixed import formatter


def test_get_adaptation():
    assert formatter.get_adaptation([{"relationType": "ADAPTATION", "node": {"idMal": [31]}}]) == [
        [31]
    ]
