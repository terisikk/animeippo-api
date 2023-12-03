from animeippo.providers.formatters import mixed_formatter


def test_get_adaptation():
    assert mixed_formatter.get_adaptation(
        [{"relationType": "ADAPTATION", "node": {"idMal": [31]}}]
    ) == [[31]]
