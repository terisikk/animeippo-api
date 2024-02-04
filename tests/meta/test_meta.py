import datetime
from collections import namedtuple

from animeippo.meta import meta


def test_simulcastscategory_current_season_getter(monkeypatch):
    class FakeDateWinter:
        @classmethod
        def today(cls):
            return datetime.datetime(2022, 2, 2)

    class FakeDateSpring:
        @classmethod
        def today(cls):
            return datetime.datetime(2022, 5, 2)

    class FakeDateSummer:
        @classmethod
        def today(cls):
            return datetime.datetime(2022, 9, 2)

    class FakeDateFall:
        @classmethod
        def today(cls):
            return datetime.datetime(2022, 12, 2)

    class FakeDateMalformed:
        @classmethod
        def today(cls):
            FakeToday = namedtuple("fakedate", ["year", "month"])
            return FakeToday(2022, 15)

    monkeypatch.setattr(datetime, "date", FakeDateWinter)
    assert meta.get_current_anime_season() == (2022, "winter")

    monkeypatch.setattr(datetime, "date", FakeDateSpring)
    assert meta.get_current_anime_season() == (2022, "spring")

    monkeypatch.setattr(datetime, "date", FakeDateSummer)
    assert meta.get_current_anime_season() == (2022, "summer")

    monkeypatch.setattr(datetime, "date", FakeDateFall)
    assert meta.get_current_anime_season() == (2022, "fall")

    # Probably not possible but for coverage
    monkeypatch.setattr(datetime, "date", FakeDateMalformed)
    assert meta.get_current_anime_season() == (2022, "?")
