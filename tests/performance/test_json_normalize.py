"""Performance comparison: fast_json_normalize vs pl.json_normalize.

As of Polars 1.39, fast_json_normalize (via pandas) is still significantly faster
than Polars' native json_normalize for nested AniList API data. This test will fail
if Polars closes the gap, signaling that we can drop the pandas dependency for this.

See: src/animeippo/providers/anilist/formatter.py
"""

import time

import polars as pl
from fast_json_normalize import fast_json_normalize

from tests import test_data

# Use the raw AniList media list format — multiply to get a realistic size
SAMPLE_DATA = test_data.ANI_SEASONAL_LIST["data"]["Page"]["media"]
# Repeat to simulate a full season (~200 items)
LARGE_SAMPLE = SAMPLE_DATA * 100
ITERATIONS = 20


def _benchmark(func, iterations=ITERATIONS):
    # Warmup
    func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return min(times)


def test_fast_json_normalize_faster_than_polars_native():
    """fast_json_normalize should be faster than pl.json_normalize.

    If this test fails, Polars has improved enough that we can switch to
    pl.json_normalize and remove the fast_json_normalize + pandas dependency.
    """

    def via_fast_json():
        return pl.from_pandas(fast_json_normalize(LARGE_SAMPLE))

    def via_polars():
        return pl.json_normalize(LARGE_SAMPLE)

    fast_time = _benchmark(via_fast_json)
    polars_time = _benchmark(via_polars)

    # Both should produce the same number of rows
    fast_result = via_fast_json()
    polars_result = via_polars()
    assert len(fast_result) == len(polars_result)

    assert fast_time < polars_time, (
        f"pl.json_normalize ({polars_time:.3f}s) is now faster than "
        f"fast_json_normalize ({fast_time:.3f}s). "
        "Consider switching to native Polars and removing the pandas dependency."
    )
