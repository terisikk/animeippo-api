import asyncio
import datetime
from concurrent.futures import ThreadPoolExecutor


def run_coroutine(coro):
    """Run an async coroutine, handling both running and non-running event loops.

    When called from within a running event loop (e.g. Jupyter, Flask),
    uses a thread pool to avoid 'loop already running' errors.
    """
    try:
        asyncio.get_running_loop()
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()
    except RuntimeError:
        return asyncio.run(coro)


def get_current_anime_season():
    today = datetime.date.today()

    season = ""

    if today.month in [1, 2, 3]:
        season = "WINTER"
    elif today.month in [4, 5, 6]:
        season = "SPRING"
    elif today.month in [7, 8, 9]:
        season = "SUMMER"
    elif today.month in [10, 11, 12]:
        season = "FALL"
    else:
        season = "?"

    return today.year, season
