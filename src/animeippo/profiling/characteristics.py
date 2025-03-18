import polars as pl


class Characteristics:
    def __init__(self, watchlist, all_genres):
        self.genre_variance = watchlist_genre_variance(watchlist, all_genres)


def watchlist_genre_variance(watchlist, all_genres):
    user_unique_genres = (
        watchlist.filter(pl.col("user_status").is_in(["completed", "watching"]))
        .explode("genres")
        .unique("genres")["genres"]
        .to_list()
    )

    # NSFW genres are not taken into account
    return len(set(user_unique_genres).union({"Hentai"})) / len(all_genres) * 100.0
