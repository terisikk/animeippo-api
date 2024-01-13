import numpy as np
import polars as pl
import scipy.spatial.distance as scdistance


def distance(x_orig, y_orig, metric="jaccard"):
    """
    Calculate pairwise distance between two series.
    Just a wrapper for scipy cdist for a matching
    signature with analysis.similairty."""
    return scdistance.cdist(x_orig, y_orig, metric=metric)


def similarity(x_orig, y_orig, metric="jaccard"):
    """Calculate similarity between two series."""
    distances = distance(x_orig, y_orig, metric=metric)
    return 1 - distances  # This is incorrect for distances that are not 0-1


def categorical_similarity(features1, features2, metric="jaccard", columns=None):
    """Calculate similarity between two series of categorical features. Assumes a series
    that contains vector-encoded representation of features."""
    similarities = pl.DataFrame(
        similarity(
            # Polars seems to have a bug where to_numpy gets a cached value
            # instead of the actual conversion, thus np.array(x.to_list()),
            # not to_numpy().
            np.stack(features1.to_list()),
            np.stack(features2.to_list()),
            metric=metric,
        )
    )

    if columns is not None:
        similarities.columns = columns

    return similarities


def similarity_of_anime_lists(features1, features2, metric="jaccard"):
    similarities = categorical_similarity(features1, features2, metric=metric)

    return similarities.mean_horizontal(ignore_nulls=True)