import polars as pl
import scipy.spatial.distance as scdistance


def distance(x_orig, y_orig, metric="cosine"):
    """
    Calculate pairwise distance between two series.
    Just a wrapper for scipy cdist for a matching
    signature with analysis.similairty."""
    return scdistance.cdist(x_orig, y_orig, metric=metric)


def similarity(x_orig, y_orig, metric="cosine"):
    """Calculate similarity between two series."""
    distances = distance(x_orig, y_orig, metric=metric)
    return 1 - distances  # This is incorrect for distances that are not 0-1


def categorical_similarity(features1, features2, metric="cosine", columns=None):
    """Calculate similarity between two series of categorical features. Assumes a series
    that contains vector-encoded representation of features."""
    similarities = pl.DataFrame(
        similarity(
            features1.struct.unnest().fill_null(0).to_numpy(),
            features2.struct.unnest().fill_null(0).to_numpy(),
            metric=metric,
        )
    )

    if columns is not None:
        similarities.columns = columns

    return similarities
