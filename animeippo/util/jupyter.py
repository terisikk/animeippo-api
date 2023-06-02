import IPython.display as idisplay
import numpy as np
import pandas as pd

from ..recommendation import util as pdutil


def pandas_display_all_clusters(dataframe):
    gdf = dataframe.explode("features")

    descriptions = pdutil.extract_features(gdf["features"], gdf["cluster"], 2)

    with pd.option_context("display.max_rows", None):
        for cluster in np.sort(dataframe["cluster"].unique()):
            idisplay.display(cluster)
            idisplay.display(descriptions.iloc[cluster].tolist())
            idisplay.display(filter_by_cluster(dataframe, cluster))


def filter_by_cluster(dataframe, clusterno):
    return dataframe[dataframe["cluster"] == clusterno]
