import IPython.display as idisplay
import numpy as np
from . import pandas as pdutil


def pandas_display_all_clusters(dataframe):
    descriptions = pdutil.extract_features(dataframe.explode("genres"), dataframe["cluster"])

    for cluster in np.sort(dataframe["cluster"].unique()):
        idisplay.display(descriptions.iloc[cluster].tolist())
        idisplay.display(filter_by_cluster(dataframe, cluster))


def filter_by_cluster(dataframe, clusterno):
    return dataframe[dataframe["cluster"] == clusterno]
