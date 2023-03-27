import IPython.display as idisplay
import numpy as np


def pandas_display_all_clusters(dataframe, descriptions):
    for cluster in np.sort(dataframe["cluster"].unique()):
        idisplay.display(descriptions.iloc[cluster][0].tolist())
        idisplay.display(filter_by_cluster(dataframe, cluster))


def filter_by_cluster(dataframe, clusterno):
    return dataframe[dataframe["cluster"] == clusterno]
