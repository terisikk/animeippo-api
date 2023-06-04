import numpy as np
import sklearn.preprocessing as skpre


class CategoricalEncoder:
    def __init__(self, classes):
        self.classes = classes
        self.mlb = skpre.MultiLabelBinarizer(classes=classes)
        self.mlb.fit(None)

    def encode(self, series, dtype=bool):
        return np.array(self.mlb.transform(series), dtype=dtype)
