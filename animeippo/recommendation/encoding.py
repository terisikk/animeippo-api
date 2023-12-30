import sklearn.preprocessing as skpre
import numpy as np
import ast


class CategoricalEncoder:
    """Encodes categorical data (e.g. genres or similar) to a vector representation
    of whether a dataframe row has a category from all possible categories or not.
    """

    def fit(self, classes, class_field="features"):
        self.classes = classes
        self.class_field = class_field
        self.mlb = skpre.MultiLabelBinarizer(classes=classes)
        self.mlb.fit(None)

    def encode(self, dataframe, dtype=bool):
        return self.mlb.transform(dataframe[self.class_field]).astype(dtype)


class WeightedCategoricalEncoder:
    """Encoded categorical data (e.g. genres or similar) to a vector representation
    from 0 to 1 of how much each category for all possible categories applies to a
    dataframe row."""

    def fit(self, classes, class_field="features", weight_field="ranks"):
        self.class_field = class_field
        self.weight_field = weight_field
        self.classes = sorted(classes)
        self.initial_encoding = dict.fromkeys(self.classes, 0)

    def encode(self, dataframe):
        return dataframe[self.weight_field].map_elements(self.get_weights)

    def get_weights(self, row):
        encoding = self.initial_encoding.copy()
        encoding.update(ast.literal_eval(row))

        return np.fromiter(encoding.values(), dtype=float)
