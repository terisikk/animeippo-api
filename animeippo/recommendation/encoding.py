import sklearn.preprocessing as skpre


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
        return self.mlb.transform(dataframe[self.class_field]).astype(dtype).tolist()


class WeightedCategoricalEncoder:
    """Encoded categorical data (e.g. genres or similar) to a vector representation
    from 0 to 1 of how much each category for all possible categories applies to a
    dataframe row."""

    def fit(self, classes, class_field="features", weight_field="ranks"):
        self.class_field = class_field
        self.weight_field = weight_field
        self.classes = sorted(classes)

    def encode(self, dataframe):
        return dataframe.apply(self.get_weights, axis=1)

    def get_weights(self, row):
        weights = row[self.weight_field]
        return [weights.get(cls, 0) for cls in self.classes]
