import sklearn.preprocessing as skpre


class CategoricalEncoder:
    def fit(self, classes, class_field="features"):
        self.classes = classes
        self.class_field = class_field
        self.mlb = skpre.MultiLabelBinarizer(classes=classes)
        self.mlb.fit(None)

    def encode(self, dataframe, dtype=bool):
        return self.mlb.transform(dataframe[self.class_field]).astype(dtype).tolist()


class WeightedCategoricalEncoder:
    def fit(self, classes, class_field="features", weight_field="ranks"):
        self.class_field = class_field
        self.weight_field = weight_field
        self.classes = dict.fromkeys(sorted(classes))

    def encode(self, dataframe):
        weight_mapped = dataframe.apply(self.get_weights, axis=1)

        return weight_mapped.apply(self._encode_field).to_list()

    def get_weights(self, row):
        # Weight is 1 when a feature exists but does not have a weight, for example anilist genres
        return {
            feature: row[self.weight_field].get(feature, 1) for feature in row[self.class_field]
        }

    def _encode_field(self, field):
        # Weight is 0 when a feature does not exist
        return [field.get(cls, 0) for cls in self.classes.keys()]
