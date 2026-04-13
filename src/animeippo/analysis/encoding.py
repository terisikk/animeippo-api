import polars as pl


class CategoricalEncoder:
    """Encodes categorical data (e.g. genres or similar) to a vector representation
    of whether a dataframe row has a category from all possible categories or not.
    """

    def fit(self, classes, class_field="features"):
        self.classes = sorted(classes)
        self.class_field = class_field
        self.dtype = pl.Struct(dict.fromkeys(self.classes, pl.UInt8))

    def encode(self, dataframe):
        features_df = dataframe.select(
            pl.col(self.class_field).list.eval(pl.element().cast(pl.Utf8)).alias(self.class_field)
        )

        binary_struct = pl.struct(
            [
                pl.when(pl.col(self.class_field).list.contains(feature))
                .then(1)
                .otherwise(0)
                .alias(feature)
                for feature in self.classes
            ]
        )

        return features_df.select(binary_struct.cast(self.dtype)).to_series()


class WeightedCategoricalEncoder:
    """Encoded categorical data (e.g. genres or similar) to a vector representation
    from 0 to 1 of how much each category for all possible categories applies to a
    dataframe row."""

    def fit(self, classes, class_field="features", weight_field="clustering_ranks"):
        self.class_field = class_field
        self.weight_field = weight_field
        self.classes = sorted(classes)
        self.dtype = pl.Struct(dict.fromkeys(self.classes, pl.UInt8))

    def encode(self, dataframe):
        return dataframe.select(pl.col("clustering_ranks").cast(self.dtype)).to_series()
