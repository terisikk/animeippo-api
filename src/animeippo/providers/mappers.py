import polars as pl


class DefaultMapper:
    def __init__(self, name, default=None):
        self.name = name
        self.default = default

    def map(self, series):
        return series.get_column(self.name) if self.name in series.columns else pl.lit(self.default)


class SelectorMapper:
    def __init__(self, selector):
        self.selector = selector

    def map(self, dataframe):
        try:
            return dataframe.select(self.selector).to_series()
        except pl.exceptions.ColumnNotFoundError:
            return pl.lit(None)


class QueryMapper:
    def __init__(self, query):
        self.query = query

    def map(self, dataframe):
        try:
            return self.query(dataframe)
        except pl.exceptions.ColumnNotFoundError:
            return pl.lit(None)


class SingleMapper:
    def __init__(self, name, func, default=None, dtype=pl.String):
        self.name = name
        self.func = func
        self.default = default
        self.dtype = dtype

    def map(self, dataframe):
        if self.name not in dataframe.columns and len(dataframe) > 0:
            return pl.lit(self.default)

        return dataframe[self.name].map_elements(self.row_wrapper, return_dtype=self.dtype)

    def row_wrapper(self, row):
        try:
            return self.func(row)
        except (TypeError, ValueError, AttributeError, KeyError) as error:
            print(f"Error extracting {self.name}: {error}")
            return self.default


class MultiMapper:
    def __init__(self, columns, func, default=None):
        self.columns = columns
        self.func = func
        self.default = default

    def map(self, dataframe):
        if any(column not in dataframe.columns for column in self.columns):
            return pl.lit(self.default)

        return dataframe.select(self.columns).map_rows(self.row_wrapper).to_series()

    def row_wrapper(self, row):
        try:
            return self.func(*row)
        except (TypeError, ValueError, AttributeError, KeyError) as error:
            print(f"Error extracting with function {self.func}: {error}")
            return self.default
