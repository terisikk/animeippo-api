import pandas as pd


class DefaultMapper:
    def __init__(self, name, default=pd.NA):
        self.name = name
        self.default = default

    def map(self, series):
        return series.get(self.name, self.default)


class SingleMapper:
    def __init__(self, name, func, default=pd.NA):
        self.name = name
        self.func = func
        self.default = default

    def map(self, dataframe):
        if self.name not in dataframe.columns and len(dataframe) > 0:
            dataframe[self.name] = self.default
            return dataframe[self.name]

        return dataframe[self.name].apply(self.row_wrapper, args=(self.func, self.default))

    def row_wrapper(self, row, func, default=None, args=None):
        args = args or []

        try:
            return func(row, *args)
        except (TypeError, ValueError, AttributeError, KeyError) as error:
            print(error)
            return default


class MultiMapper:
    def __init__(self, func, default=pd.NA):
        self.func = func
        self.default = default

    def map(self, dataframe):
        return dataframe.apply(self.row_wrapper, args=(self.func, self.default), axis=1)

    def row_wrapper(self, row, func, default=None, args=None):
        args = args or []

        try:
            return func(row, *args)
        except (TypeError, ValueError, AttributeError, KeyError) as error:
            print(error)
            return default
