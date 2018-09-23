import numpy

import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import LabelBinarizer, Imputer, StandardScaler


# rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
# class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
#     def __init__(self, add_bedrooms_per_room = True):
#         # no *args or **kargs
#         self.add_bedrooms_per_room = add_bedrooms_per_room
#     def fit(self, X, y=None):
#         return self
#         # nothing else to do
#     def transform(self, X, y=None):
#         rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
#         population_per_household = X[:, population_ix] / X[:, household_ix]
#         if self.add_bedrooms_per_room:
#             bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
#             return numpy.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
#         else:
#             return numpy.c_[X, rooms_per_household, population_per_household]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names].values


class HousingPrice:

    def data(self):
        data = pandas.read_csv("../data/012/housing.csv")
        imputer = Imputer(strategy="median")
        data["total_bedrooms"] = imputer.fit_transform(data[["total_bedrooms"]])

        data['rooms_per_household'] = data["total_rooms"] / data["households"]
        data['population_per_household'] = data["population"] / data["households"]

        data["income_cat"] = numpy.ceil(data["median_income"] / 1.5)
        data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)

        X = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9,10,11]]
        y = data.iloc[:, [8]]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for train_index, test_index in sss.split(data, data["income_cat"]):
            X_train = X.loc[train_index]
            X_test = X.loc[test_index]
            y_train = y.loc[train_index]
            y_test = y.loc[test_index]

        # from sklearn_pandas import DataFrameMapper
        # mapper = DataFrameMapper([('ocean_proximity', LabelBinarizer())], df_out=True)
        # X_train = X_train.join(mapper.fit_transform(X_train))
        # X_test = X_test.join(mapper.fit_transform(X_test))

        return X_train, X_test, y_train, y_test, data

    def prepare_data(self):
        X_train, X_test, y_train, y_test, data = self.data()

        num_attribs = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                       'households', 'median_income', 'population_per_household',
                       'rooms_per_household']
        cat_attribs = ["ocean_proximity"]

        num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('std_scaler', StandardScaler()),
        ])

        from sklearn_pandas import DataFrameMapper
        cat_pipeline = Pipeline([
            ('label_binarizer', DataFrameMapper([(cat_attribs, LabelBinarizer())])),
        ])

        full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ])

        # housing_prepared = full_pipeline.fit_transform(data)
        # print(housing_prepared)
        return full_pipeline.fit_transform(X_train), full_pipeline.fit_transform(X_test), y_train.values.ravel(), y_test.values.ravel()
