from util import HousingPrice
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, Imputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
        # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names].values

# class LabelBinarizerWrapper(BaseEstimator, TransformerMixin):
#     def __init__(self, attribute_names):
#         self.attribute_names = attribute_names
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         return X[self.attribute_names].values

data = pandas.read_csv("../data/012/housing.csv")
# imputer = Imputer(strategy="median")
# data["total_bedrooms"] = imputer.fit_transform(data[["total_bedrooms"]])

# housing_num = data.drop("ocean_proximity", axis=1)
# num_attribs = list(housing_num)
num_attribs = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

from sklearn_pandas import DataFrameMapper
cat_pipeline = Pipeline([
    ('label_binarizer', DataFrameMapper([('ocean_proximity', LabelBinarizer())])),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(data)
print(housing_prepared)

housing_labels=data["median_house_value"]
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)

some_data = data.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = housing_prepared[:5]#full_pipeline.fit_transform(some_data)
print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))

# some_data = data.iloc[:5]
# some_labels = data['median_house_value'].iloc[:5]
# full_pipeline.fit(some_data, some_labels)
# some_data_prepared = full_pipeline.transform(some_data)

# print(some_data)
# print(some_data_prepared)