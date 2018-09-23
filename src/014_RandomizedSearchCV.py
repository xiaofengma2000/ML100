import numpy
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

from housing import HousingPrice
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint

X_train, X_test, y_train, y_test = HousingPrice().prepare_data()

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10],
     'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
# specify parameters and distributions to sample from
param_dist = {"n_estimators": [3, 10, 30],
              "max_features": [2, 4, 6, 8],
              "n_estimators": [3, 10],
              "bootstrap": [True, False],
              }

grid_search = RandomizedSearchCV(forest_reg, param_dist, cv=5,
                                 scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
best_reg = grid_search.best_estimator_

some_data = X_test
some_labels = y_test
print("Predictions:\t", best_reg.predict(some_data))
print("Labels:\t\t", some_labels)
housing_predictions = best_reg.predict(some_data)
mse = mean_squared_error(some_labels, housing_predictions)
print("RMSE:\t\t", numpy.sqrt(mse))
