import numpy
from sklearn.ensemble import RandomForestRegressor
from housing import HousingPrice
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = HousingPrice().prepare_data()

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10],
     'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
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

from sklearn.externals import joblib
joblib.dump(best_reg, "../model/housing_rf_model")
