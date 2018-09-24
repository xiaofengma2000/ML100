import numpy

from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

from housing import  HousingPrice

best_reg = joblib.load("../model/housing_rf_model")
X_train, X_test, y_train, y_test = HousingPrice().prepare_data()

some_data = X_test
some_labels = y_test
print("Predictions:\t", best_reg.predict(some_data))
print("Labels:\t\t", some_labels)
housing_predictions = best_reg.predict(some_data)
mse = mean_squared_error(some_labels, housing_predictions)
print("RMSE:\t\t", numpy.sqrt(mse))

