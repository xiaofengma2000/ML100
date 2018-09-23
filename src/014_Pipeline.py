import numpy
from housing import HousingPrice

X_train, X_test, y_train, y_test = HousingPrice().prepare_data()
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

some_data = X_test[:5]
some_labels = y_test[:5]

print("Predictions:\t", lin_reg.predict(some_data))
print("Labels:\t\t", some_labels)

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(some_data)
lin_mse = mean_squared_error(some_labels, housing_predictions)
print("linear RMSE:\t\t", numpy.sqrt(lin_mse))

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
lin_mse = mean_squared_error(some_labels, tree_reg.predict(some_data))
print("Tree   RMSE:\t\t", numpy.sqrt(lin_mse))
