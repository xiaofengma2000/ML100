import numpy
from housing import HousingPrice
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = HousingPrice().prepare_data()
tree_reg = DecisionTreeRegressor()
scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

print("Score :\t\t", scores)
print("Mean :\t\t", scores.mean())
print("Standard Devlation :\t\t", scores.std())
