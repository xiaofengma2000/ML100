from pandas.plotting import scatter_matrix

from util import HousingPrice
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test, data = HousingPrice().data()

#below two descirrption should look very similar
print(X_train['median_income'].describe())
print(X_test['median_income'].describe())

print(data['total_bedrooms'].describe())

# X_train.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()

corr_matrix = data.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# attributes = ["median_house_value", "median_income"]
# scatter_matrix(data[attributes], figsize=(12, 8))
# plt.show()
