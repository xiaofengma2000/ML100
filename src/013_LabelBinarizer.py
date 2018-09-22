from util import HousingPrice
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

X_train, X_test, y_train, y_test, data = HousingPrice().data()

# from sklearn_pandas import DataFrameMapper
# mapper = DataFrameMapper([('ocean_proximity', LabelBinarizer())], df_out=True)
# X_train = X_train.join(mapper.fit_transform(X_train))

# moved above section to HousingPrice().data()

print(X_train)
