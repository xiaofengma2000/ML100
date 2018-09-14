import numpy,pandas
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pandas.read_csv("C:/work/python/100/data/003/50_Startups.csv", na_values=['no info', '.', 0])
print("raw data ; \n", data)

imp=Imputer(missing_values="NaN", strategy="mean")
data["Marketing Spend"]=imp.fit_transform(data[["Marketing Spend"]])
data["R&D Spend"]=imp.fit_transform(data[["R&D Spend"]])
print("Imputer data ; \n", data)

regr1 = LinearRegression()
regr1.fit(data.iloc[:, [1]].values, data.iloc[:, -1].values)
splitx1 = numpy.split(data.iloc[:, [1]].values, [30])
splity1 = numpy.split(data.iloc[:, -1].values, [30])

regr2 = LinearRegression()
regr2.fit(data.iloc[:, [2]].values, data.iloc[:, -1].values)
splitx2 = numpy.split(data.iloc[:, [2]].values, [30])
splity2 = numpy.split(data.iloc[:, -1].values, [30])

# # plt.scatter(splitx1[1], splity1[1], color='gray')
# plt.plot(splitx1[1], regr1.predict(splitx1[1]), color='blue', linewidth=3, label="R&D")
# # plt.scatter(splitx2[1], splity2[1], color='gray')
# plt.plot(splitx2[1], regr2.predict(splitx2[1]), color='lightblue', linewidth=3, label="Spend")
# plt.xticks(())
# plt.yticks(())
# plt.show()

data = pandas.read_csv("C:/work/python/100/data/003/50_Startups.csv", na_values=['no info', '.'])
X = data.iloc[ : , :-1].values
Y = data.iloc[ : ,  4 ].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
print(X)

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
print(X)

# R&D is part of spend
X = X[: , 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)


# from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(Y_test, y_pred)
# print(confusion_matrix)

print(Y_test)
print(y_pred)
index_x = numpy.reshape(numpy.arange(len(Y_test)), (-1, 1))
plt.scatter(index_x, Y_test, color='black')
plt.scatter(index_x, y_pred, color='blue')
# plt.plot(Y_test, y_pred, color='lightblue', linewidth=3, label="Spend")
plt.show()

