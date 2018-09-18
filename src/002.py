import numpy,pandas
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

data = pandas.read_csv("../data/002/studentscores.csv", na_values=['no info', '.'])
X = data.iloc[:, 0:1].values
Y = data.iloc[:, 1].values
print("X : \n", X)
print("Y : \n", Y)

# imp=Imputer(missing_values="NaN", strategy="mean")
# data["Hours"]=imp.fit_transform(data[["Hours"]])
# data["Scores"]=imp.fit_transform(data[["Scores"]])
# print(data)

splitx = numpy.split(X, [20])
splity = numpy.split(Y, [20])
training_x = splitx[0]
test_x = splitx[1]
training_y = splity[0]
test_y = splity[1]
print("training set : \n", training_x)
print("test set : \n", test_x)

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(training_x, training_y)

score_pred = regr.predict(test_x)
print("test : \n", test_x)
print("pred : \n", score_pred)

# plt.scatter(data, data["Scores"],  color='black')
plt.scatter(training_x, training_y, color='gray')
plt.plot(training_x, regr.predict(training_x), color='lightblue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

plt.scatter(test_x, test_y, color='black')
plt.plot(test_x, score_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
