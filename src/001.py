import numpy,pandas
from sklearn.preprocessing import Imputer

data = pandas.read_csv("../data/001/Data.csv", na_values=['no info', '.'])
print(data.head(0))
print(data["Country"][0])
print(data.head(4))

imp=Imputer(missing_values="NaN", strategy="most_frequent") #mean,median,most_frequent
# imp.fit(data[["Age"]])
print("after filling missing data")
data["Age"]=imp.fit_transform(data[["Age"]])
data["Salary"]=imp.fit_transform(data[["Salary"]])
print(data)

from sklearn.preprocessing import LabelEncoder
print("after encoding data")
le = LabelEncoder()
le.fit(['Yes', 'No'])
data["Purchased"] = le.transform(data["Purchased"])
print(data.head(10))

# from sklearn.cross_validation import train_test_split
# tts = train_test_split()
# X, y = np.arange(10).reshape((5, 2)), range(5)
split = numpy.split(data, [8])
print("training data")
print(split[0])
print("testing data")
print(split[1])

from sklearn.preprocessing import StandardScaler
print("processing data")
scaler = StandardScaler()
print(scaler.fit(data[["Age"]]))
print(scaler.mean_)
print(scaler.transform(data[["Age"]]))
print(scaler.fit(data[["Salary"]]))
print(scaler.mean_)
print(scaler.transform(data[["Salary"]]))
