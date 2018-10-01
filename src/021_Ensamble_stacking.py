import numpy

from sklearn import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

m_data = fetch_mldata("MNIST original")
X = m_data["data"]
y = m_data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 7, random_state=42)
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=1 / 6, random_state=42)

mlp = MLPClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
eratree = ExtraTreesClassifier(random_state=42)

estimators = [rf, eratree, mlp]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)

X_val_predictions = numpy.empty((len(X_val), len(estimators)), dtype=numpy.float32)
for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

stacking = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
stacking.fit(X_val_predictions, y_val)

print("MLPClassifier test Score : \t", accuracy_score(y_test, mlp.predict(X_test)))
print("RandomForestClassifier test Score : \t", accuracy_score(y_test, rf.predict(X_test)))
print("ExtraTreesClassifier test Score : \t", accuracy_score(y_test, eratree.predict(X_test)))

X_test_predictions = numpy.empty((len(X_test), len(estimators)), dtype=numpy.float32)
for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)
print("Stacking test Score : \t", accuracy_score(y_test, stacking.predict(X_test_predictions)))

#Result
# MLPClassifier test Score : 	 0.9586
# RandomForestClassifier test Score : 	 0.9482
# ExtraTreesClassifier test Score : 	 0.9491
# Stacking test Score : 	 0.9487
