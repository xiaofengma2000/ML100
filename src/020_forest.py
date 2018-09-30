import numpy

from sklearn import datasets
from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.base import clone
data = datasets.make_moons(n_samples=10000, noise=0.4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2, random_state=42)

grid = GridSearchCV(DecisionTreeClassifier(), {"max_leaf_nodes":[3,4,5,6,7]}, cv=5)
grid.fit(X_train, y_train)

rs = ShuffleSplit(n_splits=100, test_size=0.9, random_state=42)
scores = []
y_pred_list = numpy.empty([100, len(y_test)])
from sklearn.metrics import accuracy_score
for index, (train_index, _) in enumerate(rs.split(X_train)):
    X_train_2 = [X_train[i] for i in train_index]
    y_train_2 = [y_train[i] for i in train_index]
    tree = clone(grid.best_estimator_) # DecisionTreeClassifier(max_leaf_nodes=4)
    tree.fit(X_train_2, y_train_2)
    predict = tree.predict(X_test)
    scores.append(accuracy_score(y_test, predict))
    y_pred_list[index] = predict

print("Mean value of 100 DecisionTreeClassifier : \t\t", numpy.mean(scores))

from scipy.stats import mode
y_pred_majority_votes, n_votes = mode(y_pred_list, axis=0)
print("Combined 100 DecisionTreeClassifier : \t\t\t\t", accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))

from sklearn.ensemble import VotingClassifier
