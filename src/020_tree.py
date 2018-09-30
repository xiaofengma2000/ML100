from sklearn import datasets
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

data = datasets.make_moons(n_samples=10000, noise=0.4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2, random_state=42)

tree = DecisionTreeClassifier()
grid = GridSearchCV(tree, {"max_leaf_nodes":[3,4,5,6,7]}, cv=5)
grid.fit(X_train, y_train)
print(grid.cv_results_)
print(grid.best_params_)
from sklearn.metrics import confusion_matrix, f1_score
print("Final score matrix:\n", confusion_matrix(y_test, grid.best_estimator_.predict(X_test)))
print("Final F1 score :\t\t", f1_score(y_test, grid.best_estimator_.predict(X_test)))


