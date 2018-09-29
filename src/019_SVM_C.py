# SVM Classification

import numpy

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn import datasets

data = datasets.load_diabetes()
X = data['data']
y = (data['target'] > 100).astype(numpy.int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR


def print_score(l_reg):
    score = cross_val_score(l_reg, X_train, y_train, cv=3)
    print("Score \t\t:", score)
    print("Mean \t\t:", numpy.mean(score))


print("\nLinearSVC")
print_score(LinearSVC())
print("\nSVC")
print_score(SVC())
print("\nSVC -- poly")
print_score(SVC(kernel="poly", degree=3, coef0=5, C=5))
print("\nSVC -- RBF")
print_score(SVR(kernel="rbf", gamma=1, C=5))
