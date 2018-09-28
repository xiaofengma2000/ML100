import numpy

from housing import HousingPrice
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

X_train, X_test, y_train, y_test = HousingPrice().prepare_data()

from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR

def print_score(l_reg):
    score = cross_val_score(l_reg, X_train, y_train, cv=3)
    print("Score \t\t:",score)
    print("Mean \t\t:",numpy.mean(score))

print("\nLinearSVR")
print_score(LinearSVR())
print("\nSVR")
print_score(SVR())
print("\nSVR -- poly")
print_score(SVR(kernel="poly",degree=3, coef0=1, C=5))
print("\nSVR -- RBF")
print_score(SVR(kernel="rbf", gamma=5, C=0.001))

