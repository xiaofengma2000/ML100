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

mean = numpy.mean(y_train)
#turn it to be classifier problem, high price and low price
y_train = (y_train < mean).astype(numpy.int)

print("\nLinearSVC")
print_score(LinearSVC())
print("\nSVC")
print_score(SVC())
print("\nSVC -- poly")
print_score(SVC(kernel="poly", degree=3, coef0=5, C=5))
print("\nSVC -- RBF")
print_score(SVR(kernel="rbf", gamma=1, C=5))
