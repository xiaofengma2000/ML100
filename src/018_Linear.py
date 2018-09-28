import numpy

from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn import datasets

data = datasets.load_diabetes()
# print(list(data.keys()))
# print(data['target'])
# print(data['feature_names'])
# print(data['data'])
X = data['data']
y = (data['target'] > 100).astype(numpy.int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge, LogisticRegression

def print_score(l_reg):
    score = cross_val_score(l_reg, X_train, y_train, cv=3)
    print(score)

print_score(LinearRegression())
print_score(Ridge(alpha=1, solver="cholesky"))
print_score(Lasso(alpha=1))
print_score(ElasticNet(alpha=1, l1_ratio=0.5))
print_score(LogisticRegression(C=10))
print_score(LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10))


