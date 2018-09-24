import matplotlib
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import SGDClassifier


mnist = fetch_mldata('MNIST original')

X_train, X_test, y_train, y_test = train_test_split(mnist["data"], mnist["target"], test_size=0.2, random_state=10)

shuffle_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
print(X_train.shape)
print(y_train.shape)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
score = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print("cross_val_score \t\t",score)

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
from sklearn.metrics import confusion_matrix
score2 = confusion_matrix(y_train, y_train_pred)
print("cross_val_predict \t\t\n",score2)

from sklearn.metrics import f1_score,precision_score, recall_score
print("precision_score \t\t",precision_score(y_train, y_train_pred, average="macro"))
print("recall_score \t\t",recall_score(y_train, y_train_pred, average="macro"))
print("f1_score \t\t",f1_score(y_train, y_train_pred, average="macro"))

from sklearn.metrics import precision_recall_curve
y_scores = sgd_clf.decision_function(X_train)
print(y_scores)




