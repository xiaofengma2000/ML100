import matplotlib
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import SGDClassifier


mnist = fetch_mldata('MNIST original')

# print(mnist['target'].shape)
# X, y = mnist["data"], mnist["target"]
# some_digit = X[36000]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image,
#            cmap = matplotlib.cm.binary,
#            interpolation="nearest")
# plt.axis("off")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(mnist["data"], mnist["target"], test_size=0.2, random_state=10)

shuffle_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
print(X_train.shape)
print(y_train.shape)

sgd_clf = SGDClassifier(random_state=42)
y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)
sgd_clf.fit(X_train, y_train_5)

from sklearn.model_selection import cross_val_score
score = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print("cross_val_score \t\t",score)

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
from sklearn.metrics import confusion_matrix
score2 = confusion_matrix(y_train_5, y_train_pred)
print("cross_val_predict \t\t\n",score2)
from sklearn.metrics import f1_score,precision_score, recall_score
print("precision_score \t\t",precision_score(y_train_5, y_train_pred))
print("recall_score \t\t",recall_score(y_train_5, y_train_pred))
print("f1_score \t\t",f1_score(y_train_5, y_train_pred))

from sklearn.metrics import precision_recall_curve
y_scores = sgd_clf.decision_function(X_train)
print(y_scores)
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

#PR Curve
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

#ROC
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
# plot_roc_curve(fpr, tpr)
# plt.show()

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]   # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="bottom right")
plt.show()




