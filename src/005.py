import numpy,pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pandas.read_csv("C:/work/python/100/data/006/Social_Network_Ads.csv")

x = data.iloc[:, 1:4].values
y = data.iloc[:, -1].values

# labelencoder = LabelEncoder()
# x[: , 0] = labelencoder.fit_transform(x[ : , 0])

x[:,0]=LabelEncoder().fit_transform(x[:,0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
classifier = KNeighborsClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# logreg = LogisticRegression()
# logreg.fit(x_train, y_train)
# print(classification_report(y_test, logreg.predict(x_test)))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='KNN (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Tennis characteristic')
plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
plt.show()

