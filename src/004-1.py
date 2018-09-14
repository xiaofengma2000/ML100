import numpy,pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def first_set_result (row):
   if row['ST1.1'] < row['ST1.2'] :
      return 0
   return 1

rawdata=pandas.read_csv("../data/tennis/AusOpen-men-2013.csv")
# rawdata=pandas.read_csv("../data/tennis/AusOpen-women-2013.csv")

# data = rawdata[['Player1','Player2','Result','FSP.1','FSW.1','SSP.1','SSW.1','FSP.2','FSW.2','SSP.2','SSW.2']]
rawdata['ST1_Result'] = rawdata.apply (lambda row: first_set_result (row),axis=1)
data = rawdata[['Result','FSW.1','SSW.1','FSW.2','SSW.2','ST1_Result']]
print(data)

# sns.countplot(x="SSW.2", data=data, palette='hls')
# plt.show()

# print(data.groupby('Result').mean())

# 1st set result vs final result
# pandas.crosstab(data['ST1_Result'],data['Result']).plot(kind='bar')
# plt.show()


x = data.iloc[ :, 1:].values
y = data.iloc[ :, 0].values

print(x)
print(y)

onehotencoder = OneHotEncoder(categorical_features = [4])
x = onehotencoder.fit_transform(x).toarray()
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
print(logreg.score(x_test, y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Tennis characteristic')
plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
plt.show()
