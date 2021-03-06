import numpy,pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from util import SocialAd
x,y=SocialAd().data()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
classifier = KNeighborsClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from util import Report
Report().ROC(x_test, y_test, classifier)

