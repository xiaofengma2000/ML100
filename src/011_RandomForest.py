from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from util import SocialAd
x,y=SocialAd().scalerData()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
