from util import SmsSpam
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

x,y=SmsSpam().data()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
count_vect = CountVectorizer()
x_train_counts=count_vect.fit_transform(x_train)
print(x_train_counts.shape)

tf_transformer = TfidfTransformer()
x_train_tf = tf_transformer.fit_transform(x_train_counts)
print(x_train_tf.shape)

text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None))])

text_clf.fit(x_train, y_train)
y_pred = text_clf.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
