import numpy

from sklearn import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import time

m_data = fetch_mldata("MNIST original")
X = m_data["data"]
y = m_data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 7, random_state=42)

t1 = time.time()
c_score = cross_val_score(RandomForestClassifier(random_state=42), X_train, y_train, cv=5)
t2 = time.time()
print("Time spent without PCA : \t\t", t2-t1)

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
t1 = time.time()
c_score = cross_val_score(RandomForestClassifier(random_state=42), X_train_pca, y_train, cv=5)
t2 = time.time()
print("Time spent with PCA : \t\t", t2-t1)

