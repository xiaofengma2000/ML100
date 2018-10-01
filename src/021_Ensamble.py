import numpy

from sklearn import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

m_data = fetch_mldata("MNIST original")
X = m_data["data"]
y = m_data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 7, random_state=42)
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=1 / 6, random_state=42)

rf = RandomForestClassifier(random_state=42)
eratree = ExtraTreesClassifier(random_state=42)

v_s_c = VotingClassifier(
    estimators = [("rf", clone(rf)), ("eratree", clone(eratree))],
    voting="soft")
v_h_c = VotingClassifier(
    estimators = [("rf", clone(rf)), ("eratree", clone(eratree))],
    voting="hard")


rf.fit(X_train2, y_train2)
eratree.fit(X_train2, y_train2)
v_s_c.fit(X_train2, y_train2)
v_h_c.fit(X_train2, y_train2)

# print("DecisionTreeClassifier validation Score : \t", accuracy_score(y_val, rf.predict(X_val)))
print("RandomForestClassifier validation Score : \t", accuracy_score(y_val, rf.predict(X_val)))
print("ExtraTreesClassifier validation Score : \t", accuracy_score(y_val, rf.predict(X_val)))
print("Soft voting validation Score : \t", accuracy_score(y_val, rf.predict(X_val)))
print("Hard voting validation Score : \t", accuracy_score(y_val, rf.predict(X_val)))


# print("DecisionTreeClassifier test Score : \t", accuracy_score(y_test, svc.predict(X_test)))
print("RandomForestClassifier test Score : \t", accuracy_score(y_test, rf.predict(X_test)))
print("ExtraTreesClassifier test Score : \t", accuracy_score(y_test, eratree.predict(X_test)))
print("Soft voting test Score : \t", accuracy_score(y_test, v_s_c.predict(X_test)))
print("Hard voting test Score : \t", accuracy_score(y_test, v_h_c.predict(X_test)))

#Result
# RandomForestClassifier validation Score : 	 0.9467
# ExtraTreesClassifier validation Score : 	 0.9467
# Soft voting validation Score : 	 0.9467
# Hard voting validation Score : 	 0.9467

# RandomForestClassifier test Score : 	 0.9434
# ExtraTreesClassifier test Score : 	 0.9444
# Soft voting test Score : 	 0.9582
# Hard voting test Score : 	 0.9372
