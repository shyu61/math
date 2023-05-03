from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier
# import numpy as np
from decision_tree import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

clf = DecisionTree()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"scratch: {accuracy_score(y_test, y_pred)}")
print(f"depth: {clf.get_depth()}")

graph = clf.visualize_tree()
graph.view()

# ==================
# sk_clf = DecisionTreeClassifier(max_depth=100)
# sk_clf.fit(X_train, y_train)
# y_pred = sk_clf.predict(X_test)
# print(f"scikit_learn: {accuracy_score(y_test, y_pred)}")

# scratch: 0.9122807017543859
# scikit_learn: 0.8947368421052632
