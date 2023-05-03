import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_boston_dataset():
    data = fetch_openml(name='Boston', version=1, as_frame=False)
    X, y = data.data, data.target
    return X, y

# powered by chatGPT
class SimpleGOSS:
    def __init__(self, n_trees=100, learning_rate=0.01, a=0.2, b=0.1, max_depth=4):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.a = a
        self.b = b
        self.max_depth = max_depth
        self.trees = []
        self.costs = []

    # MSE
    def _calc_cost(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def _calc_gradients(self, y, y_pred):
        return y - y_pred

    def _fit_tree(self, X, y, sample_weights):
        tree = DecisionTreeRegressor(max_depth=self.max_depth)
        tree.fit(X, y, sample_weight=sample_weights)
        return tree

    def _goss_sampling(self, X, y, y_pred, grads):
        top_n = int(self.a * len(X))
        rand_n = int(self.b * len(X))

        abs_gradients = np.abs(self._calc_gradients(y, y_pred))
        top_indices = np.argpartition(abs_gradients, -top_n)[-top_n:]

        rest_indices = np.setdiff1d(np.arange(len(X)), top_indices)
        rest_sample_grads = grads[rest_indices]
        rest_indices = np.random.choice(rest_indices, size=rand_n, replace=False,
                                    p=rest_sample_grads / rest_sample_grads.sum())
        return np.concatenate([top_indices, rest_indices])

    def fit(self, X, y):
        np.random.seed(42)

        self.F0 = y.mean()
        Fm = np.repeat(self.F0, X.shape[0])

        for _ in range(self.n_trees):
            sample_weights = np.abs(self._calc_gradients(y, Fm))
            sampled_indices = self._goss_sampling(X, y, Fm, sample_weights)
            X_sampled, y_sampled = X[sampled_indices], y[sampled_indices]
            gradients_sampled = self._calc_gradients(y_sampled, Fm[sampled_indices])
            tree = self._fit_tree(X_sampled, gradients_sampled, sample_weights[sampled_indices])
            y_pred_update = self.learning_rate * tree.predict(X)
            Fm += y_pred_update
            self.trees.append(tree)
            self.costs.append(self._calc_cost(y[sampled_indices], Fm[sampled_indices]))
        return self

    def predict(self, X):
        Fm = np.repeat(self.F0, X.shape[0])
        pred = Fm + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)
        return pred
