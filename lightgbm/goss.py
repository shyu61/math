import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

class SimpleGOSS:
    def __init__(self, n_trees=100, learning_rate=0.01, a=0.2, b=0.1, max_depth=4, random_state=42):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.a = a
        self.b = b
        self.max_depth = max_depth
        self.trees = []
        self.costs = []
        self.selected_data_points = []
        self.all_grads_abs = []
        # self.a_borders = []
        self.random_state = random_state

    def plot_grad_with_iter(self, iteration):
        if iteration >= self.n_trees:
            raise ValueError("Invalid iteration number")

        selected_indices = self.selected_data_points[iteration]
        grads = self.all_grads_abs[iteration]
        selected_grads = grads[selected_indices]

        fig, ax = plt.subplots()
        ax.scatter(np.arange(len(grads)), grads, label="All data points")
        ax.scatter(selected_indices, selected_grads, label="Selected data points", color="red")
        ax.set_xlabel("Data point index")
        ax.set_ylabel("Gradient")
        ax.legend()

        plt.close()

        return fig

    def plot_grads(self):
        selected_grads = np.array([self.all_grads_abs[i][self.selected_data_points[i]] for i in range(self.n_trees)])
        selected_grads_mean = selected_grads.mean(axis=1)

        not_selected_grads = np.array([self.all_grads_abs[i][np.setdiff1d(np.arange(len(self.all_grads_abs[i])), self.selected_data_points[i])] for i in range(self.n_trees)])
        not_selected_grads_mean = not_selected_grads.mean(axis=1)

        fig, ax = plt.subplots()
        ax.plot(np.arange(self.n_trees), selected_grads_mean, label="Selected data points")
        ax.plot(np.arange(self.n_trees), not_selected_grads_mean, label="Not selected data points")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient")
        ax.legend()

        plt.close()

        return fig
        
    # MSE
    def _calc_cost(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    # MSEの負の勾配
    # MSEの勾配は: -2 * (y - y_pred) / len(y)
    # MSEの負の勾配を使っても良いが、定数倍は勾配降下法の学習率で調整できるため、簡潔に残差を使う。
    # ただし残差はMSEの負の勾配に比例するので、採用して問題ない。
    def _calc_gradients(self, y, y_pred):
        return y - y_pred
        # return 2 * (y - y_pred) / len(y)

    def _goss_sampling(self, X, top_n, rand_n, grads):
        top_indices = np.argpartition(np.abs(grads), -top_n)[-top_n:]
        rand_indices = np.setdiff1d(np.arange(len(X)), top_indices)
        # 論文に忠実な実装
        # rand_indices = np.random.choice(np.setdiff1d(np.arange(len(X)), top_indices), size=rand_n, replace=False)
        # こっちの方が精度が出る
        rand_indices = np.random.choice(rand_indices, size=rand_n, replace=False, p=np.abs(grads[rand_indices]) / np.abs(grads[rand_indices]).sum())
        used_indices = np.concatenate([top_indices, rand_indices])
        return used_indices, top_indices

    def fit(self, X, y):
        self.used_cnt = pd.Series(np.zeros(len(X)), dtype=int)

        # pd.DataFrameだとうまく動かないので、numpyに変換
        X = np.array(X)
        y = np.array(y)

        np.random.seed(self.random_state)

        self.F0 = y.mean()
        Fm = np.repeat(self.F0, X.shape[0])

        top_n = int(self.a * len(X))
        rand_n = int(self.b * len(X))

        for _ in range(self.n_trees):
            grads = self._calc_gradients(y, Fm)

            used_indices, top_indices = self._goss_sampling(X, top_n, rand_n, grads)

            # 学習に使われた回数をカウント
            self.used_cnt[used_indices] += 1
            self.selected_data_points.append(used_indices)
            self.all_grads_abs.append(np.abs(grads))
            # self.a_borders.append(np.sort(np.abs(grads[top_indices]))[0])

            # 重みを計算
            # ランダムサンプリングしたデータに重みをつける
            # 論文に忠実な実装
            # top_wight = np.repeat(self.a / top_n, top_n)
            # rand_weight = np.repeat((1 - self.a) / self.b, rand_n)
            # weight = np.concatenate([top_wight, rand_weight])
            # こっちの方が精度が出る
            weight = np.abs(grads[used_indices])

            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X[used_indices], grads[used_indices], sample_weight=weight)

            self.costs.append(self._calc_cost(y[used_indices], Fm[used_indices]))

            # Fmを更新
            Fm += self.learning_rate * tree.predict(X)

            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X)

        Fm = np.repeat(self.F0, X.shape[0])
        pred = Fm + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)
        return pred
