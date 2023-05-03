import numpy as np
from sklearn.tree import DecisionTreeRegressor

# 勾配法と言いつつ、勾配は計算していない。残差による再学習を繰り返す。
class SimpleGBDT:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.random_state = random_state
        self.costs = []
    
    # MSE
    def _calc_cost(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def _calc_gradients(self, y, y_pred):
        return y - y_pred

    def fit(self, X, y):
        np.random.seed(self.random_state)

        self.F0 = y.mean()
        # 初期予測値は、目的変数の平均値
        Fm = np.repeat(self.F0, y.shape[0])

        # 残差を0に近づけるように学習を繰り返す
        # ポイントは、yを予測するモデルを作っているのではなく、残差を予測するモデルを作っているということ。
        # つまり、残差を予測するモデルを作って、それを予測値に足していくことで、残差を0に近づけている。
        for _ in range(self.n_estimators):
            grads = self._calc_gradients(y, Fm)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, grads)

            self.costs.append(self._calc_cost(y, Fm))

            # 予測値を更新
            # predictは残差の予測値を返す。つまりFが残差分更新されると、次の残差は0に近づく。
            Fm += self.learning_rate * tree.predict(X)

            self.trees.append(tree)

    def predict(self, X):
        Fm = np.repeat(self.F0, X.shape[0])
        # 予測値は、初期予測値 + 残差の予測値の総和
        # 前のモデルの残差の残差を予測するモデルを作ったので、それらを足し合わせることで、より正確な残差を予測できる。
        # それを最初の予測値に足せば、予測値 + 正確な残差 = 正確な予測値が得られる。
        pred = Fm + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)
        return pred
        # return np.where(pred > 0, 1, 0)
