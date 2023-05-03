# 参考: https://qiita.com/omiita/items/1735c1d048fe5f611f80

import numpy as np

class LinearRegressionGD:
    """
    最急勾配降下法:
    - 全部のデータを使って損失を計算し、勾配を求める。
    - 最小値ではない局所解に陥る可能性がある。
    - 1回の更新に時間がかかる。
    """
    def __init__(self, n_iter=100, eta=0.1):
        self.n_iter = n_iter
        self.eta = eta
        self.w = None
        self.bias = 0
        self.costs = []
    
    # MSE
    def __cost(self, y, pred):
        return np.mean((y - pred) ** 2)

    def __gradient(self, X, y, pred) -> tuple[float, np.array]:
        db = np.mean(pred - y)
        dw = np.dot(X.T, (pred - y)) / X.shape[0]
        return db, dw

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            pred = self.predict(X)

            cost = self.__cost(y, pred)
            db, dw = self.__gradient(X, y, pred)

            self.bias -= self.eta * db
            self.w -= self.eta * dw
            self.costs.append(cost)

    def predict(self, X):
        return np.dot(X, self.w) + self.bias

class LinearRegressionSGD:
    """
    確率的勾配降下法:
    - 1回の更新に、1つのデータしか使わない。
    - ランダム性が生まれるため、局所解に陥る可能性が低い。（別のデータでは勾配が増えたりするため局所解から抜け出せる）
    - 収束に時間がかかる。並列処理できない。
    """
    def __init__(self, n_iter=10, eta=0.1):
        self.n_iter = n_iter
        self.eta = eta
        self.w = None
        self.bias = 0
        self.costs = []

    # MSE
    def __cost(self, yi, pred):
        return np.mean((yi - pred) ** 2)

    def __gradient(self, xi, yi, pred) -> tuple[float, np.array]:
        db = np.mean(pred - yi)
        dw = np.dot(xi.T, (pred - yi))
        return db, dw

    def fit(self,X, y):
        self.w = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            costs = []
            # 1データずつ更新する
            for xi, yi in zip(X, y):
                pred = self.predict(xi)

                cost = self.__cost(yi, pred)
                db, dw = self.__gradient(xi, yi, pred)

                self.bias -= self.eta * db
                self.w -= self.eta * dw
                costs.append(cost)
            
            self.costs.append(np.mean(costs))

    def predict(self, X):
        return np.dot(X, self.w) + self.bias


class LinearRegressionMiniBatch:
    """
    ミニバッチ勾配降下法:
    - 最急勾配降下法とSGDの折衷案。
    """
    def __init__(self, epoch=10, eta=0.1, batch_size=10):
        self.epoch = epoch
        self.eta = eta
        self.batch_size = batch_size
        self.w = None
        self.bias = 0
        self.costs = []

    # MSE    
    def __cost(self, y_batch, pred):
        return np.mean((y_batch - pred) ** 2)

    def __gradient(self, X_batch, y_batch, pred) -> tuple[float, np.array]:
        db = np.mean(pred - y_batch)
        dw = np.dot(X_batch.T, (pred - y_batch)) / X_batch.shape[0]
        return db, dw
    
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])

        for _ in range(self.epoch):
            costs = []
            shuffled_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                pred = self.predict(X_batch)
                cost = self.__cost(y_batch, pred)
                db, dw = self.__gradient(X_batch, y_batch, pred)

                self.bias -= self.eta * db
                self.w -= self.eta * dw
                costs.append(cost)
            
            self.costs.append(np.mean(costs))

    def predict(self, X):
        return np.dot(X, self.w) + self.bias
