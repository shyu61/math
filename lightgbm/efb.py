import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import networkx as nx

class SimpleEFB:
    def __init__(self, n_trees=100, learning_rate=0.01, max_depth=4, max_bin=255, random_state=42):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bin = max_bin
        self.trees = []
        self.costs = []
        self.bundles = {}
        self.feature_bundles = []
        self.random_state = random_state

    # MSE
    def _calc_cost(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def _calc_gradients(self, y, y_pred):
        return y - y_pred
    
    def _create_weighted_feature_graph(self, X: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        feats = list(X.columns)
        for feat in feats:
            G.add_node(feat)

        for i, feature_i in enumerate(feats):
            for j, feature_j in enumerate(feats):
                # 重複して評価しないため
                if i < j:
                    # nonzeroかつ同じ値の数をカウント（衝突数）
                    # 数値特徴量の場合はビン化してから衝突数をカウント
                    X_i = pd.cut(X[feature_i], bins=self._get_bins(X[feature_i]), labels=False).astype(int)
                    X_j = pd.cut(X[feature_j], bins=self._get_bins(X[feature_j]), labels=False).astype(int)

                    non_zero_mask = (X_i != 0) & (X_j != 0)
                    conflicts = (X_i[non_zero_mask] == X_j[non_zero_mask]).sum()
                    # non_zero_mask = (X[feature_i] != 0) & (X[feature_j] != 0)
                    # conflicts = (X[feature_i][non_zero_mask] == X[feature_j][non_zero_mask]).sum()

                    # 衝突がある場合のみweightを衝突数とするedgeを追加
                    if conflicts > 0:
                        G.add_edge(feature_i, feature_j, weight=conflicts)
        return G

    def _greedy_bundling(self, G: nx.Graph, total_sample_cnt, threshold=None) -> dict:
        # https://github.com/microsoft/LightGBM/issues/4114#issuecomment-813201652
        if threshold is None:
            threshold = int(total_sample_cnt / 10000)

        bundles = {}
        # (edge数 * weight) でソート
        sortedNodes = sorted(G.degree(weight='weight'), key=lambda x: x[1], reverse=True)

        # より多くの衝突がある特徴量から処理を行う
        for feat, degree in sortedNodes:
            needNew = True
            for i in range(len(bundles)):
                conflicts = 0
                # 既存のバンドル（bundles[i]）に含まれる全ての特徴量との衝突数を合計する
                for f in bundles[i]:
                    if f in G[feat]:
                        conflicts += G[feat][f]['weight']

                # 値の衝突が閾値以下であれば、マージする
                # 閾値以上であれば、新しいバンドルを追加する
                if conflicts <= threshold:
                    bundles[i].append(feat)
                    # bundle内で最も衝突が小さいものにマージするのではなく、早い者勝ちでマージする
                    needNew = False
                    break
            if needNew:
                idx = len(bundles)
                bundles[idx] = [feat]
        
        return bundles
        
    def _get_bins(self, X: pd.Series) -> int:
        unique_cnt = X.nunique()
        return min(unique_cnt, self.max_bin)

    def _merge_exclusive_feature(self, X: pd.DataFrame, bundles: dict) -> pd.DataFrame:
        df = pd.DataFrame()

        for i, feats in enumerate(bundles.values()):
            bin_ranges = {}
            total_bin = 0

            # offsetを計算
            for feat in feats:
                # 各特徴量のbin数を加算していくことで、offsetを計算する
                total_bin += self._get_bins(X[feat])
                bin_ranges[feat] = total_bin

            bin = pd.Series(np.zeros(len(X), dtype=int))
            for feat in feats:
                bin_df = pd.cut(X[feat], bins=self._get_bins(X[feat]), labels=False)
                zero_mask = bin_df == 0

                # bin値が0の場合は、offsetを加算しない
                # そもそもここで加算している特徴量同士は、同時にnonzero値を取らないことが前提であるため、
                # offsetの加算により、どの特徴量のbin値かが一意に特定できる。0にoffsetを加算すると、この一意性が失われる。
                bin_df += bin_ranges[feat]
                bin_df[zero_mask] = 0

                bin += bin_df

            df[i] = bin

        return df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        np.random.seed(self.random_state)

        X = X.copy()
        y = y.copy()

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        self.F0 = y.mean()
        Fm = np.repeat(self.F0, X.shape[0])

        G = self._create_weighted_feature_graph(X)
        self.bundles = self._greedy_bundling(G, X.shape[0])
        X = self._merge_exclusive_feature(X, self.bundles)

        X = np.array(X)
        y = np.array(y)

        for _ in range(self.n_trees):
            grads = self._calc_gradients(y, Fm)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, grads)

            self.costs.append(self._calc_cost(y, Fm))

            # Fmを更新
            Fm += self.learning_rate * tree.predict(X)

            self.trees.append(tree)
        return self

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X = X.reset_index(drop=True)

        X = self._merge_exclusive_feature(X, self.bundles)

        X = np.array(X)

        Fm = np.repeat(self.F0, X.shape[0])
        pred = Fm + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)
        return pred
