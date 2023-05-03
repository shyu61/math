import numpy as np
from collections import Counter
import graphviz
from IPython.display import display

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, n_samples=None, sample_indices=None, gain=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        # 回帰ならそのleaf-nodeに含まれるデータの平均値、分類なら最頻値、leaf-nodeじゃないならNone
        self.value = value
        self.n_samples = n_samples
        # leaf-nodeにsampleのindexを保持するかどうか
        self.sample_indices = sample_indices
        self.gain = gain

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_samples_split=2, max_depth=100, n_features=None, keep_sample_indices=False, criterion='entropy'):
        self.max_samples_split = max_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.keep_sample_indices = keep_sample_indices
        self.criterion = criterion
    
    def get_leaf_nodes(self, node: Node = None):
        if node is None:
            node = self.root

        if node.is_leaf_node():
            return [node]

        leaf_nodes = []
        if node.left is not None:
            leaf_nodes.extend(self.get_leaf_nodes(node.left))
        if node.right is not None:
            leaf_nodes.extend(self.get_leaf_nodes(node.right))
        
        return leaf_nodes

    
    def get_depth(self, node: Node = None) -> int:
        if node is None:
            node = self.root
        if node.is_leaf_node():
            return 0
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))

    # def print_tree(self, node=None, depth=0, feature_names=None):
    #     if node is None:
    #         node = self.root
    #     indent = "  " * depth
    #     if node.is_leaf_node():
    #         print(f"{indent}Leaf Node: Value={node.value}, Data Points={node.n_samples}")
    #     else:
    #         if feature_names is not None:
    #             feature = feature_names[node.feature]
    #         else:
    #             feature = node.feature
    #         print(f"{indent}Node: Feature={feature}, Threshold={node.threshold}, Data Points={node.n_samples}")
    #         self.print_tree(node.left, depth + 1, feature_names)
    #         self.print_tree(node.right, depth + 1, feature_names)
    
    def export_graphviz(self, graph, node=None, counter=None):
        if node is None:
            node = self.root
            counter = {"node_count": 0}

        # 各ノードに一意な名前をつける
        node_name = f"node{counter['node_count']}"
        counter["node_count"] += 1

        if node.is_leaf_node():
            label = f"Leaf Node\nValue={node.value}\nSamples={node.n_samples}"
            # leaf nodeは四角で表示
            graph.node(node_name, label=label, shape="box")
        else:
            label = f"Node\nFeature={node.feature}\nThreshold={node.threshold}\nSamples={node.n_samples}\nGain={node.gain:.4f}"
            # 普通のノードは円形で表示
            graph.node(node_name, label=label)

            # 子ノードを再帰的に描画
            # 同時に、edgeを引くために子ノードの名前を取得
            left_node_name = self.export_graphviz(graph, node.left, counter)
            right_node_name = self.export_graphviz(graph, node.right, counter)

            # 子ノードと親ノードの接続矢印を引く
            graph.edge(node_name, left_node_name, label="True")
            graph.edge(node_name, right_node_name, label="False")

        return node_name

    def visualize_tree(self, output=None, size=('7,7'), filename='tree.gv'):
        graph = graphviz.Digraph()
        self.export_graphviz(graph)

        if output == 'widget':
            graph.attr(size=size)
            display(graph)
        else:
            graph.render(filename, view=True)

        return graph
    
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])

        # keep_sample_indices=True の場合は、leaf-nodeに含まれるデータのindexを保持する
        indices = None
        if self.keep_sample_indices:
            indices = np.arange(X.shape[0])

        self.root = self.__grow_tree(X, y, original_indices=indices)
    
    def __grow_tree(self, X, y, depth=0, original_indices=None) -> Node:
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # stop criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.max_samples_split):
            leaf_value = self.__most_common_label(y)
            indices = None
            if self.keep_sample_indices:
                indices = original_indices
            return Node(value=leaf_value, n_samples=n_samples, sample_indices=indices)
    
        # n_featuresだけ特徴量を非復元抽出する
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # どの特徴量のどの閾値で分割するかを決める
        best_gain, best_feature, best_threshold = self.__best_split(X, y, feat_idxs)

        # 子ノードの作成
        left_idxs, right_idxs = self.__split(X[:, best_feature], best_threshold)
        # original_indices: originalのデータに対するindex
        # left_idxs, right_idxs: スライスで生成されたXに対するindex
        left = self.__grow_tree(X[left_idxs, :], y[left_idxs], depth+1, original_indices=original_indices[left_idxs])
        right = self.__grow_tree(X[right_idxs, :], y[right_idxs], depth+1, original_indices=original_indices[right_idxs])

        # ノードが終端まで伸び切るまで、ここには到達しない（left, rightが再帰的に__grow_treeを呼んでいるため）
        return Node(best_feature, best_threshold, left, right, n_samples=n_samples, gain=best_gain)
    
    def __best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # 情報利得を計算
                gain = self.__information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold
        
        return best_gain, split_idx, split_threshold
    
    def __information_gain(self, y, X_column, threshold):
        # 親nodeの不純度
        if self.criterion == 'entropy':
            parent_entropy = self.__entropy(y)
        elif self.criterion == 'gini':
            parent_entropy = self.__gini(y)
        else:
            raise NotImplementedError()

        # 左右の子ノードに分割するデータのインデックスを取得
        left_idx, right_idx = self.__split(X_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # 子ノードの不純度
        n = len(y)
        n_left, n_right = len(left_idx), len(right_idx)

        if self.criterion == 'entropy':
            e_left, e_right = self.__entropy(y[left_idx]), self.__entropy(y[right_idx])
        elif self.criterion == 'gini':
            e_left, e_right = self.__gini(y[left_idx]), self.__gini(y[right_idx])
        else:
            raise NotImplementedError()

        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # IGを計算
        information_gain = parent_entropy - child_entropy
        return information_gain

    def __split(self, X_column, threshold):
        left_idx = np.argwhere(X_column <= threshold).flatten()
        right_idx = np.argwhere(X_column > threshold).flatten()
        return left_idx, right_idx

    def __entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def __gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum(ps**2)
        # return 1 - np.sum([p**2 for p in ps])
    
    # 回帰問題の時は、分散減少をIG計算に使用する
    # def __variance_reduction(self, parent, left, right):
    #     parent_var = np.var(parent)
    #     left_var = np.var(left)
    #     right_var = np.var(right)

    #     left_weight = len(left) / len(parent)
    #     right_weight = len(right) / len(parent)

    #     ig = parent_var - (left_weight * left_var + right_weight * right_var)
    #     return ig

    def __most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self.__traverse_tree(x, self.root) for x in X])

    # 再帰的に木を降りていく
    def __traverse_tree(self, x, node: Node):
        if node.is_leaf_node():
            return node.value

        # データポイントxの中で、分割に使用された特徴量であるnode.featureの値を取り出す
        # その値と、分割に使用された閾値node.thresholdを比較する
        if x[node.feature] <= node.threshold:
            # 閾値より小さいなら左のノードへ、大きいなら右のノードへ
            return self.__traverse_tree(x, node.left)
        return self.__traverse_tree(x, node.right)
