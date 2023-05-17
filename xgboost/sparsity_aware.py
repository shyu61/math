from typing import Tuple
import pandas as pd
import numpy as np

### Xgboost内部で使われているsparsity-aware split findingアルゴリズム
# 欠損値を送る方向をあらかじめ学習しておくことで、分割点の探索空間を絞ることができる。
# 事前学習方法:
#  - 欠損値を左右それぞれに寄せて、best_gain, best_splitを探索する
#  - 左右どちらのbest_gainがより大きかったかで、方向を決定する
# 事前学習も、欠損値を除外したデータでループを回して探索するので、結果的に、事前学習でも本学習でも欠損値に対する分割点探索を行う必要がなく、計算量を削減できる。
###


### 損失関数: 平滑化指数損失関数
# L(y, y_pred) = - [y * exp(-y_pred) + (1 - y) * exp(y_pred)]


def calc_grad(y_true, y_pred):
    return -y_true * np.exp(-y_pred) + (1 - y_true) * np.exp(y_pred)


def calc_hessian(y_true, y_pred):
    return y_true * np.exp(-y_pred) + (1 - y_true) * np.exp(y_pred)


def calc_gain(
        grad,
        hessian,
        grad_left,
        grad_right,
        hessian_left,
        hessian_right,
        lam):
    left = grad_left**2 / (hessian_left + lam)
    right = grad_right**2 / (hessian_right + lam)
    return left + right - (grad**2 / (hessian + lam))


def main(lam=1.0):
    train = pd.read_csv('./input/boston.csv')

    ik = train.dropna()
    n_feats = train.shape[1]
    best_gain = 0
    grad, hessian = np.sum(calc_grad(train)), np.sum(calc_hessian(train))

    for k in train.columns:
        ik = ik[k]
        # 欠損値をleftへ移動
        # 欠損値を除去したデータでループを回していることがポイント
        # 先に欠損値を含む全体でgradを計算しておくことで、欠損値を探索せずにgainを計算できる
        ik = ik.sort_values(ascending=True)
        grad_left, hessian_left = 0, 0
        for i in ik:
            # 一つずつ点が増えるごと（ループが回るにつれて）にgainも加算されていく
            grad_left += np.sum(calc_grad(i))
            hessian_left += np.sum(calc_hessian(i))

            # rightノードのgainは全体から、leftノードのgainを引くことで求める
            # つまりrightノードは欠損値を含めたgain値になる
            grad_right = grad - grad_left
            hessian_right = hessian - hessian_left
            new_gain = calc_gain(grad, hessian, grad_left, grad_right, hessian_left, hessian_right, lam)
            if new_gain > best_gain:
                best_gain = new_gain
                best_direction = 'left'

        # 欠損値をrightへ移動
        ik = ik.sort_values(ascending=False)
        for i in ik:
            grad_right, hessian_right = 0, 0
            grad_right += np.sum(calc_grad(i))
            hessian_right += np.sum(calc_hessian(i))

            grad_right = grad - grad_left
            hessian_right = hessian - hessian_left
            new_gain = calc_gain(grad, hessian, grad_left, grad_right, hessian_left, hessian_right, lam)
            if new_gain > best_gain:
                best_gain = new_gain
                best_direction = 'right'
    
    return best_direction, best_gain


if __name__ == '__main__':
    main()
