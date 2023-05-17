import numpy as np
import pandas as pd


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
    parent = grad**2 / (hessian + lam)
    left = grad_left**2 / (hessian_left + lam)
    right = grad_right**2 / (hessian_right + lam)

    return left + right - parent


def main():
    train = pd.read_csv('./input/boston.csv')
    best_gain = 0
    # 親ノードのgrad, hessianを計算
    # 親ノードの含まれる全てのデータの合計
    grad, hessian = np.sum(calc_grad(train)), np.sum(calc_hessian(train))
    for k in train.columns:
        grad_left, hessian_left = 0, 0
        for j in train[k]:
            # GLの計算方法は、leftノードに含まれる全ての点でのgradの合計
            # したがって、このループの累積値になる
            grad_left += np.sum(calc_grad(j))
            hessian_left += np.sum(calc_hessian(j))

            # 親のgrad, hessianからleftノードのgrad, hessianを引くことで、rightノードのgrad, hessianを計算
            grad_right = grad - grad_left
            hessian_right = hessian - hessian_left

            new_gain = calc_gain(grad, hessian, grad_left, grad_right, hessian_left, hessian_right, 1.0)
            if new_gain > best_gain:
                best_gain = new_gain

    return best_gain

# 各ノードごとに呼び出す
# 決して、決定木ごとではない
if __name__ == '__main__':
    main()
