from typing import Tuple
import pandas as pd
import numpy as np

def extract_not_missing(train: pd.DataFrame) -> pd.DataFrame:
    ik = train.dropna()
    return ik

def calc_grad():
    pass

def calc_hess():
    pass

def calc_gain():
    pass

# 各特徴量ごとに実施する
def determine_default_direction(df: pd.Series, lambda_ = 1.0) -> Tuple(float, str):
    """
    df: 欠損のないある1つの特徴量列
    """
    best_gain = float('-inf')
    best_direction = None

    # goto left
    df = df.sort_values(ascending=True)
    for i in df:
        gain = calc_gain(i) # 計算には、欠損データも含める。じゃないと降順と昇順でgainが全く同じになってしまう
        if gain > best_gain:
            best_gain = gain
            best_direction = 'left'

    # goto right
    df = df.sort_values(ascending=False)
    for i in df:
        gain = calc_gain(i)
        if gain > best_gain:
            best_gain = gain
            best_direction = 'right'
            break # early_stop

    # 降順でも昇順でも、best_gainは同じ値になるのでは？
    # だとすると、欠損値をどちらに割り当てるかをこのロジックで判定できないのでは？
    return best_gain, best_direction


def sparsity_aware(ik: pd.DataFrame, lambda_ = 1.0):
    g, h = 0, 0
    h_l, h_r = 0, 0
    n_feats = ik.shape[1]
    directions = {}
    for k in range(n_feats):
        # goto right
        g_l, h_l = 0, 0
        score_l = 0
        k_df = ik.loc[:, k].sort_values(ascending=True)
        for i in k_df:
            g_l += calc_grad(i)
            h_l += calc_hess(i)
            g_r -= calc_grad(i)
            h_r -= calc_hess(i)
            score_l = np.max(score_l, g_l**2 / (h_l + lambda_) + g_r**2 / (h_r + lambda_) - g**2 / (h + lambda_))
        
        # goto left
        g_r, h_r = 0, 0
        score_r = 0
        k_df = ik.loc[:, k].sort_values(ascending=False)
        for i in k_df:
            g_r += calc_grad(i)
            h_r += calc_hess(i)
            g_l -= calc_grad(i)
            h_l -= calc_hess(i)
            score_r = np.max(score_r, g_l**2 / (h_l + lambda_) + g_r**2 / (h_r + lambda_) - g**2 / (h + lambda_))
        
        if score_l > score_r:
            directions[k] = 'left'
        else:
            directions[k] = 'right'
    return directions

if __name__ == '__main__':
    train = pd.read_csv('./input/boston.csv')
    ik = extract_not_missing(train)
    sparsity_aware(ik)
