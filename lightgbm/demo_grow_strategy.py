import re
import xgboost as xgb
import lightgbm as lgb
import streamlit as st
import pandas as pd
import numpy as np
from graphviz import Digraph
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

def train_xgboost(X, y):
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6)
    model.fit(X, y)
    return model

def train_lightgbm(X, y):
    model = lgb.LGBMRegressor(n_estimators=100, max_depth=6)
    model.fit(X, y)
    return model

def tree_to_digraph(graph, model_type):
    dot_source = graph.source
    G = Digraph()
    G.attr(bgcolor="black")

    G.attr("node", style="filled", fillcolor="darkblue", fontcolor="white", color="white")
    G.attr("edge", color="white")

    # グラフのサイズを設定 (幅、高さ)
    G.attr(size="6,4")  # 6インチ幅、4インチ高さ
    # グラフのアスペクト比を固定
    G.attr(ratio="fill")

    source_lines = str(dot_source).splitlines()
    # Remove 'digraph tree {'
    source_lines.pop(0)
    # Remove the closing brackets '}'
    source_lines.pop(-1)

    out = []
    if model_type == "xgboost":
        for line in source_lines:
            line = line.strip()
            line = re.sub(r'\[\s?label=".*?"\s?\]', "", line)
            out.append(line)
    elif model_type == "lightgbm":
        for line in source_lines:
            line = line.strip()
            line = re.sub(r'\[\s?label=.*?\]', "", line)
            out.append(line)

    G.body += out

    return G

def lgb_tree_to_digraph(model, tree_index):
    lgb_graph = lgb.create_tree_digraph(model.booster_, tree_index=tree_index, orientation='vertical')
    dot_source = lgb_graph.source

    dot_source = lgb_graph.source
    source_lines = str(dot_source).splitlines()
    # Remove 'digraph tree {'
    source_lines.pop(0)
    # Remove the closing brackets '}'
    source_lines.pop(-1)

    G = Digraph()
    G.attr(bgcolor="black")
    G.attr("node", style="filled", fillcolor="darkblue", fontcolor="white", color="white")
    G.attr("edge", color="white")
    
    # グラフのサイズを設定 (幅、高さ)
    G.attr(size="6,4")  # 6インチ幅、4インチ高さ
    # グラフのアスペクト比を固定
    G.attr(ratio="fill")

    out = []
    for line in source_lines:
        line = line.strip()
        line = re.sub(r'\[\s?label=.*?\]', "", line)
        out.append(line)

    G.body += out

    return G

def app():
    # データの読み込み
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="Species")

    # モデルの学習
    xgb_model = train_xgboost(X, y)
    lgb_model = train_lightgbm(X, y)

    # Streamlitウィジェットの追加
    st.title("Tree Visualization")
    st.sidebar.header("Settings")
    tree_index = st.sidebar.slider("Select a tree index", 0, 99, 27)
    # max_depth = st.sidebar.slider("Select max depth", 1, 10, 10)

    st.markdown("""
        <style>
            .graphviz_svg {
                margin-top: 50px !important;
                margin-bottom: 50px !important;
                margin-left: 50px !important;
                margin-right: 50px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.header("XGBoost Depth-wise Tree")
        xgb_graph = xgb.to_graphviz(xgb_model.get_booster(), num_trees=tree_index)
        xgb_tree = tree_to_digraph(xgb_graph, model_type="xgboost")
        st.graphviz_chart(xgb_tree)
    with col2:
        st.header("LightGBM Leaf-wise Tree")
        lgb_graph = lgb.create_tree_digraph(lgb_model.booster_, tree_index=tree_index, orientation='vertical')
        lgb_tree = tree_to_digraph(lgb_graph, model_type="lightgbm")
        st.graphviz_chart(lgb_tree)   

if __name__ == "__main__":
    app()
