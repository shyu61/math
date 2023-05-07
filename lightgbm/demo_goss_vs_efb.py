import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt
import matplotlib.style as style

st.set_page_config(layout="wide")
style.use('dark_background')

from goss import SimpleGOSS
from gbdt import SimpleGBDT
from efb import SimpleEFB

data = pd.read_csv("./data/boston.csv")
X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.title("GBDT vs GOSS vs EFB Demo")

st.write("### Dataset")
st.write(data.head())
st.write(f"Number of rows: {data.shape[0]}")

results = {"GBDT": {}, "GOSS": {}, "EFB": {}}
trained = False

@st.cache_data
def cached_model():
    return results

@st.cache_data
def component(algorithm: str):
    model = None
    if algorithm == "GBDT":
        model = SimpleGBDT(n_estimators=70, learning_rate=0.1, max_depth=7, random_state=42)
    elif algorithm == "GOSS":
        model = SimpleGOSS(n_trees=70, learning_rate=0.1, a=0.2, b=0.4, max_depth=7, random_state=42)
    elif algorithm == "EFB":
        model = SimpleEFB(n_trees=700, learning_rate=0.002, max_depth=9, random_state=42, max_bin=4)

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 保存する結果を更新
    results[algorithm] = {"mse": mse, "elapsed_time": elapsed_time, "model": model}

    if algorithm == "GOSS":
        data_reduction = 1 - (model.a + model.b)
        results[algorithm]["data_reduction"] = data_reduction
    elif algorithm == "GBDT":
        results[algorithm]["data_reduction"] = 1
    elif algorithm == "EFB":
        results[algorithm]["data_reduction"] = 1

    results[algorithm]["costs"] = model.costs

    st.write(f"### {algorithm} Results")
    st.write(f"MSE: {results[algorithm]['mse']:.4f}")
    st.write(f"Training time: {results[algorithm]['elapsed_time']:.2f} seconds")

    st.write(f"Data reduction: {results[algorithm]['data_reduction'] * 100:.2f}%")

    fig, ax = plt.subplots()
    ax.plot(results[algorithm]["costs"])
    ax.set_title("Cost Function")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    st.pyplot(fig)


@st.cache_data
def plot_grads():
    st.title("GOSS gradients in all iters")
    fig2 = results["GOSS"]["model"].plot_grads()
    st.pyplot(fig2)


col1, col2, col3 = st.columns(3)

with col1:
    component("GBDT")
with col2:
    component("GOSS")
with col3:
    component("EFB")

cached_model()

tree_index = st.slider("Select a tree index", 0, 69, 0)

results = cached_model()
if "model" in results["GOSS"]:
    col1, col2 = st.columns(2)

    with col1:
        st.title(f"GOSS gradients with iter {tree_index}")
        fig1 = results["GOSS"]["model"].plot_grad_with_iter(tree_index)
        st.pyplot(fig1)
    with col2:
        plot_grads()
