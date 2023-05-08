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

st.title("GBDT, GBDT_GOSS, GBDT_EFB Demo")

st.write("### Dataset")
st.write(data.head())
st.write(f"Number of rows: {data.shape[0]}")

results = {"GBDT": {}, "GBDT_GOSS": {}, "GBDT_EFB": {}}
trained = False

@st.cache_data
def cached_model():
    return results

@st.cache_data
def component(algorithm: str):
    model = None
    n_iter = 0
    if algorithm == "GBDT":
        n_iter = 300
        model = SimpleGBDT(n_estimators=n_iter, learning_rate=0.005, max_depth=5, random_state=42, max_bin=20)
    elif algorithm == "GBDT_GOSS":
        n_iter = 50
        model = SimpleGOSS(n_trees=n_iter, learning_rate=0.05, a=0.25, b=0.1, max_depth=8, random_state=42, max_bin=20)
    elif algorithm == "GBDT_EFB":
        n_iter = 200
        model = SimpleEFB(n_trees=n_iter, learning_rate=0.01, max_depth=7, random_state=42, max_bin=20)

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 保存する結果を更新
    results[algorithm] = {"mse": mse, "elapsed_time": elapsed_time, "model": model, "elapsed_time_per_iter": elapsed_time / n_iter}

    if algorithm == "GBDT_GOSS":
        data_reduction = 1 - (model.a + (1 -  model.a) * model.b)
        results[algorithm]["data_reduction"] = f"{data_reduction * 100:.2f}%"
    elif algorithm == "GBDT":
        results[algorithm]["data_reduction"] = f"{1 * 100:.2f}%"
    elif algorithm == "GBDT_EFB":
        results[algorithm]["data_reduction"] = f"Original feats: {X.shape[1]}, bundled feats: {len(model.bundles.keys())}"

    results[algorithm]["costs"] = model.costs

    st.write(f"### {algorithm} Results")
    st.write(f"MSE: {results[algorithm]['mse']:.4f}")
    # st.write(f"Training time: {results[algorithm]['elapsed_time']:.2f} seconds")
    st.write(f"Training time per iteration: {results[algorithm]['elapsed_time_per_iter']:.3e} seconds")

    st.write(f"Training Dataset percent: {results[algorithm]['data_reduction']}")

    fig, ax = plt.subplots()
    ax.plot(results[algorithm]["costs"])
    ax.set_title("Cost Function")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    st.pyplot(fig)


@st.cache_data
def plot_grads():
    st.title("GBDT_GOSS grads in all iters")
    fig2 = results["GBDT_GOSS"]["model"].plot_grads()
    st.pyplot(fig2)


col1, col2, col3 = st.columns(3)

with col1:
    component("GBDT")
with col2:
    component("GBDT_GOSS")
with col3:
    component("GBDT_EFB")

cached_model()

# tree_index = st.slider("Select a tree index", 0, 69, 0)
tree_index = st.selectbox("Select a tree index", options=list(range(70)), index=0)

results = cached_model()
if "model" in results["GBDT_GOSS"]:
    col1, col2 = st.columns(2)

    with col1:
        st.title(f"GBDT_GOSS grads with iter {tree_index}")
        fig1 = results["GBDT_GOSS"]["model"].plot_grad_with_iter(tree_index)
        st.pyplot(fig1)
    with col2:
        plot_grads()
