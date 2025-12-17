import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from exploratory_analysis import (
    plot_rul_distribution,
    plot_sensor_trends,
    summarize_dataset,
)

from data_processing import load_cmapss, add_rul, create_sliding_windows
from models import RULPredictor
from utils import compute_metrics

# Streamlit Page Config 
st.set_page_config(page_title="NASA Turbofan RUL - LSTM", layout="wide")
st.title("🚀 NASA Turbofan Jet Engine RUL Prediction (using LSTM)")


# Streamlit default menu & footer for a cleaner UI
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: visible;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#  Sidebar Controls 
st.sidebar.header("Model & Data Settings")

rul_cap = st.sidebar.slider("RUL Cap", min_value=50, max_value=200, value=125, step=5)
seq_len = st.sidebar.slider("Sequence Length (time steps)", min_value=20, max_value=80, value=50, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("LSTM Hyperparameters")

hidden_size = st.sidebar.slider("Hidden Size", min_value=32, max_value=256, value=64, step=32)
num_layers = st.sidebar.slider("Number of LSTM Layers", min_value=1, max_value=3, value=2, step=1)
epochs = st.sidebar.slider("Training Epochs", min_value=3, max_value=30, value=10, step=1)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=64, step=16)

st.sidebar.markdown("---")
st.sidebar.caption("Model: LSTM regressor for Remaining Useful Life (RUL).")


#  Data Loading 
st.write(" 1. Load Data")

uploaded = st.file_uploader("Upload a CMAPSS train file (.txt)", type=["txt"])

try:
    if uploaded is not None:
        # Safely read uploaded file
        df = pd.read_csv(uploaded, sep=r"\s+", header=None)

        if df.shape[1] != 26:
            raise ValueError(
                f"Unexpected number of columns: {df.shape[1]} (expected 26)."
            )

        df.columns = ["unit", "cycle"] + [f"op_{i+1}" for i in range(3)] + [
            f"sensor_{i+1}" for i in range(21)
        ]
        st.success("Custom training file uploaded successfully.")

    else:
        st.info("Using default dataset: data/train_FD001.txt")
        df = load_cmapss("data/train_FD001.txt")

except FileNotFoundError:
    st.error("Dataset file not found. Please check the file path.")
    st.stop()

except ValueError as ve:
    st.error(f"Invalid dataset format: {ve}")
    st.stop()

except Exception as e:
    st.error(f"Unexpected error while loading data: {e}")
    st.stop()

st.write("Data preview:")
st.dataframe(df.head())




#  RUL Calculation 
st.write(" 2. Compute Remaining Useful Life (RUL)")
df = add_rul(df, cap=rul_cap)

#  Exploratory Data Analysis 
st.write("### Exploratory Data Analysis")

if st.checkbox("Run Exploratory Data Analysis"):
    plot_rul_distribution(df)
    plot_sensor_trends(df, unit_id=1)
    summarize_dataset(df)

    st.success("EDA completed. Plots and summaries saved to /results folder.")

st.write("Added 'RUL' column (capped at", rul_cap, ").")
st.write(df[["unit", "cycle", "RUL"]].head())



#  Create LSTM Sequences 
st.write(" 3. Create LSTM Sequences")

with st.spinner(f"Creating sliding windows (sequence length = {seq_len})..."):
    X, y = create_sliding_windows(df, seq_len=seq_len)

st.write(f"Sequences shape: **X = {X.shape}**, Targets shape: **y = {y.shape}**")

if X.shape[0] == 0:
    st.error("No sequences were created. Try reducing the sequence length.")
    st.stop()


#  Train / Validation Split 
st.write(" 4. Train / Validation Split (80% / 20%)")

num_samples = X.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

split_idx = int(0.8 * num_samples)
train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

st.write(f"Training samples: **{X_train.shape[0]}**, Validation samples: **{X_val.shape[0]}**")


#  Train LSTM Button 
st.write(" 5. Train LSTM Model & Evaluate")

if st.button("Train LSTM Model"):
    input_size = X.shape[2]

    with st.spinner("Training LSTM model... (check terminal for epoch logs)"):
        model = RULPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            lr=1e-3,
        )
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # Predict on validation set
        preds = model.predict(X_val)

    #  Metrics 
    metrics = compute_metrics(y_val, preds)
    st.subheader("Evaluation Metrics (Validation Set)")
    st.write(f"**RMSE:** {metrics['rmse']:.2f} cycles")
    st.write(f"**R² Score:** {metrics['r2']:.3f}")

    # Plot 
    st.subheader("Predicted vs. True RUL (Validation Set)")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_val, preds, alpha=0.4)
    max_val = max(np.max(y_val), np.max(preds))
    ax.plot([0, max_val], [0, max_val], "r--")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Predicted vs True RUL (LSTM)")
    ax.grid(True)
    fig.savefig("results/plots/predicted_vs_true.png", dpi=300)
    st.pyplot(fig)
else:
    st.info("Click **Train LSTM Model** to start training and see evaluation results.")
