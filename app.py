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
st.title("🚀 NASA Turbofan Jet Engine RUL Prediction (Using LSTM)")

# Hide Streamlit default menu/footer
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar Controls

st.sidebar.header("Model & Data Settings")

rul_cap = st.sidebar.slider("RUL Cap", 50, 200, 125, 5)
seq_len = st.sidebar.slider("Sequence Length", 20, 80, 50, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("LSTM Hyperparameters")

hidden_size = st.sidebar.slider("Hidden Size", 32, 256, 64, 32)
num_layers = st.sidebar.slider("LSTM Layers", 1, 3, 2)
epochs = st.sidebar.slider("Epochs", 3, 30, 10)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 64, 16)


#  Data Loading

st.header("1. Load Data")

uploaded = st.file_uploader("Upload a CMAPSS dataset (.txt)", type=["txt"])

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded, sep=r"\s+", header=None)
        if df.shape[1] != 26:
            raise ValueError("Dataset must contain exactly 26 columns.")
        df.columns = (
            ["unit", "cycle"]
            + [f"op_{i+1}" for i in range(3)]
            + [f"sensor_{i+1}" for i in range(21)]
        )
        st.success("Custom dataset loaded successfully.")
        is_test_data = "test" in uploaded.name.lower()
    else:
        st.info("Using default dataset: data/train_FD001.txt")
        df = load_cmapss("data/train_FD001.txt")
        is_test_data = False

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.dataframe(df.head())


#  RUL Calculation

st.header("2. Compute Remaining Useful Life (RUL)")
df = add_rul(df, cap=rul_cap)
st.dataframe(df[["unit", "cycle", "RUL"]].head())


# Exploratory Data Analysis

st.subheader("Exploratory Data Analysis")

if st.checkbox("Run Exploratory Data Analysis"):
    plot_rul_distribution(df)
    plot_sensor_trends(df, unit_id=1)
    summarize_dataset(df)
    st.success("EDA completed. Outputs saved to results/ folder.")


#  Create LSTM Sequences

st.header("3. Create LSTM Sequences")

with st.spinner("Creating sliding windows..."):
    X, y = create_sliding_windows(df, seq_len=seq_len)

st.write(f"X shape: {X.shape}")
st.write(f"y shape: {y.shape}")

if len(X) == 0:
    st.error("No sequences created. Reduce sequence length.")
    st.stop()


#  Train / Validation Split

st.header("4. Train / Validation Split (80 / 20)")

indices = np.random.permutation(len(X))
split = int(0.8 * len(X))

X_train, y_train = X[indices[:split]], y[indices[:split]]
X_val, y_val = X[indices[split:]], y[indices[split:]]

st.write(f"Training samples: {len(X_train)}")
st.write(f"Validation samples: {len(X_val)}")


#  Train & Evaluate Model

st.header("5. Train LSTM Model & Evaluate")

if st.button("Train LSTM Model"):
    input_size = X.shape[2]

    with st.spinner("Training LSTM model..."):
        model = RULPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            lr=1e-3,
        )
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        preds = model.predict(X_val)

    # Metrics
 
    if len(y_val) == 0 or np.isnan(y_val).any():
        st.warning(
            "This dataset does not contain per-cycle RUL labels. "
            "Predictions shown without RMSE/R²."
        )
    else:
        metrics = compute_metrics(y_val, preds)
        st.subheader("Evaluation Metrics (Validation Set)")
        st.write(f"**RMSE:** {metrics['rmse']:.2f} cycles")
        st.write(f"**R² Score:** {metrics['r2']:.3f}")

        # Plot
        
        st.subheader("Predicted vs True RUL")
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
    st.info("Click **Train LSTM Model** to begin training.")
