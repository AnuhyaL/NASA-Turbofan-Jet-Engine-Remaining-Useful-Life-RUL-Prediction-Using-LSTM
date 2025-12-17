import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import math
import os


def compute_metrics(y_true, y_pred):
    """Compute and save evaluation metrics."""
    os.makedirs("results", exist_ok=True)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    with open("results/metrics.txt", "a") as f:
        f.write("\n\n===== MODEL EVALUATION =====\n")
        f.write(f"RMSE: {rmse:.2f} cycles\n")
        f.write(f"R2 Score: {r2:.3f}\n")

    return {"rmse": rmse, "r2": r2}



def batch_generator(X, y, batch_size):
    """
    Generator that yields batches of data.
    """
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]


def plot_pred_vs_true(y_true, y_pred, title="Predicted vs True RUL"):
    """
    Scatter plot comparing predicted and true RUL with y=x reference line.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([0, max_val], [0, max_val], "r--")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
