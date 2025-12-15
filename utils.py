import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import math


def compute_metrics(y_true, y_pred):
    """
    Compute RMSE and R^2 for regression.
    """
    # mean_squared_error returns MSE; take sqrt for RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {"rmse": float(rmse), "r2": float(r2)}


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
