"""
Exploratory Data Analysis (EDA) for NASA CMAPSS Dataset
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

def ensure_dirs():
    os.makedirs("results/plots", exist_ok=True)

def plot_rul_distribution(df):
    """Plot and save RUL distribution."""
    ensure_dirs()
    plt.figure(figsize=(6, 4))
    plt.hist(df["RUL"], bins=50, alpha=0.7)
    plt.xlabel("Remaining Useful Life (cycles)")
    plt.ylabel("Frequency")
    plt.title("RUL Distribution")
    plt.grid(True)
    plt.savefig("results/plots/rul_distribution.png", dpi=300)
    plt.close()


def plot_sensor_trends(df, unit_id=1, sensors=None):
    """Plot sensor trends for a single engine."""
    ensure_dirs()
    if sensors is None:
        sensors = ["sensor_1", "sensor_2", "sensor_3"]

    unit_df = df[df["unit"] == unit_id]

    plt.figure(figsize=(8, 4))
    for s in sensors:
        plt.plot(unit_df["cycle"], unit_df[s], label=s)

    plt.xlabel("Cycle")
    plt.ylabel("Sensor Value")
    plt.title(f"Sensor Trends for Engine {unit_id}")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/sensor_trends.png", dpi=300)
    plt.close()

def summarize_dataset(df):
    """Save dataset summary statistics."""
    ensure_dirs()
    summary = df.describe()

    with open("results/metrics.txt", "w") as f:
        f.write("===== DATASET SUMMARY =====\n\n")
        f.write(summary.to_string())


