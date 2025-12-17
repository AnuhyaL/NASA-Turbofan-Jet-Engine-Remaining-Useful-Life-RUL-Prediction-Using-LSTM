# NASA-Turbofan-Jet-Engine-Remaining-Useful-Life-RUL-Prediction-Using-LSTM

**Students Information**

Name: Anuhya Lanke, Drashti Sheta

Email: alanke@stevens.edu, dsheta@stevens.edu

# Project Overview

Predictive maintenance is a critical component of modern aerospace systems, enabling early fault detection, reduced downtime, and optimized maintenance costs. One of the most important predictive maintenance tasks is Remaining Useful Life (RUL) estimation, which predicts how many operational cycles remain before an engine failure occurs.

This project presents an end-to-end machine learning and deep learning solution for predicting the RUL of turbofan jet engines using the NASA C-MAPSS dataset. Both a traditional machine learning model (Random Forest Regressor) and a deep learning model (Long Short-Term Memory – LSTM) were implemented and evaluated. Based on performance comparison, the LSTM model was selected as the final solution due to its superior ability to capture temporal degradation patterns.

## ▶ Run the Application (No Download Required)

The Streamlit web application is deployed and can be accessed directly using the link below:

🔗 [https://nasa-turbofan-jet-engine.streamlit.app/]

No local installation or code download is required to run the project.


# Problem Description

The NASA C-MAPSS dataset contains multivariate time-series sensor readings collected from multiple turbofan engines operating under varying conditions. Each engine degrades over time until failure, but the degradation patterns are complex and non-linear.
**
Objective:** To predict the Remaining Useful Life (RUL) of an engine at any given operational cycle using historical sensor data.

# Key Challenges:

• Modeling long-term temporal dependencies

• Handling multivariate sensor data

• Learning degradation trends without explicit failure indicators

• Improving prediction accuracy over traditional ML methods

# Project Structure

NASA-RUL-LSTM/

├── app.py # Streamlit web application

├── data_processing.py # Data loading, RUL computation, preprocessing

├── models.py # LSTM model definition and training logic

├── utils.py # Evaluation metrics and visualization

├── README.md # Project documentation

All necessary files are included to run, evaluate, and interact with the project.

# Workflow & Methodology

# Data Loading

The project uses the NASA C-MAPSS datasets (FD001–FD004).

Each dataset contains:

• Engine unit ID

• Cycle number

• 3 operational settings

• 21 sensor measurements

Users can either:

• Use the default dataset, or

• Upload their own dataset through the Streamlit interface

# Remaining Useful Life (RUL) Calculation

**For each engine:**

• RUL is computed as the difference between the final failure cycle and the current cycle

• An optional RUL cap is applied to reduce extreme values and stabilize training

# Data Preprocessing & Feature Engineering

• Sensor and operational features are normalized using standard scaling

• Sliding-window sequences are created to transform time-series data into fixed-length sequences

• These sequences are used as inputs for the LSTM model

# Model Development

# Random Forest Regressor (Baseline Model)

A Random Forest regression model was implemented as a baseline to evaluate traditional machine learning performance. 

**Approach:**

• Extracted statistical features (mean, standard deviation) from sensor windows

• Used these handcrafted features for regression

**Observed Performance:**

• RMSE: ~35–40 cycles

• R² Score: ~0.20–0.30


**Limitations:**

• Does not explicitly model temporal dependencies

• Relies on handcrafted features

• Struggles with long-term degradation trends

• Significantly lower accuracy compared to sequence-based models

• Due to these limitations, Random Forest was not selected as the final model.

# LSTM Model (Final Model)

A multi-layer Long Short-Term Memory (LSTM) network was implemented to directly model sequential sensor behavior across engine cycles.

Since the C-MAPSS datasets differ in operating conditions and sensor behavior, model hyperparameters and data preprocessing settings such as sequence length, RUL cap, and normalization parameters were adjusted accordingly to ensure stable training and optimal performance.

Why LSTM?

• Captures long-term temporal dependencies

• Learns degradation patterns automatically

• Handles multivariate time-series data effectively

• Provides significantly improved prediction accuracy

# Model Training & Evaluation

**Training Configuration:**

• Loss Function: Mean Squared Error (MSE)

• Optimizer: Adam

**Tunable hyperparameters:**

• Sequence length

• Batch size

• Number of hidden units

• Number of epochs

**Evaluation Metrics:**

• RMSE (Root Mean Squared Error)

• R² Score (Coefficient of Determination)

# Results Summary

**Models**

Random Forest -RMSE (cycles) = ~35-40, R² Score = ~0.20-0.30

LSTM(Final)- RMSE (cycles) = 8.68, R² Score = 0.959

The LSTM model shows a substantial improvement over the baseline, confirming the importance of temporal modeling for RUL prediction.

# Streamlit Web Application

• An interactive Streamlit-based interface is included, allowing users to:

• Upload C-MAPSS datasets

• Configure RUL cap and sequence length

• Tune LSTM hyperparameters

• Train models interactively

• View evaluation metrics and prediction plots

# How to Run the Project

• git clone

• cd NASA-RUL-LSTM

• pip install -r requirements.txt

• streamlit run app.py

**(The dataset is in the .zip file so need to exact it before uploading it in the code)**

# RECOMMENDED SETTINGS FOR EACH DATASET

🔹 FD001 (Baseline / Simple case)

Files used

train_FD001.txt

test_FD001.txt

RUL_FD001.txt

Parameter,	Value

RUL Cap	- 125

Sequence Length	- 30–40

Epochs	- 10–15

Batch Size	- 64

Hidden Size	- 64

LSTM Layers	- 1–2

🔹 FD002 (Multiple operating conditions)

Files used

train_FD002.txt

test_FD002.txt

RUL_FD002.txt

Parameter,	Value

RUL Cap	- 130–150

Sequence Length	- 40–50

Epochs	- 15–20

Batch Size	- 64

Hidden Size	- 64–128

LSTM Layers	- 2

🔹 FD003  (MAIN DATASET – RECOMMENDED)

Files used

train_FD003.txt

test_FD003.txt

RUL_FD003.txt

Parameter, Value, Reason

RUL Cap - 125 - Standard in literature

Sequence Length	- 50 -	Captures degradation trend

Epochs	- 20–30 -	Harder faults

Batch Size	- 64	- Stable

Hidden Size	- 64 or 128	- Enough capacity

LSTM Layers	- 2	- Avoid overfitting

🔹 FD004 (Most difficult)

Files used

train_FD004.txt

test_FD004.txt

RUL_FD004.txt

Parameter,	Value

RUL Cap -	150

Sequence Length	- 60

Epochs	- 30

Batch Size	- 32–64

Hidden Size	- 128

LSTM Layers	- 2–3

# Contributions

Anuhya Lanke

1 - Added data loading & RUL calculation

2	- Implemented LSTM model

3	- Added Streamlit UI

4	- Added Exploratory Data Analysis

5	- Updated README and results

Drashti Sheta

1 - Improved Random Forest baseline

2 - Added model evaluation metrics

3 - Created prediction plots

4 - Final cleanup and testing

# Advanced Python & ML Features Used

• Object-Oriented Programming (custom model classes)

• Sliding-window sequence generation

• Exception handling using try–except

• Generator-style batch processing

**Libraries:**

• PyTorch

• Streamlit

• scikit-learn

• pandas

• NumPy

# Dataset

• NASA C-MAPSS Turbofan Engine Degradation Dataset

• Public benchmark dataset from NASA Prognostics Center of Excellence

# Results

The proposed LSTM model was evaluated on the NASA CMAPSS FD003 dataset using RMSE and R² metrics. The model achieved an RMSE of approximately 8–15 cycles and an R² score between 0.90 and 0.96, indicating strong predictive accuracy. The Predicted vs. True RUL plots show close alignment with the ideal diagonal, confirming effective learning of degradation trends. Compared to a Random Forest baseline (RMSE ≈ 35–40, R² ≈ 0.20–0.30), the LSTM model significantly outperforms by capturing temporal dependencies in sensor data.

# Conclusion

This project demonstrates a complete end-to-end predictive maintenance pipeline for turbofan engine Remaining Useful Life prediction. By comparing a traditional Random Forest baseline with a deep learning LSTM model, the study highlights the critical importance of temporal modeling in degradation analysis. The LSTM-based approach significantly outperforms the baseline, achieving high accuracy and robustness, making it suitable for real-world aerospace maintenance applications.

