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

No local installation or code download is required to run the project but dataset download is required.

**Instruction to download manually are given below as How to run the code** 


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

# How to run the code

**Instructions for Downloading, Running, and Verifying the Project** 

* Download the project by clicking Code → Download ZIP from the GitHub repository and extract it locally, or clone it using git clone.

* Ensure Python 3.12 or 3.13 is installed on the system before running the project.

* Install all required dependencies using pip install -r requirements.txt or manually install NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, Streamlit, and Pytest.

* Verify the project directory structure includes all required files: NASA_RUL_Main.ipynb, app.py, data_processing.py, models.py, utils.py, and the data/ and tests/ folders.

* Open the main Jupyter Notebook (NASA_RUL_Main.ipynb) and run all cells sequentially to execute the complete end-to-end workflow.

* Confirm that the notebook performs data loading, RUL computation, sliding-window sequence creation, LSTM training, and evaluation.

* Launch the interactive web interface using streamlit run app.py to visualize predictions and metrics.

* Run unit tests from the project root using python -m pytest to verify data loading and model functionality.

* Check the generated evaluation results in the results/metrics.txt file and observe prediction plots produced during execution.
  
* Or
  
*  after downloading the code exact it into a folder and open command prompt in it and type streamlit run app.py
  
* it will open streamlit GUI in the default browser then browse the file (dataset in train_FD001-FD004)
  
* then adjust the setting in the data according to the choice of dataset (recommendations given below) and click on the explanatory analysis and also click on the train the model
  
* It takes few minutes of time and give the plots

* when you do it manually you see the MSE for each epochs in the command prompt

**(The dataset is in the .zip file so need to exact it before uploading it in the code if downloading the code and doing manually)**

**The test datasets (FD001–FD004) do not contain per-cycle RUL labels. Therefore, RMSE and R² metrics are reported only on training/validation data. For test datasets, the model generates RUL predictions that can be compared against the provided RUL text files when available.**

**Required to test the dataset on only train_FD001-FD004 in the data**

**Note: The dataset is provided in .zip format as Dataset.zip and folder as data in the Github for direct access**

# Unit Testing (PyTest)

* This project uses PyTest to validate data processing and model functionality before full execution.

* The file test_data.py verifies the dataset loading and preprocessing pipeline.

* It tests successful loading of the NASA CMAPSS dataset using load_cmapss.

* It confirms correct computation of the Remaining Useful Life (RUL) using add_rul.

* It ensures required columns and dataset structure are handled properly.

* The file test_model.py validates the LSTM-based RUL prediction model.

* It tests model initialization, training execution, and prediction output shape.

* It also verifies the __str__ method for proper model description.

* All tests are executed from the project root using python -m pytest.

* Successful test execution confirms correctness of data handling and model behavior.
  

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



# Conclusion

This project demonstrates a complete end-to-end predictive maintenance pipeline for turbofan engine Remaining Useful Life prediction. By comparing a traditional Random Forest baseline with a deep learning LSTM model, the study highlights the critical importance of temporal modeling in degradation analysis. The LSTM-based approach significantly outperforms the baseline, achieving high accuracy and robustness, making it suitable for real-world aerospace maintenance applications.

