# Predictive_Maintainance

# Predictive Maintenance using Machine Learning

## Project Overview

This project focuses on predicting machine failures using machine learning techniques on industrial IoT sensor data. The objective is to identify potential failures before they occur, enabling proactive maintenance and reducing downtime.

The project includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Handling Class Imbalance using SMOTE
- Model Training and Evaluation
- Ablation Study
- Cross-Validation using Stratified K-Fold
- LightGBM and Random Forest Models

---

## Dataset

Dataset: AI4I 2020 Predictive Maintenance Dataset

Features include:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type

Target Variable:

- **Machine Failure**
  - 0 → No Failure
  - 1 → Failure

---

## Project Workflow

### Week 1

#### Data Preprocessing

- Checked missing values
- Checked duplicate records
- Removed unnecessary columns:
  - UDI
  - Product ID
  - Failure Type Columns (TWF, HDF, PWF, OSF, RNF)

#### Feature Encoding

- Converted Machine Type into numerical values using LabelEncoder

#### Exploratory Data Analysis

- Failure distribution analysis
- Correlation matrix
- Histograms
- Outlier detection

#### Outlier Treatment

- Applied IQR Method to reduce the effect of extreme values

---

### Week 2

#### Time-Series Feature Engineering

Created:

- Time Index
- Rolling Mean Features
- Rolling Standard Deviation Features
- Rolling Variance Features
- Lag Features
- Sensor Change Features

#### External Context Simulation

Generated additional operational features:

- Ambient Temperature
- Humidity
- Load Density
- Voltage

#### Interaction Features

Created:

- Temperature Difference
- Torque × Tool Wear
- Speed × Torque
- Ambient Process Gap

---

## Ablation Study

Four feature groups were evaluated:

| Model | Features |
|---------|----------|
| A | Internal Sensor Features |
| B | Internal + External Features |
| C | Internal + External + Interaction Features |
| D | Full Feature Set |

Evaluation Metric:

- Macro F1 Score

---

## Week 3

### Handling Class Imbalance

Applied:

- SMOTE (Synthetic Minority Oversampling Technique)

### Cross Validation

Used:

- Stratified 5-Fold Cross Validation

Benefits:

- Preserves class distribution
- Reduces overfitting risk
- Produces reliable model evaluation

---

## Machine Learning Models

### Random Forest Classifier

Used for:

- Baseline Model
- Ablation Study

### LightGBM Classifier

Parameters:

```python
LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42
)