# FraudLens: AI-Based Financial Transaction Anomaly Detection

FraudLens is an unsupervised machine learning pipeline designed to detect fraudulent and anomalous financial transactions in large-scale banking and mobile money logs. Powered by **Isolation Forest**, the model isolates suspicious behaviors without relying on labeled fraud cases during training.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset Summary](#dataset-summary)
- [Feature Engineering](#feature-engineering)
- [Model Architecture & Methodology](#model-architecture--methodology)
- [Performance & Results](#performance--results)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage & Execution](#usage--execution)
- [Outputs & Exported Reports](#outputs--exported-reports)

---

## 🔍 Overview

Financial transaction fraud is rare, fast-evolving, and heavily imbalanced (typically < 0.15% of all transactions). Traditional supervised classifiers often overfit to known fraud patterns and fail to detect novel fraud tactics.

**FraudLens** addresses this by leveraging an **unsupervised anomaly detection approach**:
- Learns the structural topology of normal financial behavior.
- Isolates transactions with irregular amounts, unexpected balance discrepancies, or abnormal balance-to-transaction ratios.
- Ranks transactions with an **Anomaly Score** to prioritize high-risk activities for compliance and fraud operations teams.

---

## 🚀 Key Features

- **Time-Independent Anomaly Detection**: Intentionally removes temporal counters (`step`) to focus on structural and behavioral transaction discrepancies rather than historical timing quirks.
- **Financial Balance Accounting Features**: Computes origin and destination balance errors to catch immediate zero-balance drains and uncredited inflows.
- **Fast Training & Scalability**: Trains on 300,000 transactions in under 2 seconds while scaling inference across millions of records.
- **Automated Forensic Export**: Automatically exports flagged high-risk transactions with their corresponding anomaly scores to `flagged_transactions.csv`.
- **Single-Transaction Real-Time Inference**: Interactive inference function to evaluate incoming transaction payloads on demand.

---

## 📊 Dataset Summary

The system is evaluated on the **PaySim** mobile money simulation dataset:

| Attribute | Details |
| :--- | :--- |
| **Total Transactions** | 6,362,620 rows |
| **Cleaned Dataset** | 6,362,604 rows (duplicates & invalid entries handled) |
| **Transaction Types** | `CASH_OUT`, `PAYMENT`, `CASH_IN`, `TRANSFER`, `DEBIT` |
| **Actual Fraud Count** | 8,197 transactions (~0.1288%) |
| **Evaluated Sample** | 300,000 transactions |

---

## 🛠️ Feature Engineering

FraudLens constructs 7 domain-specific features in addition to raw balance amounts:

1. **`origin_balance_change`**: Absolute difference between sender's balance before and after (`|oldbalanceOrg - newbalanceOrig|`).
2. **`destination_balance_change`**: Absolute difference between recipient's balance before and after (`|newbalanceDest - oldbalanceDest|`).
3. **`amount_origin_ratio`**: Ratio of transferred amount relative to the sender's pre-transaction balance.
4. **`amount_destination_ratio`**: Ratio of transferred amount relative to the recipient's pre-transaction balance.
5. **`log_amount`**: Log-transformed transaction amount (`log1p`) to compress extreme wealth distributions.
6. **`origin_balance_error`**: Divergence from expected balance calculation:  
   $$\text{Error}_{\text{orig}} = \text{newbalanceOrig} - (\text{oldbalanceOrg} - \text{amount})$$
7. **`destination_balance_error`**: Divergence from expected destination balance calculation:  
   $$\text{Error}_{\text{dest}} = \text{newbalanceDest} - (\text{oldbalanceDest} + \text{amount})$$

---

## 🧠 Model Architecture & Methodology

- **Algorithm**: `sklearn.ensemble.IsolationForest`
- **Contamination Rate**: `0.01` (1% prioritized anomaly threshold)
- **Estimators (`n_estimators`)**: `100`
- **Unsupervised Paradigm**: The ground truth label `isFraud` is **strictly excluded** during model training and feature compilation. It is used solely at the end of the pipeline to evaluate precision and recall against historical audits.

---

## 📈 Performance & Results

### Sample Detection Results (300,000 Transactions)

- **Total Assessed**: 300,000
- **Normal Transactions**: 297,000 (99.0%)
- **Flagged Anomalies**: 3,000 (1.0%)
- **Actual Fraud in Sample**: 367 transactions

### Metrics Against Ground Truth

```
              precision    recall  f1-score   support

      Normal       1.00      0.99      0.99    299633
       Fraud       0.03      0.25      0.05       367

    accuracy                           0.99    300000
   macro avg       0.51      0.62      0.52    300000
weighted avg       1.00      0.99      0.99    300000
```

> **Note**: In an unsupervised setting without label supervision, capturing **25% of actual frauds** in just the top 1% flagged transactions demonstrates high sensitivity to severe outliers and zero-balance fraud schemes.

---

## 📁 Project Structure

```
.
├── PS_20174392719_1491204439457_log.csv      # PaySim dataset (extracted)
├── PS_20174392719_1491204439457_log.csv.zip  # Compressed dataset archive
├── flagged_transactions.csv                 # Output file with flagged anomalies
├── ShauriyaGarg_FraudLens.ipynb            # Main end-to-end pipeline notebook
├── ShauriyaGarg_ProjectReport.docx          # Project report document
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open Workspace
Ensure your terminal or command prompt is pointed to the project root:
```bash
cd /path/to/Internship
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Dataset Preparation
If `PS_20174392719_1491204439457_log.csv` is still compressed in `.zip` format:
```bash
# Python one-liner to extract
python -c "import zipfile; zipfile.ZipFile('PS_20174392719_1491204439457_log.csv.zip').extractall('.')"
```

---

## 💻 Usage & Execution

### Running the Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook ShauriyaGarg_FraudLens.ipynb
```
Execute the cells in sequence to:
1. Load and inspect the dataset.
2. Clean duplicate and invalid records.
3. Compute engineered financial features.
4. Train the Isolation Forest model.
5. Generate anomaly scores and evaluation reports.
6. Export flagged records to `flagged_transactions.csv`.
7. Test sample transactions with real-time scoring.

---

## 📄 Outputs & Exported Reports

- **`flagged_transactions.csv`**: Contains transactions identified as suspicious, including:
  - Account IDs (`nameOrig`, `nameDest`)
  - Transaction characteristics (`type`, `amount`, balances)
  - Engineered metrics (`amount_origin_ratio`, balance changes)
  - Computed `ANOMALY_SCORE` (lower scores indicate higher irregularity)
  - `STATUS`: Flagged for review
