# FraudLens: AI-Based Financial Transaction Anomaly Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Model-Isolation%20Forest-orange.svg)](https://scikit-learn.org/)
[![IBM SkillsBuild](https://img.shields.io/badge/Internship-IBM%20SkillsBuild-purple.svg)](https://skillsbuild.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Academic Internship Project** for **IBM SkillsBuild Internship (September 2026)**  
> **Author:** Shauriya Garg (B.Tech Computer Science & Engineering - Data Science, Batch 2028)  
> **GitHub Repository:** [https://github.com/ShauriyaDeveloper1/IBMSkillBuild-Internship](https://github.com/ShauriyaDeveloper1/IBMSkillBuild-Internship)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Project Description](#-project-description)
- [Dataset & Link](#-dataset--link)
- [Technologies Used](#-technologies-used)
- [Key Features & Architectural Decisions](#-key-features--architectural-decisions)
- [Domain Feature Engineering](#-domain-feature-engineering)
- [Model Architecture & Methodology](#-model-architecture--methodology)
- [Performance & Evaluation Results](#-performance--evaluation-results)
- [Project Structure](#-project-structure)
- [Setup & Run Instructions](#-setup--run-instructions)
- [Outputs & Exported Reports](#-outputs--exported-reports)
- [Future Enhancements](#-future-enhancements)

---

## 🔍 Project Overview

Financial fraud represents billions of dollars in annual losses worldwide. Fraudulent transactions occur infrequently (typically under 0.15% of all events), evolve dynamically, and frequently disguise themselves as routine transactions. 

**FraudLens** is an AI-powered financial anomaly detection system engineered to detect suspicious transactions in massive transaction streams without requiring pre-labeled training data. Using an **unsupervised Isolation Forest** model, FraudLens isolates behavioral and financial balance discrepancies, assigns each transaction an **Anomaly Score**, and prioritizes high-risk operations for forensic compliance review.

---

## 📖 Project Description

Traditional fraud detection systems rely heavily on supervised classification or rigid, rule-based heuristics. However:
1. **Rule-based systems** are brittle, require manual updates, and generate high false-alarm rates.
2. **Supervised classifiers** often overfit to historical fraud patterns and fail to detect novel, zero-day fraud schemes.
3. **Severe Class Imbalance** (~99.87% normal vs. ~0.13% fraud) leads supervised models to bias strongly toward the majority class.

**FraudLens** circumvents these challenges by:
- Operating in a **strictly unsupervised regime**: The ground-truth fraud label (`isFraud`) is **never** shown to the model during training.
- Learning the latent geometric distribution of legitimate transactions.
- Identifying transactions that isolate easily through randomized recursive partitioning (the core principle of tree-based isolation).
- Exporting the top suspicious records to a dedicated CSV file for forensic investigator review.

---

## 📊 Dataset & Link

This project utilizes the **PaySim Synthetic Financial Dataset**, which simulates mobile money transactions based on actual financial transaction logs from an African financial service.

- 🔗 **Official Dataset Link:** [Kaggle - PaySim Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **File Name:** `PS_20174392719_1491204439457_log.csv` (or `.zip`)
- **Total Transactions:** 6,362,620 rows across 11 attributes
- **Cleaned Dataset:** 6,362,604 valid rows after removing duplicates and invalid records
- **Transaction Types:** `CASH_OUT`, `PAYMENT`, `CASH_IN`, `TRANSFER`, `DEBIT`
- **Class Distribution:**
  - Legitimate Transactions: **6,354,407** (99.8712%)
  - Fraudulent Transactions: **8,197** (0.1288%)
- **Evaluated Sample Size:** 300,000 transactions for optimal balance of speed and representation.

---

## 🛠️ Technologies Used

| Technology / Library | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.10+` | Core programming language |
| **Pandas** | `>=2.0.0` | High-performance data ingestion, cleaning, and tabular transformations |
| **NumPy** | `>=1.24.0` | Vectorized numerical operations and ratio computations |
| **Scikit-Learn** | `>=1.3.0` | Isolation Forest model implementation and evaluation metrics |
| **Matplotlib** | `>=3.7.0` | Plot generation, histograms, and figure exports |
| **Seaborn** | `>=0.12.0` | Statistical data visualization, scatter plots, and confusion matrix heatmaps |
| **Jupyter Notebook** | `>=7.0.0` | Interactive execution environment and pipeline development |

---

## 🚀 Key Features & Architectural Decisions

1. **Time-Invariant Modeling (`step` dropped):**  
   The synthetic time index column (`step`) was deliberately omitted. Fraud should be identified based on **transactional anomalies and balance integrity**, rather than an arbitrary simulation timestamp.
2. **Balance Error Accounting:**  
   Calculates mathematical balance discrepancies where money disappeared from sender accounts without reaching recipient accounts—a hallmark of fraud.
3. **High-Speed Scalability:**  
   Trains an ensemble of 100 isolation trees on 300,000 transactions in under **2 seconds**, enabling near-real-time batch inference.
4. **Interactive Single-Transaction Prediction:**  
   Includes a reusable scoring function allowing compliance officers to input a single transaction payload and immediately obtain an anomaly status and numerical anomaly score.

---

## ⚙️ Domain Feature Engineering

FraudLens derives 7 domain-specific features to expose deceptive transaction behavior:

| Engineered Feature | Mathematical Formula | Financial Rationale |
| :--- | :--- | :--- |
| **`origin_balance_change`** | $\| \text{oldbalanceOrg} - \text{newbalanceOrig} \|$ | Tracks the absolute amount drained from the sender's account. |
| **`destination_balance_change`** | $\| \text{newbalanceDest} - \text{oldbalanceDest} \|$ | Tracks the absolute amount credited to the recipient's account. |
| **`amount_origin_ratio`** | $\frac{\text{amount}}{\text{oldbalanceOrg} + 1}$ | Identifies full-balance draining (ratio $\approx 1$) vs. standard transactions. |
| **`amount_destination_ratio`** | $\frac{\text{amount}}{\text{oldbalanceDest} + 1}$ | Highlights sudden large deposits into historically dormant recipient accounts. |
| **`log_amount`** | $\ln(1 + \text{amount})$ | Compresses heavy-tailed distribution of transaction values. |
| **`origin_balance_error`** | $\text{newbalanceOrig} - (\text{oldbalanceOrg} - \text{amount})$ | Detects discrepancies between reported balance changes and actual transferred amounts. |
| **`destination_balance_error`** | $\text{newbalanceDest} - (\text{oldbalanceDest} + \text{amount})$ | Identifies uncredited recipient balances and phantom transfers. |

---

## 🧠 Model Architecture & Methodology

- **Algorithm:** Isolation Forest (`sklearn.ensemble.IsolationForest`)
- **Number of Estimators:** `100` trees
- **Contamination Rate:** `0.01` (Top 1.0% highest-risk transactions flagged)
- **Random State:** `42` (ensuring 100% reproducibility)
- **Parallelization:** `n_jobs=-1` (utilizes all available CPU cores)
- **Scoring Protocol:**  
  - Negative values signify anomalous, outlying behavior (shorter average tree path length).
  - Positive values signify typical, dense clusters of normal transactions.

---

## 📈 Performance & Evaluation Results

While training was strictly unsupervised, performance was validated against the ground-truth `isFraud` column on the 300,000-transaction sample:

### Classification Metrics

```
              precision    recall  f1-score   support

      Normal       1.00      0.99      0.99    299633
       Fraud       0.03      0.25      0.05       367

    accuracy                           0.99    300000
   macro avg       0.51      0.62      0.52    300000
weighted avg       1.00      0.99      0.99    300000
```

### Key Highlights
- **25% Recall on Unseen Fraud:** Without seeing a single fraud label during training, the model successfully isolated **25% of all actual fraud occurrences** inside just the top 1% flagged subset.
- **Top Suspicious Category:** The most severe anomalies identified were massive transfers ($\ge \$10,000,000$) originating from accounts with zero balance or accounts that were completely emptied in a single transaction.

---

## 📁 Project Structure

```
.
├── .gitignore                             # Excludes large CSV dataset (>100MB) & cache
├── README.md                              # Comprehensive project documentation
├── requirements.txt                       # Python dependencies with version constraints
├── ShauriyaGarg_FraudLens.ipynb           # End-to-end pipeline notebook (23 structured cells)
├── ShauriyaGarg_ProjectReport.docx         # Formal internship project report
├── flagged_transactions.csv                # Exported forensic report of flagged anomalies
└── PS_20174392719_1491204439457_log.csv     # PaySim dataset (download from Kaggle link above)
```

---

## 💻 Setup & Run Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ShauriyaDeveloper1/IBMSkillBuild-Internship.git
cd IBMSkillBuild-Internship
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
1. Download the PaySim dataset from [Kaggle PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1).
2. Extract the archive into the project directory as `PS_20174392719_1491204439457_log.csv`.

### 5. Launch and Run the Pipeline
Open the notebook in Jupyter Notebook or VS Code:
```bash
jupyter notebook ShauriyaGarg_FraudLens.ipynb
```
Select **Run All** to execute all 23 pipeline stages sequentially:
1. Data loading and initial inspection.
2. Removal of temporal columns and duplicate cleaning.
3. Exploratory Data Analysis (EDA) visualizations.
4. Domain feature engineering.
5. Model training on 300,000 sampled transactions.
6. Anomaly score generation and thresholding.
7. Model evaluation against ground truth.
8. Exporting flagged records to `flagged_transactions.csv`.
9. Interactive sample prediction.

---

## 📄 Outputs & Exported Reports

- **`flagged_transactions.csv`**: Contains the top 3,000 transactions flagged for manual review, complete with:
  - Account IDs (`nameOrig`, `nameDest`)
  - Transaction parameters (`type`, `amount`, balances)
  - Engineered metrics (`amount_origin_ratio`, balance changes)
  - `ANOMALY_SCORE`: Quantified degree of abnormality
  - `STATUS`: Actionable label (`Flagged for Review`)

---

## 🔮 Future Enhancements

- **Autoencoder Integration:** Benchmark deep autoencoder neural networks against Isolation Forest.
- **Graph Neural Networks (GNNs):** Model account relationship networks (`nameOrig` $\rightarrow$ `nameDest`) to uncover organized fraud rings.
- **Live Stream Processing:** Deploy model inference using FastAPI / Apache Kafka for real-time transaction screening.

---

## 👤 Author & Acknowledgments

- **Author:** Shauriya Garg  
- **Batch:** B.Tech Computer Science Engineering (Data Science) - 2028  
- **Internship Program:** IBM SkillsBuild Internship  
- **Dataset Credits:** PaySim Synthetic Financial Datasets (Edgar Lopez-Rojas)
