# 🏥 Healthcare Claims Analysis & Outcome Prediction

This project explores medical claims data to uncover cost drivers, denial patterns, and predict claim outcomes using machine learning. It combines real-world healthcare logic with statistical reasoning and classification modeling — a full-stack data science workflow.

---

## 📦 Dataset Overview

The project uses two key datasets:

- `claims_data.csv`: Contains 1000 health insurance claims with fields such as diagnosis code, procedure, billed and paid amounts, insurance type, claim status, and outcome.
- `icd10.csv`: ICD-10 code reference with descriptions, allowing us to map cryptic diagnosis codes to meaningful medical terms.

---

## 📊 Exploratory Data Analysis (EDA)

We explored:
- Most common diagnoses by frequency and cost
- Top reasons for claim denial
- Claim outcomes by insurance type
- Billed vs. paid discrepancies
- Denial patterns by provider and diagnosis

All diagnosis codes were joined with ICD-10 descriptions to enhance interpretability.

---

## 🧪 Hypothesis Testing

To add statistical rigor:
- ✅ **t-test**: Tested if average billed amount differed between *Denied* and *Paid* claims
- ✅ **Chi-squared test**: Assessed association between *Insurance Type* and *Outcome*

These tests helped move from simple observation to evidence-backed insight.

---

## 🤖 Predictive Modeling

Used machine learning to predict the **claim outcome** (`Paid`, `Denied`, `Partially Paid`):

- Preprocessed categorical variables with one-hot encoding
- Trained models:
  - `RandomForestClassifier`
  - `LogisticRegression (multinomial)`
- Evaluated with F1-score and confusion matrix

While accuracy was modest (~32%), the project revealed key limitations due to feature sparsity and label imbalance — ideal for showcasing real-world complexity.

---

## 🛠 Tech Stack

- Python (Pandas, Scikit-learn, SciPy)
- SQL (for hypothesis prototyping & logic)
- Jupyter Notebooks
- Visualizations (Seaborn, Matplotlib)

---

## 📌 Next Steps

- Deploy model with Streamlit for real-time predictions
- Add SHAP explainability
- Build a Power BI or Tableau dashboard
- Try advanced encodings or XGBoost

---
