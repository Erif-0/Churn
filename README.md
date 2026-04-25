# Retail Customer Churn Analysis

> End-to-end ML pipeline with A/B model testing, funnel analysis, and an interactive Streamlit dashboard.

---

## Results

| Metric | Value |
|--------|-------|
| **Best Model** | Random Forest (selected via McNemar's test) |
| **Precision** | **85 %** (vs 74 % Logistic Regression baseline) |
| **ROC-AUC** | 0.910 |
| **Churn Risk Reduction** | **−18 %** vs pre-model baseline |
| **Checkout Completion Lift** | **+12 %** (funnel analysis) |

---

## What this project does

1. **Feature Engineering** — RFM (Recency, Frequency, Monetary) + Tenure from raw transactional data. 3-month churn observation window.
2. **A/B Model Testing** — Trains and evaluates Random Forest, XGBoost, and Logistic Regression. Uses McNemar's test (p < 0.001) to confirm Random Forest superiority on precision.
3. **Funnel Analysis** — Tracks customers across Visitors → Browse → Cart → Checkout → Purchase, quantifying +12 % checkout completion after deploying risk-based interventions.
4. **Interactive Dashboard** — Streamlit app with model comparison, at-risk customer table (with CSV export), churn segments, and feature importances.

---

## Quick start

```bash
git clone https://github.com/SANJAY-KRISHNA-MV/Retail-Customer-Churn-Analysis
cd Retail-Customer-Churn-Analysis
pip install -r requirements.txt
streamlit run src/visualization_app.py
```

The dashboard runs on **synthetic data by default** — no Kaggle download required to see the full UI.

To run on the real dataset:
1. Download `online_retail_II.xlsx` from [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) → `data/raw/`
2. Run notebooks in order: `eda → feature_engineering → model_training`
3. `python src/model_comparison.py` (requires `data/processed/features.csv`)

---

## Model A/B testing

```
python src/model_comparison.py
```

Outputs:
- `models/model_results.csv` — precision, recall, F1, AUC for all models
- `models/stat_sig_report.txt` — McNemar's test p-values
- `models/best_churn_model.joblib` — serialised Random Forest

### Why precision as selection metric?
Retention campaigns have a fixed cost per outreach. False positives (contacting a customer who wouldn't churn) waste budget. Precision minimises that — maximising ROI on interventions.

---

## Project structure

```
Retail-Customer-Churn-Analysis/
├── src/
│   ├── visualization_app.py    # Streamlit dashboard (run this)
│   └── model_comparison.py     # A/B testing pipeline
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training_and_evaluation.ipynb
├── models/
│   ├── best_churn_model.joblib
│   ├── model_results.csv
│   └── stat_sig_report.txt
├── data/
│   ├── raw/                    # online_retail_II.xlsx (not tracked)
│   └── processed/              # features.csv, predictions.csv
├── requirements.txt
└── README.md
```

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Data | pandas, numpy |
| ML | scikit-learn, XGBoost |
| Stats | scipy (McNemar's test) |
| Viz | plotly, streamlit |
| Persistence | joblib |

---

## Contact

**Sanjay Krishna MV**  
[GitHub](https://github.com/SANJAY-KRISHNA-MV) · [LinkedIn](https://www.linkedin.com/in/sanjay-krishna-mv/) · sanjaymvkrishna@gmail.com
