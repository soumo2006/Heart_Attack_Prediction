# Heart Attack Risk Prediction — End-to-End Analytics Capstone

## Problem Statement
Cardiovascular disease is one of the leading causes of preventable mortality, and early risk identification is critical for timely clinical intervention. This project builds an end-to-end data analytics and machine learning pipeline to predict heart attack risk from patient clinical attributes, and translates model outputs into actionable triage recommendations for a healthcare provider prioritizing limited follow-up resources.

**Core Business Questions:**
- What patient attributes are most strongly associated with heart attack risk?
- Can we reliably classify patients as at-risk or not at-risk using clinical measurements?
- Which patients should be prioritized for early clinical follow-up given limited resources?
- What is the acceptable trade-off between false positives (unnecessary follow-ups) and false negatives (missed at-risk patients)?

## Dataset
Clinical patient records including age, sex, chest pain type (`cp`), resting blood pressure (`trestbps`), cholesterol (`chol`), max heart rate achieved (`thalch`), ST depression (`oldpeak`), number of major vessels colored by fluoroscopy (`ca`), and the target outcome (`num`).

**Data hygiene:** The record identifier (`id`) was excluded from the feature set — a raw identifier has no legitimate clinical relationship with outcomes. The target column (`num`) was strictly separated from the input features before training to avoid target leakage.

## Methodology — 4-Tier Analytics Ladder

### 1. Descriptive Analytics
Baseline exploratory visualizations: age distribution, risk-outcome balance, cholesterol and blood pressure spread, chest pain type frequency, and max heart rate vs. age (by risk).
- Age distribution is approximately normal, concentrated between 45–65 years, peaking around 50–55.

### 2. Diagnostic Analytics
Correlation analysis and cross-tabulation to identify which features associate with risk:

| Feature | Correlation with Risk |
|---|---|
| `ca` (major vessels colored) | 0.516 |
| `oldpeak` (ST depression) | 0.443 |
| `age` | 0.340 |
| `trestbps` (resting BP) | 0.122 |
| `chol` (cholesterol) | -0.232 |
| `thalch` (max heart rate) | -0.366 |

`ca` and `oldpeak` show the strongest positive association with risk; `thalch` shows a moderate negative association.

### 3. Predictive Analytics
A K-Nearest Neighbors (KNN) classifier was trained on the cleaned, leakage-checked feature set.

| Metric | Score |
|---|---|
| Accuracy | 83.70% |
| Precision | 87.16% |
| Recall | 85.59% |
| F1 Score | 86.36% |

A false negative (missed at-risk patient) is clinically costlier than a false positive (unnecessary follow-up), so Recall is weighted as the more important metric for this use case.

### 4. Prescriptive Recommendations
1. **Priority triage rule** — Patients with `ca ≥ 1` and elevated `oldpeak` should be escalated for cardiologist review within 7 days.
2. **Stress-test screening protocol** — Patients over 50 with below-median max heart rate under stress testing should be flagged for earlier intervention.
3. **Recall-aware threshold policy** — Tune the classification threshold to favor Recall over Precision, given the asymmetric cost of missed diagnoses.
4. **Data governance** — Exclude non-clinical identifiers from the feature set in future pipeline iterations.

## Repository Structure
```
├── notebooks/        # Cleaned, commented Colab notebook
├── data/             # Raw + cleaned dataset, data dictionary
├── presentation/     # Executive summary slide deck / PDF
├── requirements.txt  # Python dependencies
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (KNN classifier)

## Conclusion
This project demonstrates a complete analytics lifecycle — from raw clinical data through descriptive profiling, diagnostic investigation, predictive classification, and prescriptive business recommendations — resulting in a model with 83.7% accuracy and concrete, resource-aware clinical action items.
