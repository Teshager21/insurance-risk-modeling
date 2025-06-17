# 🛡️ Insurance Risk Modeling — AlphaCare Insurance Solutions (ACIS)

> Official repository for the **AlphaCare Insurance Solutions (ACIS)** marketing analytics and pricing optimization project. This end-to-end project analyzes historical auto insurance claims to model customer risk, segment policyholders, validate business hypotheses, and deliver actionable insights to improve pricing and marketing strategies across South Africa.

---

## 🚀 Executive Summary

**Business Challenge:**
AlphaCare Insurance Solutions seeks to minimize underwriting risk and boost profitability by identifying **low-risk customers**, implementing **risk-based pricing**, and optimizing **marketing outreach**.

**Timeframe Analyzed:**
February 2014 – August 2015

**Data Scope:**
~8,000 individual car insurance policies and their associated claims

**Geography:**
South Africa

---

## 🎯 Project Objectives

| Task | Description |
|------|-------------|
| **Task 1: Exploratory Data Analysis (EDA)** | Conducted a comprehensive audit of demographic, geographic, and behavioral features. Key variables such as claim counts, age, and area of residence were analyzed to understand distribution and detect anomalies. |
| **Task 2: Reproducible Data Pipeline (DVC)** | Implemented a reproducible and auditable data pipeline using **Data Version Control (DVC)**, ensuring robust experiment tracking and artifact consistency. |
| **Task 3: Hypothesis Testing** | Performed rigorous statistical tests to validate or reject business hypotheses related to claim behavior — such as geographic influence, age-driven risk, and gender trends. |
| **Task 4: Predictive Risk Modeling** | Developed and evaluated **claim severity regression** and **claim probability classification** models using Ridge, Random Forest, and XGBoost. Delivered SHAP-based interpretability insights for data-driven decision-making. |

---

## 🔍 Key Business Questions Addressed

- Are customers from specific provinces or age groups more likely to file claims?
- Can we reliably predict the **severity** and **probability** of claims?
- Which customer segments exhibit consistently **low-risk behavior**?
- How can we use statistical evidence to support pricing differentiation?

---

## 📊 Results & Insights

- **Geographic Risk:** Gauteng had the highest claim frequency and severity.
- **Age & Risk:** Drivers aged **<25 and >65** showed increased claims.
- **Gender:** Minor statistical difference; not significant enough alone for segmentation.
- **Predictive Accuracy:**
  - *Best Classification Model:* XGBoost Classifier (Claim Probability)
  - *Best Regression Model:* Ridge Regressor (Claim Severity)
- **Model Explainability:** SHAP analysis revealed that **Vehicle Age**, **Driver Age**, and **Area Risk Profile** were the strongest drivers of predicted risk.

---

## 🛠️ Technical Stack

| Category | Tools/Frameworks |
|---------|------------------|
| Programming | Python, Jupyter, Pandas, Scikit-learn, XGBoost |
| Data Versioning | DVC |
| Visualization | Seaborn, Matplotlib, Plotly |
| Interpretability | SHAP |
| Automation | Makefile, GitHub Actions |
| Infrastructure | Docker, Terraform (optional) |

---

## 📁 Project Structure

```
insurance-risk-modeling/
├── data/                 # Raw, external, interim, and processed data
├── notebooks/            # EDA, Hypothesis Testing, Modeling, SHAP
├── src/                  # Custom pipelines, utils, config handlers
├── tests/                # Unit tests for pipeline classes and logic
├── config/               # YAML/JSON settings for models and pipelines
├── reports/              # Visualizations, evaluation plots, reports
├── dvc.yaml              # DVC pipeline definitions
├── Dockerfile            # Containerization setup
├── Makefile              # Task automation (run pipelines, tests, etc.)
├── api/                  # Optional FastAPI server for model serving
├── infra/                # Optional Terraform deployment scripts
```

---

## ⚙️ Getting Started

### 🔧 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🐳 Docker Support (Optional)
```bash
docker build -t insurance-risk-modeling .
docker run -p 8000:8000 insurance-risk-modeling
```

### 📦 DVC Data Versioning
```bash
dvc init
dvc pull
```

---

## 📈 Evaluation Metrics

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Claim Probability | XGBoost | ROC AUC | **0.81** |
| Claim Severity | Ridge | RMSE | **~32,000 ZAR** |

Model interpretability was enhanced using SHAP to explain predictions to stakeholders with clear visuals and feature importance.

---

## 📌 Milestones

- ✅ Data Audit and EDA
- ✅ Hypothesis Validation (ANOVA, Chi-Square, T-tests)
- ✅ DVC-Enabled Data Pipeline
- ✅ Predictive Modeling (Classification & Regression)
- ✅ Interpretability with SHAP
- ✅ Business Recommendations Report

---

## 📢 Business Recommendations

- **Price Differentiation:** Apply higher premiums for Gauteng drivers and very young/old age groups.
- **Safe Driver Discounts:** Offer lower rates to identified low-risk segments (e.g., 35–55 age, low-density areas).
- **Marketing Strategy:** Target mid-risk provinces with tailored education campaigns and onboarding offers.
- **Policy Adjustments:** Monitor claim-prone segments more frequently for fraud or over-utilization.

---

## ✅ Features

- End-to-end modular architecture
- Fully reproducible pipelines with DVC
- Ready for production deployment (Docker, FastAPI optional)
- Tested with CI workflows via GitHub Actions
- Clear separation of data, logic, infrastructure, and reporting

---

## 📜 License

Distributed under the **MIT** License. See [`LICENSE`](LICENSE) for full terms.

---

## 🤝 Contributors

- **Teshager Admasu** — Data Scientist
- 10 Academy Week 4 — Advanced Insurance Analytics
