# insurance-risk-modeling
> Welcome to the official repository for the **AlphaCare Insurance Solutions (ACIS)** marketing analytics project. This project aims to optimize pricing and marketing strategies by analyzing customer claim history and predicting risk levels using car insurance data from South Africa.

---

## 🚀 Overview

**Business Goal:** Reduce losses and improve pricing strategies by identifying low-risk customers and optimizing premium pricing.

**Duration:** February 2014 – August 2015
**Dataset Size:** ~8,000 policyholders
**Region:** South Africa (Car Insurance)

---

## 🧠 Project Objectives

1. **Exploratory Data Analysis (EDA):** Understand claim patterns, demographic trends, and correlations.
2. **Hypothesis & A/B Testing:** Evaluate pricing and marketing strategies.
3. **Predictive Modeling:** Build models to predict likelihood of filing a claim.
4. **Segmentation:** Classify customers into risk categories.
5. **Marketing Insights:** Provide actionable insights to optimize campaign targeting.

---

## 🛠️ Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```


### 2. Docker (Optional)

Build and run using Docker:

```bash
docker build -t insurance-risk-modeling .
docker run -p 8000:8000 insurance-risk-modeling
```



### 3. DVC Setup (Optional)

Initialize and pull versioned data:

```bash
dvc init
dvc pull
```




## 📁 Project Structure

```
insurance-risk-modeling/
├── data/                 # Data folders (raw, processed, external, etc.)
├── notebooks/            # Jupyter notebooks for exploration and reporting
├── src/                  # Python package: data, features, models, utils
├── tests/                # Unit & integration tests
├── config/               # Environment-specific configs
├── reports/              # Generated outputs and visualizations
├── api/                  # FastAPI backend (if enabled)
├── edge/                 # Edge deployment tools (e.g., quantization)
├── infra/                # Terraform infrastructure code
├── .github/              # Workflows, PR templates, issue templates
├── Makefile              # Automation commands
├── Dockerfile            # Containerization (if enabled)
├── dvc.yaml              # DVC pipelines (if enabled)
```

## 📌 Milestones

- ✅ Project Setup & Data Loading
- ✅ EDA & Initial Reports
- ✅ Hypothesis Testing
- ⏳ Predictive Modeling
- ⏳ Insights & Recommendations Report

---

## ✅ Features

- Clean, modular structure
- Integrated DVC for data versioning

- Docker for reproducible environments
- MLFlow-ready experiment tracking
- GitHub Actions CI/CD pipeline
- Infrastructure-as-Code with Terraform

## 📜 License

Distributed under the **MIT** License. See `LICENSE` for more information.
