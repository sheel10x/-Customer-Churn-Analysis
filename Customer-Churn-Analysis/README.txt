# 📉 Telco Customer Churn Prediction with XGBoost & SHAP

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange)

## 📌 Project Overview
This project implements an end-to-end Machine Learning pipeline to predict customer churn in the telecommunications industry. It utilizes **SHAP (SHapley Additive exPlanations)** to provide deep interpretability, identifying the root causes of customer attrition.

## 🛠️ Tech Stack
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Modeling:** XGBoost Classifier
* **Handling Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Explainability:** SHAP (Beeswarm, Force, and Waterfall plots)

## 📊 Key Results (10-Fold CV)
The model was evaluated using Stratified 10-Fold Cross-Validation.

| Metric | Score |
| :--- | :--- |
| **ROC-AUC** | **0.8424** |
| **Accuracy** | 78.61% |
| **Recall** | 65.27% |
| **Precision** | 58.80% |

## 🔍 Model Interpretability

### 1. Feature Impact (Beeswarm Plot)
![Beeswarm Plot](Results/shap_beeswarm_plot.png)
* **Contract_Month-to-month:** High values (red) increase churn risk.
* **Tenure:** Low tenure (blue) is associated with higher churn.
* **Fiber Optic:** Fiber Optic users show a higher likelihood of churn.

### 2. Individual Prediction Explanation (Waterfall Plot)
![Waterfall Plot](Results/shap_waterfall_plot_client_100.png)

## 🚀 How to Run
1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Customer-Churn-Analysis.git](https://github.com/YOUR_USERNAME/Customer-Churn-Analysis.git)
    cd Customer-Churn-Analysis
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis pipeline**
    ```bash
    python churn_analysis.py
    ```

## 📂 Project Structure
* `churn_analysis.py`: Main pipeline script.
* `Data/`: Cleaned and transformed datasets.
* `Results/`: Generated plots and SHAP values.