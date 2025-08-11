# Bank Marketing Classifiers Comparison

## Project Overview
This repository contains a Jupyter Notebook (`Bank_Marketing.ipynb`) that compares the performance of classification models—K-Nearest Neighbors (KNN), Logistic Regression (LR), Decision Trees (DT), and Support Vector Machines (SVM)—on the UCI Bank Marketing dataset. The analysis predicts client subscriptions to term deposits via telemarketing campaigns, following CRISP-DM methodology for business understanding, data exploration, feature engineering, modeling, hyperparameter tuning, and evaluation. Visualizations include EDA plots, ROC curves, and an interactive Plotly bar chart for tuned metrics.

## Business Problem
Banks face high costs and low response rates (~11%) in telemarketing campaigns for term deposits. The objective is to build predictive models identifying likely subscribers, enabling targeted calls to reduce contacts (e.g., 20-30% fewer) while maintaining successes amid economic challenges like the 2008-2010 crisis. Among the compared models (LR, DT, KNN, SVM), the Decision Tree (DT) proved to be the best overall, offering a balanced F1-score (~0.08 tuned), interpretability through decision rules (e.g., prioritizing prior successes), and fast training (~0.17s), making it ideal for efficient, actionable targeting in campaigns.

## Data Source
- Dataset: From UCI Machine Learning Repository - [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing) (included as `Data/bank-additional-full.csv`—41,188 samples, 20 features + binary target 'y').
- Features: Client (age, job), campaign (contact, duration), economic (emp.var.rate).
### Note book: Bank_Marketing.ipynb 
## Methods and Tools
- **Libraries**: pandas, numpy, scikit-learn (modeling, GridSearchCV, metrics), matplotlib/seaborn (EDA plots), plotly (interactive visuals).
- **Steps**:
  - EDA: Distributions, correlations, imbalance handling.
  - Features: Impute 'unknown', bin age, create 'contacted_before'; drop low-impact.
  - Models: Baseline (Dummy), defaults, tuned (GridSearchCV, F1 scoring).
  - Metrics: Accuracy, F1, AUC-ROC for interactive bars.

## Key Findings
- Tuned Results (Test F1/AUC):
  | Model | Best Params | Tuned Test F1 | Tuned Test AUC |
  |-------|-------------|---------------|----------------|
  | LR    | {'model_C': 0.1} | 0.0000 | 0.6264 |
  | DT    | {'model_max_depth': 10, 'model_min_samples_split': ...} | 0.0822 | 0.6135 |
  | KNN   | {'model_n_neighbors': 3} | 0.1255 | 0.5492 |
  | SVM   | {'model_C': 10, 'model_kernel': 'rbf'} | 0.0333 | 0.5525 |

- DT excels for interpretability/fast tuning; SVM highest AUC but slow. Imbalance limits F1; features like prior success/economic stability key drivers.
- Insights: Target retirees/students via cellular; A/B test for 30% cost savings.

