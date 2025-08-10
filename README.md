# Advancing Fairness and Explainability in AI for Autism Diagnosis
This project extends an existing autism spectrum disorder (ASD) diagnostic pipeline by:
- Integrating multiple imputation strategies - Comparing Mean, Median, and KNN Imputation to handle missing data and analyze their impact on model robustness.
- Conducting fairness analysis - Evaluating bias across gender groups using metrics such as demographic parity difference and equalized odds difference, and then applying bias mitigation using Threshold Optimizer to minimize prediction disparities.
- Enhancing explainability - Generating SHAP plots to visualize feature importance and individual prediction contributions to improve interpretability.

The pipeline is built using GBM, SVM, and XGB machine learning models with phenotypic and neuroimaging features from the ABIDE II dataset.
