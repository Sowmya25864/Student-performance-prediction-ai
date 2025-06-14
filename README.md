# Student-performance-prediction-ai
AI-based project that predicts student pass/fail outcomes and risk zones using ensemble machine learning models.

# Student Performance Prediction Using AI

This AICTE internship project uses machine learning to predict whether a student will pass or fail and assigns them to risk zones (Failing, At-Risk, Average, Topper) based on academic scores and background information.

---

# Project Files

- `student_model.ipynb` – Full training pipeline with feature engineering, model training, evaluation, and visualization.
- `final_predictions.csv` – Output predictions from the ensemble model.
- `evaluation_metrics.png` – Visual summary: Confusion Matrix and ROC Curve.
- `model_accuracy_comparison.png` – Bar chart comparing base model accuracies.
- `correlation_heatmap.png` – Heatmap showing correlations between numeric features.
---

# Dataset Features

The dataset includes:

- Gender, Race/Ethnicity
- Parental education level
- Lunch type, Test preparation course
- Math, Reading, Writing scores

Engineered Features:

- `avg_score`, `score_std`, `study_efficiency`, and `risk_zone` (label binning)

Target Variables:

- `pass_fail`: Whether student passed (avg_score ≥ 60)
- `risk_level`: Category label (Failing, At-Risk, Average, Topper)

---

# Machine Learning Approach

- Label encoding for categorical data
- Feature scaling using StandardScaler
- Dual output classification using MultiOutputClassifier
- Stacking ensemble using:
  - Base models: Random Forest, SVM, KNN
  - Meta model: Logistic Regression

---

# Performance Summary

| Task             | Metric        | Value  |
|------------------|---------------|--------|
| Pass/Fail        | Accuracy      | ~90%   |
| Risk Level       | Accuracy      | ~85%   |
| Visualization    | Confusion Matrix, ROC Curve, Accuracy Chart |

---

# Risk Intervention Logic

Based on predicted risk zone:

| Risk Zone  | Intervention                     |
|------------|----------------------------------|
| Failing    | Immediate academic support       |
| At-Risk    | Provide mentoring and guidance   |
| Average    | Normal academic monitoring       |
| Topper     | Recommend gifted programs        |

---

# Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Optional for notebooks:
```bash
pip install jupyter
```

---

# Future Improvements

- Add behavioral/psychological features
- Use SHAP for model explainability
- Deploy the model using Streamlit or Gradio

---

# References

- [Kaggle Dataset: Student Performance](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Scikit-learn documentation
