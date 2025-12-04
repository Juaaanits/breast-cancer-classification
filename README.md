
# Breast Cancer Classification using Logistic Regression

A machine learning project that classifies breast cancer tumors as malignant or benign using logistic regression on the Breast Cancer Wisconsin dataset.

## 📌 Project Description

This repository contains a full pipeline for binary classification of breast cancer using logistic regression. The workflow includes:

- Data cleaning and preprocessing
- Feature selection and transformation
- Model training with stratified train-test split
- Model evaluation using multiple metrics
- Visualization of ROC curves and confusion matrix
- Suggestions for improving performance

## 📁 Dataset

- **Source**: UCI Machine Learning Repository 
- **Features**: 30 numeric features (mean, standard error, worst)
- **Target**: Diagnosis (`M` = malignant, `B` = benign)

## 🚀 Installation

Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 🧪 Usage

Run the Jupyter notebook or Colab version step by step. It includes:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
```

Train/test split with stratification:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- ROC Curve and AUC
- Confusion Matrix
- False Alarm Rate

## 📌 Key Findings

- Most influential features: `concave points_mean`, `texture_mean`, `area_mean`
- Stratified train-test split helps preserve class balance
- ROC curve helps visualize generalization and overfitting
- Logistic regression performs well but can be improved with tuning

## ⚠️ Challenges

- Missing headers when loading from some sources
- Solved by manually adding headers and grouping features

## ✅ Future Work

- Try other models: SVM, Decision Trees, Random Forests
- Feature scaling and selection
- Hyperparameter tuning
- Addressing class imbalance with SMOTE or weighting

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🙌 Acknowledgments

- UCI Machine Learning Repository

---

**Author**: Juanito M. Ramos II  
