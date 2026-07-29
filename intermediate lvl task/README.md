# 💳 Loan Approval Prediction Model — End-to-End ML & Deep Learning Pipeline

An end-to-end Machine Learning and Deep Learning pipeline built to classify loan approval status (`Loan_Status`: Approved / Rejected) based on applicant demographic, financial, and credit history features.

---

## 📊 Overview & Key Highlights

This project implements a complete binary classification workflow using both classic machine learning models and a custom Deep Neural Network (Keras/TensorFlow).

### 🏆 Models Evaluated
- **Logistic Regression** (Linear baseline)
- **Random Forest Classifier** (Ensemble baseline & tuned via GridSearchCV)
- **Gradient Boosting Classifier** (Advanced ensemble)
- **Support Vector Machine (SVM)** (Kernel-based classifier)
- **Deep Neural Network (Keras)** (Multi-layer Perceptron with Batch Normalization & Dropout)

---

## 🔧 Project Pipeline Structure

### 1. Data Preprocessing & Missing Value Imputation
- **Categorical Variables** (`Gender`, `Married`, `Dependents`, `Self_Employed`): Imputed using mode.
- **Numerical / Ordinal Variables** (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`): Imputed using median.

### 2. Feature Engineering & Transformation
- **Log Transformations**: Applied `np.log1p` to skewed continuous variables (`LoanAmount`, `ApplicantIncome`, `CoapplicantIncome`, `TotalIncome`).
- **Domain-Specific Engineered Features**:
  - `TotalIncome` = `ApplicantIncome` + `CoapplicantIncome`
  - `EMI` = `LoanAmount` / `Loan_Amount_Term`
  - `BalanceIncome` = `TotalIncome` - (`EMI` × 1000)
  - `Income_Loan_Ratio` = `TotalIncome` / (`LoanAmount` + 1)

### 3. Encoding & Feature Scaling
- **Categorical Encoding**: `LabelEncoder` applied to categorical features (`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`).
- **Feature Scaling**: `StandardScaler` applied across training and test sets to normalize continuous inputs for linear, SVM, and Neural Network models.

---

## 🧠 Neural Network Architecture (Keras)

```
Input Layer (13 Features)
    ↓
Dense (128 units, ReLU) → Batch Normalization → Dropout (0.3)
    ↓
Dense (64 units, ReLU)  → Batch Normalization → Dropout (0.3)
    ↓
Dense (32 units, ReLU)  → Dropout (0.2)
    ↓
Output Layer (1 unit, Sigmoid)
```

- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Binary Cross-Entropy
- **Callbacks**:
  - `EarlyStopping` (patience = 15, restore best weights)
  - `ReduceLROnPlateau` (factor = 0.5, patience = 5)

---

## 📈 Evaluation Metrics

All models are evaluated and ranked based on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Score**

---

## 📁 File Structure

```
intermediate lvl task/
├── loan_approval_prediction.py   # Full ML & Deep Learning pipeline script
├── loan_prediction.csv           # Dataset
└── README.md                     # Project documentation (this file)
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow
```

### 2. Execute the Pipeline

```bash
python loan_approval_prediction.py
```

---

## 📦 Dependencies

- `python >= 3.8`
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`

---

## 👤 Author

**Rishi Chaurasia**  
ShadowFox Internship — Intermediate Level Task
