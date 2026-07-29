# 🏠 Boston House Price Prediction Model

An end-to-end Machine Learning pipeline developed to accurately predict real estate values in Boston using socio-economic and structural features. This project covers rigorous exploratory data analysis (EDA), custom feature engineering, outlier mitigation, and hyperparameter tuning.

---

## 📊 Key Results

| Metric | Value |
|---|---|
| **Best Model** | Tuned Gradient Boosting Regressor |
| **Optimization** | 5-Fold Grid Search Cross-Validation |
| **R² Score** | *(Insert your R² score here)* |
| **RMSE** | *(Insert your RMSE here)* |
| **Top Value Drivers** | Room Count (`RM`) and Lower Status Population % (`LSTAT`) |

---

## 🔧 Project Pipeline Structure

### 1. Advanced Preprocessing

- Handled extreme outliers across heavily skewed variables (`CRIM`, `ZN`, `B`) using **Interquartile Range (IQR) capping**.
- Applied **log transforms** (`np.log1p`) to mitigate structural skewness in linear contexts.
- Engineered custom domain indicators:
  - `ROOM_AGE_RATIO` — captures the interaction between room count and property age.
  - `TAX_PER_DIS` — captures tax burden relative to distance from employment centers.

---

### 2. Model Selection & Comparison

Evaluated three core architectures across **5 Cross-Validation Folds**:

| Model | Type | Role |
|---|---|---|
| Linear Regression | Parametric | Baseline |
| Decision Tree Regressor | Non-linear | Baseline |
| **Gradient Boosting Regressor** | Advanced Ensemble | **Best Performer ✅** |

---

### 3. Hyperparameter Fine-Tuning

Utilized **GridSearchCV** to automatically map out optimal parameters for:
- Learning rate
- Number of estimators
- Maximum tree depths

---

## 📁 File Structure

```
beginner lvl task/
├── BostonHousingPricePrediction.ipynb   # Main notebook
└── README.md                            # Project documentation
```

---

## 🚀 How to Run

1. Open `BostonHousingPricePrediction.ipynb` in **Jupyter Notebook** or **Google Colab**.
2. Run all cells sequentially from top to bottom.
3. Ensure the following dependencies are installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 📦 Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 👤 Author

**Rishi Chaurasia**  
ShadowFox Internship — Beginner Level Task
