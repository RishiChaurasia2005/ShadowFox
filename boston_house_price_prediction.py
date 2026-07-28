import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "HousingData.csv")

print("=" * 70)
print("STEP A: DATA LOADING & PREPROCESSING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape : {df.shape}")
print(f"Features      : {list(df.columns[:-1])}")
print(f"Target        : {df.columns[-1]}")

print("\n-- Missing Values --")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Count": missing, "Percent": missing_pct})
print(missing_df[missing_df["Count"] > 0].to_string())

print("\n-> Imputing missing values with column medians ...")
df.fillna(df.median(), inplace=True)
print(f"  Missing values after imputation: {df.isnull().sum().sum()}")

print("\n-- Descriptive Statistics --")
print(df.describe().round(2).to_string())

print("\n-- Outlier Detection (IQR method) --")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    if n_outliers > 0:
        print(f"  {col:8s}: {n_outliers:3d} outliers  "
              f"(range [{lower:.2f}, {upper:.2f}])")

print("\n-> Capping outliers using IQR bounds ...")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)
print("  Outliers capped successfully.")

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n-- Train / Test Split --")
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing samples  : {X_test.shape[0]}")

print("\n" + "=" * 70)
print("STEP B & C: MODEL SELECTION & TRAINING")
print("=" * 70)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
    ),
    "AdaBoost": AdaBoostRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
    "SVR": SVR(kernel="rbf", C=10, gamma="scale"),
}

results = []

print(f"\n{'Model':<25s} {'MAE':>8s} {'MSE':>10s} {'RMSE':>8s} {'R2 Score':>10s}")
print("-" * 65)

for name, model in models.items():
    if name in ("SVR", "Lasso Regression", "Ridge Regression", "Linear Regression"):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    })
    print(f"  {name:<23s} {mae:8.4f} {mse:10.4f} {rmse:8.4f} {r2:10.4f}")

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

print("\n" + "=" * 70)
print("STEP D: MODEL EVALUATION")
print("=" * 70)

print("\n-- 5-Fold Cross-Validation (top 3 models) --")
top3 = results_df.head(3)["Model"].tolist()
for name in top3:
    model = models[name]
    if name in ("SVR", "Lasso Regression", "Ridge Regression", "Linear Regression"):
        X_cv = scaler.fit_transform(X)
    else:
        X_cv = X
    scores = cross_val_score(model, X_cv, y, cv=5, scoring="r2")
    print(f"  {name:<23s}  mean R2 = {scores.mean():.4f}  (+/- {scores.std():.4f})")

print("\n" + "=" * 70)
print("STEP E: FINE-TUNING - Gradient Boosting Regressor")
print("=" * 70)

param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.15],
    "max_depth": [3, 4, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3],
}

gbr = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(
    gbr, param_grid, cv=5, scoring="r2",
    n_jobs=-1, verbose=0
)

print("  Running GridSearchCV ... (this may take a moment)")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\n  Best Parameters: {grid_search.best_params_}")
print(f"  Best CV R2 Score: {grid_search.best_score_:.4f}")

y_pred_best = best_model.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"\n-- Fine-Tuned Model - Test Set Performance --")
print(f"  MAE     : {mae_best:.4f}")
print(f"  MSE     : {mse_best:.4f}")
print(f"  RMSE    : {rmse_best:.4f}")
print(f"  R2 Score: {r2_best:.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
  Dataset           : {DATA_PATH}
  Samples           : {len(df)}
  Features          : {X.shape[1]}
  Best Model        : Gradient Boosting Regressor (fine-tuned)
  Best Parameters   : {grid_search.best_params_}
  Test R2 Score     : {r2_best:.4f}
  Test RMSE         : {rmse_best:.4f}
  Test MAE          : {mae_best:.4f}
""")

print("-- Final Model Ranking --")
print(results_df.to_string(index=False))
print("\nPipeline complete.")
