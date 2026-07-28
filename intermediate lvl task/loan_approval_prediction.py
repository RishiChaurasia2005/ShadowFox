import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

np.random.seed(42)
tf.random.set_seed(42)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_prediction.csv")
df = pd.read_csv(data_path)

if df.columns[0] not in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome']:
    cols_to_drop = [c for c in df.columns if c in ['Loan_ID'] or 'Unnamed' in str(c)]
    first_col = df.columns[0]
    if first_col not in ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']:
        cols_to_drop.append(first_col)
    df.drop(columns=[c for c in set(cols_to_drop) if c in df.columns], inplace=True)

for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    df[col].fillna(df[col].median(), inplace=True)

df["LoanAmount_Log"] = np.log1p(df["LoanAmount"])
df["ApplicantIncome_Log"] = np.log1p(df["ApplicantIncome"])
df["CoapplicantIncome_Log"] = np.log1p(df["CoapplicantIncome"])

df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["TotalIncome_Log"] = np.log1p(df["TotalIncome"])
df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]
df["BalanceIncome"] = df["TotalIncome"] - (df["EMI"] * 1000)
df["Income_Loan_Ratio"] = df["TotalIncome"] / (df["LoanAmount"] + 1)

cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

feature_cols = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Credit_History", "Property_Area", "LoanAmount_Log", "TotalIncome_Log",
    "EMI", "BalanceIncome", "Income_Loan_Ratio", "Loan_Amount_Term"
]

X = df[feature_cols]
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]) if hasattr(model, "predict_proba") else None
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc, "model": model, "y_pred": y_pred}

evaluate_model("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), X_train_scaled, X_test_scaled, y_train, y_test)
evaluate_model("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1), X_train_scaled, X_test_scaled, y_train, y_test)
evaluate_model("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42), X_train_scaled, X_test_scaled, y_train, y_test)
evaluate_model("SVM", SVC(kernel="rbf", probability=True, random_state=42), X_train_scaled, X_test_scaled, y_train, y_test)

nn_model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop, lr_scheduler], verbose=0)

y_pred_nn_prob = nn_model.predict(X_test_scaled).flatten()
y_pred_nn = (y_pred_nn_prob >= 0.5).astype(int)

results["Neural Network (Keras)"] = {
    "Accuracy": accuracy_score(y_test, y_pred_nn),
    "Precision": precision_score(y_test, y_pred_nn),
    "Recall": recall_score(y_test, y_pred_nn),
    "F1-Score": f1_score(y_test, y_pred_nn),
    "ROC-AUC": roc_auc_score(y_test, y_pred_nn_prob),
    "model": nn_model,
    "y_pred": y_pred_nn
}

param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [5, 8],
    "min_samples_split": [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test_scaled)
acc_tuned = accuracy_score(y_test, y_pred_tuned)

if acc_tuned > results["Random Forest"]["Accuracy"]:
    results["Random Forest (Tuned)"] = {
        "Accuracy": acc_tuned,
        "Precision": precision_score(y_test, y_pred_tuned),
        "Recall": recall_score(y_test, y_pred_tuned),
        "F1-Score": f1_score(y_test, y_pred_tuned),
        "ROC-AUC": roc_auc_score(y_test, best_rf.predict_proba(X_test_scaled)[:, 1]),
        "model": best_rf,
        "y_pred": y_pred_tuned
    }

comparison_df = pd.DataFrame({
    name: {k: v for k, v in metrics.items() if k not in ["model", "y_pred"]}
    for name, metrics in results.items()
}).T.sort_values("Accuracy", ascending=False)

print(comparison_df.to_string())
