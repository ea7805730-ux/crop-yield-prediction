"""Crop Yield Prediction - training script.

Uses rainfall, temperature, humidity, soil pH, soil nutrients (N, P, K),
soil type, crop type and fertilizer use to predict yield (tonnes/hectare).

Usage:
    python train.py                       # uses data/crop_yield.csv, or makes demo data
    python train.py --data my_file.csv --target Yield_t_ha
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_demo_data(n=3000, seed=42):
    """Synthetic dataset so the project runs without any download."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Rainfall_mm": rng.normal(900, 250, n).clip(200, 2000),
        "Temperature_C": rng.normal(26, 4, n).clip(10, 42),
        "Humidity_pct": rng.normal(65, 12, n).clip(20, 100),
        "Soil_pH": rng.normal(6.5, 0.7, n).clip(4.5, 8.5),
        "Nitrogen": rng.normal(80, 25, n).clip(10, 200),
        "Phosphorus": rng.normal(45, 15, n).clip(5, 120),
        "Potassium": rng.normal(50, 15, n).clip(5, 120),
        "Fertilizer_kg_ha": rng.normal(120, 40, n).clip(0, 300),
        "Soil_Type": rng.choice(["Loamy", "Clay", "Sandy", "Silty"], n),
        "Crop": rng.choice(["Rice", "Wheat", "Maize", "Cotton"], n),
    })
    soil_bonus = df["Soil_Type"].map({"Loamy": 1.0, "Silty": 0.6, "Clay": 0.2, "Sandy": -0.5})
    crop_base = df["Crop"].map({"Rice": 4.0, "Wheat": 3.2, "Maize": 5.0, "Cotton": 2.0})
    df["Yield_t_ha"] = (
        crop_base
        + 0.0018 * df["Rainfall_mm"] - 0.0000009 * (df["Rainfall_mm"] - 1000) ** 2
        - 0.02 * (df["Temperature_C"] - 27) ** 2
        + 0.015 * df["Nitrogen"] + 0.008 * df["Phosphorus"] + 0.006 * df["Potassium"]
        + 0.004 * df["Fertilizer_kg_ha"]
        - 0.35 * (df["Soil_pH"] - 6.5) ** 2
        + soil_bonus
        + rng.normal(0, 0.4, n)
    ).clip(0.2)
    return df


def load_data(path):
    if os.path.exists(path):
        print(f"Loading dataset: {path}")
        return pd.read_csv(path)
    print("No dataset found -> generating demo data (data/crop_yield.csv)")
    df = make_demo_data()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return df


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"{name:20s} R2={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    return r2, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crop_yield.csv")
    ap.add_argument("--target", default="Yield_t_ha")
    args = ap.parse_args()

    df = load_data(args.data).dropna()
    X, y = df.drop(columns=[args.target]), df[args.target]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results, fitted = {}, {}
    print("\nModel comparison (test set):")
    for name, reg in models.items():
        pipe = Pipeline([("prep", pre), ("model", reg)]).fit(X_train, y_train)
        r2, pred = evaluate(name, pipe, X_test, y_test)
        results[name], fitted[name] = r2, (pipe, pred)

    best = max(results, key=results.get)
    best_pipe, best_pred = fitted[best]
    print(f"\nBest model: {best} (R2={results[best]:.3f})")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(best_pipe, "models/best_model.joblib")
    print("Saved -> models/best_model.joblib")

    # Plot 1: actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_pred, alpha=0.4)
    lims = [y_test.min(), y_test.max()]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual yield (t/ha)")
    plt.ylabel("Predicted yield (t/ha)")
    plt.title(f"Actual vs Predicted - {best}")
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
    plt.close()

    # Plot 2: feature importance (tree models only)
    reg = best_pipe.named_steps["model"]
    if hasattr(reg, "feature_importances_"):
        names = best_pipe.named_steps["prep"].get_feature_names_out()
        imp = pd.Series(reg.feature_importances_, index=names).sort_values().tail(12)
        imp.plot(kind="barh", figsize=(7, 5), title="Top feature importances")
        plt.tight_layout()
        plt.savefig("outputs/feature_importance.png", dpi=150)
        plt.close()
    print("Saved plots -> outputs/")


if __name__ == "__main__":
    main()
