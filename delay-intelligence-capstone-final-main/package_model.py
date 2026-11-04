"""Rebuild and package the final Part C classifier for reusable scoring."""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.shipment_prep import clean_shipments
from part_c_classification import FEATURES, TARGET

ROOT = Path(__file__).resolve().parent


def main():
    data = pd.read_csv(ROOT / "data" / "modelling_table.csv")
    X, y = data[FEATURES].copy(), data[TARGET].astype(int).copy()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    config = pd.read_csv(ROOT / "outputs" / "part_c_final_config.csv").iloc[0]
    max_depth = None if str(config["max_depth"]).lower() in {"unlimited", "none", "nan"} else int(config["max_depth"])
    model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=int(config["selector_k"]))),
        ("model", RandomForestClassifier(n_estimators=int(config["n_estimators"]), max_depth=max_depth,
                                          min_samples_leaf=int(config["min_samples_leaf"]), random_state=42)),
    ])
    model.fit(X_train, y_train)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "delay_model.pkl")
    pd.DataFrame({"threshold": [float(config["threshold"])]}).to_csv(models_dir / "delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(models_dir / "feature_order.csv", index=False)

    raw = pd.read_csv(ROOT / "data" / "raw_shipments.csv")
    _, medians = clean_shipments(raw)
    pd.DataFrame([{"column": k, "median": v} for k, v in medians.items()]).to_csv(models_dir / "impute_values.csv", index=False)

    reloaded = joblib.load(models_dir / "delay_model.pkl")
    p_memory = model.predict_proba(X_test)[:, 1]
    p_reload = reloaded.predict_proba(X_test)[:, 1]
    diff = float(np.max(np.abs(p_memory - p_reload))) if len(p_memory) else 0.0
    pd.DataFrame([
        {"metric": "test_rows_scored", "value": len(X_test)},
        {"metric": "reload_max_abs_probability_diff", "value": diff},
        {"metric": "threshold", "value": float(config["threshold"])},
    ]).to_csv(ROOT / "outputs" / "part_d_reload_check.csv", index=False)
    print(f"Packaged model; test rows={len(X_test)}, reload max abs diff={diff:.12g}")

if __name__ == "__main__":
    main()
