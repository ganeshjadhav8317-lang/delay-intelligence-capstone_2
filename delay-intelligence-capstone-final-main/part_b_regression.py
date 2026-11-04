"""Part B: leakage-safe regression modelling for shipment delivery time.

Run from the repository root:
    python part_b_regression.py

Reads data/modelling_table.csv and writes Part B diagnostics to outputs/.
All candidate estimators are fitted on the training partition. The test set is
used only once, after selecting the final model from validation performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
MODELLING_TABLE_PATH = ROOT / "data" / "modelling_table.csv"
OUTPUT_DIR = ROOT / "outputs"

# Keep this feature order pinned exactly as specified in Part A.
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]
TARGET = "delivery_hours"
TREE_DEPTHS = [1, 2, 3, 4, 6, 8, 12, None]
MODEL_NAMES = [
    "mean_baseline",
    "linear_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
]


def load_modelling_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load Part A's modelling table and return ordered X and target y."""
    if not MODELLING_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODELLING_TABLE_PATH}. Run Part A first: "
            "python part_a_clean_eda.py"
        )
    modelling = pd.read_csv(MODELLING_TABLE_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in modelling.columns]
    if missing:
        raise ValueError("Missing required modelling columns: " + ", ".join(missing))
    X = modelling[FEATURES].copy()
    y = modelling[TARGET].copy()
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Missing values found in FEATURES or delivery_hours.")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    """Create reproducible 70% / 15% / 15% partitions."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    sizes = (len(X_train), len(X_val), len(X_test))
    if sizes != (840, 180, 180):
        raise ValueError(f"Unexpected split sizes {sizes}; expected (840, 180, 180).")
    return X_train, X_val, X_test, y_train, y_val, y_test


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the requested regression metrics, with RMSE derived from MSE."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_candidates(best_tree_depth: int | None) -> dict[str, object]:
    """Construct the four requested candidate estimators."""
    return {
        "linear_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "decision_tree": DecisionTreeRegressor(
            max_depth=best_tree_depth, random_state=42
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_modelling_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Scaling diagnostics: fit on training rows only; reuse the fitted scaler.
    diagnostic_scaler = StandardScaler()
    X_train_scaled = diagnostic_scaler.fit_transform(X_train)
    X_val_scaled = diagnostic_scaler.transform(X_val)
    X_test_scaled = diagnostic_scaler.transform(X_test)
    checks = [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", float(np.abs(X_train_scaled.mean(axis=0)).max())),
        ("test_scaled_max_abs_mean", float(np.abs(X_test_scaled.mean(axis=0)).max())),
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv(
        OUTPUT_DIR / "part_b_checks.csv", index=False
    )

    # Baseline: predict the training-target mean for each validation parcel.
    baseline_prediction = np.full(len(y_val), y_train.mean(), dtype=float)
    metric_rows = [
        {"model": "mean_baseline", "split": "val", **regression_metrics(y_val, baseline_prediction)}
    ]

    # Tree-depth diagnostic includes a fully grown (unlimited-depth) tree.
    tree_diagnostic_rows = []
    for depth in TREE_DEPTHS:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        train_prediction = tree.predict(X_train)
        val_prediction = tree.predict(X_val)
        tree_diagnostic_rows.append(
            {
                "max_depth": "unlimited" if depth is None else depth,
                "train_rmse": regression_metrics(y_train, train_prediction)["RMSE"],
                "val_rmse": regression_metrics(y_val, val_prediction)["RMSE"],
            }
        )
    tree_depth_df = pd.DataFrame(tree_diagnostic_rows)
    tree_depth_df.to_csv(OUTPUT_DIR / "part_b_tree_depth.csv", index=False)
    best_tree_row = min(
        tree_diagnostic_rows,
        key=lambda row: (
            row["val_rmse"],
            float("inf") if row["max_depth"] == "unlimited" else row["max_depth"],
        ),
    )
    best_tree_depth = (
        None if best_tree_row["max_depth"] == "unlimited" else int(best_tree_row["max_depth"])
    )

    candidates = make_candidates(best_tree_depth)
    fitted_candidates = {}
    validation_rmse = {}
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        fitted_candidates[name] = estimator
        predictions = estimator.predict(X_val)
        metrics = regression_metrics(y_val, predictions)
        validation_rmse[name] = metrics["RMSE"]
        metric_rows.append({"model": name, "split": "val", **metrics})

    # Learning-rate experiment for gradient boosting.
    learning_rate_rows = []
    for learning_rate in (0.01, 0.1, 1.0):
        gb = GradientBoostingRegressor(
            n_estimators=100, learning_rate=learning_rate, random_state=42
        )
        gb.fit(X_train, y_train)
        pred = gb.predict(X_val)
        metrics = regression_metrics(y_val, pred)
        learning_rate_rows.append(
            {
                "learning_rate": learning_rate,
                "val_rmse": metrics["RMSE"],
                "val_r2": metrics["R2"],
            }
        )
    pd.DataFrame(learning_rate_rows).to_csv(
        OUTPUT_DIR / "part_b_gb_learning_rate.csv", index=False
    )

    # Cross-validation: compare the two validation-RMSE leaders on train + val.
    ranked_names = sorted(validation_rmse, key=lambda name: (validation_rmse[name], name))
    top_two = ranked_names[:2]
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    shared_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for name in top_two:
        cv_scores = cross_val_score(
            candidates[name], X_train_val, y_train_val, cv=shared_kfold
        )
        cv_rows.append(
            {
                "model": name,
                "mean_r2": float(np.mean(cv_scores)),
                "std_r2": float(np.std(cv_scores)),
            }
        )
    pd.DataFrame(cv_rows).to_csv(OUTPUT_DIR / "part_b_cv.csv", index=False)

    # Choose the validation-RMSE winner and evaluate it once on untouched test rows.
    final_model_name = ranked_names[0]
    final_model = fitted_candidates[final_model_name]
    test_metrics = regression_metrics(y_test, final_model.predict(X_test))
    metric_rows.append(
        {"model": final_model_name, "split": "test", **test_metrics}
    )
    pd.DataFrame(
        metric_rows, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]
    ).to_csv(OUTPUT_DIR / "part_b_metrics.csv", index=False)

    print("Part B - Regression modelling completed")
    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Training scaled max absolute column mean: {checks[3][1]:.8f}")
    print(f"Test scaled max absolute column mean: {checks[4][1]:.8f}")
    print(f"Best decision-tree depth: {best_tree_row['max_depth']}")
    print("Validation RMSE ranking:")
    for rank, name in enumerate(ranked_names, start=1):
        print(f"  {rank}. {name}: {validation_rmse[name]:.4f} hours")
    print(f"Final model: {final_model_name}")
    print(f"Final test RMSE: {test_metrics['RMSE']:.4f} hours")
    print("Wrote Part B checks, tree-depth, metrics, learning-rate, and CV CSVs.")


if __name__ == "__main__":
    main()
