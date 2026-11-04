"""Part C: delay classification, feature selection, tuning, and alert threshold.

Run from the repository root:
    python part_c_classification.py

Reads data/modelling_table.csv and writes Part C results to outputs/.
The test partition is kept out of feature selection, tuning, and threshold
selection. It is evaluated once, after the final configuration is chosen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODELLING_TABLE_PATH = ROOT / "data" / "modelling_table.csv"
OUTPUT_DIR = ROOT / "outputs"

# Pinned Part A / Part B feature order. Do not reorder.
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
TARGET = "is_delayed"
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]
TREE_DEPTHS = [2, 3, 4, 5, 6, 8, 10]
SELECT_K_VALUES = [5, 7, 10, 12, 17]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_modelling_data() -> tuple[pd.DataFrame, pd.Series]:
    if not MODELLING_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODELLING_TABLE_PATH}. Run Part A first: "
            "python part_a_clean_eda.py"
        )
    table = pd.read_csv(MODELLING_TABLE_PATH)
    missing = [column for column in FEATURES + [TARGET] if column not in table]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    X = table[FEATURES].copy()
    y = table[TARGET].astype("int64").copy()
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Missing values found in FEATURES or is_delayed.")
    if set(y.unique()) != {0, 1}:
        raise ValueError("is_delayed must contain both binary classes 0 and 1.")
    return X, y


def stratified_split(X: pd.DataFrame, y: pd.Series):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    sizes = (len(X_train), len(X_val), len(X_test))
    delayed = (int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))
    if sizes != (840, 180, 180):
        raise ValueError(f"Unexpected split sizes {sizes}; expected (840, 180, 180).")
    if delayed != (195, 42, 41):
        raise ValueError(
            f"Unexpected delayed counts {delayed}; expected (195, 42, 41)."
        )
    return X_train, X_val, X_test, y_train, y_val, y_test


def classification_metrics(y_true, y_pred) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def model_row(name: str, y_true, y_pred) -> dict[str, object]:
    return {"model": name, **classification_metrics(y_true, y_pred)}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_modelling_data()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    # 1. Blind baseline and three classifiers, all using the default 0.5 rule.
    blind_predictions = np.zeros(len(y_val), dtype="int64")
    model_rows = [model_row("blind_all_on_time", y_val, blind_predictions)]

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )
    logistic.fit(X_train, y_train)
    model_rows.append(
        model_row("logistic_regression", y_val, logistic.predict(X_val))
    )

    tree_candidates = []
    for depth in TREE_DEPTHS:
        tree = DecisionTreeClassifier(
            criterion="gini", max_depth=depth, random_state=42
        )
        tree.fit(X_train, y_train)
        pred = tree.predict(X_val)
        tree_candidates.append(
            {"depth": depth, "f1": f1_score(y_val, pred, zero_division=0), "model": tree}
        )
    best_tree = sorted(tree_candidates, key=lambda row: (-row["f1"], row["depth"]))[0]
    tree_name = f"decision_tree_depth_{best_tree['depth']}"
    model_rows.append(model_row(tree_name, y_val, best_tree["model"].predict(X_val)))

    default_forest_17 = RandomForestClassifier(n_estimators=100, random_state=42)
    default_forest_17.fit(X_train, y_train)
    model_rows.append(
        model_row("random_forest_default", y_val, default_forest_17.predict(X_val))
    )
    models_df = pd.DataFrame(
        model_rows,
        columns=["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"],
    )
    models_df.to_csv(OUTPUT_DIR / "part_c_models.csv", index=False)

    # 2. Feature scores and the five bookkeeping/noise columns.
    selector_all = SelectKBest(score_func=f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = np.nan_to_num(selector_all.scores_, nan=0.0, posinf=0.0, neginf=0.0)
    rf_importances = default_forest_17.feature_importances_
    score_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "f_score": f_scores,
            "rf_importance": rf_importances,
            "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES],
        }
    )
    score_df["f_rank"] = score_df["f_score"].rank(
        method="min", ascending=False
    ).astype(int)
    score_df = score_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    score_df.to_csv(OUTPUT_DIR / "part_c_feature_scores.csv", index=False)

    selectk_rows = []
    for k in SELECT_K_VALUES:
        selectk_pipeline = Pipeline(
            [
                ("selector", SelectKBest(score_func=f_classif, k=k)),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )
        selectk_pipeline.fit(X_train, y_train)
        val_pred = selectk_pipeline.predict(X_val)
        selectk_rows.append(
            {"k": k, "val_f1": f1_score(y_val, val_pred, zero_division=0)}
        )
    pd.DataFrame(selectk_rows).to_csv(OUTPUT_DIR / "part_c_selectk.csv", index=False)

    # 3. Fit feature selector on training rows only; use its transform for tuning.
    selector_7 = SelectKBest(score_func=f_classif, k=7)
    X_train_7 = selector_7.fit_transform(X_train, y_train)
    X_val_7 = selector_7.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5
    )
    grid.fit(X_train_7, y_train)
    grid_val_pred = grid.best_estimator_.predict(X_val_7)

    randomized = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=9,
        cv=5,
        random_state=42,
    )
    randomized.fit(X_train_7, y_train)
    randomized_val_pred = randomized.best_estimator_.predict(X_val_7)

    default_forest_7 = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    default_forest_7.fit(X_train_7, y_train)
    default_val_pred = default_forest_7.predict(X_val_7)

    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": len(grid.cv_results_["params"]),
            "cv_folds": 5,
            "total_fits": len(grid.cv_results_["params"]) * 5,
            "best_params": json.dumps(grid.best_params_, sort_keys=True),
            "best_cv_accuracy": float(grid.best_score_),
            "val_f1": float(f1_score(y_val, grid_val_pred, zero_division=0)),
        },
        {
            "search": "randomized_search",
            "candidates": len(randomized.cv_results_["params"]),
            "cv_folds": 5,
            "total_fits": len(randomized.cv_results_["params"]) * 5,
            "best_params": json.dumps(randomized.best_params_, sort_keys=True),
            "best_cv_accuracy": float(randomized.best_score_),
            "val_f1": float(f1_score(y_val, randomized_val_pred, zero_division=0)),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": np.nan,
            "val_f1": float(f1_score(y_val, default_val_pred, zero_division=0)),
        },
    ]
    tuning_columns = [
        "search", "candidates", "cv_folds", "total_fits", "best_params",
        "best_cv_accuracy", "val_f1",
    ]
    pd.DataFrame(tuning_rows, columns=tuning_columns).to_csv(
        OUTPUT_DIR / "part_c_tuning.csv", index=False
    )

    # 4. Choose by validation F1; deterministic tie priority as specified.
    tie_priority = {
        "default_forest_same_7_features": 0,
        "grid_search": 1,
        "randomized_search": 2,
    }
    winner = sorted(
        tuning_rows,
        key=lambda row: (-row["val_f1"], tie_priority[row["search"]]),
    )[0]
    chosen_from = str(winner["search"])
    if chosen_from == "grid_search":
        chosen_params = grid.best_params_
    elif chosen_from == "randomized_search":
        chosen_params = randomized.best_params_
    else:
        chosen_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
        }

    final_model = Pipeline(
        [
            ("selector", SelectKBest(score_func=f_classif, k=7)),
            (
                "model",
                RandomForestClassifier(random_state=42, **chosen_params),
            ),
        ]
    )
    final_model.fit(X_train, y_train)

    # 5. Select alert threshold using validation recall only.
    val_probabilities = final_model.predict_proba(X_val)[:, 1]
    threshold_rows = []
    for threshold in THRESHOLDS:
        val_predictions = (val_probabilities >= threshold).astype("int64")
        threshold_rows.append(
            {"threshold": threshold, **classification_metrics(y_val, val_predictions)}
        )
    thresholds_df = pd.DataFrame(
        threshold_rows,
        columns=["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"],
    )
    thresholds_df.to_csv(OUTPUT_DIR / "part_c_thresholds.csv", index=False)
    eligible = [row["threshold"] for row in threshold_rows if row["recall"] >= 0.80]
    chosen_threshold = max(eligible) if eligible else 0.1

    # 6. One final test evaluation at the selected threshold.
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= chosen_threshold).astype("int64")
    final_test_metrics = classification_metrics(y_test, test_predictions)
    final_test_df = pd.DataFrame(
        [{"threshold": chosen_threshold, **final_test_metrics}],
        columns=["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"],
    )
    final_test_df.to_csv(OUTPUT_DIR / "part_c_final_test.csv", index=False)

    final_config = pd.DataFrame(
        [
            {
                "selector_k": 7,
                "n_estimators": chosen_params["n_estimators"],
                "max_depth": (
                    "unlimited" if chosen_params.get("max_depth") is None
                    else chosen_params.get("max_depth")
                ),
                "min_samples_leaf": chosen_params["min_samples_leaf"],
                "threshold": chosen_threshold,
                "chosen_from": chosen_from,
            }
        ],
        columns=[
            "selector_k", "n_estimators", "max_depth", "min_samples_leaf",
            "threshold", "chosen_from",
        ],
    )
    final_config.to_csv(OUTPUT_DIR / "part_c_final_config.csv", index=False)

    # Persist compact run facts for README reporting.
    noise_importance_share = float(
        score_df.loc[score_df["is_noise"], "rf_importance"].sum()
    )
    noise_outside_top7 = bool(
        (score_df.loc[score_df["is_noise"], "f_rank"] > 7).all()
    )
    blind_metrics = model_rows[0]
    threshold_row = next(
        row for row in threshold_rows if row["threshold"] == chosen_threshold
    )
    selectk_best = max(selectk_rows, key=lambda row: (row["val_f1"], -row["k"]))
    default_tuning_f1 = next(
        row["val_f1"] for row in tuning_rows
        if row["search"] == "default_forest_same_7_features"
    )
    tuning_best_f1 = max(row["val_f1"] for row in tuning_rows)

    print("Part C - Classification completed")
    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(
        "Delayed counts: "
        f"train={int(y_train.sum())}, val={int(y_val.sum())}, test={int(y_test.sum())}"
    )
    print(
        f"Blind baseline accuracy={blind_metrics['accuracy']:.4f}; "
        f"precision={blind_metrics['precision']:.4f}; recall={blind_metrics['recall']:.4f}"
    )
    print(f"Noise-column combined RF importance: {noise_importance_share:.4f}")
    print(f"All five noise columns outside top 7 F-score: {noise_outside_top7}")
    print(f"Best SelectKBest k by validation F1: {selectk_best['k']}")
    print(
        f"Tuning winner: {chosen_from}; validation F1={winner['val_f1']:.4f}; "
        f"untuned F1={default_tuning_f1:.4f}; best tuned/default F1={tuning_best_f1:.4f}"
    )
    print(
        f"Chosen threshold={chosen_threshold:.1f}; validation precision="
        f"{threshold_row['precision']:.4f}; recall={threshold_row['recall']:.4f}; "
        f"F1={threshold_row['f1']:.4f}"
    )
    print(
        f"Test precision={final_test_metrics['precision']:.4f}; "
        f"recall={final_test_metrics['recall']:.4f}; "
        f"F1={final_test_metrics['f1']:.4f}"
    )
    print("Wrote all requested Part C CSV outputs.")


if __name__ == "__main__":
    main()
