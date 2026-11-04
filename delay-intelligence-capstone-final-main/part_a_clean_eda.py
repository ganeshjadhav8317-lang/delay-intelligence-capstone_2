"""Part A: clean shipment data, build features, and produce EDA checks.

Run from the repository root:
    python part_a_clean_eda.py

Creates:
    data/modelling_table.csv
    outputs/part_a_checks.csv
    outputs/part_a_delay_by_group.csv
    outputs/part_a_correlations.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Make `src` importable when this script is run from the repository root
# or invoked directly by its file path.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shipment_prep import clean_shipments, build_features


# These are the 17 model inputs specified for the capstone.
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


def _declared_value_numeric(series: pd.Series) -> pd.Series:
    """Parse the raw rupee-formatted declared-value column for pre-fill checks."""
    return pd.to_numeric(
        series.astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _round_metric_values(checks: dict[str, object], digits: int = 4) -> pd.DataFrame:
    """Build the required two-column checks table and round numeric outputs."""
    rows = []
    for metric, value in checks.items():
        if isinstance(value, (float, np.floating)):
            value = round(float(value), digits)
        elif isinstance(value, (int, np.integer)):
            value = int(value)
        rows.append({"metric": metric, "value": value})
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> None:
    raw_path = ROOT / "data" / "raw_shipments.csv"
    hubs_path = ROOT / "data" / "hubs.csv"
    modelling_path = ROOT / "data" / "modelling_table.csv"
    checks_path = ROOT / "outputs" / "part_a_checks.csv"
    delay_by_group_path = ROOT / "outputs" / "part_a_delay_by_group.csv"
    correlations_path = ROOT / "outputs" / "part_a_correlations.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Could not find input file: {raw_path}")
    if not hubs_path.exists():
        raise FileNotFoundError(f"Could not find input file: {hubs_path}")

    raw = pd.read_csv(raw_path)
    hubs = pd.read_csv(hubs_path)

    required_raw = {
        "shipment_id", "hub_id", "service_type", "payment_mode",
        "parcel_weight_kg", "distance_km", "declared_value",
        "dispatch_time", "delivery_hours",
    }
    missing_columns = required_raw - set(raw.columns)
    if missing_columns:
        raise ValueError(
            "raw_shipments.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # ------------------------------------------------------------------
    # BEFORE metrics: after exact-duplicate removal, before normalization
    # and before any imputation.
    # ------------------------------------------------------------------
    deduplicated_raw = raw.drop_duplicates().copy()

    service_type_spellings_before = int(
        deduplicated_raw["service_type"].nunique()
    )
    missing_distance_before = int(deduplicated_raw["distance_km"].isna().sum())

    # This count is the actual missing-value count. Sentinel rows are
    # non-missing values (999 kg) and are counted separately.
    missing_parcel_weight_before = int(
        deduplicated_raw["parcel_weight_kg"].isna().sum()
    )
    sentinel_weight_count = int(
        deduplicated_raw["parcel_weight_kg"].gt(100).sum()
    )
    missing_declared_value_before = int(
        _declared_value_numeric(deduplicated_raw["declared_value"]).isna().sum()
    )
    missing_payment_mode_before = int(
        deduplicated_raw["payment_mode"].isna().sum()
    )

    duplicate_rows_removed = int(len(raw) - len(deduplicated_raw))

    # Call the required project cleaning function. It also removes exact
    # duplicates, normalizes service_type, parses values/timestamps,
    # converts sentinels to missing, and median-fills numeric gaps.
    clean, medians = clean_shipments(raw)

    service_type_spellings_after = int(clean["service_type"].nunique())

    # Attach hub data and derive all model features using the required
    # left-merge feature builder.
    modelling = build_features(clean, hubs)

    # Target is created only after feature construction. It is not used
    # by either cleaning or feature engineering.
    modelling["is_delayed"] = (
        modelling["delivery_hours"] > modelling["promised_hours"]
    ).astype("int64")

    # Confirm the modelling table is one row per shipment.
    if modelling["shipment_id"].duplicated().any():
        duplicated_ids = int(modelling["shipment_id"].duplicated().sum())
        raise ValueError(
            f"Modelling table contains {duplicated_ids} duplicate shipment IDs."
        )

    # The modelling table must contain all 17 FEATURES plus the target
    # columns used by the Part A completeness check.
    check_columns = FEATURES + ["delivery_hours", "is_delayed"]
    missing_model_columns = set(check_columns) - set(modelling.columns)
    if missing_model_columns:
        raise ValueError(
            "Modelling table is missing required feature/target columns: "
            + ", ".join(sorted(missing_model_columns))
        )

    # Declared-value distribution is calculated on the median-filled
    # modelling-table column, not just on originally observed values.
    declared_value = modelling["declared_value_inr"].astype(float)
    log_declared_value = np.log(declared_value)

    # Calculate quartiles and IQR from unrounded delivery-hour values.
    hours = modelling["delivery_hours"].astype(float)
    q1 = float(hours.quantile(0.25))
    q3 = float(hours.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    hours_outlier_count = int(
        ((hours < lower_bound) | (hours > upper_bound)).sum()
    )

    checks = {
        "raw_rows": int(len(raw)),
        "rows_after_deduplication": int(len(deduplicated_raw)),
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_distance_before": missing_distance_before,
        "missing_parcel_weight_before": missing_parcel_weight_before,
        "missing_declared_value_before": missing_declared_value_before,
        "missing_payment_mode_before": missing_payment_mode_before,
        "sentinel_weight_count": sentinel_weight_count,
        "service_type_spellings_before": service_type_spellings_before,
        "service_type_spellings_after": service_type_spellings_after,
        "historical_median_distance_km": medians["distance_km"],
        "historical_median_parcel_weight_kg": medians["parcel_weight_kg"],
        "historical_median_declared_value_inr": medians["declared_value_inr"],
        "modelling_rows": int(len(modelling)),
        "unique_shipments": int(modelling["shipment_id"].nunique()),
        "nulls_in_features_and_target": int(
            modelling[check_columns].isna().sum().sum()
        ),
        "declared_value_mean_inr": float(declared_value.mean()),
        "declared_value_median_inr": float(declared_value.median()),
        "log_declared_value_mean": float(log_declared_value.mean()),
        "log_declared_value_median": float(log_declared_value.median()),
        "hours_min": float(hours.min()),
        "hours_max": float(hours.max()),
        "hours_mean": float(hours.mean()),
        "hours_median": float(hours.median()),
        "hours_q1": q1,
        "hours_q3": q3,
        "hours_iqr": iqr,
        "hours_lower_bound": lower_bound,
        "hours_upper_bound": upper_bound,
        "hours_outlier_count": hours_outlier_count,
        "delayed_shipments": int(modelling["is_delayed"].sum()),
        "on_time_shipments": int((modelling["is_delayed"] == 0).sum()),
        "delayed_rate": float(modelling["is_delayed"].mean()),
    }

    # Create output directories and write both deliverables.
    modelling_path.parent.mkdir(parents=True, exist_ok=True)
    checks_path.parent.mkdir(parents=True, exist_ok=True)

    modelling.to_csv(modelling_path, index=False)
    _round_metric_values(checks).to_csv(checks_path, index=False)

    # ------------------------------------------------------------------
    # Delay evidence table: compare delay performance across service,
    # hub operating tier, and distance bands.
    # ------------------------------------------------------------------
    distance_labels = ["0-200", "201-400", "401-600", "601+"]
    delay_evidence_source = modelling.copy()
    delay_evidence_source["distance_band"] = pd.cut(
        delay_evidence_source["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=distance_labels,
        right=True,
        include_lowest=True,
    )

    delay_group_rows = []
    for dimension in ["service_type", "hub_tier", "distance_band"]:
        grouped = delay_evidence_source.groupby(
            dimension,
            observed=False,
            dropna=False,
        ).agg(
            shipments=("shipment_id", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean"),
        ).reset_index()

        grouped = grouped.rename(columns={dimension: "group"})
        grouped.insert(0, "dimension", dimension)
        delay_group_rows.append(grouped)

    delay_by_group = pd.concat(delay_group_rows, ignore_index=True)
    delay_by_group = delay_by_group[
        ["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]
    ]
    delay_by_group["group"] = delay_by_group["group"].astype("string")
    delay_by_group.to_csv(delay_by_group_path, index=False)

    # Correlations across every numeric column in the modelling table.
    # index_label preserves the matrix row names in the first CSV column.
    numeric_modelling = modelling.select_dtypes(include=[np.number])
    correlations = numeric_modelling.corr()
    correlations.to_csv(correlations_path, index=True, index_label="column")

    print("Part A cleaning and EDA completed.")
    print(f"Modelling table: {modelling_path.relative_to(ROOT)}")
    print(f"Checks file:     {checks_path.relative_to(ROOT)}")
    print(f"Delay evidence:  {delay_by_group_path.relative_to(ROOT)}")
    print(f"Correlations:    {correlations_path.relative_to(ROOT)}")
    print(f"Unique shipments: {len(modelling):,}")
    print(f"Duplicate rows removed: {duplicate_rows_removed:,}")
    print(f"Median values: {medians}")
    print(f"Nulls in features and target: {checks['nulls_in_features_and_target']:,}")


if __name__ == "__main__":
    main()
