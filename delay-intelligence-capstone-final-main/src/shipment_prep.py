"""Shipment data cleaning and median imputation utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


MEDIAN_COLUMNS = (
    "distance_km",
    "parcel_weight_kg",
    "declared_value_inr",
)


def clean_shipments(
    raw: pd.DataFrame,
    medians: Optional[dict[str, float]] = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Clean raw shipment records and impute numeric gaps with medians.

    When ``medians`` is omitted, calculate them from the cleaned historical
    table before imputation. When supplied, reuse those values to clean later
    batches without recalculating statistics from those batches.

    Returns the cleaned DataFrame and a dictionary containing exactly the
    keys ``distance_km``, ``parcel_weight_kg``, and ``declared_value_inr``.
    """
    clean_df = raw.copy()

    # 1. Remove exact duplicate rows.
    clean_df = clean_df.drop_duplicates().copy()

    # 2. Normalize service names.
    clean_df["service_type"] = (
        clean_df["service_type"].astype("string").str.strip().str.title()
    )

    # 3. Replace missing payment modes with an explicit category.
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")

    # 4. Convert rupee-formatted declared value to numeric INR.
    clean_df["declared_value_inr"] = pd.to_numeric(
        clean_df["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )
    clean_df = clean_df.drop(columns=["declared_value"])

    # 5. Convert dispatch timestamp to datetime.
    clean_df["dispatch_time"] = pd.to_datetime(
        clean_df["dispatch_time"], errors="coerce"
    )

    # 6. Treat weights above 100 kg as scanner-fault sentinels.
    clean_df.loc[
        clean_df["parcel_weight_kg"] > 100, "parcel_weight_kg"
    ] = float("nan")

    # 7. Calculate medians on historical rows or reuse supplied medians.
    if medians is None:
        median_values: dict[str, float] = {}
        for column in MEDIAN_COLUMNS:
            value = clean_df[column].median()
            if pd.isna(value):
                raise ValueError(
                    f"Cannot calculate median for {column}: "
                    "no non-missing values remain."
                )
            median_values[column] = float(value)
    else:
        missing_keys = set(MEDIAN_COLUMNS) - set(medians)
        if missing_keys:
            raise ValueError(
                "medians is missing required keys: "
                + ", ".join(sorted(missing_keys))
            )
        median_values = {
            column: float(medians[column]) for column in MEDIAN_COLUMNS
        }

    for column in MEDIAN_COLUMNS:
        # Cast to float first: medians can be fractional even when source
        # values (such as declared_value) were inferred as nullable integers.
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce").astype(float)
        clean_df[column] = clean_df[column].fillna(median_values[column])

    return clean_df, median_values


PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96,
}


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attach hub attributes and create shipment model features.

    The merge is a left join, so every input shipment is retained. This
    function deliberately does not reference ``delivery_hours`` and can
    therefore be used for undelivered shipments as well as historical data.
    """
    hub_columns = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity",
    ]

    features = clean.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left",
        sort=False,
    )

    # Map the promised delivery window from the service-level definition.
    features["promised_hours"] = (
        features["service_type"].map(PROMISED_HOURS).astype("int64")
    )

    # Binary operational indicators.
    features["is_cod"] = (
        features["payment_mode"].eq("COD").astype("int64")
    )
    features["is_weekend"] = (
        features["dispatch_time"].dt.dayofweek.ge(5).astype("int64")
    )

    # Dispatch hour is an integer (generated data: 6 through 21).
    features["dispatch_hour"] = (
        features["dispatch_time"].dt.hour.astype("int64")
    )

    # Natural logarithm of the numeric declared value in INR.
    features["log_declared_value"] = np.log(
        features["declared_value_inr"].astype(float)
    )

    return features
