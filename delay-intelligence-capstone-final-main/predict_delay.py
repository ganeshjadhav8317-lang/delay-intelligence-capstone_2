"""Score new shipments and produce a deterministic operational action list."""
from pathlib import Path
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

ROOT = Path(__file__).resolve().parent


def main():
    models = ROOT / "models"
    model = joblib.load(models / "delay_model.pkl")
    threshold = float(pd.read_csv(models / "delay_threshold.csv").iloc[0, 0])
    feature_order = pd.read_csv(models / "feature_order.csv")["feature"].tolist()
    medians_df = pd.read_csv(models / "impute_values.csv")
    medians = dict(zip(medians_df["column"], medians_df["median"].astype(float)))

    raw = pd.read_csv(ROOT / "data" / "new_shipments.csv")
    hubs = pd.read_csv(ROOT / "data" / "hubs.csv")
    segments = pd.read_csv(ROOT / "outputs" / "hub_segments.csv")[["hub_id", "segment"]]
    clean, _ = clean_shipments(raw, medians=medians)
    features = build_features(clean, hubs)
    X = features[feature_order]
    probabilities = model.predict_proba(X)[:, 1]
    rounded = [round(float(p), 4) for p in probabilities]
    flagged = [int(p >= threshold) for p in rounded]

    actions = pd.DataFrame({
        "shipment_id": clean["shipment_id"].values,
        "hub_id": clean["hub_id"].values,
        "delay_probability": rounded,
        "flagged": flagged,
    })
    actions = actions.merge(segments, on="hub_id", how="left", sort=False)
    if actions["shipment_id"].tolist() != clean["shipment_id"].tolist():
        raise ValueError("Hub-segment join changed the original new-shipment order")
    if actions["segment"].isna().any():
        raise ValueError("At least one new shipment has no matching hub segment")
    actions["action"] = [
        "Escalate to hub manager" if f and s == "Escalate" else
        "Proactive customer call" if f else "No action"
        for f, s in zip(actions["flagged"], actions["segment"])
    ]
    actions = actions[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    if len(actions) != len(raw):
        raise ValueError(f"Expected {len(raw)} scored parcels, got {len(actions)}")
    actions.to_csv(ROOT / "outputs" / "new_shipment_actions.csv", index=False, lineterminator="\n", float_format="%.4f")
    print(f"Scored {len(actions)} new shipments; flagged={sum(flagged)}; hub-manager escalations={(actions['action'] == 'Escalate to hub manager').sum()}")

if __name__ == "__main__":
    main()
