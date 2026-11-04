"""Part D: hub segmentation, separation evidence, and tier mismatch analysis."""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "modelling_table.csv"
OUT = ROOT / "outputs"
PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]


def mean_pair_distance(matrix, mask):
    distances = []
    for i, j in combinations(range(len(matrix)), 2):
        if mask[i] != mask[j]:
            distances.append(float(np.linalg.norm(matrix[i] - matrix[j])))
    return float(np.mean(distances)) if distances else float("nan")


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    profile = (df.groupby("hub_id", as_index=False, sort=True)
        .agg(shipments=("shipment_id", "count"),
             mean_delivery_hours=("delivery_hours", "mean"),
             delay_rate=("is_delayed", "mean"),
             mean_distance_km=("distance_km", "mean"),
             mean_n_stops=("n_stops", "mean"))
        .sort_values("hub_id", kind="stable").reset_index(drop=True))
    if len(profile) != 24:
        raise ValueError(f"Expected 24 hubs, found {len(profile)}")
    profile.to_csv(OUT / "hub_profile.csv", index=False)

    scaler = StandardScaler()
    matrix = scaler.fit_transform(profile[PROFILE_COLUMNS])
    if profile["hub_id"].tolist() != sorted(profile["hub_id"].tolist()):
        raise ValueError("Hub profile must be sorted by ascending hub_id")

    separation_rows = []
    for k in [2, 3, 4, 5]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix)
        across = mean_pair_distance(matrix, labels)
        within_distances = []
        for i, j in combinations(range(len(matrix)), 2):
            if labels[i] == labels[j]:
                within_distances.append(float(np.linalg.norm(matrix[i] - matrix[j])))
        within = float(np.mean(within_distances)) if within_distances else float("nan")
        sizes = sorted(pd.Series(labels).value_counts().tolist(), reverse=True)
        separation_rows.append({"K": k, "across_over_within": across / within,
                                "cluster_sizes": "-".join(map(str, sizes))})
    pd.DataFrame(separation_rows).to_csv(OUT / "part_d_separation.csv", index=False)

    labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(matrix)
    profile["cluster"] = labels
    cluster_delay = profile.groupby("cluster")["delay_rate"].mean().sort_values()
    names = {cluster: name for cluster, name in zip(cluster_delay.index, ["Standard", "Watch", "Escalate"])}
    profile["segment"] = profile["cluster"].map(names)

    hubs = pd.read_csv(ROOT / "data" / "hubs.csv")[["hub_id", "hub_city", "hub_tier"]]
    segments = profile[["hub_id", "segment"]].merge(hubs, on="hub_id", how="left", sort=False)
    segments = segments[["hub_id", "segment", "hub_city", "hub_tier"]].sort_values("hub_id", kind="stable")
    segments.to_csv(OUT / "hub_segments.csv", index=False)

    segment_profile = (profile.groupby("segment", as_index=False)
        .agg(hubs=("hub_id", "count"),
             mean_delivery_hours=("mean_delivery_hours", "mean"),
             delay_rate=("delay_rate", "mean"),
             mean_distance_km=("mean_distance_km", "mean"),
             mean_n_stops=("mean_n_stops", "mean")))
    order = pd.CategoricalDtype(["Standard", "Watch", "Escalate"], ordered=True)
    segment_profile["segment"] = segment_profile["segment"].astype(order)
    segment_profile = segment_profile.sort_values("segment").reset_index(drop=True)
    segment_profile["segment"] = segment_profile["segment"].astype(str)
    segment_profile.to_csv(OUT / "segment_profile.csv", index=False)

    expected = {1: "Standard", 2: "Watch", 3: "Escalate"}
    mismatch = segments[segments.apply(lambda r: expected.get(int(r["hub_tier"]), "") != r["segment"], axis=1)]
    mismatch.to_csv(OUT / "tier_vs_segment_mismatch.csv", index=False)

    print("Part D hub segmentation completed")
    print(segment_profile.to_string(index=False))
    print(f"Tier/segment mismatches: {len(mismatch)}")
    print(pd.DataFrame(separation_rows).to_string(index=False))

if __name__ == "__main__":
    main()
