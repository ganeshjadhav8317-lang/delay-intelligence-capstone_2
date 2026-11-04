"""Run the complete Delay Intelligence Capstone: Parts A through D."""
from part_a_clean_eda import main as run_part_a
from part_b_regression import main as run_part_b
from part_c_classification import main as run_part_c
from part_d_segments import main as run_part_d
from package_model import main as package_final_model
from predict_delay import main as score_new_shipments


def main():
    steps = [
        ("Part A: data cleaning, feature engineering, and EDA", run_part_a),
        ("Part B: regression modelling and model selection", run_part_b),
        ("Part C: delay classification and alert threshold", run_part_c),
        ("Part D: hub segmentation and tier mismatch analysis", run_part_d),
        ("Package final classifier", package_final_model),
        ("Score new shipments and create action list", score_new_shipments),
    ]
    print("=" * 76)
    print("DELAY INTELLIGENCE CAPSTONE — COMPLETE WORKFLOW (PARTS A–D)")
    print("=" * 76)
    for i, (label, function) in enumerate(steps, start=1):
        print(f"\n[{i}/{len(steps)}] {label}\n")
        function()
    print("\n" + "=" * 76)
    print("COMPLETE WORKFLOW FINISHED SUCCESSFULLY")
    print("Parts A–D, model package, and new-shipment action list are ready.")
    print("=" * 76)

if __name__ == "__main__":
    main()
