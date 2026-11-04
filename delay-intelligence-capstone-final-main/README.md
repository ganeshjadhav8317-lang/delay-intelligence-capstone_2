# Delay Intelligence Capstone

## How to run

The complete workflow is available through one command. From the repository root, install the dependencies and run:

```bash
pip install -r requirements.txt
python generate_data.py
python run_all.py
```

`run_all.py` executes Part A first (cleaning, feature engineering, EDA, and modelling-table generation), then Part B (leakage-safe regression, diagnostics, model comparison and final test evaluation), Part C (stratified classification, feature selection, tuning and alert-threshold selection), and Part D (hub segmentation, model packaging and new-shipment scoring). You do not need to run the two parts separately.

The individual scripts remain available for independent execution: `python part_a_clean_eda.py`, `python part_b_regression.py`, and `python part_c_classification.py`.

## Part A — Data cleaning, exploratory analysis, and delay evidence

Run the complete Part A pipeline from the repository root:

```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
```

The pipeline cleans the historical shipment data, builds the modelling table, computes data-quality checks, and writes the delay evidence and numeric correlation matrix under `outputs/`.

## Data findings

The following findings are calculated from the generated Part A output files (`outputs/part_a_delay_by_group.csv`, `outputs/part_a_correlations.csv`, and `outputs/part_a_checks.csv`).

1. **Service type:** Express shipments have the highest delay rate at **0.4897 (48.97%)**, compared with Standard at 0.2056 and Economy at 0.0857.

2. **Hub operating tier:** The delay rate rises from **0.0400 (4.00%) in Tier 1** to **0.2150 (21.50%) in Tier 2** and **0.4400 (44.00%) in Tier 3**. This is a 40.00 percentage-point increase from Tier 1 to Tier 3.

3. **Distance band:** Delay rate increases with distance: **0.0219 (2.19%)** for 0–200 km, **0.1368 (13.68%)** for 201–400 km, **0.4364 (43.64%)** for 401–600 km, and **0.8523 (85.23%)** for 601+ km.

4. **Strongest correlations with delivery time:** Ranking the numeric columns by absolute Pearson correlation with `delivery_hours` (excluding `delivery_hours` itself), the top three are **`promised_hours` (+0.8372)**, **`distance_km` (+0.4013)**, and **`n_stops` (+0.3247)**. These are associations in this dataset and should not be interpreted as proof of causation.

5. **Why log-transform declared value:** In the filled modelling column, `declared_value_inr` has a mean of **₹1,533.0542** and median of **₹1,121.5000**, indicating a mean pulled above the median by higher values; after the natural-log transformation, the mean and median are much closer at **7.0277** and **7.0224**, respectively, so the transformation reduces the influence of the long right tail and makes the distribution less skewed.

### Generated outputs

- `data/modelling_table.csv` — cleaned, feature-engineered modelling dataset with the delay target.
- `outputs/part_a_checks.csv` — cleaning, missing-value, target, and distribution checks.
- `outputs/part_a_delay_by_group.csv` — shipment counts, delayed counts, delay rates, and mean delivery hours by service type, hub tier, and distance band.
- `outputs/part_a_correlations.csv` — Pearson correlation matrix for numeric modelling-table columns; the first column stores the row labels.

### Notes

- The delay target is defined as `delivery_hours > promised_hours`.
- The distance bands use cut edges `[0, 200, 400, 600, 2000]`, right-inclusive intervals, and labels `0-200`, `201-400`, `401-600`, and `601+`.
- Correlation describes linear association, not causation.

## Part B — Regression data split

Run the Part B preparation script from the repository root:

```bash
python part_b_regression.py
```

The script reads `data/modelling_table.csv`, selects the 17 regression inputs in the pinned `FEATURES` order, and uses `delivery_hours` as the target. It creates a reproducible 70% / 15% / 15% train-validation-test split with `random_state=42` in both split operations. For the 1,200-row Part A modelling table, the resulting partitions contain 840 training rows, 180 validation rows, and 180 test rows.

No scaler is fitted in this split-preparation step. When scaling is introduced in the subsequent modelling workflow, fit it on the training partition only and use that fitted scaler to transform validation and test data.

## Part B — Regression modelling and model selection

Run Part B after generating the Part A modelling table:

```bash
python part_b_regression.py
```

The script reads `data/modelling_table.csv`, uses the 17 pinned `FEATURES` in `part_b_regression.py`, and predicts `delivery_hours`. It creates the same reproducible 840 / 180 / 180 training, validation, and test partitions. The diagnostic `StandardScaler` is fitted on the 840 training rows only; that fitted scaler is then reused to transform validation and test rows. The linear-regression candidate is also implemented as a scaler-plus-regression `Pipeline`, so its preprocessing is fitted only on the rows supplied to each fit (and remains fold-safe during cross-validation).

## Model results (regression)

The following results are read from the generated Part B CSV files.

1. **Mean baseline:** The validation baseline has **R² = -0.0212**. A negative R² means predicting the training-target mean for every validation parcel performs slightly worse than using the actual mean of the validation targets as a constant prediction. It is therefore a weak reference point, not a useful operational model. Its validation RMSE is **18.8552 hours**.

2. **Validation winner:** **Linear Regression** has the lowest validation RMSE at **4.8912 hours** (MAE **3.9222 hours**, R² **0.9313**). Gradient Boosting is a very close second at **4.9165 hours**. The selected final model is Linear Regression because the specified selection rule is the lowest validation RMSE.

3. **Decision-tree depth diagnostic:** Depths **1–4** show high bias: both training and validation RMSE remain relatively high, indicating that shallow trees do not capture enough structure. Depth **6** gives the best validation RMSE (**6.2796 hours**) and is the selected decision-tree depth. Depths **8 and 12**, and the unlimited tree, show increasingly large train–validation gaps (approximately **3.454**, **5.993**, and **7.837 hours**, respectively), indicating high variance/overfitting. The unlimited tree reaches zero training RMSE while its validation RMSE rises to **7.8368 hours**.

4. **Gradient-boosting learning rate:** **0.1** is best, with validation RMSE **4.9165 hours** and R² **0.9306**. At **0.01**, learning is too slow for the fixed 100 estimators and the model underfits (RMSE **10.0018 hours**). At **1.0**, the large updates overshoot and generalize less well (RMSE **7.3701 hours**). The middle rate gives the best balance for this estimator and iteration count.

5. **Cross-validation comparison and shipping decision:** The two validation leaders were evaluated with the same shuffled 5-fold `KFold(random_state=42)` on the combined 1,020 training-plus-validation rows. Linear Regression has mean CV R² **0.9261** (standard deviation **0.0048**); Gradient Boosting has mean CV R² **0.9353** (standard deviation **0.0045**). Thus, the CV R² ranking favours Gradient Boosting, while the validation RMSE ranking narrowly favours Linear Regression. **Linear Regression remains the model selected for this capstone**, because the stated final-model rule selects the lowest validation RMSE; it also achieves a strong untouched test result. The CV result is a useful caution that Gradient Boosting may be more robust across folds, and the small validation RMSE difference should not be overstated.

6. **Final untouched test result:** The selected Linear Regression model, fitted on the 840 training rows, was evaluated once on the 180 test rows. It achieved **MAE 3.5671 hours, MSE 20.4624 hours², RMSE 4.5235 hours, and R² 0.9493**. The test set was not used to choose the model.

### Generated Part B outputs

- `outputs/part_b_checks.csv` — split sizes and maximum absolute scaled-column means for train and test.
- `outputs/part_b_tree_depth.csv` — training and validation RMSE across the requested tree depths.
- `outputs/part_b_metrics.csv` — validation metrics for the baseline and four candidates, plus the selected model's final test metrics.
- `outputs/part_b_gb_learning_rate.csv` — validation RMSE and R² for the three learning rates.
- `outputs/part_b_cv.csv` — mean and standard deviation of 5-fold R² for the two validation leaders.

**Leakage-control note:** The test set is held out from model selection and cross-validation. The final candidate is selected using validation RMSE, and the selected estimator is evaluated on the test partition only after selection.


## Part C — Delay classifier and alert threshold

Run Part C after Part A has generated `data/modelling_table.csv`:

```bash
python part_c_classification.py
```

The classifier uses the same 17 pinned `FEATURES` as Parts A and B and predicts `is_delayed`. It creates stratified 840 / 180 / 180 training, validation and test partitions with 195 / 42 / 41 delayed shipments, respectively. Both split operations use `random_state=42` and stratify on the target. The test partition is excluded from feature scoring, model comparison, feature-count selection, hyperparameter tuning and alert-threshold selection. It is evaluated once after the final configuration is fixed.

## Model results (classification)

The following findings are calculated from the generated Part C CSV files.

1. **Blind baseline:** Predicting every validation shipment as on-time gives accuracy **0.7667 (76.67%)**, but precision, recall and F1 are all **0.0000**. It identifies none of the 42 delayed validation parcels. The apparently reasonable accuracy merely reflects the 138 on-time parcels in the validation set; it is useless for detecting delays.

2. **Noise/bookkeeping columns:** The five noise columns (`label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, and `shelf_slot_no`) account for a combined **0.2244 (22.44%)** of the 17-feature random-forest importance. All five are outside the top seven by training-set F-score (their ranks are 8, 12, 14, 16 and 17). This is a useful reminder that tree-based importance can assign non-zero importance to bookkeeping variables even when univariate F-scores place them low.

3. **Number of selected features:** The validation F1 scores for `k = 5, 7, 10, 12, 17` are **0.6988, 0.8333, 0.7857, 0.8049 and 0.7848**, respectively. Seven features perform best in this experiment, so all subsequent tuning and the final model use `SelectKBest(f_classif, k=7)`.

4. **Hyperparameter tuning:** The untuned 100-tree forest on the selected seven features achieves validation F1 **0.8333**. Grid search (27 candidates, 5 folds, 135 fits) achieves **0.7750**; randomized search (9 candidates, 5 folds, 45 fits) achieves **0.7848**. Neither search beats the untuned forest on the 180-row validation set. This does not prove tuning is generally harmful: it shows that the validation sample is small and noisy, and that extra search complexity did not improve this particular held-out F1 result. The prescribed tie/selection rule therefore chooses `default_forest_same_7_features`.

5. **Alert threshold and operational trade-off:** The chosen threshold is **0.5**, the highest tested threshold whose validation recall is at least 0.80. At this threshold, validation precision is **0.7955**, recall is **0.8333**, and F1 is **0.8140** (35 true alerts among 44 alerts, with 7 missed delayed parcels). Lowering the threshold to 0.4 raises recall to 0.8810 but lowers precision to 0.7400, generating more unnecessary calls. Raising it to 0.6 improves precision to 0.8889 but recall falls to 0.7619, below the operational 0.80 requirement.

6. **Final test result:** At the selected 0.5 threshold, the untouched test set gives precision **0.6889**, recall **0.7561**, and F1 **0.7209**. Test recall is **7.72 percentage points lower** than validation recall (0.8333). The test partition contains only **41 delayed parcels**, so each additional missed parcel changes recall by about 2.44 percentage points. The test result is therefore less precise statistically and should be interpreted cautiously; it is reported as observed, not tuned after the fact.

### Generated Part C outputs

- `outputs/part_c_models.csv` — blind baseline, logistic regression, selected-depth decision tree, and default random forest at the 0.5 rule.
- `outputs/part_c_feature_scores.csv` — training-only F-scores/ranks, full-feature forest importances, and noise-column flags.
- `outputs/part_c_selectk.csv` — validation F1 across the requested feature counts.
- `outputs/part_c_tuning.csv` — grid search, randomized search, and untuned seven-feature forest comparison.
- `outputs/part_c_thresholds.csv` — validation confusion matrix and metrics at each exact decimal threshold.
- `outputs/part_c_final_test.csv` — one final test evaluation at the chosen threshold.
- `outputs/part_c_final_config.csv` — selected feature count, forest parameters, threshold, and winning candidate.

**Leakage-control note:** `SelectKBest` is fitted on training rows only before tuning. The final selector and classifier are fitted together in a pipeline on training rows. The validation partition is used for model and threshold selection; the test partition is not accessed until the final one-time evaluation.

## Part D — Hub segments, model packaging, and operational scoring

Part D turns the classification result into an operational workflow. It profiles the 24 hubs, clusters them into three business playbooks, checks whether their existing tier labels agree with observed operating behaviour, packages the final Part C classifier, and scores the 48 new shipments.

### Part D commands

From a fresh clone, run these commands in this order from the repository root:

```bash
pip install -r requirements.txt
python generate_data.py
python run_all.py
```

`run_all.py` runs Part A, Part B, Part C, Part D segmentation, model packaging, and new-shipment scoring in sequence. To regenerate the reusable classifier explicitly after Part C has produced `outputs/part_c_final_config.csv`, run:

```bash
python package_model.py
```

To score the new-shipment batch again using the saved model and hub segments, run:

```bash
python predict_delay.py
```

`models/delay_model.pkl` is regenerated by running `python package_model.py` (and automatically by `python run_all.py`). The committed pickle loads reliably only with the scikit-learn version pinned in `requirements.txt`. If the installed scikit-learn version differs, the grader should re-run `python package_model.py` in that environment rather than relying on the committed pickle. The model is rebuilt from the committed modelling table, final Part C configuration, and fixed random seeds.

### Hub segmentation and separation evidence

The four hub-level clustering inputs are mean delivery hours, delay rate, mean distance, and mean number of stops. They are standardized together, with the 24 hubs ordered by ascending `hub_id`, before K-Means. The business requires exactly three playbooks, so the production segmentation uses K=3. The resulting segments are:

| Segment | Hubs | Mean delivery hours | Mean delay rate | Mean distance (km) | Mean stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.24 | 2.57% | 188.93 | 55.37 |
| Watch | 8 | 56.81 | 15.50% | 314.18 | 70.72 |
| Escalate | 9 | 64.65 | 46.00% | 470.48 | 84.34 |

The **Escalate** hubs and their historical delay rates are: HUB13 Nagpur (46%), HUB15 Kochi (36%), HUB17 Patna (46%), HUB18 Ranchi (38%), HUB19 Guwahati (62%), HUB20 Raipur (48%), HUB21 Bhubaneswar (40%), HUB23 Siliguri (44%), and HUB24 Agartala (54%). These are hub-level historical rates, not guaranteed future outcomes.

The tier-to-segment mismatches are:

| Hub | City | Existing tier | Observed segment | Historical delay rate |
|---|---|---:|---|---:|
| HUB07 | Kolkata | 1 | Watch | 14% |
| HUB13 | Nagpur | 2 | Escalate | 46% |
| HUB15 | Kochi | 2 | Escalate | 36% |
| HUB22 | Dehradun | 3 | Watch | 20% |

The tier label should therefore be treated as an existing capacity/operating classification, not as a stand-alone proxy for current delay risk. In particular, HUB07's Tier 1 label should not suppress Watch-level attention; HUB13 and HUB15 merit Escalate-level attention despite Tier 2 labels; and HUB22's Tier 3 label should not automatically trigger the highest playbook without considering its observed segment.

The across-to-within distance ratios from `outputs/part_d_separation.csv` are K=2: **2.894**, K=3: **5.393**, K=4: **6.305**, and K=5: **6.887**. K=3 gives clear separation between within-cluster and across-cluster distances. Higher K increases this ratio, but it also fragments the hub population into four or five groups, while the business explicitly needs three playbooks. The incremental ratio gain diminishes from about 0.912 (K=3 to 4) to 0.582 (K=4 to 5); K=3 is therefore retained as the operationally mandated, interpretable compromise, not claimed to be the mathematical maximum.

## Operations recommendation

Use **0.5** as the current alert threshold, as selected in Part C by the highest tested threshold that retains at least 0.80 validation recall. On the held-out test set, precision is **68.89%** and recall is **75.61%**: approximately **6.9 of every 10 flagged parcels** were actually delayed, and approximately **7.6 of every 10 delayed parcels** were caught. Test F1 is **72.09%**. These are test estimates; the test set contains only 41 delayed parcels.

In the 48-parcel new batch, **8 parcels were flagged**. Of those, **6 require “Escalate to hub manager”** because their hubs belong to the Escalate segment; the other 2 flagged parcels receive “Proactive customer call”. The remaining 40 receive “No action”. The detailed parcel-level list is `outputs/new_shipment_actions.csv`.

### Model artefacts and scoring consistency

- `models/delay_model.pkl` — fitted Part C pipeline (SelectKBest with k=7 followed by the selected Random Forest).
- `models/delay_threshold.csv` — operational alert threshold.
- `models/feature_order.csv` — exact 17-feature input order.
- `models/impute_values.csv` — historical medians for distance, parcel weight, and declared value.
- `outputs/part_d_reload_check.csv` — verifies the saved model reloads and reproduces in-memory test probabilities. The current maximum absolute probability difference is **0.0** across 180 test rows.
- `outputs/new_shipment_actions.csv` — 48 new parcels, in input-file order, with probability, flag, hub segment, and action.

`predict_delay.py` uses the same `clean_shipments` and `build_features` functions as historical preparation, reuses stored historical medians instead of refitting them on the new batch, follows the saved feature order, and joins the fixed hub segments. Re-running the scoring script produces byte-identical `new_shipment_actions.csv` output for unchanged inputs and artefacts.

## Production plan

**Production-readiness characteristics mapped to repository evidence**

| Characteristic | Concrete repository item |
|---|---|
| Accuracy | `outputs/part_c_final_test.csv` reports held-out precision, recall, F1, and confusion-matrix counts; the threshold is chosen against an explicit recall requirement. |
| Reliability | `outputs/part_d_reload_check.csv` records 180 test rows scored and a zero maximum probability difference after saving and reloading the model. |
| Reproducibility | Fixed `random_state=42` settings, pinned package versions, deterministic hub ordering, and the single `run_all.py` entry point. |
| Performance (speed) | `predict_delay.py` scores the 48-row batch in one vectorized `predict_proba` call; record actual runtime in the deployment environment before setting a production SLA. |
| Maintainability | Cleaning/feature engineering is centralized in `src/shipment_prep.py`; segmentation, packaging, and scoring have separate scripts and CSV artefacts. |
| Consistency | Both historical training and new-batch inference call the same `clean_shipments` and `build_features`; inference loads training medians and the persisted feature order. |
| Monitorability | `outputs/new_shipment_actions.csv`, `outputs/part_c_final_test.csv`, `outputs/hub_profile.csv`, and `outputs/segment_profile.csv` provide auditable parcel- and hub-level measures. |

**Operational monitoring triggers (initial proposed thresholds)**

1. **Weekly delayed share of delivered parcels:** investigate if the rolling 7-day delayed share exceeds **28%** or rises by more than **5 percentage points** against the comparable baseline. The generated historical share is about 23.17%, so 28% is an initial alert, not a permanent service-level target.
2. **Weekly flagged share:** review capacity and threshold if more than **25%** of parcels are flagged in a rolling 7-day period, or if the share doubles versus the preceding four-week average. The current 48-parcel batch flag rate is 16.7% (8/48).
3. **Recall on matured outcomes:** once delivery outcomes are known, calculate recall on the flagged-model cohort weekly; trigger model/threshold review if rolling four-week recall falls below **0.80**. Report the denominator (number of actually delayed parcels) alongside recall.
4. **Input drift:** trigger investigation if a key feature's population stability index (PSI) exceeds **0.20** for two consecutive weekly comparisons, or if a hub's delay rate changes by at least **10 percentage points** versus its matched baseline.

**Retraining and event handling**

Retrain on a **quarterly** cadence using the latest matured shipment outcomes, with an additional review after a sustained monitoring trigger, a major network/process change, or a material feature-definition change. Keep a time-based holdout for evaluation and do not use parcels whose delivery outcome is not yet known. Recheck calibration, precision, recall, false negatives, and workload before promoting a replacement model.

The **15–28 September sale window** (`sale_event_flag`) should be evaluated against previous comparable sale-event windows, not ordinary weeks. Compare sale-period delay share, flagged share, feature distributions, and recall with the same event type and similar operating conditions. Data drift here could appear as changed distance/stop distributions, hub mix, parcel weight/value, service mix, or a different relationship between those features and delays; a higher sale-period volume alone is not necessarily model drift.

## Limitations

1. The shipment data is synthetic. It is useful for testing the end-to-end workflow but cannot establish real-world performance or operational causality.
2. The held-out test set contains only 41 delayed parcels. A small number of changed predictions can materially move precision and recall; confidence intervals and larger live evaluation samples are needed.
3. The alert threshold is selected from one 180-row validation split and a coarse grid of nine thresholds. Its measured recall may not generalize to other weeks, hubs, or sale periods; threshold stability and probability calibration should be checked on later data.
4. K-Means forces every hub into one of three clusters and is sensitive to the selected profile variables and scaling. The segment names are operational interpretations of cluster delay-rate ordering, not intrinsic hub categories.
5. The model uses historical hub attributes and shipment features; sudden capacity changes, new routes, policy changes, or unrecorded operational disruptions may not be represented.
