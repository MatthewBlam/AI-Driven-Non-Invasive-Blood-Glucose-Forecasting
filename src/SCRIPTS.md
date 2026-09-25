# What each file does

## Data pipeline

| File | What it does |
|---|---|
| `preprocess.py` | Recovers the true 5-minute sensor readings from the interpolated public series; builds the clean data cache |
| `audit.py` | Data-quality audit; verifies every preprocessing claim numerically |
| `dataset.py` | Sliding windows, subject-level train/val/test splits, scaling; the PyTorch data module |
| `meals.py` | Loads and cleans the meal log |
| `meal_audit.py` | Data-quality audit of the meal log |
| `meal_features.py` | Builds the 21 meal features (timing, macros, meal type) used by B2 |
| `activity.py` | Loads and aligns per-minute heart rate and activity data |
| `activity_audit.py` | Data-quality audit of the activity data |
| `activity_features.py` | Builds the 11 physiology features (HR, energy, wear gaps) used by B3 |

## Models and training

| File | What it does |
|---|---|
| `model.py` | The LSTM (plus bidirectional and attention variants) |
| `train.py` | Trains a model, pooled runs or leave-one-subject-out folds |
| `sweep.py` | The 88-run hyperparameter sweep |
| `baselines.py` | Persistence, constant, and ridge yardsticks every model must beat |

## Evaluation and analysis

| File | What it does |
|---|---|
| `evaluate.py` | Loads any checkpoint, rebuilds its exact split and features, returns predictions |
| `metrics_table.py` | The full results table: RMSE, NRMSE, skill, Parkes zones, for every model and both LOSO arms |
| `analyze_loso.py` | Per-subject LOSO analysis: who improves, by glycemic class, significance tests |
| `compare_architectures.py` | LSTM vs bidirectional vs attention, paired by seed |
| `aggregate_horizons.py` | Multi-seed results across the +30/+60/+120 min horizons |
| `per_horizon_eval.py` | Per-horizon scores for the multi-horizon model |
| `ridge_meal_screen.py` | Fast linear screen: do meal features carry signal? (yes) |
| `ridge_activity_screen.py` | Fast linear screen: do activity features carry signal? (no) |
| `ablation_meals.py` | The B2 ablations: encodings, feature ladder, where the gain concentrates |
| `ablation_activity.py` | The B3 ablations: the measured activity null, stratified |
| `conformal.py` | Uncertainty ranges via conformal prediction; coverage checks on unseen patients |

## Figures and tables

| File | What it does |
|---|---|
| `figures.py` | All eleven paper figures |
| `plot_attention_weights.py` | What the attention model attends to (mostly the newest readings) |
| `plot_input_window.py` | One real hour of input, one row per baseline |
| `plot_carb_response.py` | Post-meal glucose response curves by carb load |
| `poster_figures.py` | Poster-sized variants of the four poster figures |
| `cohort_table.py` | Table 1: participant characteristics |
| `compute_footprint.py` | What the study cost to train (runs, hours, hardware) |

## One command

| File | What it does |
|---|---|
| `rebuild_all.py` | Rebuilds every table and figure from saved checkpoints (~18 min, no training) |
