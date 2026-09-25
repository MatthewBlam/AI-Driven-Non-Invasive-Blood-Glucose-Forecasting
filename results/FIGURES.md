# What each figure shows

## B1 (glucose only)

| Figure | What it shows |
|---|---|
| `loso_per_subject.png` | Error for each of the 44 held-out subjects; the model beats persistence for 41 |
| `error_bins.png` | Error by glucose level and trend; the model predicts falls well but misses the start of rises |
| `pred_vs_actual.png` | The easiest and hardest subjects side by side; the model will not follow extreme excursions |
| `pred_vs_actual_pooled.png` | The same picture for the mixed-subject split |
| `dispersion.png` | Predictions vary about 13% less than reality, for all 44 subjects |
| `rmse_vs_horizon.png` | Error as the forecast reaches further ahead; persistence degrades faster than the model |
| `parkes_grid.png` | The clinical view: 96% of predictions in the harmless zone, vs 93.5% for persistence |
| `sweep_null.png` | Every hyperparameter setting lands within seed noise; tuning does not help |
| `learning_curves.png` | All runs converge; nothing was under-trained |
| `seed_pairs.png` | Model vs persistence, paired seed by seed |
| `data_integrity.png` | The data discovery: 80% of the published 1-minute readings were interpolation, not sensor |
| `attention_weights.png` | Attention concentrates on the newest readings; with glucose-only input, "look at the present" is nearly optimal |

## B2 (adding meals)

| Figure | What it shows |
|---|---|
| `meal_ablation.png` | Every meal-feature setup helps on 11 of 11 seeds; the simple summary encoding wins |
| `meal_strata.png` | Where meals help: forecast skill jumps from 9.1% to 22.5% right after eating |
| `postprandial_trace.png` | One real post-meal stretch; B2 starts the rise that B1 misses |
| `loso_b2_per_subject.png` | 43 of 44 held-out subjects improve; B1's two failures flip to wins |
| `horizon_meal_benefit.png` | The meal benefit peaks at +60 minutes ahead and is gone by +120 |

## B3 (adding physiology)

| Figure | What it shows |
|---|---|
| `activity_ablation.png` | The headline null: activity features add nothing, and feeding them in naively hurts |
| `activity_screen_horizons.png` | A fast linear check at all three horizons; no activity signal anywhere |
| `activity_strata.png` | The null checked situation by situation, with a clean fasting-and-inactive control |

## Uncertainty (conformal prediction)

| Figure | What it shows |
|---|---|
| `conformal_coverage.png` | The 90% interval promise holds on the honest forward-in-time split |
| `conformal_strata.png` | The one-size interval fails exactly where glucose moves; per-situation intervals repair it |
| `conformal_subjects.png` | Coverage per person; calibrating on someone's own first days fixes the spread |
| `conformal_width.png` | What honest intervals cost in mg/dL, and what meal features buy back |

## Poster

| Figure | What it shows |
|---|---|
| `input_window.png` | One real hour of input, one row per baseline |
| `carb_response.png` | Post-meal glucose response by carb load; the response saturates above about 40 g |
| `poster_*.png` | Poster-sized variants of the four poster figures |
