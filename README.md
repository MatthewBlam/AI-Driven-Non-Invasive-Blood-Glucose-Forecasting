# AI-Driven-Non-Invasive-Blood-Glucose-Forecasting

Can a glucose forecaster do better if it knows about your last meal, or how
active you have been? This project trains a small LSTM (17k parameters) to
predict blood glucose 30 to 120 minutes ahead for the 44 subjects of the
public CGMacros dataset, comparing three input sets: CGM history alone, plus
a summary of the most recent meal, plus fitness tracker physiology (heart
rate and activity calories) on top of that. The published 1-minute glucose
series turned out to be 80% interpolation, which leaks future values into
the input, so the true 5-minute sensor readings were recovered first and
everything runs on those.

## Headline results

- 12.5% lower error than the no-change baseline on patients the model never
  saw; 43 of 44 people improve
- Meal information doubles forecast skill right after eating, from 9.1% to
  22.5%
- 97% of predictions are clinically harmless on the Parkes error grid, vs
  93.5% for the baseline
- When you ate matters more than what you ate: meal timing alone carries
  about 80% of the benefit, and among macronutrients only carbs add signal
- The model was never the bottleneck: four architectures and an 88-run
  hyperparameter sweep all tie, so the gains had to come from better inputs
- Activity data (heart rate, calories) adds nothing, a carefully measured
  null
- Every forecast carries an uncertainty range that holds up on unseen
  patients

## What is here

| Path | Contents |
|---|---|
| `src/` | All source code, from preprocessing to figures. `src/SCRIPTS_PUBLIC.md` says what each file does in one line; `src/SCRIPTS.md` is the full reference |
| `results/` | Every figure and result table. `results/FIGURES_PUBLIC.md` says what each figure shows in one line; `results/FIGURES.md` is the full guide. The CSVs are the per-seed and per-subject numbers behind every claim above |
| `REFERENCES.md` | Dataset, methods, and background citations |

The raw CGMacros data is not included; download it from PhysioNet (see
`REFERENCES.md`) to run the pipeline yourself.
