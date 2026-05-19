# EfficientKAN Study Log for 16QAM BER Equalization

Date started: 2026-05-10

This file is the running research log for the focused EfficientKAN study. The goal is to understand the behavior of B-spline EfficientKAN in two formulations:

1. Regression equalization: input IQ window -> corrected `(I, Q)`.
2. Classification/detection: input IQ window -> 16 logits for 16QAM symbols.

The broader RBF/FastKAN work remains important, but this document focuses specifically on EfficientKAN so that the story does not get mixed with other architectures.

## Current Motivation

Previous experiments showed:

- `efficient_kan_baseline` works as a nonlinear equalizer, but its final BER is worse than RBF/FastKAN.
- `efficient_kan_baseline` achieved approximately `3.514662e-03` BER from raw baseline `9.741273e-03`, i.e. `63.92%` relative improvement.
- `efficient_kan_residual` was close but slightly worse, likely because the residual branch mixed standardized RX coordinates with differently scaled target coordinates.
- `efficient_kan_features` was much weaker, meaning handcrafted statistics destroyed useful temporal structure.
- `cnn_kan` improved strongly, showing that learned temporal features before KAN help.
- RBF/FastKAN achieved about `89%` relative improvement, so the KAN family is promising, but EfficientKAN needs a careful isolated study.

The key research question now:

```text
How good can plain B-spline EfficientKAN become for 16QAM BER equalization, and when is regression better than direct 16-class classification?
```

## Fixed Experimental Setup

Unless explicitly varied:

```text
Dataset directory: symbols_new
Train files: 1..61
Validation file: 62
Test files: 63, 64
Baseline BER: 9.741273e-03
CONTEXT_K = 32
SEQ_LEN = 65
INPUT_DIM = 2
Flattened input dim = 130
POWER_NORMALIZE = True
BER_SCALE_SEARCH = True for regression models
Optimizer = Adam
Loss for regression = MSE
Loss for classifier = CrossEntropy
SAVE_BEST_BY = val_ber
```

Important comparison rule:

- Regression models output `(I, Q)`, then BER is computed by nearest 16QAM constellation point.
- Classifier models output 16 logits, then BER is computed by `argmax`.
- Loss curves between regression and classification are not directly comparable because MSE and CE have different units. Compare them primarily by BER, SER/accuracy, convergence speed, parameter count, and samples/sec.

## Architectures Under Study

### A. EfficientKAN Regression

Code model name:

```text
efficient_kan_baseline
```

Structure:

```text
Input: 130 real-valued IQ-window features
EfficientKAN: 130 -> hidden -> ... -> hidden -> 2
Output: corrected [I, Q]
Decision: nearest 16QAM point after optional BER scale search
Loss: MSE
```

Default:

```text
hidden = 128
layers = 2
grid_size = 8
spline_order = 3
grid_range = [-3.0, 3.0]
```

Known result:

```text
Params = 432,640
BER = 3.514662e-03
Relative improvement = 63.92%
Speed ~= 309k samples/s
```

### B. EfficientKAN Classifier

Code model name:

```text
kan_classifier
```

Structure:

```text
Input: 130 real-valued IQ-window features
EfficientKAN: 130 -> hidden -> ... -> hidden -> 16
Output: 16 logits, one for each 16QAM constellation point
Decision: argmax over 16 logits
Loss: CrossEntropy
```

Difference from regression:

```text
Regression: output_dim = 2, loss = MSE, decision = nearest constellation point.
Classifier: output_dim = 16, loss = CrossEntropy, decision = argmax.
```

Known result:

```text
Params = 455,936
BER = TBD
Relative improvement = TBD
Speed = TBD
```

## Metrics To Record

For every run, record:

- `model_type`
- experiment ID
- date
- git commit or manual code state if no commit
- seed
- train files / validation file / test files
- `CONTEXT_K`
- hidden dimension
- number of EfficientKAN layers
- `grid_size`
- `spline_order`
- `grid_range`
- learning rate
- scheduler settings
- epoch count
- train loss at best epoch
- best validation BER
- best validation loss
- test BER at best checkpoint
- final test BER
- baseline BER
- relative improvement
- absolute BER improvement
- equalizer scale for regression models
- accuracy / SER
- parameter count
- samples/sec
- epoch time
- notes or anomalies

## Primary Plots

These are the main plots for the report.

### 1. BER vs Hidden Dimension

Purpose:

```text
Does larger width help EfficientKAN, and does classifier need different width than regression?
```

Sweep:

```text
hidden_dim in [32, 64, 96, 128, 192, 256]
models = [efficient_kan_baseline, kan_classifier]
fixed grid_size = 8
fixed spline_order = 3
fixed layers = 2
fixed lr = 1e-3 initially
```

Plot:

```text
x-axis: hidden_dim
y-axis: best validation/test BER, log scale
curves: regression vs classifier
secondary labels: parameter count
```

Expected interpretation:

- If BER saturates early, more hidden units are not useful.
- If classifier benefits from larger width more than regression, class separation may require more capacity.
- If regression is consistently better at similar parameter count, direct IQ equalization is preferable.

### 2. BER vs Learning Rate

Purpose:

```text
Find stable LR range and whether current 1e-3 is too high or too low.
```

Sweep:

```text
lr in [3e-4, 5e-4, 1e-3, 2e-3, 3e-3]
models = [efficient_kan_baseline, kan_classifier]
fixed hidden_dim = 128
fixed grid_size = 8
fixed spline_order = 3
fixed layers = 2
```

Plot:

```text
x-axis: learning rate
y-axis: best validation/test BER, log scale
curves: regression vs classifier
```

Additional plot:

```text
epoch -> val_ber
for each lr
```

Expected interpretation:

- If `1e-3` gives long plateau and late improvement only after LR decay, try smaller initial LR or faster scheduler.
- Classifier may need different LR because CE gradients differ from MSE gradients.

### 3. BER vs Grid Size

Purpose:

```text
Understand how spline resolution affects BER.
```

Sweep:

```text
grid_size in [4, 6, 8, 12, 16]
models = [efficient_kan_baseline, kan_classifier]
fixed hidden_dim = 128
fixed spline_order = 3
fixed layers = 2
fixed lr = best from LR sweep or 1e-3
```

Plot:

```text
x-axis: grid_size
y-axis: BER, log scale
curves: regression vs classifier
secondary axis or labels: parameter count
```

Expected interpretation:

- Low grid size may underfit nonlinear correction.
- High grid size may overfit/noisify and increase parameters.
- If RBF-KAN remains much better, spline basis may be less suitable than Gaussian RBF basis for this noisy IQ task.

### 4. BER vs Spline Order

Purpose:

```text
Check whether cubic splines are actually necessary.
```

Sweep:

```text
spline_order in [1, 2, 3, 4]
models = [efficient_kan_baseline, kan_classifier]
fixed hidden_dim = 128
fixed grid_size = 8
fixed layers = 2
```

Plot:

```text
x-axis: spline_order
y-axis: BER, log scale
```

Expected interpretation:

- `order = 1`: continuous but not smooth, piecewise linear.
- `order = 2`: smoother.
- `order = 3`: cubic, default.
- `order = 4`: smoother but more expensive and may over-smooth or overfit.

Important: parameter count changes because each edge has:

```text
1 base weight + (grid_size + spline_order) spline weights + 1 spline scaler
```

### 5. BER vs Number of KAN Layers

Purpose:

```text
Check depth vs width tradeoff.
```

Sweep:

```text
layers in [1, 2, 3, 4]
models = [efficient_kan_baseline, kan_classifier]
fixed hidden_dim = 128
fixed grid_size = 8
fixed spline_order = 3
```

Plot:

```text
x-axis: number of hidden KAN layers
y-axis: BER, log scale
curves: regression vs classifier
```

Expected interpretation:

- More layers may improve nonlinear composition but can slow training and increase overfitting.
- If one hidden layer is close to two layers, a shallow KAN may be preferable for complexity.

### 6. BER vs Context Window

Purpose:

```text
Understand how much temporal memory EfficientKAN needs.
```

Sweep:

```text
CONTEXT_K in [8, 16, 24, 32, 48]
SEQ_LEN = 2 * CONTEXT_K + 1
models = [efficient_kan_baseline, kan_classifier]
fixed hidden_dim = 128
fixed grid_size = 8
fixed spline_order = 3
```

Plot:

```text
x-axis: SEQ_LEN or CONTEXT_K
y-axis: BER, log scale
curves: regression vs classifier
```

Expected interpretation:

- If BER improves with larger context, channel memory matters.
- If BER saturates around smaller context, use smaller window for speed/parameters.

### 7. Convergence Curves

Purpose:

```text
Show not only final BER, but training dynamics.
```

Plot for selected best runs:

```text
epoch -> train loss
epoch -> val BER
epoch -> test BER when evaluated
epoch -> learning rate
```

Important:

- Regression MSE and classifier CE cannot be compared on the same loss scale.
- BER curves are the fair comparison.

### 8. Accuracy/SER vs BER for Classifier

Purpose:

```text
Check if classifier improves symbol accuracy but not bit accuracy, or vice versa.
```

Plot:

```text
epoch -> SER
epoch -> BER
```

Interpretation:

- Because Gray coding makes nearby symbol mistakes less harmful than far mistakes, two models with similar SER can have different BER.
- Classifier should be inspected for whether mistakes are mostly nearest-neighbor or far constellation jumps.

### 9. Parameter Count vs BER

Purpose:

```text
Compare accuracy/complexity tradeoff.
```

Plot:

```text
x-axis: trainable parameters
y-axis: BER, log scale
points: each EfficientKAN configuration
marker style: regression vs classifier
```

Expected interpretation:

- Identify Pareto-optimal EfficientKAN configurations.
- Compare later against RBF/FastKAN and CNN_KAN.

### 10. Throughput vs BER

Purpose:

```text
Find architectures that improve BER without becoming too slow.
```

Plot:

```text
x-axis: samples/sec
y-axis: BER, log scale
points: selected runs
```

This is important because RBF/FastKAN is currently both strong and relatively compact.

## Secondary / Optional Plots

### Grid Range Sweep

Motivation:

EfficientKAN uses fixed grid range. Inputs are standardized RX windows, so most values should lie around a few standard deviations.

Sweep:

```text
grid_range in [[-2,2], [-2.5,2.5], [-3,3], [-4,4]]
```

Question:

```text
Does too wide a range waste grid resolution?
Does too narrow a range clip/out-of-range behavior?
```

### BER vs `scale_noise`

Sweep:

```text
EFFICIENT_KAN_SCALE_NOISE in [0.01, 0.05, 0.1, 0.2]
```

Question:

```text
Does spline initialization affect convergence or final BER?
```

### BER vs KAN Regularization

Sweep:

```text
KAN_PRUNE_L1 in [0, 1e-6, 1e-5, 1e-4]
```

Question:

```text
Can we encourage simpler spline functions without losing BER?
```

## Recommended Experiment Phases

### Phase 0: Smoke Tests

Goal:

Ensure both EfficientKAN regression and classifier train and evaluate correctly.

Runs:

```text
efficient_kan_baseline, hidden=128, lr=1e-3, epochs=5
kan_classifier, hidden=128, lr=1e-3, epochs=5
```

Checks:

- No shape errors.
- Classifier uses CE and outputs 16 logits.
- Regression uses MSE and outputs 2 coordinates.
- BER computation works for both.

### Phase 1: Coarse Hyperparameter Sweep

Goal:

Find promising ranges cheaply.

Suggested settings:

```text
epochs = 60 or 80
TEST_BER_EVERY = 10
SAVE_BEST_BY = val_ber
max_test_files = 1 if runtime is too high
```

Sweeps:

1. Hidden dimension.
2. Learning rate.
3. Grid size.

Reason:

These are likely the highest-impact variables.

### Phase 2: Medium Sweep Around Winners

Goal:

Refine best region.

Use:

```text
epochs = 120 or 150
test files = full [63, 64]
```

Sweeps:

1. Best hidden values from Phase 1.
2. Best LR values from Phase 1.
3. `grid_size` near best value.
4. `spline_order` around 2-4.
5. layers 1-3.

### Phase 3: Final Runs

Goal:

Produce reliable report numbers.

Use:

```text
epochs = 250
test files = full [63, 64]
seeds = 3 if possible
```

Final candidates:

- best EfficientKAN regression
- best EfficientKAN classifier
- default EfficientKAN regression
- RBF/FastKAN reference
- MLP/CNN reference if final comparison is needed

Report:

- mean BER over seeds
- standard deviation
- best single run
- parameter count
- samples/sec

## Proposed Experiment IDs

Use compact IDs so files and plots are easy to track.

```text
EKREG-H{hidden}-L{layers}-G{grid}-O{order}-LR{lr}
EKCLS-H{hidden}-L{layers}-G{grid}-O{order}-LR{lr}
```

Examples:

```text
EKREG-H128-L2-G8-O3-LR1e-3
EKCLS-H128-L2-G8-O3-LR1e-3
EKREG-H192-L2-G12-O3-LR5e-4
```

## Tables To Fill During Experiments

### Run Table

| ID | Model | Hidden | Layers | Grid | Order | LR | Epochs | Params | Best Val BER | Test BER | Improvement | Samples/s | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EKREG-H128-L2-G8-O3-LR1e-3 | regression | 128 | 2 | 8 | 3 | 1e-3 | 250 | 432,640 | TBD | 3.514662e-03 | 63.92% | ~309k | previous run |
| EKCLS-H128-L2-G8-O3-LR1e-3 | classifier | 128 | 2 | 8 | 3 | 1e-3 | TBD | 455,936 | TBD | TBD | TBD | TBD | needs run |

### Hidden Sweep Table

| Model | Hidden | Params | Best Val BER | Test BER | Samples/s | Notes |
|---|---:|---:|---:|---:|---:|---|
| regression | 32 | TBD | TBD | TBD | TBD | |
| regression | 64 | TBD | TBD | TBD | TBD | |
| regression | 96 | TBD | TBD | TBD | TBD | |
| regression | 128 | 432,640 | TBD | 3.514662e-03 | ~309k | previous default |
| regression | 192 | TBD | TBD | TBD | TBD | |
| regression | 256 | TBD | TBD | TBD | TBD | |
| classifier | 32 | TBD | TBD | TBD | TBD | |
| classifier | 64 | TBD | TBD | TBD | TBD | |
| classifier | 96 | TBD | TBD | TBD | TBD | |
| classifier | 128 | 455,936 | TBD | TBD | TBD | |
| classifier | 192 | TBD | TBD | TBD | TBD | |
| classifier | 256 | TBD | TBD | TBD | TBD | |

### Learning Rate Sweep Table

| Model | LR | Best Val BER | Test BER | Epoch of Best | Notes |
|---|---:|---:|---:|---:|---|
| regression | 3e-4 | TBD | TBD | TBD | |
| regression | 5e-4 | TBD | TBD | TBD | |
| regression | 1e-3 | TBD | 3.514662e-03 | TBD | previous default |
| regression | 2e-3 | TBD | TBD | TBD | |
| regression | 3e-3 | TBD | TBD | TBD | |
| classifier | 3e-4 | TBD | TBD | TBD | |
| classifier | 5e-4 | TBD | TBD | TBD | |
| classifier | 1e-3 | TBD | TBD | TBD | |
| classifier | 2e-3 | TBD | TBD | TBD | |
| classifier | 3e-3 | TBD | TBD | TBD | |

## Plot File Naming

Use names like:

```text
efficientkan_outputs/ber_vs_hidden_reg_vs_cls.png
efficientkan_outputs/ber_vs_lr_reg_vs_cls.png
efficientkan_outputs/ber_vs_grid_size_reg_vs_cls.png
efficientkan_outputs/ber_vs_spline_order_reg_vs_cls.png
efficientkan_outputs/ber_vs_layers_reg_vs_cls.png
efficientkan_outputs/ber_vs_context_reg_vs_cls.png
efficientkan_outputs/params_vs_ber.png
efficientkan_outputs/speed_vs_ber.png
efficientkan_outputs/convergence_best_reg_vs_cls.png
```

## Analysis Rules

To avoid misleading conclusions:

1. Compare regression and classifier primarily by BER, not by training loss.
2. Always report parameter count because KAN parameter count grows quickly with `hidden`, `grid_size`, and `spline_order`.
3. Always report samples/sec because a slightly better BER may not justify much slower inference.
4. Use the same train/val/test split for all runs.
5. Use the same `TEST_BER_EVERY` when comparing convergence curves.
6. For final conclusions, use multiple seeds if runtime allows.
7. Mark preliminary runs clearly if they use fewer epochs or fewer test files.

## Current Hypotheses

### H1: Regression EfficientKAN may beat classifier for final BER.

Reason:

Regression preserves IQ geometry. Classifier directly optimizes class separation, but it may make far constellation mistakes if confidence is poorly calibrated.

### H2: Classifier may converge faster early.

Reason:

Cross entropy directly separates classes and does not need to learn exact coordinate reconstruction.

### H3: Smaller LR or faster LR decay may improve EfficientKAN regression.

Reason:

Previous default run improved after late LR reduction around epoch 238, suggesting `1e-3` may stay too high for too long.

### H4: Grid size may matter more than spline order.

Reason:

The current cubic basis is already smooth. Increasing grid resolution may help local nonlinear correction more than increasing smoothness/order.

### H5: EfficientKAN alone may remain worse than RBF/FastKAN with complex encoder.

Reason:

Plain EfficientKAN receives a flattened window and must learn temporal structure internally. RBF/FastKAN receives better physics-aware features.

## Next Immediate Action

Implement or use a sweep runner that can vary:

```text
model_type
EFFICIENT_KAN_HIDDEN_DIM
EFFICIENT_KAN_LAYERS
EFFICIENT_KAN_GRID_SIZE
EFFICIENT_KAN_SPLINE_ORDER
LEARNING_RATE
CONTEXT_K
EPOCHS
```

Minimum first sweep:

```text
models = [efficient_kan_baseline, kan_classifier]
hidden_dim = [64, 96, 128, 192]
lr = [5e-4, 1e-3]
epochs = 60
```

This gives:

```text
2 models * 4 hidden values * 2 LR values = 16 runs
```

If each full run is too expensive, reduce to one test file or fewer epochs for Phase 1.

## Running History

Append all future results below this section.

### 2026-05-10: Study plan created

Created the EfficientKAN-focused research plan. No new runs in this entry.

### 2026-05-16: Split protocol tightened

Found and fixed an important evaluation issue in `ber_equalization.py`.

Previous protocol risk:

```text
Config.EVAL_ON_ALL_FILES = True
compute_test_metrics() used all_x/all_y
```

That meant the reported `test_ber` could be computed on all files, including train and validation files. This explains the suspicious behavior where test BER was much lower than validation BER.

New protocol:

```text
1. Split is file-level: train+val files are separated from hold-out test files.
2. Validation is carved out inside the train+val pool using VAL_PORTION_WITHIN_TRAIN.
3. Normalization statistics are computed only from train files.
4. Test files are not evaluated during training by default.
5. Final test BER is computed only once after loading the best validation checkpoint.
```

Current split-related defaults:

```text
TRAIN_PORTION = 0.97
VAL_PORTION_WITHIN_TRAIN = 0.10
MIN_VAL_FILES = 1
EVAL_TEST_DURING_TRAINING = False
```

Interpretation for future comparisons:

```text
Old reported test curves should be treated carefully if they were produced with EVAL_ON_ALL_FILES=True.
New runs are stricter and should be compared primarily by best_val_ber and final hold-out equalized_ber.
```

### 2026-05-16: Early stopping and split randomization option

Updated `ber_equalization.py` training protocol:

```text
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 72
EARLY_STOPPING_MIN_EPOCHS = 40
EARLY_STOPPING_THRESHOLD = 0.0
```

Early stopping is based on `val_ber`, independently from the LR scheduler. The scheduler can still reduce LR every `DECAY_STEPS`, but training stops if validation BER does not improve for the configured patience window.

Also added an optional reproducible file split randomization:

```text
RANDOMIZE_FILE_SPLIT = False
SPLIT_SEED = 42
```

Default is kept deterministic/chronological for continuity. For robustness experiments, enable `RANDOMIZE_FILE_SPLIT=True` and run several `SPLIT_SEED` values. This directly tests the concern that one validation file can be unusually noisy or unusually easy.

### 2026-05-17: Per-file windows and per-file BER diagnostics

Updated `ber_equalization.py` to avoid artificial context windows across CSV file boundaries.

Previous risk:

```text
load_files() concatenated several files
make_windows() then unfolded across the concatenated tensor
```

That creates invalid windows at file boundaries: part of the context comes from one file and part from the next file.

New behavior:

```text
1. Each CSV is loaded separately.
2. Power normalization still uses train files only.
3. Mean/std normalization still uses train files only.
4. Context windows are built separately inside each file.
5. File windows are concatenated only after window construction.
```

Added final per-file diagnostics:

```text
val_file_equalized_ber_mean/std/worst
test_file_equalized_ber_mean/std/worst
val_file_equalized_ber_by_file
test_file_equalized_ber_by_file
```

Also reduced the default training block/batch size:

```text
TRAIN_BLOCK_SIZE = 8192
```

Rationale: smaller stochastic batches may generalize better than very large batches while still being GPU-friendly.

### 2026-05-19: Experiment suite for KAN/MLP figures

Prepared `ber_equalization.py` for the planned experiment figures.

New switch:

```text
RUN_KAN_EXPERIMENT_SUITE = True
```

When enabled, it runs a dedicated suite and saves:

```text
kan_experiment_suite_all.csv
ber_vs_grid.csv / ber_vs_grid.png
ber_vs_spline_order.csv / ber_vs_spline_order.png
kan_mlp_vs_hidden_grid16.csv / kan_mlp_ber_vs_hidden_grid16.png
kan_mlp_vs_window.csv / kan_mlp_ber_vs_window.png
kan_mlp_vs_layers.csv / kan_mlp_ber_vs_layers.png
ber_vs_complexity.csv / ber_vs_complexity.png
```

Experiment defaults:

```text
EXPERIMENT_FIXED_GRID = 16
EXPERIMENT_FIXED_SPLINE_ORDER = 3
EXPERIMENT_HIDDEN_VALUES = [64, 96, 128, 192]
EXPERIMENT_WINDOW_VALUES = [8, 16, 24, 32, 48]
EXPERIMENT_GRID_VALUES = [4, 8, 12, 16, 20]
EXPERIMENT_SPLINE_ORDER_VALUES = [1, 2, 3, 4]
EXPERIMENT_LAYER_VALUES = [1, 2, 3]
```

Important implementation detail:

```text
MLP_LAYERS was added so MLP can be swept by number of layers.
For KAN vs MLP hidden/layer sweeps, the suite applies comparable hidden/layer overrides to both model families.
```
