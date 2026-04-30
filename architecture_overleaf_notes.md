# Architecture Notes for 16QAM BER Equalization

Date: 2026-04-29

This file summarizes the neural equalizer architectures implemented in `ber_equalization.py`. It is intended as a draft source for an Overleaf report section: architecture descriptions, main parameters, training objective, and currently observed results.

## Common Experimental Setup

The task is symbol-wise equalization for a 16QAM signal. For every central received symbol, the model receives a temporal window of neighboring received IQ samples and predicts either corrected IQ coordinates or one of 16 constellation classes.

### Input Representation

Current configuration:

```text
CONTEXT_K = 32
SEQ_LEN = 2 * CONTEXT_K + 1 = 65
INPUT_DIM = 2
```

Each sample contains a received IQ window:

```text
x_t = [I_{t-32}, Q_{t-32}, ..., I_t, Q_t, ..., I_{t+32}, Q_{t+32}]
```

For simple feed-forward models, this window is flattened into a real-valued vector:

```text
dim(x_t) = 65 * 2 = 130
```

The regression target is the transmitted normalized IQ symbol:

```text
y_t = [I_t^{TX}, Q_t^{TX}]
```

### Evaluation

For regression models, the model outputs corrected IQ coordinates:

```text
\hat{y}_t = [\hat{I}_t, \hat{Q}_t]
```

The predicted point is mapped to the nearest 16QAM constellation point. BER is computed by comparing the Gray labels of predicted and true constellation classes.

For classifier models, the model outputs 16 logits directly:

```text
z_t \in R^{16}
```

The predicted class is:

```text
\hat{c}_t = argmax_i z_{t,i}
```

BER is then computed from the Gray labels of `\hat{c}_t` and the true class.

### Dataset Split Used in Current Experiments

```text
Train files: 1..61
Validation file: 62
Test files: 63, 64
Raw baseline BER: 9.741273e-03
```

## Summary Table

| Model | Type | Input to head | Output | Loss | Params | Best/Final BER | Relative improvement | Speed |
|---|---:|---|---|---|---:|---:|---:|---:|
| `efficient_kan_baseline` | B-spline KAN regression | flattened IQ window, 130 | 2 IQ values | MSE | 432,640 | 3.514662e-03 | 63.92% | ~309k samp/s |
| `efficient_kan_residual` | B-spline KAN residual regression | flattened IQ window, 130 | 2 IQ correction values | MSE | 432,641 | 3.571934e-03 | 63.33% | ~309k samp/s |
| `efficient_kan_features` | B-spline KAN feature regression | 17 handcrafted features | 2 IQ values | MSE | 244,608 | 7.480829e-03 | 23.20% | ~348k samp/s |
| `cnn_kan` | CNN + B-spline KAN regression | CNN fused vector, 132 | 2 IQ values | MSE | 470,017 | 2.154846e-03 at epoch 30 | ~77.88% at epoch 30 | ~154k samp/s |
| `complex_fastkan` / RBF-KAN | lightweight complex encoder + RBF-KAN | fused complex temporal features, 158 | 2 IQ values | MSE | ~243,358 | ~1.56e-03 inferred | ~84% | not recorded here |
| `kan_classifier` | B-spline KAN classifier | flattened IQ window, 130 | 16 logits | cross entropy | 455,936 | TBD | TBD | TBD |
| `mlp` | MLP regression baseline | flattened IQ window, 130 | 2 IQ values | MSE | ~10,722 | TBD | TBD | TBD |
| `cnn` | CNN + MLP regression baseline | CNN fused vector, 132 | 2 IQ values | MSE | ~44,899 | TBD | TBD | TBD |

Note: `complex_fastkan` parameter count is calculated from the current configuration. Its BER is inferred from the reported `84%` relative improvement over the baseline BER `9.741273e-03`, giving approximately `1.56e-03`.

## Architecture 1: EfficientKAN Baseline

Code class:

```text
EfficientKANBaselineEqualizer
```

### Structure

This is the simplest B-spline KAN equalizer. It receives the full flattened IQ window and directly predicts the corrected IQ coordinates.

```text
Input: 130 real values
KAN: 130 -> 128 -> 128 -> 2
Output: corrected [I, Q]
```

### Main Parameters

```text
EFFICIENT_KAN_HIDDEN_DIM = 128
EFFICIENT_KAN_LAYERS = 2
EFFICIENT_KAN_GRID_SIZE = 8
EFFICIENT_KAN_SPLINE_ORDER = 3
EFFICIENT_KAN_GRID_RANGE = [-3.0, 3.0]
```

The KAN layer contains a base path and a spline path. For each edge, the model learns a univariate nonlinear function represented by spline coefficients. In this implementation, the grid is fixed during normal forward passes.

### Training Objective

Regression MSE:

```text
L = || \hat{y}_t - y_t ||_2^2
```

### Result and Interpretation

Observed result:

```text
BER = 3.514662e-03
Relative improvement = 63.92%
Params = 432,640
Speed = ~309k samples/s
```

This model proves that a plain KAN can perform nonlinear equalization. However, it must learn temporal feature extraction and nonlinear compensation simultaneously from a raw flattened window. It improves slowly and appears to plateau near `3.5e-03`.

## Architecture 2: EfficientKAN Residual

Code class:

```text
EfficientKANResidualEqualizer
```

### Structure

The residual version predicts a correction term instead of the final IQ coordinates directly.

```text
Input: 130 real values
KAN: 130 -> 128 -> 128 -> 2
Output: raw_center + learned_correction
```

Mathematically:

```text
 
```

where `alpha` is a learned scalar `residual_scale`.

### Main Parameters

Same KAN parameters as `efficient_kan_baseline`, plus:

```text
residual_scale: one trainable scalar
```

### Training Objective

Regression MSE:

```text
L = || \hat{y}_t - y_t ||_2^2
```

### Result and Interpretation

Observed result:

```text
BER = 3.571934e-03
Relative improvement = 63.33%
Params = 432,641
Speed = ~309k samples/s
```

This did not outperform the direct KAN baseline. The likely reason is scale mismatch: `raw_center` is taken from standardized RX windows, while the target TX symbol is power-normalized but not standardized in the same way. A residual architecture is still promising, but the residual base should be a properly scaled RX center or a linear/FIR estimate.

## Architecture 3: EfficientKAN with Handcrafted Featuresста

Code class:

```text
EfficientKANFeatureEqualizer
```

### Structure

This model compresses the full IQ window into 17 manually selected features and feeds them into a KAN.

```text
Input window: 65 x 2
Feature extraction: 17 real features
KAN: 17 -> 128 -> 128 -> 2
Output: corrected [I, Q]
```

The 17 features are:

```text
center IQ: 2
local mean IQ: 2
local std IQ: 2
global mean IQ: 2
global std IQ: 2
center power: 1
global power mean: 1
global power std: 1
mean(I*Q): 1
std(I*Q): 1
edge delta IQ: 2
Total: 17
```

### Main Parameters

```text
KAN_FEATURE_RADIUS = 2
EFFICIENT_KAN_HIDDEN_DIM = 128
EFFICIENT_KAN_LAYERS = 2
```

### Training Objective

Regression MSE.

### Result and Interpretation

Observed result:

```text
BER = 7.480829e-03
Relative improvement = 23.20%
Params = 244,608
Speed = ~348k samples/s
```

This is a useful negative result. The model is faster and smaller, but the handcrafted summary statistics remove too much temporal structure. For this equalization problem, mean/std/power features are not enough. KAN needs either the full window or learned temporal features.

## Architecture 4: CNN + EfficientKAN

Code class:

```text
CNNKANEqualizer
```

### Structure

This architecture first uses a temporal CNN to extract local features from the IQ sequence. A KAN head then performs nonlinear regression from the fused CNN features to corrected IQ.

```text
Input: 65 x 2 IQ window
CNN: Conv1d(2 -> 64, k=5)
CNN: Conv1d(64 -> 64, k=5)
CNN: Conv1d(64 -> 64, k=3)
Pooling: attention-like global context over time
Fused vector: center CNN feature + global CNN context + raw center + raw mean
Fused dimension: 64 + 64 + 2 + 2 = 132
KAN: 132 -> 128 -> 128 -> 2
Output: corrected [I, Q]
```

### Main Parameters

```text
HIDDEN_DIM = 64
EFFICIENT_KAN_HIDDEN_DIM = 128
EFFICIENT_KAN_GRID_SIZE = 8
EFFICIENT_KAN_SPLINE_ORDER = 3
```

### Training Objective

Regression MSE.

### Result and Interpretation

Observed partial result:

```text
BER at epoch 30 = 2.154846e-03
Relative improvement at epoch 30 ~= 77.88%
Params = 470,017
Speed = ~154k samples/s
```

This strongly outperforms the pure EfficientKAN baseline early in training. The result shows that learned temporal features before KAN are very valuable. The drawback is speed: the CNN front-end roughly halves throughput compared with the plain KAN models.

## Architecture 5: Complex FastKAN / RBF-KAN

Code class:

```text
ComplexFastKANEqualizer
```

This is currently the most promising architecture family.

### Structure

The model uses a lightweight complex-aware temporal encoder followed by an RBF/FastKAN head.

```text
Input: 65 x 2 IQ window
Feature channels: real, imag, magnitude, power, real*imag
Lightweight temporal encoder:
  Conv1d(5 -> 48, k=1)
  depthwise temporal blocks with dilations [1, 2, 4]
Sequence features:
  hidden features + raw real + raw imag + magnitude
Fused vector:
  center features
  sequence mean
  sequence std
  raw center
  raw mean
  center power
FastKANHead:
  158 -> 96 -> 96 -> 2
Output: corrected [I, Q]
```

### Main Parameters

```text
COMPLEX_LIGHT_CHANNELS = 48
COMPLEX_LIGHT_DILATIONS = [1, 2, 4]
COMPLEX_LIGHT_KERNEL_SIZE = 3
FASTKAN_HIDDEN_DIM = 96
FASTKAN_LAYERS = 2
FASTKAN_NUM_GRIDS = 8
FASTKAN_GRID_RANGE = [-2.5, 2.5]
FASTKAN_BASE_ACT = silu
```

Unlike EfficientKAN, this head uses Gaussian RBF expansion:

```text
phi_j(x) = exp(-((x - c_j) * s)^2)
```

The RBF basis gives local smooth nonlinear corrections, which is well matched to noisy 16QAM point clouds.

### Training Objective

Regression MSE with optional KAN regularization:

```text
L = MSE + lambda * regularization_loss
```

### Result and Interpretation

Observed result from experiment:

```text
Relative improvement ~= 84%
Estimated BER ~= 1.56e-03
Params ~= 243,358
```

This is currently the strongest direction. The result suggests that the winning factor is not simply "KAN" in the abstract, but the combination:

```text
physics-aware complex temporal features + RBF-KAN nonlinear head
```

This architecture is more accurate than plain EfficientKAN and likely faster than the heavier CNN_KAN path.

## Architecture 6: KAN Classifier

Code class:

```text
EfficientKANClassifierEqualizer
```

### Structure

The classifier uses the same B-spline EfficientKAN block as the plain KAN baseline, but changes the output dimension from 2 to 16.

```text
Input: 130 real values
KAN: 130 -> 128 -> 128 -> 16
Output: 16 class logits
```

The output is not corrected IQ. Instead, each output value is a score for one 16QAM constellation point.

```text
z = [score_0, score_1, ..., score_15]
\hat{c} = argmax(z)
```

### Main Parameters

Same as EfficientKAN baseline, except:

```text
output_dim = 16
```

### Training Objective

Cross entropy:

```text
L = CE(z, c_true)
```

The target IQ point is first mapped to the nearest 16QAM class.

### Result and Interpretation

Observed result:

```text
BER = TBD
Relative improvement = TBD
Params = 455,936
```

This model is not a traditional equalizer because it does not output a corrected IQ point. It is better described as a symbol detector. It may optimize BER directly, but it is less flexible than regression if the modulation format changes.

## Architecture 7: MLP Baseline

Code class:

```text
MLPRxEqualizer
```

### Structure

This is a simple feed-forward regression baseline.

```text
Input: 130 real values
Linear: 130 -> 64
LayerNorm + GELU + Dropout
Linear: 64 -> 32
LayerNorm + GELU + Dropout
Linear: 32 -> 2
Output: corrected [I, Q]
```

### Main Parameters

```text
HIDDEN_DIM = 64
DROPOUT = 0.2
```

### Training Objective

Regression MSE.

### Result and Interpretation

Observed result:

```text
BER = TBD
Relative improvement = TBD
Params ~= 10,722
```

This model is important as a lightweight baseline. If KAN only slightly beats this, KAN is not justified. If KAN strongly beats this at comparable speed, the KAN nonlinear edge functions are likely useful.

## Architecture 8: CNN Baseline

Code class:

```text
CNNRxEqualizer
```

### Structure

This model tests whether temporal convolution alone is enough, without a KAN head.

```text
Input: 65 x 2 IQ window
CNN: Conv1d(2 -> 64, k=5)
CNN: Conv1d(64 -> 64, k=5)
CNN: Conv1d(64 -> 64, k=3)
Pooling: attention-like global context over time
Fused vector:
  center CNN feature
  global CNN context
  raw center
  raw mean
MLP head: 132 -> 64 -> 32 -> 2
Output: corrected [I, Q]
```

### Main Parameters

```text
HIDDEN_DIM = 64
DROPOUT = 0.2
```

### Training Objective

Regression MSE.

### Result and Interpretation

Observed result:

```text
BER = TBD
Relative improvement = TBD
Params ~= 44,899
```

This is the key comparison for `cnn_kan`. If `cnn_kan` beats this model, then the KAN head is adding value beyond temporal CNN features.

## Main Comparative Conclusions

### Current Ranking From Available Results

Based on observed or inferred BER:

```text
1. complex_fastkan / RBF-KAN      ~= 1.56e-03
2. cnn_kan                       ~= 2.15e-03 at epoch 30
3. efficient_kan_baseline        ~= 3.51e-03
4. efficient_kan_residual        ~= 3.57e-03
5. efficient_kan_features        ~= 7.48e-03
```

MLP, CNN, and KAN classifier require final runs before being placed in the ranking.

### Main Interpretation

The most important empirical observation is:

```text
KAN is useful, but it performs best when it receives good temporal/complex features.
```

The plain B-spline KAN can learn equalization, but it is not the best current model. The RBF/FastKAN with lightweight complex temporal encoding is strongest so far. The handcrafted-feature KAN is weak because it removes temporal structure.

### Best Research Direction

The best next direction is:

```text
Lightweight complex temporal encoder
  -> compact RBF/FastKAN head
  -> optional pruning
  -> optional symbol-aware regression loss
```

This direction is consistent with both our experiments and recent KAN-based optical equalization literature.

## Suggested LaTeX Table Skeleton

```latex
\begin{table}[t]
\centering
\caption{Comparison of neural equalizer architectures for 16QAM BER equalization.}
\begin{tabular}{lcccc}
\hline
Model & Parameters & Output & BER & Improvement \\
\hline
EfficientKAN & 432,640 & IQ & $3.51\times10^{-3}$ & 63.92\% \\
EfficientKAN-Residual & 432,641 & IQ & $3.57\times10^{-3}$ & 63.33\% \\
EfficientKAN-Features & 244,608 & IQ & $7.48\times10^{-3}$ & 23.20\% \\
CNN-KAN & 470,017 & IQ & $2.15\times10^{-3}$ & 77.88\% \\
Complex FastKAN & $\sim$243,358 & IQ & $\sim1.56\times10^{-3}$ & $\sim$84\% \\
KAN Classifier & 455,936 & class & TBD & TBD \\
MLP & $\sim$10,722 & IQ & TBD & TBD \\
CNN & $\sim$44,899 & IQ & TBD & TBD \\
\hline
\end{tabular}
\end{table}
```

## Suggested Text for Report

The results indicate that KAN-based equalizers are effective for nonlinear 16QAM equalization, but the choice of input representation and basis functions is crucial. A plain B-spline KAN operating on the flattened IQ window improves BER by approximately 64% relative to the unequalized baseline. However, compressing the input into handcrafted statistics significantly degrades performance, suggesting that temporal structure in the received window is essential. The best observed performance is obtained by a lightweight complex temporal encoder followed by an RBF/FastKAN head, reaching approximately 84% relative BER improvement. This supports the hypothesis that KAN is most effective when used as a compact nonlinear correction block after physically meaningful temporal feature extraction.
