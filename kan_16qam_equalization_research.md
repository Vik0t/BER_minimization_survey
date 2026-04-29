# KAN for 16QAM BER Equalization: Research Notes

Date: 2026-04-27

This note preserves the current research thread: what prior work suggests about KAN-based optical equalization, what our experiments have shown so far, and which KAN architectures look most promising for 16QAM BER equalization.

## Problem Setup

We are working on BER equalization for a 16QAM signal. The current training pipeline builds windows of normalized received IQ samples:

- Input window: `SEQ_LEN = 2 * CONTEXT_K + 1 = 65`
- Input dimension per symbol: `I, Q`
- Flattened real-valued input for simple models: `65 * 2 = 130`
- Regression target: transmitted normalized `(I, Q)`
- BER metric: nearest 16QAM constellation point plus Gray bit labels

Important detail: most models are regression equalizers that output corrected `(I, Q)`. A classifier-style model outputs 16 constellation logits and is evaluated by `argmax`.

## Literature Thread

### Original KAN

Paper: Kolmogorov-Arnold Networks  
Link: https://arxiv.org/abs/2404.19756

Main idea: replace fixed node activations in MLPs with learnable univariate functions on edges. In practice, KAN layers often use spline-like basis expansions. This makes KAN attractive for nonlinear compensation, because channel nonlinearities can be represented as combinations of learned one-dimensional functions.

Relevance to our task:

- KAN is a natural candidate for nonlinear equalization.
- A plain KAN on a flattened IQ window can work, but it may have to learn both temporal feature extraction and nonlinear correction at the same time.
- This is likely harder than using a small temporal/complex encoder before KAN.

### FastKAN / RBF-KAN

Paper: FastKAN / RBF approximation of KAN-style basis functions  
Link: https://arxiv.org/abs/2405.06721

Main idea: approximate or replace spline basis functions with Gaussian RBF expansions. RBF-KAN can be faster and easier to optimize while preserving the local-function behavior that makes KAN useful.

Relevance to our task:

- Received IQ samples form noisy local clouds around constellation regions.
- Gaussian RBF basis functions are well matched to local smooth correction around those clouds.
- Our best result so far comes from the RBF/FastKAN path, not from B-spline EfficientKAN.

### KAN for Optical Equalization in PON / PAM Systems

Paper: Non-linear Equalization in 112 Gb/s PONs Using Kolmogorov-Arnold Networks  
Link: https://arxiv.org/abs/2411.19631

Main idea: KAN can be used as a nonlinear equalizer in optical access systems. The reported direction is that KAN can compete with or outperform classical/NN equalizers at favorable complexity.

Relevance to our task:

- This is not the same as coherent 16QAM, but it supports the general idea that KAN is useful for optical nonlinear equalization.
- It motivates using KAN as the nonlinear compensation block, especially when complexity matters.

### FKAN-E / KAN-E for Short-Reach IM/DD

Paper record: Kolmogorov-Arnold network for efficient equalization in short-reach IM/DD systems  
Link: https://pubmed.ncbi.nlm.nih.gov/40984485/

Main idea: FKAN-E uses Gaussian RBF-based KAN equalization and reports similar or better BER than KAN-E while improving training/inference efficiency.

Relevance to our task:

- This supports our empirical result that RBF/FastKAN can beat spline KAN in practical equalization.
- It also suggests that basis choice matters: KAN is not one thing; spline KAN and RBF KAN can behave quite differently.

### GDP-KAN for Coherent PDM-16QAM

Paper record: Gradient-driven pruned Kolmogorov-Arnold Network for ultralow-complexity fiber nonlinearity compensation  
Link: https://pubmed.ncbi.nlm.nih.gov/41396934/

Main setting: 8-channel WDM, 1600 km SSMF, 64 GBaud PDM-16QAM.

Main idea: use a pruned KAN for fiber nonlinearity compensation. The important part for us is not only the KAN itself, but the combination of KAN plus pruning to get low complexity.

Relevance to our task:

- This is the closest found literature match to coherent 16QAM equalization.
- It strongly suggests that a compact/pruned KAN is a good direction.
- It supports our target: high BER improvement without becoming much slower than current RBF-KAN.

### Caution on Classifier Equalizers

Paper: Neural Networks-Based Equalizers for Coherent Optical Transmission: Caveats and Pitfalls  
Link: https://arxiv.org/abs/2109.14942

Main idea: neural equalizers in coherent optical systems need careful evaluation. Classification-style equalizers can be less format-flexible than regression equalizers.

Relevance to our task:

- A 16-class KAN classifier is worth testing, but it should not automatically replace regression.
- Regression to `(I, Q)` is more modulation-format flexible and gives meaningful residual/error geometry.
- A good compromise may be regression plus a symbol-aware auxiliary loss.

## Our Current Experimental Results

Dataset split:

- Train files: `1..61`
- Validation file: `62`
- Test files: `63, 64`
- Baseline BER: approximately `9.741273e-03`

### EfficientKAN Baseline

Architecture:

- Input: flattened real-valued IQ window, `130`
- KAN: `130 -> 128 -> 128 -> 2`
- Output: corrected `(I, Q)`
- Parameters: `432,640`

Result:

- Final equalized BER: approximately `3.514662e-03`
- Relative improvement: `63.92%`

Interpretation:

- Plain B-spline EfficientKAN works.
- It learns slowly and keeps improving for many epochs.
- It appears to plateau around `3.5e-03`.
- Scheduler reduced LR very late, and the LR drop helped. This suggests the LR schedule may be too patient for this model.

### EfficientKAN Residual

Architecture:

- Input: flattened real-valued IQ window, `130`
- KAN predicts correction
- Output: `raw_center + correction`
- Parameters: `432,641`

Result:

- Final equalized BER: approximately `3.571934e-03`
- Relative improvement: `63.33%`

Interpretation:

- Slightly worse than plain EfficientKAN baseline.
- Likely issue: `raw_center` is taken from standardized input windows, while target `(I, Q)` is in power-normalized TX scale. The residual branch may be adding quantities in mismatched coordinate systems.
- Residual KAN is still worth revisiting, but only with correct scale handling.

### EfficientKAN Features

Architecture:

- Input: 17 handcrafted features
- Features include center IQ, local/global mean/std, power statistics, IQ cross term, edge delta
- KAN maps features to corrected `(I, Q)`
- Parameters: `244,608`

Result:

- Final equalized BER: approximately `7.480829e-03`
- Relative improvement: `23.20%`

Interpretation:

- This is a useful negative result.
- The handcrafted feature compression is too aggressive.
- Mean/std/power features destroy important temporal structure in the received window.
- For this task, KAN needs either the full window or a learned temporal encoder.

### CNN_KAN

Architecture:

- CNN extracts temporal features from the IQ window.
- Fused vector includes CNN center feature, attention/global context, raw center, raw mean.
- EfficientKAN head maps fused features to corrected `(I, Q)`.
- Parameters: `470,017`

Partial result from available log:

- At epoch 30: test BER approximately `2.154846e-03`
- This already beats final plain EfficientKAN baseline.

Interpretation:

- Learned temporal feature extraction before KAN is very valuable.
- Runtime is slower: about `153k samples/s` vs about `309k samples/s` for plain EfficientKAN.
- The quality gain is large, but the speed cost is also large.
- This motivates using a lighter temporal encoder, like the existing `LightweightComplexEncoder`, instead of a heavier CNN stack.

### RBF/FastKAN Result

Observed user result:

- RBF-KAN achieved about `84%` relative improvement.
- With baseline BER `9.741273e-03`, this implies BER around:

```text
9.741273e-03 * (1 - 0.84) ~= 1.56e-03
```

Architecture in current code:

- `ComplexFastKANEqualizer`
- Encoder: `LightweightComplexEncoder`
- Head: `FastKANHead`
- Basis: Gaussian RBF expansion
- Output: corrected `(I, Q)`

Interpretation:

- This is currently the strongest direction.
- The win is probably due to both parts:
  - complex/temporal encoder gives useful physics-aware features,
  - RBF-KAN head gives strong local nonlinear correction.
- This result changes the story: KAN is not weak here; the right KAN variant is strong.

## Main Research Conclusion

The best architecture family for this task is probably:

```text
lightweight complex/temporal encoder -> compact RBF/FastKAN head -> corrected I/Q
```

The weaker direction is:

```text
flat IQ window -> B-spline EfficientKAN -> corrected I/Q
```

The failed direction so far is:

```text
handcrafted summary statistics -> KAN -> corrected I/Q
```

The most important lesson: **do not remove temporal structure before KAN**. Either give KAN the full window or use a learned temporal encoder.

## Best Next Architecture Candidates

### 1. Pruned Complex RBF-KAN

Priority: highest.

Base:

```text
LightweightComplexEncoder -> FastKANHead -> I/Q
```

Add:

- Train normally.
- Prune low-value `feature_gate` inputs.
- Prune small RBF/spline weights.
- Fine-tune for 20-40 epochs.

Why:

- Supported by GDP-KAN direction.
- Should reduce complexity without losing much BER.
- Best match to goal: accurate but not slower than current RBF-KAN.

### 2. Symbol-Aware Regression RBF-KAN

Keep output as `(I, Q)`, but modify loss:

```text
loss = MSE(pred_iq, target_iq) + lambda * soft_symbol_loss(pred_iq, target_class)
```

Why:

- MSE alone optimizes geometry, not directly BER.
- Classification alone may reduce flexibility.
- Symbol-aware regression keeps both: coordinate correction and decision-region awareness.
- Runtime unchanged.

### 3. Correctly Scaled Residual RBF-KAN

Current residual EfficientKAN likely mixes standardized RX center with power-normalized target.

Better residual:

```text
base = properly scaled rx_center or LS/FIR estimate
output = base + FastKAN(delta_features)
```

Why:

- Equalizer often only needs to learn residual nonlinear correction.
- Should train faster and maybe improve BER.
- Very low runtime overhead.

### 4. Tiny TCN + RBF-KAN Head

Architecture:

```text
depthwise dilated temporal blocks -> center/mean/std features -> FastKANHead
```

Use small settings:

- `COMPLEX_LIGHT_CHANNELS = 32` or `48`
- dilations `[1, 2, 4]` or `[1, 2, 4, 8]`
- `FASTKAN_HIDDEN_DIM = 64..96`

Why:

- Should capture temporal memory better than handcrafted features.
- Should be much faster than full `cnn_kan`.
- Very close to current winning `ComplexFastKANEqualizer`.

### 5. Ring-Aware RBF-KAN

For 16QAM, inner and outer constellation points experience different nonlinear behavior.

Options:

- Add `power_center` and local power stats more explicitly.
- Use a small power-conditioned gate.
- Use two tiny RBF-KAN heads mixed by a soft gate from `|rx_center|^2`.

Why:

- 16QAM has amplitude-dependent nonlinear distortion.
- Runtime increase should be small.

## What Not To Prioritize

### Pure Handcrafted Feature KAN

Reason:

- Current result is weak: around `7.48e-03`.
- It loses temporal structure.

### Large CNN_KAN as Final Model

Reason:

- Quality looks good, but speed is about 2x slower than plain EfficientKAN.
- Useful as an upper-bound/reference, but may violate the goal of staying near RBF-KAN speed.

### Pure 16-Class Classifier as Only Model

Reason:

- Worth testing, but regression is safer for equalization.
- Classification can hide geometric errors and may be less flexible across modulation formats.

## Recommended Immediate Experiment Plan

1. Keep current `ComplexFastKANEqualizer` as the main baseline.
2. Add parameter count, final BER, best validation BER, and samples/sec to the comparison CSV.
3. Implement `PrunedComplexFastKANEqualizer` or add pruning/fine-tuning mode for existing FastKAN.
4. Add symbol-aware regression loss as a config option.
5. Fix residual scaling and test residual RBF-KAN.
6. Compare:
   - `complex_fastkan`
   - `complex_fastkan_pruned`
   - `complex_fastkan_symbol_loss`
   - `complex_fastkan_residual`
   - `cnn_kan`
   - `mlp`
   - `cnn`
   - `efficient_kan_baseline`

## Current Narrative To Preserve

At first, plain EfficientKAN showed that KAN can equalize 16QAM, but it was not best. CNN_KAN then showed that learned temporal feature extraction before KAN matters a lot. The decisive result is that RBF/FastKAN with a lightweight complex encoder beat everything so far, reaching about 84% relative BER improvement. Therefore the research direction should not be "KAN vs neural equalizers" in a generic way. The direction should be:

```text
physics-aware lightweight temporal features + fast local KAN nonlinear correction + pruning
```

This is the line most consistent with both our experiments and the closest optical KAN literature.
