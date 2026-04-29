# BER Protocol Audit

| Parameter | CNN_complex_v1.py | MXNet_Complex_FCNN_Conv_GPU_v1.ipynb | experiment1.py |
|---|---:|---:|---:|
| data_files | 64 | 32 | 32 |
| train_portion | 0.5 | 0.97 | None |
| train_files | 32 | 31 | None |
| test_files | 32 | 1 | None |
| data_size | 524288 | 524288 | None |
| radius | 30 | 10 | 40 |
| batch_size | 524288 | 524288 | 8192 |
| lrate | 0.001 | 0.001 | 0.0003 |
| seed_mode | random (randint + mx.random.seed) | random (randint + mx.random.seed) | not fixed in file (gap) |
| precision | float32 by default (float64 optional via swap_64_to_128_complex) | float32 by default (float64 optional via swap_64_to_128_complex) | float32 + cuda autocast fp16 |
| gpu_ctx | mxnet gpu(0) | mxnet gpu(0) | cpu |

## Unified Reference Split Used
- train files: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
- test files: [32]

## Test Distribution Validation
- base_dir: symbols_new
- reference_source: MXNet_Complex_FCNN_Conv_GPU_v1.ipynb
- test_files_count: 1
- test_samples_total: 524288
- test_samples_per_file_expected: 524268
- test_samples_per_file_observed: 524288
- center_cropped_symbols_expected_total: 524268
- center_cropped_symbols_observed_total: 524268

## Explicit Gaps
- seed is random in both external files (`randint(...)`), so runs are not inherently deterministic.
- SNR/channel are encoded only via source file paths, not explicit runtime parameters in code.