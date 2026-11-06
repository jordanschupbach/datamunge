### gemm f32, GPU (GFLOP/s, median of >= 5 calls)

| case | inhouse-cuda | cublas | in-house speed / best library |
|---|---|---|---|
| overhead 1x1x1 | 0.0 | 0.0 | 1.24 |
| square 256 | 2184.8 | 2347.1 | 0.93 |
| square 512 | 10084.7 | 7818.8 | 1.29 |
| square 1024 | 12620.5 | 13420.9 | 0.94 |
| square 2048 | 17677.9 | 16886.9 | 1.05 |
| square 4096 | 16854.7 | 17029.2 | 0.99 |
| conv1 fwd 32768x64x27 | 4221.0 | 3381.4 | 1.25 |
| squeeze fwd 8192x16x128 | 1889.0 | 1716.8 | 1.10 |
| expand1x1 fwd 8192x64x16 | 1931.5 | 1688.2 | 1.14 |
| expand3x3 fwd 8192x64x144 | 7454.0 | 6432.7 | 1.16 |
| fire8 fwd 2048x256x576 | 11823.5 | 10310.9 | 1.15 |
| fire fwd 2048x32x512 | 3656.6 | 2985.1 | 1.22 |
| expand3x3 dx 8192x144x64 NT | 6159.5 | 5305.1 | 1.16 |
| expand3x3 dw 144x64x8192 TN | 5281.0 | 6689.8 | 0.79 |

### gemm f64, GPU (GFLOP/s, median of >= 5 calls)

| case | inhouse-cuda | cublas | in-house speed / best library |
|---|---|---|---|
| overhead 1x1x1 | 0.0 | 0.0 | 0.78 |
| square 256 | 469.0 | 292.8 | 1.60 |
| square 512 | 633.3 | 495.8 | 1.28 |
| square 1024 | 636.3 | 558.6 | 1.14 |
| square 2048 | 626.1 | 560.1 | 1.12 |
| conv1 fwd 32768x64x27 | 494.9 | 441.7 | 1.12 |
| squeeze fwd 8192x16x128 | 285.0 | 236.2 | 1.21 |
| expand1x1 fwd 8192x64x16 | 430.2 | 365.7 | 1.18 |
| expand3x3 fwd 8192x64x144 | 622.3 | 486.7 | 1.28 |
| fire8 fwd 2048x256x576 | 653.2 | 555.1 | 1.18 |
| fire fwd 2048x32x512 | 559.9 | 303.3 | 1.85 |
| expand3x3 dx 8192x144x64 NT | 548.3 | 491.8 | 1.11 |
| expand3x3 dw 144x64x8192 TN | 474.2 | 508.1 | 0.93 |
