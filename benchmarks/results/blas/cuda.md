### gemm f32, GPU (GFLOP/s, median of >= 5 calls)

| case | inhouse-cuda | cublas | in-house speed / best library |
|---|---|---|---|
| overhead 1x1x1 | 0.0 | 0.0 | 1.09 |
| square 256 | 2222.3 | 2200.4 | 1.01 |
| square 512 | 9771.2 | 7901.2 | 1.24 |
| square 1024 | 12999.8 | 13499.3 | 0.96 |
| square 2048 | 17699.9 | 16814.6 | 1.05 |
| square 4096 | 16900.5 | 17115.2 | 0.99 |
| conv1 fwd 32768x64x27 | 4296.1 | 3548.9 | 1.21 |
| squeeze fwd 8192x16x128 | 1732.6 | 1765.5 | 0.98 |
| expand1x1 fwd 8192x64x16 | 1871.0 | 1633.8 | 1.15 |
| expand3x3 fwd 8192x64x144 | 7340.9 | 6575.3 | 1.12 |
| fire8 fwd 2048x256x576 | 12260.1 | 10759.2 | 1.14 |
| fire fwd 2048x32x512 | 3825.4 | 2990.2 | 1.28 |
| expand3x3 dx 8192x144x64 NT | 6440.4 | 5384.4 | 1.20 |
| expand3x3 dw 144x64x8192 TN | 5758.8 | 6698.4 | 0.86 |

### gemm f64, GPU (GFLOP/s, median of >= 5 calls)

| case | inhouse-cuda | cublas | in-house speed / best library |
|---|---|---|---|
| overhead 1x1x1 | 0.0 | 0.0 | 0.78 |
| square 256 | 478.3 | 294.5 | 1.62 |
| square 512 | 633.5 | 498.0 | 1.27 |
| square 1024 | 626.9 | 558.5 | 1.12 |
| square 2048 | 633.5 | 559.9 | 1.13 |
| conv1 fwd 32768x64x27 | 494.5 | 441.8 | 1.12 |
| squeeze fwd 8192x16x128 | 285.0 | 236.4 | 1.21 |
| expand1x1 fwd 8192x64x16 | 433.4 | 366.9 | 1.18 |
| expand3x3 fwd 8192x64x144 | 621.5 | 486.8 | 1.28 |
| fire8 fwd 2048x256x576 | 653.3 | 555.0 | 1.18 |
| fire fwd 2048x32x512 | 555.9 | 303.3 | 1.83 |
| expand3x3 dx 8192x144x64 NT | 545.9 | 491.8 | 1.11 |
| expand3x3 dw 144x64x8192 TN | 475.7 | 506.8 | 0.94 |
