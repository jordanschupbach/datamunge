### gemm f32, 0 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-cuda | cublas | in-house / best library |
|---|---|---|---|
| overhead 1x1x1 | 0.0 | 0.0 | 1.01 |
| square 256 | 700.3 | 2186.1 | 0.32 |
| square 512 | 3467.4 | 8491.8 | 0.41 |
| square 1024 | 4221.3 | 11196.7 | 0.38 |
| square 2048 | 10802.6 | 18205.6 | 0.59 |
| square 4096 | 9433.5 | 16978.8 | 0.56 |
| conv1 fwd 32768x64x27 | 3192.0 | 3807.0 | 0.84 |
| squeeze fwd 8192x16x128 | 817.4 | 1841.1 | 0.44 |
| expand1x1 fwd 8192x64x16 | 1276.2 | 1824.0 | 0.70 |
| expand3x3 fwd 8192x64x144 | 3398.9 | 6891.0 | 0.49 |
| fire8 fwd 2048x256x576 | 6863.6 | 11097.7 | 0.62 |
| fire fwd 2048x32x512 | 920.1 | 3127.2 | 0.29 |
| expand3x3 dx 8192x144x64 NT | 3802.8 | 5894.1 | 0.65 |
| expand3x3 dw 144x64x8192 TN | 160.6 | 7358.8 | 0.02 |

### gemm f64, 0 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-cuda | cublas | in-house / best library |
|---|---|---|---|
| overhead 1x1x1 | 0.0 | 0.0 | 0.56 |
| square 256 | 148.0 | 265.9 | 0.56 |
| square 512 | 347.8 | 498.4 | 0.70 |
| square 1024 | 302.5 | 561.7 | 0.54 |
| square 2048 | 632.7 | 562.9 | 1.12 |
| conv1 fwd 32768x64x27 | 499.3 | 444.5 | 1.12 |
| squeeze fwd 8192x16x128 | 114.9 | 238.0 | 0.48 |
| expand1x1 fwd 8192x64x16 | 376.9 | 369.8 | 1.02 |
| expand3x3 fwd 8192x64x144 | 457.9 | 489.5 | 0.94 |
| fire8 fwd 2048x256x576 | 468.3 | 558.1 | 0.84 |
| fire fwd 2048x32x512 | 167.3 | 305.1 | 0.55 |
| expand3x3 dx 8192x144x64 NT | 442.5 | 494.6 | 0.89 |
| expand3x3 dw 144x64x8192 TN | 24.0 | 510.0 | 0.05 |
