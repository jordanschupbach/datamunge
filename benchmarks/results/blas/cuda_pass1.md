### gemm f32, GPU (GFLOP/s, median of >= 5 calls)

| case | inhouse-cuda | cublas | in-house / best library |
|---|---|---|---|
| overhead 1x1x1 | 0.0 | 0.0 | 1.16 |
| square 256 | 2063.5 | 2057.2 | 1.00 |
| square 512 | 6552.3 | 7905.9 | 0.83 |
| square 1024 | 10595.8 | 13365.6 | 0.79 |
| square 2048 | 12610.0 | 17690.5 | 0.71 |
| square 4096 | 12803.1 | 16881.6 | 0.76 |
| conv1 fwd 32768x64x27 | 3628.6 | 3598.7 | 1.01 |
| squeeze fwd 8192x16x128 | 1555.6 | 1725.4 | 0.90 |
| expand1x1 fwd 8192x64x16 | 2000.6 | 1659.6 | 1.21 |
| expand3x3 fwd 8192x64x144 | 5980.5 | 6407.9 | 0.93 |
| fire8 fwd 2048x256x576 | 7956.2 | 10913.0 | 0.73 |
| fire fwd 2048x32x512 | 2734.0 | 2986.3 | 0.92 |
| expand3x3 dx 8192x144x64 NT | 4620.1 | 5425.0 | 0.85 |
| expand3x3 dw 144x64x8192 TN | 4758.9 | 6716.0 | 0.71 |

### gemm f64, GPU (GFLOP/s, median of >= 5 calls)

| case | inhouse-cuda | cublas | in-house / best library |
|---|---|---|---|
| overhead 1x1x1 | 0.0 | 0.0 | 0.84 |
| square 256 | 287.4 | 264.4 | 1.09 |
| square 512 | 350.0 | 495.1 | 0.71 |
| square 1024 | 553.8 | 558.7 | 0.99 |
| square 2048 | 631.0 | 559.9 | 1.13 |
| conv1 fwd 32768x64x27 | 495.4 | 441.0 | 1.12 |
| squeeze fwd 8192x16x128 | 222.3 | 235.6 | 0.94 |
| expand1x1 fwd 8192x64x16 | 367.7 | 362.3 | 1.01 |
| expand3x3 fwd 8192x64x144 | 459.3 | 486.1 | 0.94 |
| fire8 fwd 2048x256x576 | 470.0 | 554.9 | 0.85 |
| fire fwd 2048x32x512 | 439.3 | 302.9 | 1.45 |
| expand3x3 dx 8192x144x64 NT | 440.1 | 491.3 | 0.90 |
| expand3x3 dw 144x64x8192 TN | 386.0 | 506.8 | 0.76 |
