### gemm f32, 1 thread (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| square 256 | 287.9 | 157.0 | 54.2 | 157.2 | 284.1 | 50.7 | 37.1 | 1.01 |
| square 512 | 302.3 | 158.1 | 53.8 | 159.1 | 290.5 | 51.5 | 36.0 | 1.04 |
| square 1024 | 294.7 | 157.2 | 53.9 | 158.4 | 282.5 | 51.0 | 33.6 | 1.04 |
| square 2048 | 237.9 | 153.5 | 53.9 | 157.3 | 280.4 | 50.4 |  | 0.85 |
| square 4096 | 233.0 | 148.6 | 53.6 | 153.8 | 276.5 | 50.9 |  | 0.84 |
| conv1 fwd 32768x64x27 | 253.1 | 144.7 | 52.3 | 135.5 | 221.0 | 32.9 | 26.2 | 1.15 |
| squeeze fwd 8192x16x128 | 185.9 | 125.0 | 48.7 | 126.4 | 191.9 | 40.1 | 14.0 | 0.97 |
| expand1x1 fwd 8192x64x16 | 243.9 | 138.4 | 51.5 | 121.9 | 190.3 | 26.0 | 24.1 | 1.28 |
| expand3x3 fwd 8192x64x144 | 275.4 | 151.7 | 51.9 | 149.3 | 261.5 | 46.0 | 29.0 | 1.05 |
| fire8 fwd 2048x256x576 | 299.1 | 158.8 | 54.2 | 159.3 | 291.3 | 50.0 | 37.0 | 1.03 |
| expand3x3 dx 8192x144x64 NT | 288.1 | 155.9 | 54.4 | 151.1 | 262.2 | 43.4 | 11.3 | 1.10 |
| expand3x3 dw 144x64x8192 TN | 255.0 | 143.5 | 52.1 | 148.9 | 250.9 | 49.9 | 27.0 | 1.02 |
| tile 128x64x144 | 261.3 | 144.9 | 53.1 | 151.2 | 252.7 | 45.5 | 29.4 | 1.03 |

### gemm f64, 1 thread (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| square 256 | 139.4 | 77.6 | 28.5 | 73.3 | 119.0 | 25.3 | 18.7 | 1.17 |
| square 512 | 144.2 | 76.9 | 28.4 | 75.9 | 123.6 | 25.3 | 16.6 | 1.17 |
| square 1024 | 130.4 | 77.3 | 27.9 | 75.2 | 121.0 | 25.2 | 17.0 | 1.08 |
| square 2048 | 114.8 | 76.1 | 28.1 | 76.9 | 125.6 | 25.5 |  | 0.91 |
| square 4096 | 110.2 | 75.2 | 28.2 | 76.7 | 126.8 | 25.5 |  | 0.87 |
| conv1 fwd 32768x64x27 | 116.6 | 70.1 | 26.2 | 57.2 | 85.8 | 16.0 | 17.3 | 1.36 |
| squeeze fwd 8192x16x128 | 103.9 | 63.5 | 26.0 | 56.3 | 75.0 | 21.8 | 11.4 | 1.39 |
| expand1x1 fwd 8192x64x16 | 126.3 | 70.4 | 26.6 | 59.7 | 78.3 | 13.4 | 16.5 | 1.61 |
| expand3x3 fwd 8192x64x144 | 134.8 | 74.4 | 27.7 | 70.7 | 101.6 | 22.8 | 18.3 | 1.33 |
| fire8 fwd 2048x256x576 | 146.3 | 76.7 | 28.3 | 76.3 | 114.9 | 24.7 | 18.8 | 1.27 |
| expand3x3 dx 8192x144x64 NT | 146.7 | 78.1 | 28.5 | 74.0 | 105.7 | 21.7 | 7.2 | 1.39 |
| expand3x3 dw 144x64x8192 TN | 116.0 | 68.7 | 27.0 | 70.2 | 100.9 | 24.4 | 15.5 | 1.15 |
| tile 128x64x144 | 128.5 | 71.7 | 27.7 | 70.8 | 102.0 | 23.2 | 18.3 | 1.26 |

### gemv f32, 1 thread (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| gemv N 1024 | 62.4 | 51.8 | 40.3 | 66.6 | 65.6 | 51.8 | 5.7 | 0.94 |
| gemv T 1024 | 60.0 | 58.3 | 47.6 | 54.2 | 54.3 | 39.1 | 34.0 | 1.11 |
| gemv N 4096 | 16.1 | 18.3 | 15.3 | 17.7 | 16.6 | 13.1 | 5.4 | 1.03 |
| gemv T 4096 | 17.7 | 17.1 | 12.0 | 16.4 | 17.2 | 12.7 | 14.8 | 1.03 |
| gemv N 8192 | 13.2 | 13.2 | 10.4 | 13.8 | 13.8 | 12.3 | 4.2 | 0.96 |
| gemv T 8192 | 14.3 | 13.8 | 11.8 | 12.6 | 12.2 | 10.5 | 9.5 | 1.14 |

### gemv f64, 1 thread (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| gemv N 1024 | 37.3 | 30.1 | 20.5 | 35.1 | 35.7 | 27.2 | 5.5 | 1.04 |
| gemv T 1024 | 35.9 | 34.3 | 27.3 | 33.9 | 35.5 | 21.6 | 17.6 | 1.01 |
| gemv N 4096 | 6.9 | 6.6 | 5.1 | 7.1 | 7.1 | 6.4 | 3.0 | 0.96 |
| gemv T 4096 | 7.7 | 7.3 | 6.0 | 6.4 | 7.3 | 5.9 | 4.7 | 1.06 |
| gemv N 8192 | 6.8 | 6.7 | 5.0 | 6.5 | 6.4 | 6.3 | 3.5 | 1.05 |
| gemv T 8192 | 6.9 | 6.8 | 6.1 | 5.4 | 6.1 | 5.5 | 4.9 | 1.12 |

### dot f32, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| dot 2^16 | 268.5 | 317.2 | 165.1 | 313.2 | 323.0 | 85.5 | 21.5 | 0.98 |
| dot 2^20 | 164.7 | 142.7 | 124.5 | 143.1 | 161.0 | 79.2 | 21.2 | 1.02 |
| dot 2^24 | 35.7 | 36.1 | 34.8 | 36.9 | 35.4 | 33.8 | 20.4 | 0.98 |

### dot f64, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| dot 2^16 | 184.9 | 180.4 | 154.8 | 180.4 | 183.3 | 83.8 | 42.7 | 1.01 |
| dot 2^20 | 151.3 | 142.7 | 123.8 | 136.8 | 157.0 | 76.8 | 41.0 | 0.96 |
| dot 2^24 | 32.5 | 31.0 | 27.2 | 31.3 | 36.1 | 27.5 | 22.4 | 0.90 |

### axpy f32, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| axpy 2^16 | 331.1 | 379.2 | 218.6 | 370.3 | 443.3 | 111.7 | 60.4 | 0.86 |
| axpy 2^20 | 219.7 | 199.5 | 181.8 | 213.7 | 235.2 | 120.4 | 56.8 | 0.93 |
| axpy 2^24 | 40.3 | 39.0 | 37.4 | 39.8 | 39.7 | 37.3 | 34.2 | 1.01 |

### axpy f64, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| axpy 2^16 | 271.1 | 244.5 | 209.9 | 240.0 | 262.5 | 128.6 | 118.8 | 1.03 |
| axpy 2^20 | 220.6 | 206.3 | 167.2 | 211.5 | 210.5 | 101.1 | 104.9 | 1.04 |
| axpy 2^24 | 32.4 | 46.7 | 39.2 | 32.1 | 47.2 | 27.3 | 25.4 | 0.99 |

### scal f32, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| scal 2^16 | 381.9 | 325.0 | 164.6 | 363.3 | 439.8 | 168.3 | 63.6 | 0.87 |
| scal 2^20 | 249.0 | 230.9 | 164.6 | 247.1 | 250.4 | 164.6 | 62.2 | 0.99 |
| scal 2^24 | 40.2 | 43.2 | 42.9 | 57.2 | 54.5 | 53.6 | 47.0 | 0.76 |

### scal f64, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| scal 2^16 | 351.2 | 327.1 | 167.2 | 427.1 | 475.5 | 167.5 | 123.3 | 0.74 |
| scal 2^20 | 289.4 | 280.2 | 163.9 | 282.3 | 287.5 | 164.6 | 122.9 | 1.01 |
| scal 2^24 | 37.4 | 37.6 | 34.6 | 37.6 | 37.2 | 35.6 | 34.8 | 1.00 |

### gemm f32, 8 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| square 256 | 908.6 | 598.5 | 247.8 | 601.1 | 894.0 | 235.3 | 1.02 |
| square 512 | 1065.5 | 726.0 | 263.5 | 676.2 | 890.3 | 236.9 | 1.20 |
| square 1024 | 1051.2 | 735.9 | 264.3 | 679.9 | 1054.7 | 244.6 | 1.00 |
| square 2048 | 1058.6 | 694.6 | 265.7 | 690.1 | 1063.8 | 239.7 | 1.00 |
| square 4096 | 1040.7 | 700.9 | 258.3 | 673.7 | 1066.7 | 236.6 | 0.98 |
| conv1 fwd 32768x64x27 | 1129.8 | 937.0 | 276.7 | 878.1 | 1210.1 | 222.9 | 0.93 |
| squeeze fwd 8192x16x128 | 1011.8 | 750.9 | 264.3 | 829.2 | 1063.9 | 258.9 | 0.95 |
| expand1x1 fwd 8192x64x16 | 1197.9 | 581.3 | 254.8 | 855.7 | 1068.7 | 147.4 | 1.12 |
| expand3x3 fwd 8192x64x144 | 1516.8 | 954.9 | 278.2 | 947.4 | 1381.9 | 299.8 | 1.10 |
| fire8 fwd 2048x256x576 | 1448.7 | 887.5 | 281.8 | 1007.5 | 1421.2 | 297.9 | 1.02 |
| expand3x3 dx 8192x144x64 NT | 1578.9 | 973.1 | 190.1 | 958.5 | 1337.4 | 287.2 | 1.18 |
| expand3x3 dw 144x64x8192 TN | 986.1 | 716.1 | 250.0 | 373.6 | 449.7 | 199.1 | 2.19 |
| tile 128x64x144 | 255.1 | 143.9 | 52.3 | 433.6 | 563.3 | 156.0 | 0.45 |

### gemm f64, 8 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| square 256 | 424.2 | 306.8 | 119.9 | 288.4 | 390.1 | 118.9 | 1.09 |
| square 512 | 444.5 | 313.6 | 122.2 | 315.4 | 430.7 | 121.6 | 1.03 |
| square 1024 | 523.0 | 324.2 | 125.4 | 327.4 | 474.6 | 119.2 | 1.10 |
| square 2048 | 467.9 | 333.6 | 115.9 | 323.2 | 472.8 | 116.6 | 0.99 |
| square 4096 | 436.8 | 320.1 | 119.6 | 469.9 | 644.8 | 168.2 | 0.68 |
| conv1 fwd 32768x64x27 | 637.8 | 452.8 | 140.3 | 399.7 | 512.5 | 114.7 | 1.24 |
| squeeze fwd 8192x16x128 | 589.3 | 400.1 | 132.1 | 347.7 | 448.3 | 139.5 | 1.31 |
| expand1x1 fwd 8192x64x16 | 392.9 | 284.6 | 126.9 | 381.4 | 470.2 | 80.8 | 0.84 |
| expand3x3 fwd 8192x64x144 | 726.6 | 463.8 | 138.5 | 450.5 | 552.0 | 155.3 | 1.32 |
| fire8 fwd 2048x256x576 | 665.8 | 398.0 | 117.9 | 489.6 | 637.9 | 96.9 | 1.04 |
| expand3x3 dx 8192x144x64 NT | 745.4 | 511.7 | 143.2 | 459.5 | 507.1 | 124.8 | 1.47 |
| expand3x3 dw 144x64x8192 TN | 454.3 | 333.5 | 121.6 | 225.3 | 295.0 | 113.7 | 1.54 |
| tile 128x64x144 | 126.5 | 70.1 | 27.3 | 198.0 | 278.7 | 107.2 | 0.45 |

### gemv f32, 8 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| gemv N 1024 | 272.2 | 187.7 | 177.2 | 446.3 | 452.1 | 54.9 | 0.60 |
| gemv T 1024 | 150.2 | 141.5 | 116.5 | 229.8 | 208.1 | 41.1 | 0.65 |
| gemv N 4096 | 43.3 | 44.8 | 33.9 | 34.5 | 34.1 | 15.1 | 1.30 |
| gemv T 4096 | 31.7 | 33.1 | 38.4 | 31.3 | 30.0 | 17.9 | 1.23 |
| gemv N 8192 | 22.4 | 24.8 | 22.4 | 20.1 | 24.1 | 16.1 | 1.03 |
| gemv T 8192 | 23.5 | 22.1 | 20.7 | 20.2 | 19.5 | 13.7 | 1.16 |

### gemv f64, 8 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| gemv N 1024 | 146.5 | 124.1 | 109.5 | 186.2 | 188.6 | 28.3 | 0.78 |
| gemv T 1024 | 84.3 | 84.1 | 71.7 | 63.0 | 41.0 | 23.0 | 1.34 |
| gemv N 4096 | 11.8 | 12.2 | 13.2 | 10.3 | 11.6 | 8.0 | 1.13 |
| gemv T 4096 | 13.6 | 14.2 | 13.3 | 11.4 | 12.1 | 7.0 | 1.17 |
| gemv N 8192 | 12.7 | 10.8 | 10.8 | 7.2 | 10.2 | 7.5 | 1.25 |
| gemv T 8192 | 10.1 | 10.1 | 10.1 | 10.0 | 10.4 | 6.8 | 0.97 |

### dot f32, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| dot 2^16 | 245.7 | 242.3 | 167.2 | 260.5 | 287.6 | 84.8 | 0.85 |
| dot 2^20 | 402.4 | 384.4 | 368.8 | 140.1 | 159.3 | 78.8 | 2.53 |
| dot 2^24 | 29.8 | 26.5 | 29.7 | 25.5 | 27.2 | 20.0 | 1.10 |

### dot f64, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| dot 2^16 | 189.6 | 183.6 | 153.9 | 475.8 | 926.3 | 84.2 | 0.20 |
| dot 2^20 | 416.2 | 409.8 | 378.9 | 769.6 | 764.3 | 78.2 | 0.54 |
| dot 2^24 | 28.3 | 27.9 | 30.0 | 34.7 | 30.7 | 12.2 | 0.86 |

### axpy f32, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| axpy 2^16 | 290.7 | 258.2 | 213.9 | 800.8 | 785.6 | 117.2 | 0.36 |
| axpy 2^20 | 570.1 | 572.4 | 439.0 | 1185.8 | 1304.2 | 104.9 | 0.44 |
| axpy 2^24 | 31.3 | 29.1 | 29.9 | 35.4 | 29.2 | 18.2 | 0.88 |

### axpy f64, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| axpy 2^16 | 243.4 | 236.4 | 203.1 | 1121.1 | 1032.7 | 119.2 | 0.22 |
| axpy 2^20 | 578.9 | 310.0 | 431.7 | 1079.4 | 913.1 | 57.0 | 0.54 |
| axpy 2^24 | 34.4 | 31.9 | 28.3 | 31.4 | 31.0 | 13.2 | 1.10 |

### scal f32, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| scal 2^16 | 317.2 | 284.3 | 167.7 | 297.4 | 337.6 | 167.7 | 0.94 |
| scal 2^20 | 117.8 | 159.1 | 156.4 | 130.3 | 148.2 | 53.9 | 1.07 |
| scal 2^24 | 33.2 | 31.1 | 33.1 | 33.1 | 36.3 | 22.0 | 0.91 |

### scal f64, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| scal 2^16 | 327.1 | 282.1 | 168.0 | 298.1 | 363.5 | 168.3 | 0.90 |
| scal 2^20 | 91.6 | 108.9 | 91.1 | 146.9 | 156.1 | 68.3 | 0.70 |
| scal 2^24 | 33.9 | 29.6 | 31.6 | 32.3 | 36.4 | 20.5 | 0.93 |

### gemm f32, 16 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| square 256 | 308.8 | 503.6 | 136.7 | 6.7 | 5.6 | 0.7 | 75.66 |
| square 512 | 487.2 | 176.1 | 121.7 | 17.9 | 33.8 | 6.1 | 14.42 |
| square 1024 | 770.7 | 367.6 | 115.9 | 134.9 | 177.0 | 28.3 | 4.35 |
| square 2048 | 661.9 | 336.6 | 137.9 | 505.3 | 708.2 | 114.5 | 0.93 |
| square 4096 | 1088.7 | 739.1 | 301.1 | 736.4 | 1176.9 | 272.2 | 0.93 |
| conv1 fwd 32768x64x27 | 1034.9 | 755.1 | 265.9 | 797.0 | 1133.5 | 202.0 | 0.91 |
| squeeze fwd 8192x16x128 | 870.8 | 672.8 | 253.8 | 704.6 | 900.5 | 200.4 | 0.97 |
| expand1x1 fwd 8192x64x16 | 503.6 | 514.3 | 245.2 | 753.3 | 883.7 | 139.0 | 0.58 |
| expand3x3 fwd 8192x64x144 | 1325.0 | 894.4 | 272.4 | 896.7 | 1375.2 | 286.5 | 0.96 |
| fire8 fwd 2048x256x576 | 1456.0 | 911.4 | 285.3 | 1030.4 | 1357.2 | 300.6 | 1.07 |
| expand3x3 dx 8192x144x64 NT | 1364.9 | 928.9 | 298.0 | 896.7 | 1341.1 | 242.3 | 1.02 |
| expand3x3 dw 144x64x8192 TN | 981.7 | 687.1 | 273.7 | 453.1 | 509.2 | 199.2 | 1.93 |
| tile 128x64x144 | 250.0 | 138.2 | 51.1 | 347.9 | 309.5 | 77.8 | 0.72 |

### gemm f64, 16 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| square 256 | 324.1 | 149.7 | 97.8 | 2.8 | 3.7 | 0.7 | 86.85 |
| square 512 | 448.3 | 244.1 | 71.2 | 26.9 | 29.0 | 4.7 | 15.45 |
| square 1024 | 284.6 | 192.3 | 59.2 | 119.3 | 207.2 | 27.9 | 1.37 |
| square 2048 | 349.4 | 181.2 | 69.5 | 270.6 | 367.9 | 116.1 | 0.95 |
| square 4096 | 726.6 | 462.5 | 164.5 | 426.7 | 639.5 | 137.9 | 1.14 |
| conv1 fwd 32768x64x27 | 557.9 | 372.9 | 115.2 | 363.1 | 519.6 | 110.4 | 1.07 |
| squeeze fwd 8192x16x128 | 503.8 | 362.6 | 128.6 | 323.2 | 432.4 | 117.3 | 1.17 |
| expand1x1 fwd 8192x64x16 | 280.8 | 307.3 | 132.9 | 348.9 | 429.3 | 78.1 | 0.72 |
| expand3x3 fwd 8192x64x144 | 743.3 | 458.5 | 151.3 | 482.6 | 664.0 | 161.8 | 1.12 |
| fire8 fwd 2048x256x576 | 723.7 | 445.5 | 148.9 | 416.1 | 673.7 | 153.2 | 1.07 |
| expand3x3 dx 8192x144x64 NT | 655.1 | 448.8 | 147.5 | 433.4 | 609.9 | 151.2 | 1.07 |
| expand3x3 dw 144x64x8192 TN | 587.7 | 390.6 | 149.7 | 215.2 | 279.4 | 103.0 | 2.10 |
| tile 128x64x144 | 123.1 | 68.8 | 26.7 | 161.6 | 234.8 | 57.4 | 0.52 |

### gemv f32, 16 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| gemv N 1024 | 168.4 | 169.8 | 144.2 | 317.7 | 340.3 | 48.5 | 0.50 |
| gemv T 1024 | 112.9 | 109.3 | 86.6 | 68.9 | 69.8 | 40.2 | 1.62 |
| gemv N 4096 | 52.3 | 54.2 | 45.0 | 236.3 | 242.5 | 17.8 | 0.22 |
| gemv T 4096 | 40.4 | 40.7 | 48.4 | 66.3 | 66.1 | 17.1 | 0.73 |
| gemv N 8192 | 28.4 | 30.4 | 30.2 | 27.9 | 31.6 | 16.0 | 0.96 |
| gemv T 8192 | 24.7 | 25.5 | 26.7 | 25.3 | 21.6 | 13.7 | 1.06 |

### gemv f64, 16 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| gemv N 1024 | 124.0 | 122.7 | 89.5 | 212.9 | 165.5 | 27.4 | 0.58 |
| gemv T 1024 | 63.8 | 62.9 | 52.0 | 64.0 | 79.5 | 22.9 | 0.80 |
| gemv N 4096 | 18.2 | 18.7 | 14.0 | 23.2 | 22.4 | 9.5 | 0.81 |
| gemv T 4096 | 17.1 | 16.3 | 13.2 | 12.2 | 16.7 | 7.8 | 1.02 |
| gemv N 8192 | 12.1 | 13.4 | 11.8 | 13.4 | 14.2 | 8.3 | 0.95 |
| gemv T 8192 | 12.4 | 13.5 | 10.5 | 12.0 | 11.9 | 6.9 | 1.12 |

### dot f32, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| dot 2^16 | 302.4 | 239.0 | 158.6 | 257.8 | 315.3 | 81.9 | 0.96 |
| dot 2^20 | 342.4 | 336.7 | 323.6 | 123.7 | 148.2 | 76.6 | 2.31 |
| dot 2^24 | 65.5 | 68.7 | 64.7 | 42.7 | 40.5 | 33.9 | 1.61 |

### dot f64, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| dot 2^16 | 175.3 | 169.6 | 146.2 | 298.2 | 310.5 | 80.8 | 0.56 |
| dot 2^20 | 381.5 | 363.2 | 341.0 | 836.0 | 845.8 | 76.9 | 0.45 |
| dot 2^24 | 53.8 | 58.9 | 57.5 | 62.3 | 63.2 | 33.1 | 0.93 |

### axpy f32, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| axpy 2^16 | 298.5 | 253.2 | 212.1 | 243.0 | 235.7 | 104.8 | 1.23 |
| axpy 2^20 | 330.6 | 279.3 | 286.3 | 1309.6 | 1899.9 | 114.6 | 0.17 |
| axpy 2^24 | 64.5 | 69.9 | 69.1 | 92.7 | 79.7 | 35.6 | 0.75 |

### axpy f64, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| axpy 2^16 | 232.9 | 226.9 | 198.7 | 396.4 | 406.7 | 105.0 | 0.57 |
| axpy 2^20 | 294.1 | 348.5 | 305.8 | 1270.5 | 1090.2 | 114.7 | 0.27 |
| axpy 2^24 | 52.5 | 60.8 | 48.3 | 62.8 | 62.0 | 34.9 | 0.97 |

### scal f32, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| scal 2^16 | 363.3 | 275.4 | 162.5 | 290.6 | 387.5 | 162.5 | 0.94 |
| scal 2^20 | 513.3 | 453.1 | 287.5 | 248.0 | 251.0 | 163.8 | 2.05 |
| scal 2^24 | 131.7 | 129.3 | 125.4 | 1095.3 | 584.6 | 48.5 | 0.12 |

### scal f64, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| scal 2^16 | 364.7 | 275.4 | 162.5 | 290.8 | 367.3 | 163.5 | 0.99 |
| scal 2^20 | 502.7 | 464.4 | 358.3 | 272.8 | 286.0 | 161.6 | 1.76 |
| scal 2^24 | 75.0 | 80.3 | 77.6 | 90.6 | 95.1 | 41.4 | 0.84 |
