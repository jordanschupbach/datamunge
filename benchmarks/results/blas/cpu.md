### gemm f32, 1 thread (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| square 256 | 294.0 | 159.2 | 58.4 | 160.3 | 291.8 | 52.5 | 37.6 | 1.01 |
| square 512 | 310.4 | 162.5 | 59.0 | 163.4 | 298.3 | 52.6 | 38.3 | 1.04 |
| square 1024 | 312.7 | 164.4 | 58.9 | 166.5 | 302.3 | 52.8 | 34.4 | 1.03 |
| square 2048 | 304.5 | 164.4 | 58.6 | 165.0 | 301.2 | 52.4 |  | 1.01 |
| square 4096 | 295.0 | 161.3 | 58.9 | 162.1 | 301.6 | 52.6 |  | 0.98 |
| conv1 fwd 32768x64x27 | 260.5 | 149.2 | 56.3 | 140.9 | 228.8 | 33.6 | 26.9 | 1.14 |
| squeeze fwd 8192x16x128 | 208.0 | 165.9 | 53.5 | 130.0 | 197.0 | 41.0 | 14.3 | 1.06 |
| expand1x1 fwd 8192x64x16 | 253.8 | 146.0 | 54.7 | 131.5 | 199.2 | 26.4 | 24.7 | 1.27 |
| expand3x3 fwd 8192x64x144 | 279.3 | 155.5 | 57.6 | 155.5 | 270.1 | 47.6 | 29.9 | 1.03 |
| fire8 fwd 2048x256x576 | 307.2 | 163.3 | 58.9 | 166.0 | 305.4 | 51.3 | 37.9 | 1.01 |
| expand3x3 dx 8192x144x64 NT | 283.0 | 161.4 | 58.7 | 156.5 | 272.9 | 44.6 | 11.7 | 1.04 |
| expand3x3 dw 144x64x8192 TN | 254.5 | 146.2 | 56.3 | 150.8 | 259.7 | 50.0 | 28.0 | 0.98 |
| tile 128x64x144 | 254.9 | 146.3 | 57.2 | 153.4 | 260.5 | 46.1 | 30.0 | 0.98 |

### gemm f32, 8 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| square 256 | 940.8 | 940.8 | 273.7 | 766.0 | 894.8 | 268.8 | 1.05 |
| square 512 | 1624.1 | 1090.8 | 282.5 | 1071.9 | 1688.9 | 366.0 | 0.96 |
| square 1024 | 1877.5 | 1121.3 | 285.2 | 1135.8 | 1916.7 | 371.5 | 0.98 |
| square 2048 | 1763.8 | 1075.2 | 306.5 | 1139.1 | 1920.4 | 366.9 | 0.92 |
| square 4096 | 1771.3 | 1062.8 | 316.2 | 1103.1 | 1928.5 | 365.6 | 0.92 |
| conv1 fwd 32768x64x27 | 1674.5 | 1001.9 | 305.1 | 949.7 | 1485.9 | 238.7 | 1.13 |
| squeeze fwd 8192x16x128 | 1399.6 | 1123.5 | 284.4 | 869.0 | 1275.3 | 277.5 | 1.10 |
| expand1x1 fwd 8192x64x16 | 1275.3 | 860.1 | 300.6 | 896.4 | 1401.2 | 178.3 | 0.91 |
| expand3x3 fwd 8192x64x144 | 1885.8 | 1092.2 | 311.2 | 1066.0 | 1769.5 | 334.0 | 1.07 |
| fire8 fwd 2048x256x576 | 1674.5 | 1063.2 | 320.2 | 1123.4 | 1808.2 | 362.7 | 0.93 |
| expand3x3 dx 8192x144x64 NT | 1918.9 | 1081.8 | 319.1 | 1087.4 | 1804.3 | 314.0 | 1.06 |
| expand3x3 dw 144x64x8192 TN | 1416.7 | 865.7 | 269.4 | 557.4 | 725.9 | 304.2 | 1.95 |
| tile 128x64x144 | 645.1 | 433.7 | 140.2 | 461.7 | 582.8 | 169.3 | 1.11 |

### gemm f32, 16 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| square 256 | 1017.0 | 899.6 | 451.4 | 734.3 | 633.5 | 261.2 | 1.39 |
| square 512 | 1760.8 | 1142.9 | 476.8 | 1217.2 | 1534.8 | 464.8 | 1.15 |
| square 1024 | 2142.3 | 961.0 | 284.3 | 1503.4 | 2139.9 | 539.4 | 1.00 |
| square 2048 | 2488.0 | 1460.4 | 508.7 | 1718.8 | 2523.6 | 579.0 | 0.99 |
| square 4096 | 2543.5 | 1378.1 | 465.9 | 1747.7 | 2784.8 | 580.7 | 0.91 |
| conv1 fwd 32768x64x27 | 2455.6 | 1597.9 | 512.8 | 1482.6 | 2164.5 | 360.8 | 1.13 |
| squeeze fwd 8192x16x128 | 2086.7 | 1672.9 | 493.2 | 1233.1 | 1670.4 | 231.3 | 1.25 |
| expand1x1 fwd 8192x64x16 | 1550.4 | 961.8 | 487.8 | 983.9 | 1195.2 | 212.9 | 1.30 |
| expand3x3 fwd 8192x64x144 | 2944.1 | 1739.9 | 526.3 | 1662.0 | 2543.2 | 467.0 | 1.16 |
| fire8 fwd 2048x256x576 | 2500.3 | 1565.3 | 512.7 | 1799.8 | 2476.3 | 455.2 | 1.01 |
| expand3x3 dx 8192x144x64 NT | 2734.2 | 1721.2 | 525.4 | 1659.8 | 2396.8 | 462.0 | 1.14 |
| expand3x3 dw 144x64x8192 TN | 1376.1 | 1125.7 | 448.9 | 790.0 | 580.9 | 226.2 | 1.74 |
| tile 128x64x144 | 552.8 | 372.0 | 158.9 | 290.7 | 276.4 | 83.5 | 1.90 |

### gemm f64, 1 thread (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| square 256 | 141.7 | 79.0 | 28.9 | 74.6 | 121.9 | 26.2 | 19.2 | 1.16 |
| square 512 | 152.3 | 79.7 | 29.1 | 78.3 | 128.2 | 26.4 | 17.3 | 1.19 |
| square 1024 | 151.0 | 81.1 | 29.0 | 79.8 | 131.8 | 26.4 | 18.0 | 1.15 |
| square 2048 | 147.9 | 80.5 | 29.0 | 80.2 | 132.4 | 26.3 |  | 1.12 |
| square 4096 | 144.8 | 80.3 | 29.0 | 80.7 | 134.6 | 26.4 |  | 1.08 |
| conv1 fwd 32768x64x27 | 127.5 | 72.4 | 27.8 | 64.4 | 94.0 | 17.1 | 17.7 | 1.36 |
| squeeze fwd 8192x16x128 | 152.4 | 65.6 | 26.8 | 58.0 | 78.1 | 22.3 | 11.8 | 1.95 |
| expand1x1 fwd 8192x64x16 | 129.9 | 72.8 | 27.2 | 63.3 | 82.2 | 13.6 | 16.7 | 1.58 |
| expand3x3 fwd 8192x64x144 | 140.2 | 77.6 | 28.4 | 74.4 | 107.5 | 24.5 | 18.6 | 1.30 |
| fire8 fwd 2048x256x576 | 152.6 | 81.8 | 29.3 | 80.7 | 126.6 | 26.0 | 18.8 | 1.21 |
| expand3x3 dx 8192x144x64 NT | 149.6 | 80.4 | 29.3 | 76.5 | 110.9 | 22.5 | 7.4 | 1.35 |
| expand3x3 dw 144x64x8192 TN | 124.6 | 73.1 | 28.0 | 71.5 | 103.9 | 25.2 | 16.5 | 1.20 |
| tile 128x64x144 | 132.1 | 73.8 | 28.2 | 72.9 | 106.2 | 24.0 | 18.6 | 1.24 |

### gemm f64, 8 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| square 256 | 805.7 | 474.7 | 156.9 | 432.3 | 717.8 | 176.9 | 1.12 |
| square 512 | 781.8 | 534.6 | 158.8 | 503.3 | 803.4 | 183.0 | 0.97 |
| square 1024 | 911.5 | 541.4 | 147.7 | 536.7 | 859.5 | 185.2 | 1.06 |
| square 2048 | 780.5 | 478.6 | 154.6 | 522.5 | 842.5 | 181.4 | 0.93 |
| square 4096 | 818.1 | 496.0 | 154.0 | 535.6 | 848.3 | 183.0 | 0.96 |
| conv1 fwd 32768x64x27 | 849.4 | 501.7 | 151.2 | 428.3 | 599.8 | 121.1 | 1.42 |
| squeeze fwd 8192x16x128 | 974.4 | 453.6 | 139.9 | 363.2 | 510.2 | 151.4 | 1.91 |
| expand1x1 fwd 8192x64x16 | 790.3 | 451.5 | 145.5 | 437.8 | 569.2 | 95.4 | 1.39 |
| expand3x3 fwd 8192x64x144 | 914.3 | 522.5 | 150.8 | 508.4 | 692.1 | 169.3 | 1.32 |
| fire8 fwd 2048x256x576 | 872.6 | 528.0 | 156.9 | 539.6 | 814.6 | 181.7 | 1.07 |
| expand3x3 dx 8192x144x64 NT | 971.0 | 530.9 | 151.3 | 523.1 | 724.5 | 159.1 | 1.34 |
| expand3x3 dw 144x64x8192 TN | 638.8 | 451.9 | 146.7 | 369.2 | 461.5 | 158.4 | 1.38 |
| tile 128x64x144 | 359.5 | 217.6 | 69.5 | 222.2 | 300.0 | 112.2 | 1.20 |

### gemm f64, 16 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| square 256 | 574.7 | 453.4 | 215.6 | 363.1 | 519.1 | 141.4 | 1.11 |
| square 512 | 820.9 | 575.3 | 231.7 | 534.6 | 728.2 | 237.2 | 1.13 |
| square 1024 | 942.8 | 535.6 | 168.1 | 715.0 | 1069.2 | 274.0 | 0.88 |
| square 2048 | 1063.5 | 540.3 | 238.6 | 792.3 | 1224.8 | 285.9 | 0.87 |
| square 4096 | 1187.7 | 704.7 | 239.1 | 821.6 | 1088.8 | 289.4 | 1.09 |
| conv1 fwd 32768x64x27 | 1322.0 | 791.9 | 251.1 | 713.4 | 904.3 | 192.1 | 1.46 |
| squeeze fwd 8192x16x128 | 1494.5 | 426.0 | 240.2 | 594.8 | 739.3 | 192.5 | 2.02 |
| expand1x1 fwd 8192x64x16 | 946.1 | 662.7 | 306.1 | 619.1 | 716.9 | 132.4 | 1.32 |
| expand3x3 fwd 8192x64x144 | 1407.8 | 846.2 | 255.4 | 799.7 | 1053.6 | 255.0 | 1.34 |
| fire8 fwd 2048x256x576 | 1209.7 | 780.5 | 255.8 | 814.8 | 1202.9 | 239.2 | 1.01 |
| expand3x3 dx 8192x144x64 NT | 1477.7 | 866.1 | 259.8 | 825.7 | 1067.3 | 243.5 | 1.38 |
| expand3x3 dw 144x64x8192 TN | 616.7 | 456.6 | 222.5 | 321.6 | 352.5 | 111.7 | 1.75 |
| tile 128x64x144 | 319.9 | 211.6 | 85.8 | 166.7 | 209.5 | 61.8 | 1.53 |

### gemv f32, 1 thread (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| gemv N 1024 | 65.8 | 59.8 | 41.0 | 69.1 | 68.7 | 58.8 | 5.8 | 0.95 |
| gemv T 1024 | 63.0 | 63.1 | 51.6 | 57.8 | 66.9 | 33.4 | 29.5 | 0.94 |
| gemv N 4096 | 23.3 | 33.9 | 30.3 | 28.9 | 32.5 | 25.8 | 5.5 | 1.05 |
| gemv T 4096 | 29.1 | 29.1 | 22.1 | 26.8 | 26.5 | 23.4 | 29.9 | 0.98 |
| gemv N 8192 | 19.3 | 24.1 | 18.2 | 24.0 | 24.8 | 21.6 | 5.1 | 0.97 |
| gemv T 8192 | 24.8 | 24.9 | 21.4 | 20.6 | 20.2 | 18.6 | 17.2 | 1.21 |

### gemv f32, 8 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| gemv N 1024 | 427.1 | 322.5 | 222.7 | 352.9 | 348.2 | 57.1 | 1.21 |
| gemv T 1024 | 136.7 | 155.0 | 151.7 | 167.9 | 188.2 | 33.7 | 0.82 |
| gemv N 4096 | 112.1 | 168.6 | 54.9 | 43.0 | 56.8 | 23.3 | 2.97 |
| gemv T 4096 | 44.8 | 27.6 | 25.4 | 35.0 | 35.8 | 22.3 | 1.25 |
| gemv N 8192 | 27.2 | 33.6 | 32.2 | 24.7 | 24.4 | 21.9 | 1.36 |
| gemv T 8192 | 25.1 | 23.7 | 20.2 | 25.8 | 25.9 | 18.5 | 0.97 |

### gemv f32, 16 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| gemv N 1024 | 396.4 | 362.8 | 239.0 | 391.2 | 368.5 | 56.6 | 1.01 |
| gemv T 1024 | 47.1 | 44.6 | 45.5 | 70.9 | 76.5 | 39.1 | 0.62 |
| gemv N 4096 | 450.8 | 431.1 | 335.0 | 378.4 | 378.7 | 23.0 | 1.19 |
| gemv T 4096 | 57.0 | 41.8 | 57.3 | 80.1 | 77.4 | 22.1 | 0.72 |
| gemv N 8192 | 34.7 | 36.7 | 36.8 | 33.3 | 33.4 | 21.4 | 1.10 |
| gemv T 8192 | 25.4 | 25.5 | 22.3 | 26.7 | 28.1 | 18.7 | 0.91 |

### gemv f64, 1 thread (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| gemv N 1024 | 32.1 | 30.2 | 20.7 | 28.9 | 29.9 | 29.4 | 5.6 | 1.07 |
| gemv T 1024 | 31.8 | 31.3 | 27.4 | 31.1 | 34.9 | 14.3 | 17.7 | 0.91 |
| gemv N 4096 | 10.7 | 11.6 | 9.1 | 11.7 | 11.8 | 12.1 | 4.4 | 0.96 |
| gemv T 4096 | 13.2 | 13.1 | 11.7 | 10.8 | 11.7 | 9.9 | 8.9 | 1.13 |
| gemv N 8192 | 11.0 | 12.6 | 10.3 | 10.8 | 10.7 | 10.6 | 4.7 | 1.16 |
| gemv T 8192 | 12.2 | 11.9 | 11.3 | 10.0 | 10.3 | 9.1 | 9.5 | 1.18 |

### gemv f64, 8 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| gemv N 1024 | 200.1 | 176.8 | 127.1 | 212.1 | 214.9 | 29.4 | 0.93 |
| gemv T 1024 | 89.2 | 39.1 | 64.0 | 82.3 | 53.5 | 16.4 | 1.08 |
| gemv N 4096 | 21.2 | 23.7 | 16.8 | 14.3 | 14.8 | 11.5 | 1.61 |
| gemv T 4096 | 12.6 | 12.0 | 10.3 | 14.2 | 14.5 | 9.6 | 0.87 |
| gemv N 8192 | 15.0 | 15.2 | 15.1 | 12.8 | 12.7 | 10.7 | 1.19 |
| gemv T 8192 | 13.6 | 13.0 | 10.8 | 13.0 | 13.0 | 9.1 | 1.05 |

### gemv f64, 16 threads (GFLOP/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| gemv N 1024 | 227.0 | 192.0 | 203.0 | 230.0 | 225.3 | 29.4 | 0.99 |
| gemv T 1024 | 43.0 | 30.8 | 35.5 | 58.5 | 25.6 | 15.6 | 0.74 |
| gemv N 4096 | 27.4 | 28.4 | 26.3 | 24.8 | 23.8 | 11.3 | 1.14 |
| gemv T 4096 | 13.1 | 15.1 | 13.8 | 17.8 | 17.3 | 10.1 | 0.84 |
| gemv N 8192 | 15.1 | 15.9 | 15.8 | 14.9 | 14.8 | 10.8 | 1.07 |
| gemv T 8192 | 13.4 | 12.9 | 11.4 | 14.0 | 14.6 | 9.4 | 0.92 |

### dot f32, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| dot 2^16 | 221.8 | 240.1 | 164.0 | 219.8 | 254.0 | 86.5 | 21.7 | 0.95 |
| dot 2^20 | 158.5 | 144.2 | 132.3 | 133.6 | 145.3 | 81.8 | 21.7 | 1.09 |
| dot 2^24 | 58.6 | 58.4 | 59.1 | 61.8 | 57.8 | 58.9 | 21.1 | 0.96 |

### dot f32, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| dot 2^16 | 319.1 | 339.8 | 294.0 | 224.5 | 245.7 | 86.4 | 1.38 |
| dot 2^20 | 1046.6 | 1011.2 | 833.9 | 137.3 | 151.6 | 83.1 | 6.91 |
| dot 2^24 | 91.2 | 90.1 | 88.2 | 57.7 | 56.3 | 59.2 | 1.54 |

### dot f32, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| dot 2^16 | 293.9 | 281.4 | 278.3 | 221.7 | 259.0 | 86.5 | 1.13 |
| dot 2^20 | 1726.4 | 1500.7 | 1331.1 | 134.7 | 152.6 | 81.1 | 11.32 |
| dot 2^24 | 98.1 | 106.3 | 98.8 | 62.2 | 59.3 | 61.4 | 1.71 |

### dot f64, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| dot 2^16 | 195.6 | 204.4 | 161.5 | 175.9 | 225.5 | 86.4 | 43.4 | 0.91 |
| dot 2^20 | 161.9 | 148.2 | 138.2 | 152.5 | 166.0 | 80.6 | 42.8 | 0.97 |
| dot 2^24 | 52.4 | 51.6 | 48.5 | 52.9 | 52.7 | 35.5 | 27.0 | 0.99 |

### dot f64, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| dot 2^16 | 542.2 | 533.9 | 467.3 | 407.2 | 423.7 | 86.4 | 1.28 |
| dot 2^20 | 864.5 | 889.8 | 830.2 | 879.5 | 870.8 | 84.0 | 1.01 |
| dot 2^24 | 65.5 | 64.8 | 66.7 | 54.9 | 53.0 | 35.9 | 1.21 |

### dot f64, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| dot 2^16 | 500.8 | 505.6 | 436.2 | 312.4 | 312.4 | 86.1 | 1.62 |
| dot 2^20 | 1645.0 | 1220.5 | 1419.2 | 1472.8 | 1557.6 | 81.1 | 1.06 |
| dot 2^24 | 69.5 | 68.7 | 70.3 | 73.4 | 69.1 | 36.1 | 0.96 |

### axpy f32, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| axpy 2^16 | 286.4 | 261.6 | 218.0 | 263.4 | 301.9 | 131.1 | 60.0 | 0.95 |
| axpy 2^20 | 217.9 | 225.9 | 189.1 | 200.9 | 235.9 | 118.8 | 59.5 | 0.96 |
| axpy 2^24 | 65.1 | 64.9 | 63.0 | 65.7 | 64.0 | 57.3 | 38.3 | 0.99 |

### axpy f32, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| axpy 2^16 | 446.1 | 419.9 | 400.4 | 299.6 | 300.7 | 130.8 | 1.48 |
| axpy 2^20 | 1659.1 | 1264.7 | 1151.1 | 1336.0 | 1401.8 | 128.9 | 1.18 |
| axpy 2^24 | 77.0 | 66.2 | 67.1 | 67.0 | 66.3 | 56.7 | 1.15 |

### axpy f32, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| axpy 2^16 | 165.6 | 172.5 | 173.3 | 249.2 | 238.6 | 130.8 | 0.70 |
| axpy 2^20 | 1280.2 | 1339.0 | 1309.5 | 1788.9 | 1427.1 | 126.1 | 0.75 |
| axpy 2^24 | 94.4 | 92.3 | 100.6 | 103.1 | 97.4 | 56.6 | 0.98 |

### axpy f64, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| axpy 2^16 | 271.1 | 254.5 | 211.3 | 257.8 | 242.2 | 130.9 | 119.2 | 1.05 |
| axpy 2^20 | 217.8 | 229.6 | 203.2 | 228.3 | 232.8 | 129.6 | 112.7 | 0.99 |
| axpy 2^24 | 56.2 | 55.8 | 51.9 | 57.0 | 56.4 | 37.9 | 34.9 | 0.99 |

### axpy f64, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| axpy 2^16 | 777.1 | 710.4 | 618.3 | 539.4 | 520.0 | 130.4 | 1.44 |
| axpy 2^20 | 1269.8 | 1266.7 | 1221.1 | 1359.2 | 1334.6 | 127.6 | 0.93 |
| axpy 2^24 | 61.5 | 58.9 | 58.5 | 57.7 | 58.4 | 37.6 | 1.05 |

### axpy f64, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| axpy 2^16 | 223.6 | 246.1 | 196.7 | 422.0 | 426.6 | 130.5 | 0.58 |
| axpy 2^20 | 2319.2 | 2178.7 | 1825.5 | 1327.6 | 2167.2 | 125.3 | 1.07 |
| axpy 2^24 | 65.0 | 65.4 | 64.7 | 68.4 | 67.2 | 37.7 | 0.96 |

### scal f32, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| scal 2^16 | 294.0 | 251.6 | 168.8 | 272.6 | 333.3 | 169.9 | 61.5 | 0.88 |
| scal 2^20 | 253.9 | 249.3 | 167.3 | 275.6 | 300.3 | 168.7 | 62.8 | 0.85 |
| scal 2^24 | 63.2 | 64.7 | 65.5 | 66.4 | 64.0 | 64.9 | 52.8 | 0.99 |

### scal f32, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| scal 2^16 | 306.1 | 307.9 | 281.3 | 282.9 | 325.0 | 171.0 | 0.95 |
| scal 2^20 | 943.9 | 822.5 | 929.3 | 284.6 | 270.3 | 169.1 | 3.32 |
| scal 2^24 | 238.5 | 211.9 | 477.1 | 294.1 | 123.7 | 65.6 | 1.62 |

### scal f32, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| scal 2^16 | 109.0 | 110.2 | 112.1 | 277.0 | 327.1 | 168.3 | 0.34 |
| scal 2^20 | 1007.5 | 704.2 | 814.4 | 284.4 | 258.4 | 168.5 | 3.54 |
| scal 2^24 | 394.9 | 577.3 | 1112.2 | 1153.1 | 1317.9 | 64.5 | 0.84 |

### scal f64, 1 thread (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | reference | in-house speed / best library |
|---|---|---|---|---|---|---|---|---|
| scal 2^16 | 354.7 | 285.9 | 170.4 | 304.2 | 347.7 | 171.3 | 123.4 | 1.02 |
| scal 2^20 | 293.3 | 270.7 | 168.6 | 265.6 | 293.6 | 168.7 | 124.4 | 1.00 |
| scal 2^24 | 52.3 | 52.9 | 50.5 | 53.6 | 52.0 | 51.1 | 50.9 | 0.99 |

### scal f64, 8 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| scal 2^16 | 559.8 | 559.5 | 471.3 | 303.3 | 340.9 | 172.2 | 1.64 |
| scal 2^20 | 1814.3 | 1522.3 | 985.0 | 283.1 | 281.6 | 169.3 | 6.41 |
| scal 2^24 | 86.6 | 61.1 | 85.3 | 60.9 | 59.8 | 51.3 | 1.42 |

### scal f64, 16 threads (GB/s, median of >= 5 calls)

| case | inhouse-avx512 | inhouse-avx2 | inhouse-generic | openblas | openblas-avx512 | blis | in-house speed / best library |
|---|---|---|---|---|---|---|---|
| scal 2^16 | 136.4 | 146.0 | 131.6 | 304.2 | 343.2 | 171.6 | 0.43 |
| scal 2^20 | 1633.8 | 2469.8 | 1616.3 | 284.9 | 286.5 | 168.6 | 8.62 |
| scal 2^24 | 99.7 | 97.0 | 100.2 | 105.6 | 112.5 | 51.2 | 0.89 |
