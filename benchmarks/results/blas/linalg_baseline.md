### linalg f64, 1 thread (ms per call, lower is better, median of 5 interleaved rounds)

| case | inhouse | lapack | in-house speed / best library |
|---|---|---|---|
| cholesky n=128 | 0.042 | 0.031 | 0.72 |
| solve k=1 n=128 | 0.055 | 0.049 | 0.89 |
| solve k=n n=128 | 0.127 | 0.194 | 1.53 |
| slogdet n=128 | 0.052 | 0.046 | 0.89 |
| triangular_solve k=1 n=128 | 0.001 | 0.005 | 3.42 |
| triangular_solve k=n n=128 | 0.040 | 0.043 | 1.08 |
| qr n=128 | 0.420 | 0.330 | 0.78 |
| eigh n=128 | 1.608 | 0.687 | 0.43 |
| svd n=128 | 10.770 | 1.240 | 0.12 |
| cholesky n=256 | 0.160 | 0.154 | 0.96 |
| solve k=1 n=256 | 0.236 | 0.237 | 1.01 |
| solve k=n n=256 | 0.678 | 1.044 | 1.54 |
| slogdet n=256 | 0.227 | 0.230 | 1.01 |
| triangular_solve k=1 n=256 | 0.003 | 0.019 | 6.17 |
| triangular_solve k=n n=256 | 0.236 | 0.279 | 1.18 |
| qr n=256 | 3.153 | 1.514 | 0.48 |
| eigh n=256 | 12.714 | 3.377 | 0.27 |
| svd n=256 | 86.585 | 6.593 | 0.08 |
| cholesky n=512 | 0.926 | 0.890 | 0.96 |
| solve k=1 n=512 | 1.342 | 1.351 | 1.01 |
| solve k=n n=512 | 4.426 | 6.111 | 1.38 |
| slogdet n=512 | 1.305 | 1.326 | 1.02 |
| triangular_solve k=1 n=512 | 0.011 | 0.065 | 6.14 |
| triangular_solve k=n n=512 | 1.602 | 1.720 | 1.07 |
| qr n=512 | 26.162 | 9.366 | 0.36 |
| eigh n=512 | 111.681 | 18.570 | 0.17 |
| svd n=512 | 715.845 | 42.071 | 0.06 |
| cholesky n=1024 | 4.820 | 4.970 | 1.03 |
| solve k=1 n=1024 | 7.242 | 8.462 | 1.17 |
| solve k=n n=1024 | 32.548 | 40.462 | 1.24 |
| slogdet n=1024 | 7.099 | 8.344 | 1.18 |
| triangular_solve k=1 n=1024 | 0.043 | 0.206 | 4.82 |
| triangular_solve k=n n=1024 | 10.675 | 11.378 | 1.07 |

### linalg f64, 8 threads (ms per call, lower is better, median of 5 interleaved rounds)

| case | inhouse | lapack | in-house speed / best library |
|---|---|---|---|
| cholesky n=128 | 0.042 | 0.046 | 1.08 |
| solve k=1 n=128 | 0.055 | 0.049 | 0.91 |
| solve k=n n=128 | 0.125 | 0.101 | 0.81 |
| slogdet n=128 | 0.051 | 0.047 | 0.91 |
| triangular_solve k=1 n=128 | 0.001 | 0.005 | 3.41 |
| triangular_solve k=n n=128 | 0.039 | 0.029 | 0.73 |
| qr n=128 | 0.385 | 0.438 | 1.14 |
| eigh n=128 | 1.611 | 0.691 | 0.43 |
| svd n=128 | 10.729 | 1.243 | 0.12 |
| cholesky n=256 | 0.188 | 0.114 | 0.61 |
| solve k=1 n=256 | 0.214 | 0.182 | 0.85 |
| solve k=n n=256 | 0.453 | 0.363 | 0.80 |
| slogdet n=256 | 0.241 | 0.131 | 0.54 |
| triangular_solve k=1 n=256 | 0.004 | 0.019 | 5.46 |
| triangular_solve k=n n=256 | 0.175 | 0.093 | 0.53 |
| qr n=256 | 3.179 | 1.462 | 0.46 |
| eigh n=256 | 12.747 | 3.134 | 0.25 |
| svd n=256 | 87.473 | 6.219 | 0.07 |
| cholesky n=512 | 0.511 | 0.333 | 0.65 |
| solve k=1 n=512 | 0.999 | 0.604 | 0.60 |
| solve k=n n=512 | 2.032 | 2.003 | 0.99 |
| slogdet n=512 | 0.986 | 0.581 | 0.59 |
| triangular_solve k=1 n=512 | 0.013 | 0.064 | 4.76 |
| triangular_solve k=n n=512 | 0.651 | 0.530 | 0.81 |
| qr n=512 | 26.696 | 7.053 | 0.26 |
| eigh n=512 | 111.599 | 13.721 | 0.12 |
| svd n=512 | 717.672 | 34.945 | 0.05 |
| cholesky n=1024 | 2.295 | 1.888 | 0.82 |
| solve k=1 n=1024 | 4.293 | 3.010 | 0.70 |
| solve k=n n=1024 | 14.568 | 14.194 | 0.97 |
| slogdet n=1024 | 4.360 | 3.010 | 0.69 |
| triangular_solve k=1 n=1024 | 0.042 | 0.203 | 4.89 |
| triangular_solve k=n n=1024 | 4.118 | 4.012 | 0.97 |
