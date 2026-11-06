#pragma once

// Blocked Householder QR, symmetric eigensolver and SVD for the built-in
// linalg kernels (linalg.cpp). Not installed.
//
// LAPACK's algorithms, written around the in-house GEMM: QR as dgeqrf/dorgqr,
// eigh as dsytrd + divide and conquer (dstedc) + dormtr, the SVD as a QR,
// dgebrd + bidiagonal divide and conquer (dbdsdc) + dormbr. All matrices are
// row-major [rows, cols] as in linalg.cpp; the kernels work on column-major
// copies internally.

#include <cstdint>

namespace datamunge::nn::detail {

/// QR of a [m, n]: q [m, qc] with qc = m when full, else min(m, n); r [rr, n]
/// with rr = m when full, else min(m, n). R's diagonal has the opposite sign of
/// the reflected column's leading entry (LAPACK's convention).
void qr_householder(const double* a, double* q, double* r, std::int64_t m, std::int64_t n, bool full);

/// Eigen decomposition of the symmetric matrix whose lower triangle is a
/// [n, n]: e ascending; v [n, n] with the eigenvectors as columns, or null
/// for the values only.
void eigh_householder(const double* a, double* e, double* v, std::int64_t n);

/// dst[j * ldd + i] = src[i * lds + j] for a rows x cols block (a transpose), in cache-sized
/// tiles, split over the pool when large.
void transpose_into(const double* src, std::int64_t lds, double* dst, std::int64_t ldd, std::int64_t rows,
                    std::int64_t cols);

/// SVD of a [m, n]: s [min(m, n)] descending; u [m, uc] and v [n, vc] with
/// uc, vc = min(m, n), or m and n when full; u and v null for the values only.
void svd_householder(const double* a, double* s, double* u, double* v, std::int64_t m, std::int64_t n, bool full);

}  // namespace datamunge::nn::detail
