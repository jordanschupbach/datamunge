#pragma once

#include <datamunge/linalg/sparse_coo.hpp>
#include <datamunge/linalg/csr.hpp>
#include <datamunge/linalg/csc.hpp>
#include <datamunge/linalg/ell.hpp>
#include <datamunge/linalg/dia.hpp>
#include <datamunge/linalg/sparse_matrix.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

#include <algorithm>
#include <vector>

namespace datamunge::linalg {

// ============================================================
// diagonal() — extract the main diagonal as a dense vector
//
// For each format the main diagonal is the set of entries
// where row == col.  Duplicate (i,i) entries in uncompressed
// COO are summed (consistent with compress() semantics).
// ============================================================

template <typename T>
std::vector<T> diagonal(const SparseCOO<T>& A) {
    const std::size_t n = std::min(A.rows(), A.cols());
    std::vector<T> d(n, T{});
    const auto& rows = A.row_indices();
    const auto& cols = A.col_indices();
    const auto& vals = A.values();
    for (std::size_t k = 0; k < rows.size(); ++k)
        if (rows[k] == cols[k])
            d[rows[k]] += vals[k]; // += handles uncompressed duplicates
    return d;
}

template <typename T>
std::vector<T> diagonal(const SparseCSR<T>& A) {
    const std::size_t n = std::min(A.nrows, A.ncols);
    std::vector<T> d(n, T{});
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
            if (A.col_idx[p] == i) { d[i] = A.values[p]; break; }
    return d;
}

template <typename T>
std::vector<T> diagonal(const SparseCSC<T>& A) {
    const std::size_t n = std::min(A.nrows, A.ncols);
    std::vector<T> d(n, T{});
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p)
            if (A.row_idx[p] == j) { d[j] = A.values[p]; break; }
    return d;
}

template <typename T>
std::vector<T> diagonal(const SparseELL<T>& A) {
    const std::size_t n = std::min(A.nrows, A.ncols);
    std::vector<T> d(n, T{});
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t p = 0; p < A.max_nnz_per_row; ++p) {
            const std::size_t c = A.col_idx[i * A.max_nnz_per_row + p];
            if (c == SparseELL<T>::no_entry) break;
            if (c == i) { d[i] = A.values[i * A.max_nnz_per_row + p]; break; }
        }
    return d;
}

template <typename T>
std::vector<T> diagonal(const SparseDIA<T>& A) {
    const std::size_t n = std::min(A.nrows, A.ncols);
    std::vector<T> d(n, T{});
    // DIA offsets are sorted; the main diagonal has offset 0 (col − row = 0)
    const auto it = std::lower_bound(A.offsets.begin(), A.offsets.end(),
                                     std::ptrdiff_t{0});
    if (it == A.offsets.end() || *it != 0) return d;
    const std::size_t idx = static_cast<std::size_t>(it - A.offsets.begin());
    for (std::size_t i = 0; i < n; ++i)
        d[i] = A.data[idx * A.nrows + i];
    return d;
}

template <typename T>
std::vector<T> diagonal(const SparseMatrix<T>& A) {
    if (const auto* p = A.as_coo()) return diagonal(*p);
    if (const auto* p = A.as_csr()) return diagonal(*p);
    if (const auto* p = A.as_csc()) return diagonal(*p);
    if (const auto* p = A.as_ell()) return diagonal(*p);
    if (const auto* p = A.as_dia()) return diagonal(*p);
    return {};
}

// ============================================================
// JacobiPreconditioner — M = diag(A),  apply: z = M⁻¹ r
//
// For a symmetric positive definite A all diagonal entries are
// positive; zero entries are treated as 1 (identity component)
// to avoid division by zero.
// ============================================================

template <typename T = double>
struct JacobiPreconditioner {
    std::vector<T> inv_diag; // 1 / a_{ii}

    void apply(const std::vector<T>& r, std::vector<T>& z) const {
        for (std::size_t i = 0; i < r.size(); ++i)
            z[i] = inv_diag[i] * r[i];
    }
};

// Build a Jacobi preconditioner from any supported sparse matrix.
template <typename Matrix>
auto make_jacobi(const Matrix& A) {
    auto d = diagonal(A);
    using T = typename decltype(d)::value_type;
    JacobiPreconditioner<T> P;
    P.inv_diag.resize(d.size());
    for (std::size_t i = 0; i < d.size(); ++i)
        P.inv_diag[i] = (d[i] != T{}) ? T{1} / d[i] : T{1};
    return P;
}

template <typename T>
std::vector<T> diagonal(const DenseMatrix<T>& A) {
    const std::size_t n = std::min(A.rows(), A.cols());
    std::vector<T> d(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = A(i, i);
    return d;
}

} // namespace datamunge::linalg
