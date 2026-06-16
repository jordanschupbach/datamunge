#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <vector>

#include <datamunge/linalg/sparse_coo.hpp>

namespace datamunge {
namespace linalg {

// ============================================================
// Compressed Sparse Column (CSC)
//
// col_ptr[j]..col_ptr[j+1] is the half-open range of entries for column j.
// row_idx and values are in column-major order; within each column, entries
// are row-sorted (guaranteed when built from a compressed COO).
//
// Strengths: column slicing in O(1), natural for direct solvers (LAPACK,
//   SuiteSparse), transpose SpMV, column-oriented factorizations.
// ============================================================

template <typename T = double>
struct SparseCSC {
    using value_type = T;

    std::size_t              nrows{}, ncols{};
    std::vector<std::size_t> col_ptr;  // size ncols+1
    std::vector<std::size_t> row_idx;  // size nnz
    std::vector<T>           values;   // size nnz

    std::size_t rows() const { return nrows; }
    std::size_t cols() const { return ncols; }
    std::size_t nnz()  const { return values.size(); }

    // y = A * x  (scatter-accumulate over columns)
    void spmv(const std::vector<T>& x, std::vector<T>& y) const {
        std::fill(y.begin(), y.end(), T{0});
        for (std::size_t j = 0; j < ncols; ++j)
            for (std::size_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
                y[row_idx[k]] += values[k] * x[j];
    }

    std::vector<T> spmv(const std::vector<T>& x) const {
        std::vector<T> y(nrows, T{0});
        spmv(x, y);
        return y;
    }
};

// ============================================================
// Conversions
// ============================================================

/// COO → CSC.  Requires coo.is_compressed().
/// Internally re-sorts entries by (col, row) to build the column structure.
template <typename T>
SparseCSC<T> to_csc(const SparseCOO<T>& coo) {
    assert(coo.is_compressed() && "to_csc: call compress() first");
    const std::size_t n = coo.nnz();

    // Permutation that sorts entries by (col, row)
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return coo.col_indices()[a] != coo.col_indices()[b]
                   ? coo.col_indices()[a] < coo.col_indices()[b]
                   : coo.row_indices()[a] < coo.row_indices()[b];
    });

    SparseCSC<T> out;
    out.nrows = coo.rows();
    out.ncols = coo.cols();
    out.row_idx.resize(n);
    out.values.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.row_idx[i] = coo.row_indices()[order[i]];
        out.values[i]  = coo.values()[order[i]];
    }

    out.col_ptr.assign(coo.cols() + 1, 0);
    for (auto c : coo.col_indices()) ++out.col_ptr[c + 1];
    for (std::size_t j = 0; j < coo.cols(); ++j)
        out.col_ptr[j + 1] += out.col_ptr[j];

    return out;
}

/// CSC → COO.  Output is compressed (col-sorted, then compress() sorts by row).
template <typename T>
SparseCOO<T> to_coo(const SparseCSC<T>& csc) {
    SparseCOO<T> out(csc.nrows, csc.ncols, csc.nnz());
    for (std::size_t j = 0; j < csc.ncols; ++j)
        for (std::size_t k = csc.col_ptr[j]; k < csc.col_ptr[j + 1]; ++k)
            out.set(csc.row_idx[k], j, csc.values[k]);
    out.compress();
    return out;
}

} // namespace linalg
} // namespace datamunge
