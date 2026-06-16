#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <vector>

#include <datamunge/linalg/sparse_coo.hpp>

namespace datamunge {
namespace linalg {

// ============================================================
// ELLPACK (ELL) — row-major layout
//
// Each row stores exactly max_nnz_per_row slots.  Unused slots carry
// col = no_entry and value = 0.  Row-major layout:
//   col_idx[i * max_nnz_per_row + p]  — p-th entry of row i
//
// Strengths: no indirection on row start, simple inner loop, easy to
//   vectorize.  Best when row nnz counts are nearly uniform; large
//   variance wastes memory proportional to (max - avg) * nrows.
//
// Note: column-major ("transposed") layout is the GPU/cuSPARSE standard
//   and gives better coalescing.  Row-major is used here for CPU cache
//   efficiency (accumulating into y[i] stays in register across the
//   inner loop over p).
// ============================================================

template <typename T = double>
struct SparseELL {
    using value_type = T;

    // Sentinel column index for padded (unused) slots.
    static constexpr std::size_t no_entry = std::numeric_limits<std::size_t>::max();

    std::size_t nrows{}, ncols{};
    std::size_t max_nnz_per_row{};
    std::size_t nnz_{};               // actual non-zeros (not counting padding)
    std::vector<std::size_t> col_idx; // nrows * max_nnz_per_row, row-major
    std::vector<T>           values;  // nrows * max_nnz_per_row, row-major

    std::size_t rows()       const { return nrows; }
    std::size_t cols()       const { return ncols; }
    std::size_t nnz()        const { return nnz_; }
    std::size_t nnz_padded() const { return nrows * max_nnz_per_row; }

    // y = A * x
    void spmv(const std::vector<T>& x, std::vector<T>& y) const {
        std::fill(y.begin(), y.end(), T{0});
        for (std::size_t i = 0; i < nrows; ++i) {
            const std::size_t base = i * max_nnz_per_row;
            for (std::size_t p = 0; p < max_nnz_per_row; ++p) {
                const std::size_t c = col_idx[base + p];
                if (c != no_entry)
                    y[i] += values[base + p] * x[c];
            }
        }
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

/// COO → ELL.  Requires coo.is_compressed().
template <typename T>
SparseELL<T> to_ell(const SparseCOO<T>& coo) {
    assert(coo.is_compressed() && "to_ell: call compress() first");
    const std::size_t rows = coo.rows();

    std::vector<std::size_t> row_counts(rows, 0);
    for (auto r : coo.row_indices()) ++row_counts[r];
    const std::size_t max_per_row =
        rows > 0 ? *std::max_element(row_counts.begin(), row_counts.end()) : 0;

    SparseELL<T> out;
    out.nrows            = rows;
    out.ncols            = coo.cols();
    out.max_nnz_per_row  = max_per_row;
    out.nnz_             = coo.nnz();
    out.col_idx.assign(rows * max_per_row, SparseELL<T>::no_entry);
    out.values.assign(rows * max_per_row, T{0});

    std::vector<std::size_t> pos(rows, 0);
    for (std::size_t k = 0; k < coo.nnz(); ++k) {
        const std::size_t r          = coo.row_indices()[k];
        const std::size_t p          = pos[r]++;
        out.col_idx[r * max_per_row + p] = coo.col_indices()[k];
        out.values[r * max_per_row + p]  = coo.values()[k];
    }

    return out;
}

/// ELL → COO.
template <typename T>
SparseCOO<T> to_coo(const SparseELL<T>& ell) {
    SparseCOO<T> out(ell.nrows, ell.ncols, ell.nnz());
    for (std::size_t i = 0; i < ell.nrows; ++i) {
        const std::size_t base = i * ell.max_nnz_per_row;
        for (std::size_t p = 0; p < ell.max_nnz_per_row; ++p) {
            const std::size_t c = ell.col_idx[base + p];
            if (c != SparseELL<T>::no_entry)
                out.set(i, c, ell.values[base + p]);
        }
    }
    out.compress();
    return out;
}

} // namespace linalg
} // namespace datamunge
