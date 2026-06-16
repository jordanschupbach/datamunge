#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include <datamunge/linalg/sparse_coo.hpp>

namespace datamunge {
namespace linalg {

// ============================================================
// Diagonal (DIA) format
//
// Stores only the diagonals that contain at least one non-zero.
// offsets[d] = col - row for diagonal d (negative = below main diagonal).
// data layout: data[d * nrows + row] = value at (row, row + offsets[d]).
// Invalid (out-of-range) positions in the data array are zero.
//
// Strengths: extremely cache-friendly SpMV for banded matrices (FEM/FDM
//   stiffness matrices, tridiagonal systems).  Memory is O(num_diags * nrows).
//   Poor fit when entries are scattered across many diagonals.
// ============================================================

template <typename T = double>
struct SparseDIA {
    using value_type = T;

    std::size_t                  nrows{}, ncols{};
    std::vector<std::ptrdiff_t>  offsets; // sorted diagonal offsets
    std::vector<T>               data;    // offsets.size() * nrows, row-major per diagonal
    std::size_t                  nnz_{};  // actual non-zero count

    std::size_t rows()      const { return nrows; }
    std::size_t cols()      const { return ncols; }
    std::size_t nnz()       const { return nnz_; }
    std::size_t num_diags() const { return offsets.size(); }

    // y = A * x
    void spmv(const std::vector<T>& x, std::vector<T>& y) const {
        std::fill(y.begin(), y.end(), T{0});
        for (std::size_t d = 0; d < offsets.size(); ++d) {
            const std::ptrdiff_t off       = offsets[d];
            const std::size_t    row_start = off >= 0 ? 0 : static_cast<std::size_t>(-off);
            const std::size_t    col_start = off >= 0 ? static_cast<std::size_t>(off) : 0;
            const std::size_t    len =
                std::min(nrows - row_start, ncols - col_start);
            // Sequential reads along a diagonal; y[row] accumulates in register.
            for (std::size_t i = 0; i < len; ++i)
                y[row_start + i] += data[d * nrows + row_start + i] * x[col_start + i];
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

/// COO → DIA.  Requires coo.is_compressed().
///
/// All diagonals that contain at least one non-zero are stored.
/// If the matrix has many scattered non-zeros this will still work
/// but produces a DIA with many diagonals and poor fill ratio.
template <typename T>
SparseDIA<T> to_dia(const SparseCOO<T>& coo) {
    assert(coo.is_compressed() && "to_dia: call compress() first");
    const std::size_t n = coo.nnz();

    // Collect unique diagonal offsets (offset = col - row)
    std::vector<std::ptrdiff_t> raw;
    raw.reserve(n);
    for (std::size_t k = 0; k < n; ++k)
        raw.push_back(static_cast<std::ptrdiff_t>(coo.col_indices()[k]) -
                      static_cast<std::ptrdiff_t>(coo.row_indices()[k]));
    std::sort(raw.begin(), raw.end());
    raw.erase(std::unique(raw.begin(), raw.end()), raw.end());

    SparseDIA<T> out;
    out.nrows   = coo.rows();
    out.ncols   = coo.cols();
    out.offsets = raw;
    out.nnz_    = n;
    out.data.assign(raw.size() * coo.rows(), T{0});

    for (std::size_t k = 0; k < n; ++k) {
        const std::ptrdiff_t off   = static_cast<std::ptrdiff_t>(coo.col_indices()[k]) -
                                     static_cast<std::ptrdiff_t>(coo.row_indices()[k]);
        const auto           it    = std::lower_bound(out.offsets.begin(), out.offsets.end(), off);
        const std::size_t    d_idx = static_cast<std::size_t>(it - out.offsets.begin());
        out.data[d_idx * out.nrows + coo.row_indices()[k]] = coo.values()[k];
    }

    return out;
}

/// DIA → COO.
template <typename T>
SparseCOO<T> to_coo(const SparseDIA<T>& dia) {
    SparseCOO<T> out(dia.nrows, dia.ncols, dia.nnz());
    for (std::size_t d = 0; d < dia.offsets.size(); ++d) {
        const std::ptrdiff_t off       = dia.offsets[d];
        const std::size_t    row_start = off >= 0 ? 0 : static_cast<std::size_t>(-off);
        const std::size_t    col_start = off >= 0 ? static_cast<std::size_t>(off) : 0;
        const std::size_t    len =
            std::min(dia.nrows - row_start, dia.ncols - col_start);
        for (std::size_t i = 0; i < len; ++i) {
            const T val = dia.data[d * dia.nrows + row_start + i];
            if (val != T{0})
                out.set(row_start + i, col_start + i, val);
        }
    }
    out.compress();
    return out;
}

} // namespace linalg
} // namespace datamunge
