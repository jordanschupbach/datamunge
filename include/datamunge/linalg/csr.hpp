#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include <datamunge/linalg/sparse_coo.hpp>

namespace datamunge {
namespace linalg {

// ============================================================
// Compressed Sparse Row (CSR)
//
// row_ptr[i]..row_ptr[i+1] is the half-open range of entries for row i.
// col_idx and values are in row-major order; within each row, entries are
// column-sorted (guaranteed when built from a compressed COO).
//
// Strengths: sequential inner-loop reads for SpMV, row slicing in O(1),
//   direct compatibility with MKL, cuSPARSE, and most iterative solver libs.
// ============================================================

template <typename T = double>
struct SparseCSR {
    using value_type = T;

    std::size_t              nrows{}, ncols{};
    std::vector<std::size_t> row_ptr;  // size nrows+1
    std::vector<std::size_t> col_idx;  // size nnz
    std::vector<T>           values;   // size nnz

    std::size_t rows() const { return nrows; }
    std::size_t cols() const { return ncols; }
    std::size_t nnz()  const { return values.size(); }

    // y = A * x
    void spmv(const std::vector<T>& x, std::vector<T>& y) const {
        std::fill(y.begin(), y.end(), T{0});
        for (std::size_t i = 0; i < nrows; ++i)
            for (std::size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
                y[i] += values[k] * x[col_idx[k]];
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

/// COO → CSR.  Requires coo.is_compressed() (sorted, no zeros, no duplicates).
template <typename T>
SparseCSR<T> to_csr(const SparseCOO<T>& coo) {
    assert(coo.is_compressed() && "to_csr: call compress() first");
    SparseCSR<T> out;
    out.nrows   = coo.rows();
    out.ncols   = coo.cols();
    out.col_idx = coo.col_indices();
    out.values  = coo.values();

    out.row_ptr.assign(coo.rows() + 1, 0);
    for (auto r : coo.row_indices()) ++out.row_ptr[r + 1];
    for (std::size_t i = 0; i < coo.rows(); ++i)
        out.row_ptr[i + 1] += out.row_ptr[i];

    return out;
}

/// CSR → COO.  Output is already row-sorted and compressed.
template <typename T>
SparseCOO<T> to_coo(const SparseCSR<T>& csr) {
    SparseCOO<T> out(csr.nrows, csr.ncols, csr.nnz());
    for (std::size_t i = 0; i < csr.nrows; ++i)
        for (std::size_t k = csr.row_ptr[i]; k < csr.row_ptr[i + 1]; ++k)
            out.set(i, csr.col_idx[k], csr.values[k]);
    out.compress();
    return out;
}

} // namespace linalg
} // namespace datamunge
