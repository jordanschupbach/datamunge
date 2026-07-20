#pragma once

#include <datamunge/linalg/dense_matrix.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::filter::detail {

/// @brief `vector<vector<double>>` -- a plain, always-binding-safe representation (per this
///        codebase's established rule that linalg::DenseMatrix itself can never be a
///        SWIG-facing parameter or return type) -- converted to a DenseMatrix for internal
///        computation. Throws std::invalid_argument for a ragged (non-rectangular) input.
[[nodiscard]] inline linalg::DenseMatrix<double> to_dense(const std::vector<std::vector<double>>& m) {
    if (m.empty()) {
        return {};
    }
    const std::size_t rows = m.size();
    const std::size_t cols = m.front().size();
    linalg::DenseMatrix<double> out(rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
        if (m[i].size() != cols) {
            throw std::invalid_argument("to_dense: matrix rows must all have the same length");
        }
        for (std::size_t j = 0; j < cols; ++j) out(i, j) = m[i][j];
    }
    return out;
}

[[nodiscard]] inline std::vector<std::vector<double>> to_vector2d(const linalg::DenseMatrix<double>& m) {
    std::vector<std::vector<double>> out(m.rows(), std::vector<double>(m.cols()));
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j) out[i][j] = m(i, j);
    return out;
}

[[nodiscard]] inline linalg::DenseMatrix<double> to_column(const std::vector<double>& v) {
    linalg::DenseMatrix<double> out(v.size(), 1);
    for (std::size_t i = 0; i < v.size(); ++i) out(i, 0) = v[i];
    return out;
}

/// @brief The single column (or, if @p m has exactly one row instead, that row) of @p m as a
///        plain vector. Throws std::invalid_argument if @p m is neither Nx1 nor 1xN.
[[nodiscard]] inline std::vector<double> to_vector(const linalg::DenseMatrix<double>& m) {
    if (m.cols() == 1) {
        return m.col(0);
    }
    if (m.rows() == 1) {
        return m.row(0);
    }
    throw std::invalid_argument("to_vector: matrix is not a row or column vector");
}

} // namespace datamunge::filter::detail
