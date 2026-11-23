#pragma once

/// @file linear_model_spaces.hpp
/// @brief Column bases and nesting for full and reduced linear models.

#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace datamunge::stats::detail {

/// @brief Extended-precision matrix for model projections.
using WorkingDesignMatrix = linalg::DenseMatrix<long double>;

/// @brief Builds an orthonormal column basis after scaling predictors to comparable magnitude.
inline WorkingDesignMatrix orthonormal_design(const linalg::DenseMatrix<double>& design, std::string_view method) {
    if (design.cols() == 0)
        return WorkingDesignMatrix(design.rows(), 0);
    if (design.rows() > std::vector<long double>{}.max_size() / design.cols())
        throw std::length_error(std::string(method) + ": dense design exceeds storage capacity");
    WorkingDesignMatrix scaled(design.rows(), design.cols());
    for (std::size_t j = 0; j < scaled.cols(); ++j) {
        long double peak = 0;
        for (std::size_t i = 0; i < scaled.rows(); ++i)
            peak = std::max(peak, std::abs(static_cast<long double>(design(i,j))));
        if (peak == 0)
            throw std::invalid_argument(std::string(method) + ": zero design column");
        for (std::size_t i = 0; i < scaled.rows(); ++i)
            scaled(i,j) = design(i,j) / peak;
    }
    auto factor = linalg::svd(scaled);
    if (!factor.converged)
        throw std::runtime_error(std::string(method) + ": design factorization failed");
    if (factor.rank(scaled.rows() * std::numeric_limits<double>::epsilon()) != scaled.cols())
        throw std::invalid_argument(std::string(method) + ": aliased design");
    return std::move(factor.U);
}

/// @brief Returns coordinates of the reduced model in the full basis and rejects non-nested models.
inline WorkingDesignMatrix nested_coordinates(const WorkingDesignMatrix& full, const WorkingDesignMatrix& reduced,
    std::string_view method) {
    auto coordinates = full.transpose() * reduced;
    if (reduced.cols()) {
        const auto remainder = reduced - full * coordinates;
        const auto tolerance = 64.0L * full.rows() * std::numeric_limits<double>::epsilon();
        long double squared_error = 0;
        for (const auto value : remainder.data())
            squared_error += value * value;
        if (squared_error > tolerance * tolerance * reduced.cols())
            throw std::invalid_argument(std::string(method) + ": reduced design must be nested in the full design");
    }
    return coordinates;
}

} // namespace datamunge::stats::detail
