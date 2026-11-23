#pragma once

/// @file clustering_kmeans_detail.hpp
/// @brief Translation and common-scale normalization for matrix k-means callers.

#include <datamunge/stats/kmeans.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

/// @brief Shared numerical helpers for k-means on observation matrices.
namespace datamunge::stats::clustering_detail {

/// @brief Uniform scale and normalized coordinates translated by the first row.
struct ScaledData {
    linalg::DenseMatrix<long double> values;       ///< Scaled points relative to the first observed point.
    long double scale{}; ///< Multiplier to restore original distances.
};

/// @brief Centers coordinates at the first point and bounds the largest coordinate difference.
inline ScaledData scale_clustering_data(const linalg::DenseMatrix<double>& data) {
    const auto n = data.rows(), d = data.cols();
    ScaledData result{linalg::DenseMatrix<long double>(n,d),0};
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j) {
            const auto delta = static_cast<long double>(data(i,j))-data(0,j);
            result.values(i,j) = delta;
            result.scale = std::max(result.scale,std::abs(delta));
        }
    if (!std::isfinite(result.scale)) {
        // On platforms with a double-width long double, an opposite-sign
        // coordinate difference can exceed the available exponent range.
        result.scale = std::abs(*std::ranges::max_element(data.data(),{},
            [](double x) { return std::abs(x); }));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < d; ++j)
                result.values(i,j) = data(i,j)/result.scale-data(0,j)/result.scale;
    } else if (result.scale > 0) {
        for (auto& coordinate : result.values.data())
            coordinate /= result.scale;
    }
    return result;
}

/// @brief Partitions normalized observations using the existing DataFrame k-means solver.
inline std::vector<std::size_t> matrix_kmeans(const ScaledData& scaled, std::size_t k,
    KMeansOptions controls) {
    dstruct::DataFrame frame;
    std::vector<std::string> columns;
    for (std::size_t j = 0; j < scaled.values.cols(); ++j) {
        columns.push_back("feature_"+std::to_string(j));
        std::vector<double> values(scaled.values.rows());
        for (std::size_t i = 0; i < scaled.values.rows(); ++i)
            values[i] = static_cast<double>(scaled.values(i,j));
        frame.add_column(columns.back(),std::move(values));
    }
    controls.n_clusters = k;
    if (scaled.scale > 0)
        controls.tolerance = static_cast<double>(std::min(
            static_cast<long double>(controls.tolerance)/scaled.scale,
            static_cast<long double>(std::numeric_limits<double>::max())));
    return KMeans(frame,columns,controls).labels();
}

} // namespace datamunge::stats::clustering_detail
