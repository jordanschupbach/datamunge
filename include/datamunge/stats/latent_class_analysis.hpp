#pragma once

/// @file latent_class_analysis.hpp
/// @brief Finite mixtures of conditionally independent categorical measurements.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Maximum-likelihood latent class fitting controls.
/// @par Example
/// @code
/// const dm::stats::LatentClassOptions options{.classes=3, .seed=11};
/// std::println("{}", options.classes); // 3
/// @endcode
struct LatentClassOptions {
    std::size_t classes{2};       ///< Number of latent classes, at most the number of informative rows.
    std::size_t starts{16};       ///< Number of independently randomized starting models.
    std::size_t max_iterations{1000}; ///< Maximum EM updates in each start.
    double relative_tolerance{1e-10}; ///< Stop when improvement is at most this fraction of one plus the previous absolute log likelihood.
    std::uint64_t seed{1};        ///< Seed for local parameter initialization.
};

/// @brief Class proportions, item distributions and posterior class probabilities.
/// @par Example
/// @code
/// dm::stats::LatentClassFit fit;
/// std::println("{}", fit.converged); // false
/// @endcode
struct LatentClassFit {
    std::vector<double> class_probabilities; ///< Mixing proportions in decreasing order.
    std::vector<linalg::DenseMatrix<double>> item_probabilities; ///< One matrix per item, with classes by response levels.
    linalg::DenseMatrix<double> posterior;   ///< Input rows by classes, in the same class order.
    double log_likelihood{};                 ///< Observed-data log likelihood of the returned fit.
    std::size_t iterations{};                ///< Completed EM updates of the selected start.
    bool converged{};                        ///< True if the selected start met the likelihood tolerance.
};

/// @brief Fits a latent class model by multistart expectation maximization.
/// A component assigns independent categorical probabilities to the items. A
/// missing entry is marginalized. For a fully missing row the posterior is
/// the fitted mixing proportions. The fit uses all rows with at least one
/// observed item and returns the start with the largest log likelihood.
/// @param responses Observations by items. Categories are zero based and -1 denotes missing.
/// @param levels Number of declared response levels for each item, at least two.
/// @param options Number of classes and optimization controls.
/// @return Maximum-likelihood estimate from the best attempted start and its convergence status.
/// @throws std::invalid_argument For invalid categories, dimensions, controls or an entirely missing item.
/// @throws std::length_error If the model or posterior does not fit in dense storage.
/// @throws std::runtime_error If no attempted fit has a finite likelihood.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<std::int32_t> x(4,2,{0,0,0,1,1,1,0,1});
/// const auto fit = dm::stats::fit_latent_class_analysis(x,{2,2},{.classes=1});
/// std::println("{:.2f}", fit.item_probabilities[0](0,0)); // 0.75
/// @endcode
/// @par Reference
/// Linzer and Lewis (2011), https://doi.org/10.18637/jss.v042.i10
[[nodiscard]] LatentClassFit fit_latent_class_analysis(
    const linalg::DenseMatrix<std::int32_t>& responses, const std::vector<std::size_t>& levels,
    const LatentClassOptions& options = {});

} // namespace datamunge::stats
