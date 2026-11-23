#pragma once

/// @file marginal_means.hpp
/// @brief Reference-grid averages and simultaneous linear-contrast intervals.

#include <datamunge/linalg/dense_matrix.hpp>

#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Averaging weights, contrast coefficients, and confidence level for a reference grid.
/// @par Example
/// @code
/// const dm::stats::MarginalMeansOptions options{.groups = {"A", "A", "B", "B"},
///                                              .weights = {1, 3, 1, 3}};
/// std::println("{}", options.weights.size()); // 4
/// @endcode
/// @see LM::estimated_marginal_means()
struct MarginalMeansOptions {
    std::vector<std::string> groups{};         ///< One label per grid row. Empty pools the whole grid.
    std::vector<double> weights{};             ///< Nonnegative weights normalized within groups. Empty uses equal weights.
    linalg::DenseMatrix<double> contrasts{};   ///< Rows of coefficients on the means, in first-appearance group order.
    double confidence_level{.95};             ///< Confidence level strictly between zero and one.
    bool simultaneous{true};                  ///< Bonferroni intervals within each family. False uses pointwise intervals.
};

/// @brief One estimated grid average or linear contrast with its uncertainty.
/// @par Example
/// @code
/// const dm::stats::MarginalEstimate average{8, .5, 7, 9};
/// std::println("{:.1f}", average.standard_error); // 0.5
/// @endcode
/// @see MarginalInference
struct MarginalEstimate {
    double estimate;       ///< Estimated linear function of the regression coefficients.
    double standard_error; ///< Square root of its model-based variance.
    double lower;          ///< Lower confidence limit.
    double upper;          ///< Upper confidence limit.
};

/// @brief Joint inference for a family of averages or contrasts.
/// @par Example
/// @code
/// const dm::stats::MarginalInference family{{{8, .5, 7, 9}}, dm::linalg::DenseMatrix<double>(1, 1, .25)};
/// std::println("{:.2f}", family.covariance(0, 0)); // 0.25
/// @endcode
/// @see MarginalMeansResult
struct MarginalInference {
    std::vector<MarginalEstimate> estimates; ///< One per group or contrast row.
    linalg::DenseMatrix<double> covariance;  ///< Joint covariance in the same order as estimates.
};

/// @brief Marginal means and their requested contrasts on the same reference grid.
/// @par Example
/// @code
/// dm::dstruct::DataFrame fit_data, grid;
/// fit_data.add_column("y", std::vector<double>{1, 2, 5, 6});
/// fit_data.add_column("x", std::vector<double>{0, 1, 2, 3});
/// grid.add_column("x", std::vector<double>{0, 3});
/// const auto result = dm::stats::LM(fit_data, "y ~ x").estimated_marginal_means(grid);
/// std::println("{:.1f}", result.means.estimates.front().estimate); // 3.5
/// @endcode
/// @see LM::estimated_marginal_means()
struct MarginalMeansResult {
    std::vector<std::string> groups; ///< Labels in first-appearance order, or one empty label for a pooled grid.
    MarginalInference means;        ///< Estimated averages, with covariance in group order.
    MarginalInference contrasts;    ///< Estimated contrasts, with covariance in contrast-matrix row order.
};

} // namespace datamunge::stats
