#pragma once

/// @file graphical_lasso.hpp
/// @brief Sparse Gaussian precision estimation and conditional-dependence edges.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Numerical controls for the positive-definite graphical lasso solver.
/// @par Example
/// @code
/// const dm::stats::GraphicalLassoOptions options{.max_iterations=100};
/// std::println("{}", options.max_iterations); // 100
/// @endcode
struct GraphicalLassoOptions {
    std::size_t max_iterations{2000}; ///< Positive iteration limit per connected component.
    double tolerance{1e-8};           ///< Positive standardized optimality tolerance below one.
};

/// @brief A nonzero entry of the estimated off-diagonal precision.
/// @par Example
/// @code
/// const dm::stats::GraphicalLassoEdge edge{0,1,.4};
/// std::println("{:.1f}", edge.partial_correlation); // 0.4
/// @endcode
struct GraphicalLassoEdge {
    std::size_t first;          ///< Smaller original column index, numbered from zero.
    std::size_t second;         ///< Larger original column index.
    double partial_correlation; ///< Signed negative precision entry normalized by the two precision diagonals.
};

/// @brief Sparse precision, its covariance inverse, and convergence diagnostics.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> s(2,2,{1,.6,.6,1});
/// const auto fit = dm::stats::graphical_lasso(s,.2);
/// std::println("{:.1f}", fit.edges.front().partial_correlation); // 0.4
/// @endcode
struct GraphicalLassoResult {
    linalg::DenseMatrix<double> precision;  ///< Sparse penalized inverse covariance in input-variable order.
    linalg::DenseMatrix<double> covariance; ///< Inverse of the returned precision.
    std::vector<GraphicalLassoEdge> edges;   ///< Nonzero off-diagonal entries, sorted by endpoint indices.
    std::size_t iterations;                 ///< Largest iteration count across the connected components.
    double objective;                       ///< Trace minus log determinant plus the off-diagonal penalty in original units.
    double max_kkt_violation;                ///< Largest optimality violation after standardizing covariance diagonals to one.
    bool converged;                          ///< The returned estimates met the optimality tolerance in every component.
};

/// @brief Fits a graphical lasso precision directly from an empirical covariance.
///
/// Minimizes trace(S times precision) minus log determinant of precision
/// plus the penalty times the sum of absolute off-diagonal entries, counting
/// both symmetric entries. Diagonals are unpenalized. The solver splits
/// independent components by exact covariance thresholding, then solves
/// each penalized component by alternating direction multipliers.
/// An iteration-limited fit returns a feasible sparse estimate and its
/// measured optimality violation.
/// @param empirical_covariance Finite symmetric positive-semidefinite covariance with positive diagonal.
/// @param penalty Nonnegative penalty in the units of the supplied covariance.
/// @param options Per-component iteration limit and the standardized optimality tolerance.
/// @return Precision, covariance, conditional-dependence edges and solver diagnostics.
/// @throws std::invalid_argument Incorrect shape, nonfinite or asymmetric input, or invalid controls.
/// @throws std::domain_error The covariance is materially indefinite, a variable has zero variance, or zero penalty requires an invertible covariance.
/// @throws std::length_error The dense workspace exceeds storage capacity.
/// @throws std::runtime_error A symmetric eigensolver failed to converge.
/// @throws std::overflow_error A fitted matrix entry exceeds the reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> s(2,2,{1,.6,.6,1});
/// std::println("{:.1f}", dm::stats::graphical_lasso(s,.2).covariance(0,1)); // 0.4
/// @endcode
/// @par References
/// - Friedman, Hastie and Tibshirani (2008), https://doi.org/10.1093/biostatistics/kxm045
/// - Mazumder and Hastie (2012), https://www.jmlr.org/papers/v13/mazumder12a.html
[[nodiscard]] GraphicalLassoResult graphical_lasso(const linalg::DenseMatrix<double>& empirical_covariance,
    double penalty, const GraphicalLassoOptions& options = {});

/// @brief Fits graphical lasso to observations after centering columns and dividing scatter by the row count.
/// @param observations Finite data with at least two rows and positive variance in each column.
/// @param penalty Nonnegative penalty in covariance units.
/// @param options Per-component iteration limit and standardized optimality tolerance.
/// @return The fitted graphical lasso and its sparse conditional-dependence graph.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,2,{1,1,1,-1,-1,1,-1,-1});
/// std::println("{}", dm::stats::fit_graphical_lasso(x,.2).edges.size()); // 0
/// @endcode
[[nodiscard]] GraphicalLassoResult fit_graphical_lasso(const linalg::DenseMatrix<double>& observations,
    double penalty, const GraphicalLassoOptions& options = {});

} // namespace datamunge::stats
