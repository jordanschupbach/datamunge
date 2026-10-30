#pragma once

/// @file grid_posterior.hpp
/// @brief Deterministic posterior summaries of low-dimensional models by tensor-grid quadrature,
///        as ground truth for MCMC and importance sampling.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::bayes {

/// Posterior summaries computed on a grid.
struct GridPosterior {
    /// @brief \f$\log\int p(y\mid\theta)p(\theta)\,d\theta\f$ over the box (the log evidence when
    ///        the log density includes all normalising constants).
    double log_evidence{0.0};
    /// @brief Posterior mean of each coordinate.
    std::vector<double> mean;
    /// @brief Posterior standard deviation of each coordinate.
    std::vector<double> sd;
    /// @brief Posterior covariance matrix.
    std::vector<std::vector<double>> covariance;
    /// @brief Cell centres along each coordinate.
    std::vector<std::vector<double>> marginal_grid;
    /// @brief Marginal posterior density of each coordinate at its cell centres.
    std::vector<std::vector<double>> marginal_density;
    /// @brief Posterior mass in the outermost layer of cells; it should be negligible, otherwise
    ///        the box cuts off part of the posterior.
    double edge_mass{0.0};
};

/// @brief Integrates \f$\exp(\text{log\_density})\f$ over the box \f$\prod_j[\ell_j,u_j]\f$ with
///        the midpoint rule on \f$n^d\f$ equal cells.
///
/// With cell volume \f$\Delta\f$ and centres \f$\theta_c\f$, \f$Z\approx\Delta\sum_c e^{f(\theta_c)}\f$
/// and posterior moments are the weighted averages with weights \f$e^{f(\theta_c)}/\sum e^f\f$.
/// The midpoint rule's error is \f$O(h^2)\f$ per coordinate for smooth integrands. The cost is
/// \f$n^d\f$ evaluations, so this is practical for up to about four parameters.
/// @param log_density Unnormalised log posterior; only evaluate() is used.
/// @param lower Lower corner of the box.
/// @param upper Upper corner (each entry greater than the matching @p lower).
/// @param points Cells per coordinate (at least 3).
/// @return Evidence, moments, marginals and edge mass.
/// @throws std::invalid_argument for mismatched or empty bounds, an empty box, too few points,
///         or more than 5e7 cells.
GridPosterior grid_posterior(optim::ArbitraryFunction& log_density, const std::vector<double>& lower,
                             const std::vector<double>& upper, std::size_t points);

} // namespace datamunge::bayes
