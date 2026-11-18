#pragma once

/// @file sem.hpp
/// @brief Observed linear Gaussian structural equation models.

#include <datamunge/causal/dag.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <cstdint>

namespace datamunge::causal {

/// @brief Fixed node values under a joint intervention, expressed as (node, value) pairs.
using Intervention = std::vector<std::pair<std::size_t, double>>;

/// @brief Means and covariances implied by a linear structural equation model.
struct SEMMoments {
    std::vector<double> mean; ///< Means in node order.
    linalg::DenseMatrix<double> covariance; ///< Covariance in node order.
};

/// @brief Recursive linear Gaussian structural model with independent disturbances.
/// @par Example
/// See fit_linear_sem() for a complete fit.
/// @see fit_linear_sem(), DAG
class LinearSEM {
public:
    /// @brief The fitted model's graph.
    [[nodiscard]] const DAG& graph() const noexcept { return graph_; }
    /// @brief Path coefficients indexed by (child, parent), with zero for absent edges.
    [[nodiscard]] const linalg::DenseMatrix<double>& coefficients() const noexcept { return coefficients_; }
    /// @brief Fitted intercepts in node order.
    [[nodiscard]] const std::vector<double>& intercepts() const noexcept { return intercepts_; }
    /// @brief Residual variance estimates (RSS divided by sample size) in node order.
    [[nodiscard]] const std::vector<double>& residual_variances() const noexcept { return variances_; }

    /// @brief Implied population moments with optional fixed-node interventions.
    /// @param fixed Distinct nodes and finite values replacing their structural equations.
    /// @return Mean and covariance in node order.
    /// @throws std::invalid_argument Invalid interventions.
    /// @par Example
    /// See fit_linear_sem() for an interventional moment calculation.
    /// @see sample()
    [[nodiscard]] SEMMoments moments(const Intervention& fixed = {}) const;

    /// @brief Draws observations or interventional outcomes from the fitted model.
    /// @param n Number of draws.
    /// @param seed Local random engine seed.
    /// @param fixed Distinct nodes and finite intervention values.
    /// @return Draws in rows and nodes in columns.
    /// @throws std::invalid_argument Invalid interventions.
    /// @par Example
    /// After fitting, `model.sample(100, 42, {{0, 2.0}})` draws 100 cases with node 0 set to 2.
    /// @see moments()
    [[nodiscard]] linalg::DenseMatrix<double> sample(std::size_t n, std::uint64_t seed,
                                                     const Intervention& fixed = {}) const;

    /// @brief Sums products of path coefficients over directed paths from exposure to outcome.
    /// @param exposure Intervention node.
    /// @param outcome Distinct response node.
    /// @return Change in response mean for a one-unit increase in the intervention.
    /// @throws std::invalid_argument Invalid or identical endpoints.
    /// @par Example
    /// After fitting, `model.total_effect(0, 1)` summarizes the effect of node 0 on node 1.
    /// @see moments()
    [[nodiscard]] double total_effect(std::size_t exposure, std::size_t outcome) const;

    /// @brief Reuses the disturbances inferred from a fully observed unit under a new intervention.
    /// @param observed Finite factual observation in node order.
    /// @param fixed Distinct nodes and finite counterfactual intervention values.
    /// @return Counterfactual values in node order.
    /// @throws std::invalid_argument Invalid observation or interventions.
    /// @par Example
    /// After fitting a two-node model, `model.counterfactual({1.0, 3.0}, {{0, 2.0}})`
    /// predicts the same unit under a new value of node 0.
    /// @see moments()
    [[nodiscard]] std::vector<double> counterfactual(const std::vector<double>& observed,
                                                      const Intervention& fixed) const;

private:
    friend LinearSEM fit_linear_sem(const DAG&, const linalg::DenseMatrix<double>&);
    DAG graph_; ///< Structural graph.
    linalg::DenseMatrix<double> coefficients_; ///< Fitted (child, parent) slopes.
    std::vector<double> intercepts_; ///< Fitted location parameters.
    std::vector<double> variances_; ///< Fitted disturbance variances.

    /// @brief Allocates coefficient and disturbance storage for the supplied graph.
    /// @param graph Structural graph.
    explicit LinearSEM(const DAG& graph);
};

/// @brief Fits intercept and parent slopes for each observed DAG node by least squares.
/// @param graph Assumed causal DAG with independent structural errors.
/// @param data Finite complete observations in rows and nodes in columns.
/// @return A fitted model for observational, interventional and counterfactual calculations.
/// @throws std::invalid_argument Invalid data dimensions, nonfinite data or insufficient rows.
/// @throws std::runtime_error Rank-deficient parent regressions.
/// @par Example
/// @code
/// const dm::causal::DAG g(2, {{0, 1}});
/// const dm::linalg::DenseMatrix<double> x(4, 2, {0, 1, 1, 4, 2, 4, 3, 7});
/// const auto model = dm::causal::fit_linear_sem(g, x);
/// const auto changed = model.moments({{0, 2.0}});
/// @endcode
/// @par References
/// - https://escholarship.org/uc/item/36w8n7pg
/// @see LinearSEM, DAG
[[nodiscard]] LinearSEM fit_linear_sem(const DAG& graph, const linalg::DenseMatrix<double>& data);

} // namespace datamunge::causal
