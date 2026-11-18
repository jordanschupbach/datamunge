#pragma once

/// @file discovery.hpp
/// @brief Gaussian conditional independence tests and PC-stable discovery.

#include <datamunge/causal/dag.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

namespace datamunge::causal {

/// @brief A Gaussian conditional independence test using Fisher's transform.
struct GaussianCITest {
    double correlation{}; ///< Sample partial correlation.
    double p_value{}; ///< Two-sided asymptotic p-value for zero partial correlation.
};

/// @brief Tests conditional independence between two Gaussian variables.
/// @param data Independent complete observations in rows and variables in columns.
/// @param x First variable column.
/// @param y Distinct second variable column.
/// @param given Distinct conditioning columns excluding the endpoints.
/// @return Partial correlation and two-sided Fisher-Z p-value.
/// @throws std::invalid_argument Invalid columns, nonfinite data, constant columns or insufficient rows.
/// @throws std::runtime_error Singular conditioning correlation matrix.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> data(4, 2, {0, 0, 1, 0, 0, 1, 1, 1});
/// const auto test = dm::causal::gaussian_ci_test(data, 0, 1);
/// @endcode
/// @par References
/// - https://jmlr.org/papers/v8/kalisch07a.html
/// @see pc_stable()
[[nodiscard]] GaussianCITest gaussian_ci_test(const linalg::DenseMatrix<double>& data,
    std::size_t x, std::size_t y, const std::vector<std::size_t>& given = {});

/// @brief Search limits for Gaussian PC-stable discovery.
struct PCOptions {
    double alpha{0.01}; ///< Significance threshold used for edge removal.
    std::size_t max_conditioning{3}; ///< Maximum size of a conditioning set.
    std::size_t max_tests{100000}; ///< Maximum number of independence tests.
};

/// @brief Partially directed graph returned by PC-stable.
struct PCResult {
    std::size_t nodes{}; ///< Number of tested variables.
    std::vector<std::pair<std::size_t, std::size_t>> directed; ///< Parent-to-child arrows.
    std::vector<std::pair<std::size_t, std::size_t>> undirected; ///< Remaining edges, smaller endpoint first.
    std::size_t tests{}; ///< Number of conditional independence tests.
    std::size_t orientation_conflicts{}; ///< Number of edges with competing arrow proposals.

    /// @brief Draws estimated arrows and undirected edges at supplied node coordinates.
    /// @param positions Diagram coordinates in variable order.
    /// @param labels Optional labels in variable order.
    /// @return Native diagram.
    /// @throws std::invalid_argument Mismatched positions or labels.
    /// @par Example
    /// After fitting a two-node graph, `learned.diagram({{0, 0}, {5, 0}}, {"A", "B"})`
    /// draws both nodes and any learned edge.
    /// @see DAG::diagram()
    [[nodiscard]] graphics::Diagram diagram(const std::vector<graphics::DiagramPoint>& positions,
                                             const std::vector<std::string>& labels = {}) const;
};

/// @brief Learns a Gaussian PC-stable skeleton, orients colliders and propagates arrow constraints.
/// @param data Independent complete observations in rows and variables in columns.
/// @param options Significance threshold and search limits.
/// @return Partially directed graph and search diagnostics. Interpret under acyclicity,
/// causal sufficiency, faithfulness and Gaussian conditional independence.
/// @throws std::invalid_argument Invalid inputs, constant columns or fewer than four rows.
/// @throws std::runtime_error Singular conditioning correlation matrix.
/// @throws std::length_error Search exceeds its test budget.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4, 2, {0, 0, 1, 0, 0, 1, 1, 1});
/// const auto learned = dm::causal::pc_stable(x);
/// @endcode
/// @par References
/// - https://jmlr.org/papers/v8/kalisch07a.html
/// - https://jmlr.org/papers/v15/colombo14a.html
/// @see gaussian_ci_test(), DAG
[[nodiscard]] PCResult pc_stable(const linalg::DenseMatrix<double>& data, PCOptions options = {});

} // namespace datamunge::causal
