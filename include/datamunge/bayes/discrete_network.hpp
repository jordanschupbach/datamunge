#pragma once

/// @file discrete_network.hpp
/// @brief Discrete Bayesian networks, exact queries and Chow–Liu tree fitting.

#include <datamunge/causal/dag.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <cstdint>

namespace datamunge::bayes {

/// @brief Assignments of discrete states to nodes as (node, state) pairs.
using DiscreteAssignments = std::vector<std::pair<std::size_t, std::size_t>>;

/// @brief A categorical Bayesian network with one conditional probability table per node.
/// @par Example
/// See fit_discrete_network() for a complete fit and query.
/// @see fit_discrete_network(), fit_chow_liu()
class DiscreteBayesianNetwork {
public:
    /// @brief Constructs a Bayesian network with known conditional probability tables.
    /// @param graph A DAG indexing the random variables.
    /// @param cardinalities Positive state counts in node order; states are zero-based.
    /// @param tables Flat tables in node order. Parent states are indexed in ascending parent
    /// node order with the last parent varying fastest, then the child state varies fastest.
    /// @throws std::invalid_argument Invalid dimensions or probabilities.
    /// @par Example
    /// @code
    /// const dm::bayes::DiscreteBayesianNetwork model(dm::causal::DAG(1), {2}, {{.3, .7}});
    /// @endcode
    DiscreteBayesianNetwork(const causal::DAG& graph, const std::vector<std::size_t>& cardinalities,
                            const std::vector<std::vector<double>>& tables);

    /// @brief Graph defining the network factorization.
    [[nodiscard]] const causal::DAG& graph() const noexcept { return graph_; }
    /// @brief State counts in node order.
    [[nodiscard]] const std::vector<std::size_t>& cardinalities() const noexcept { return cardinalities_; }
    /// @brief Normalized conditional probability tables in node order.
    [[nodiscard]] const std::vector<std::vector<double>>& tables() const noexcept { return tables_; }

    /// @brief Computes an exact marginal for a single node, optionally conditioned or intervened.
    /// @param query Node to summarize.
    /// @param evidence Observed node states, with each node listed at most once.
    /// @param intervention Fixed states that replace the assigned nodes' mechanisms. Causal
    /// interpretation requires that the graph and mechanisms represent the causal system.
    /// @param max_assignments Maximum number of joint assignments enumerated after ancestor pruning.
    /// @return Normalized probabilities in query state order.
    /// @throws std::invalid_argument Invalid state, node, or zero budget.
    /// @throws std::domain_error Zero-probability evidence or evidence contradicting an intervention.
    /// @throws std::length_error Exact inference budget exceeded.
    /// @par Example
    /// See fit_discrete_network() for an observational query. In that example,
    /// `model.marginal(1, {}, {{0, 1}})` fixes node 0 to state 1 instead.
    /// @see sample()
    [[nodiscard]] std::vector<double> marginal(std::size_t query, const DiscreteAssignments& evidence = {},
        const DiscreteAssignments& intervention = {}, std::size_t max_assignments = 1000000) const;

    /// @brief Draws states by ancestral sampling using a local seeded engine.
    /// @param n Number of cases.
    /// @param seed Local engine seed.
    /// @param intervention Nodes and their fixed states, replacing their mechanisms.
    /// @return Draws in rows, nodes in columns.
    /// @throws std::invalid_argument Invalid interventions.
    /// @par Example
    /// In the constructor example, `model.sample(100, 42)` draws 100 states.
    /// @see marginal()
    [[nodiscard]] linalg::DenseMatrix<std::size_t> sample(std::size_t n, std::uint64_t seed,
        const DiscreteAssignments& intervention = {}) const;

private:
    causal::DAG graph_; ///< Graph defining the parent order.
    std::vector<std::size_t> cardinalities_; ///< Positive state counts by node.
    std::vector<std::vector<double>> tables_; ///< Normalized probabilities by node.

    /// @brief Locates the start of a conditional table row.
    /// @param node Child variable.
    /// @param state Joint assignment.
    /// @return Offset for the supplied parent states.
    [[nodiscard]] std::size_t offset(std::size_t node, const std::vector<std::size_t>& state) const;
};

/// @brief Fits a network on a supplied graph by Dirichlet smoothing of complete categorical observations.
/// @param graph Known network structure.
/// @param cardinalities Positive state counts in node order.
/// @param data Complete discrete cases in rows and zero-based states in node columns.
/// @param alpha Symmetric Dirichlet prior concentration per child state, finite and positive.
/// @return Network of posterior mean probabilities, suitable for single-case prediction.
/// @throws std::invalid_argument Invalid cardinalities, dimensions, states or prior.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<std::size_t> x(3, 2, {0, 0, 1, 1, 1, 1});
/// const auto model = dm::bayes::fit_discrete_network(dm::causal::DAG(2, {{0, 1}}), {2, 2}, x);
/// const auto prediction = model.marginal(1, {{0, 1}});
/// @endcode
/// @par References
/// - https://faculty.ist.psu.edu/vhonavar/Courses/dsmethods/bnl-heckerman.pdf
/// @see DiscreteBayesianNetwork, fit_chow_liu()
[[nodiscard]] DiscreteBayesianNetwork fit_discrete_network(const causal::DAG& graph,
    const std::vector<std::size_t>& cardinalities, const linalg::DenseMatrix<std::size_t>& data,
    double alpha = 1.0);

/// @brief Learns a discrete tree by maximizing empirical pairwise mutual information and fits its tables.
/// @param data One or more complete cases in rows and zero-based states in columns.
/// @param cardinalities Positive state counts in column order.
/// @param root Tree root used to orient the factorization. Edge directions depend on this choice.
/// @param alpha Symmetric Dirichlet prior concentration per child state for the fitted tables.
/// @return Discrete network with a spanning tree structure.
/// @throws std::invalid_argument Invalid data, root, cardinalities or prior.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<std::size_t> x(3, 2, {0, 0, 1, 1, 1, 1});
/// const auto tree = dm::bayes::fit_chow_liu(x, {2, 2});
/// @endcode
/// @par References
/// - https://doi.org/10.1109/TIT.1968.1054142
/// @see fit_discrete_network(), causal::pc_stable()
[[nodiscard]] DiscreteBayesianNetwork fit_chow_liu(const linalg::DenseMatrix<std::size_t>& data,
    const std::vector<std::size_t>& cardinalities, std::size_t root = 0, double alpha = 1.0);

} // namespace datamunge::bayes
