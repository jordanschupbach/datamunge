#pragma once

/// @file dag.hpp
/// @brief Directed acyclic graphs, d-separation and back-door adjustment checks.

#include <datamunge/plot/diagram.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/// @brief Causal graphs, effect estimation, structural models and discovery.
namespace datamunge::causal {

/// @brief An owned directed acyclic graph indexed by node numbers from zero.
/// @par Example
/// @code
/// const dm::causal::DAG graph(3, {{0, 1}, {0, 2}, {1, 2}});
/// @endcode
/// @see LinearSEM
class DAG {
public:
    /// @brief Constructs and validates a DAG and computes a deterministic topological order.
    /// @param nodes Positive node count.
    /// @param edges Directed (parent, child) pairs. Defaults to an edgeless graph.
    /// @throws std::invalid_argument For zero nodes, invalid indices, duplicate edges or cycles.
    /// @par Example
    /// See the DAG class example.
    DAG(std::size_t nodes, const std::vector<std::pair<std::size_t, std::size_t>>& edges = {});

    /// @brief Number of indexed nodes.
    /// @return Positive node count.
    [[nodiscard]] std::size_t size() const noexcept { return parents_.size(); }

    /// @brief Returns the parents in ascending node order.
    /// @param node Node index.
    /// @return Borrowed sorted parent indices.
    /// @throws std::out_of_range Invalid node.
    [[nodiscard]] const std::vector<std::size_t>& parents(std::size_t node) const { return parents_.at(node); }

    /// @brief Returns the cached topological order.
    /// @return Borrowed order with each parent before its children.
    [[nodiscard]] const std::vector<std::size_t>& order() const noexcept { return order_; }

    /// @brief Checks graphical conditional independence by ancestral moralization.
    /// @param x First endpoint index.
    /// @param y Distinct second endpoint index.
    /// @param given Distinct conditioning nodes, excluding x and y.
    /// @return True when all paths are blocked given the supplied set.
    /// @throws std::invalid_argument Invalid indices, repeated nodes or endpoint conditioning.
    /// @par Example
    /// @code
    /// const dm::causal::DAG g(3, {{0, 2}, {1, 2}});
    /// std::print("{}", g.d_separated(0, 1)); // true
    /// @endcode
    /// @par References
    /// - https://proceedings.mlr.press/r0/pearl95a.html
    /// @see is_backdoor_adjustment()
    [[nodiscard]] bool d_separated(std::size_t x, std::size_t y,
                                   const std::vector<std::size_t>& given = {}) const;

    /// @brief Checks Pearl’s back-door criterion for one exposure and one outcome.
    /// @param exposure Treatment node.
    /// @param outcome Distinct outcome node.
    /// @param given Candidate adjustment set of distinct non-endpoint nodes.
    /// @return True when the set excludes exposure descendants and blocks all back-door paths.
    /// @throws std::invalid_argument Invalid endpoints or adjustment indices.
    /// @par Example
    /// @code
    /// const dm::causal::DAG g(3, {{0, 1}, {0, 2}, {1, 2}});
    /// std::print("{}", g.is_backdoor_adjustment(1, 2, {0})); // true
    /// @endcode
    /// @par References
    /// - https://doi.org/10.1093/biomet/82.4.669
    /// @see d_separated()
    [[nodiscard]] bool is_backdoor_adjustment(std::size_t exposure, std::size_t outcome,
                                              const std::vector<std::size_t>& given) const;

    /// @brief Finds a minimum-cardinality back-door adjustment set by bounded search.
    /// @param exposure Treatment node.
    /// @param outcome Distinct outcome node.
    /// @param max_size Maximum number of adjustment nodes to try, default four.
    /// @param max_sets Maximum number of candidate sets evaluated, default 10000.
    /// @return First minimum-cardinality set in index order, or nullopt if none within max_size.
    /// @throws std::invalid_argument Invalid endpoints or a zero search budget.
    /// @throws std::length_error If the search budget is exhausted before a result is known.
    /// @par Example
    /// @code
    /// const dm::causal::DAG g(3, {{0, 1}, {0, 2}, {1, 2}});
    /// std::print("{}", g.backdoor_adjustment(1, 2)->front()); // 0
    /// @endcode
    /// @par References
    /// - https://doi.org/10.1093/biomet/82.4.669
    /// @see is_backdoor_adjustment()
    [[nodiscard]] std::optional<std::vector<std::size_t>> backdoor_adjustment(
        std::size_t exposure, std::size_t outcome, std::size_t max_size = 4, std::size_t max_sets = 10000) const;

    /// @brief Draws the graph with caller-supplied positions and optional labels.
    /// @param positions One diagram coordinate per node.
    /// @param labels One label per node or empty for numeric labels.
    /// @return Native diagram with parent-to-child arrows.
    /// @throws std::invalid_argument Invalid counts, positions or overlapping connected nodes.
    /// @par Example
    /// @code
    /// const dm::causal::DAG g(2, {{0, 1}});
    /// auto diagram = g.diagram({{0, 0}, {4, 0}}, {"Exposure", "Outcome"});
    /// @endcode
    /// @see graphics::Diagram
    [[nodiscard]] graphics::Diagram diagram(const std::vector<graphics::DiagramPoint>& positions,
                                              const std::vector<std::string>& labels = {}) const;

private:
    std::vector<std::vector<std::size_t>> parents_; ///< Sorted incoming lists.
    std::vector<std::size_t> order_; ///< Cached topological traversal.
};

} // namespace datamunge::causal
