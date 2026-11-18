/// @file dag.cpp
/// @brief Validated directed graphs and graphical identification checks.
#include <datamunge/causal/dag.hpp>

#include "detail.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <stdexcept>

namespace datamunge::causal {

void detail::finite_matrix(const linalg::DenseMatrix<double>& matrix) {
    for (std::size_t i = 0; i < matrix.rows(); ++i)
        for (std::size_t j = 0; j < matrix.cols(); ++j)
            if (!std::isfinite(matrix(i, j))) throw std::invalid_argument("causal model: nonfinite matrix entry");
}

/// @brief Private traversal helpers.
namespace {
/// @brief Validates d-separation endpoints and builds a mask of conditioned nodes.
/// @param n Graph size.
/// @param x First endpoint.
/// @param y Distinct second endpoint.
/// @param given Conditioning indices excluding endpoints.
/// @return Mask of conditioned nodes.
/// @throws std::invalid_argument For invalid or repeated indices.
std::vector<bool> conditioning(std::size_t n, std::size_t x, std::size_t y,
                               const std::vector<std::size_t>& given) {
    if (x >= n || y >= n || x == y) throw std::invalid_argument("DAG: invalid distinct endpoints");
    std::vector<bool> mask(n);
    for (const auto i : given) {
        if (i >= n || i == x || i == y || mask[i]) throw std::invalid_argument("DAG: invalid conditioning set");
        mask[i] = true;
    }
    return mask;
}

/// @brief Marks all strict descendants of a node.
/// @param graph Valid DAG.
/// @param node Valid origin.
/// @return Descendant mask excluding the origin.
std::vector<bool> descendants(const DAG& graph, std::size_t node) {
    std::vector<bool> reached(graph.size());
    for (const auto i : graph.order())
        for (const auto p : graph.parents(i))
            if (p == node || reached[p]) reached[i] = true;
    return reached;
}

/// @brief Tests connectivity in the unconditioned ancestral moral graph.
/// @param graph Valid directed graph.
/// @param x First endpoint.
/// @param y Distinct second endpoint.
/// @param given Mask of conditioned nodes.
/// @param remove_outgoing Whether to delete all arrows from x before testing.
/// @return True when no moralized path survives removal of conditioned nodes.
bool separated(const DAG& graph, std::size_t x, std::size_t y, const std::vector<bool>& given, bool remove_outgoing) {
    auto needed = given;
    needed[x] = needed[y] = true;
    for (auto it = graph.order().rbegin(); it != graph.order().rend(); ++it)
        if (needed[*it])
            for (const auto p : graph.parents(*it))
                if (!remove_outgoing || p != x) needed[p] = true;

    std::vector<std::vector<std::size_t>> moral(graph.size());
    /// @brief Connects two unconditioned nodes in both directions.
    const auto link = [&](std::size_t a, std::size_t b) {
        if (given[a] || given[b]) return;
        moral[a].push_back(b);
        moral[b].push_back(a);
    };
    for (std::size_t i = 0; i < graph.size(); ++i) {
        if (!needed[i]) continue;
        const auto& parents = graph.parents(i);
        for (std::size_t j = 0; j < parents.size(); ++j) {
            if (remove_outgoing && parents[j] == x) continue;
            link(i, parents[j]);
            for (std::size_t k = 0; k < j; ++k)
                if (!remove_outgoing || parents[k] != x) link(parents[j], parents[k]);
        }
    }
    std::vector<bool> seen(graph.size());
    std::vector<std::size_t> pending{x};
    seen[x] = true;
    while (!pending.empty()) {
        const auto i = pending.back();
        pending.pop_back();
        if (i == y) return false;
        for (const auto neighbor : moral[i])
            if (!seen[neighbor]) {
                seen[neighbor] = true;
                pending.push_back(neighbor);
            }
    }
    return true;
}
} // namespace

DAG::DAG(std::size_t nodes, const std::vector<std::pair<std::size_t, std::size_t>>& edges) : parents_(nodes) {
    if (nodes == 0) throw std::invalid_argument("DAG: need at least one node");
    std::vector<std::vector<std::size_t>> children(nodes);
    for (const auto& [from, to] : edges) {
        if (from >= nodes || to >= nodes || from == to) throw std::invalid_argument("DAG: invalid edge");
        parents_[to].push_back(from);
        children[from].push_back(to);
    }
    std::vector<std::size_t> degree(nodes);
    std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<>> ready;
    for (std::size_t i = 0; i < nodes; ++i) {
        auto& parents = parents_[i];
        std::ranges::sort(parents);
        if (std::adjacent_find(parents.begin(), parents.end()) != parents.end())
            throw std::invalid_argument("DAG: duplicate edge");
        degree[i] = parents.size();
        if (degree[i] == 0) ready.push(i);
    }
    while (!ready.empty()) {
        const auto i = ready.top();
        ready.pop();
        order_.push_back(i);
        for (const auto child : children[i])
            if (--degree[child] == 0) ready.push(child);
    }
    if (order_.size() != nodes) throw std::invalid_argument("DAG: directed cycle");
}

bool DAG::d_separated(std::size_t x, std::size_t y, const std::vector<std::size_t>& given) const {
    return separated(*this, x, y, conditioning(size(), x, y, given), false);
}

bool DAG::is_backdoor_adjustment(std::size_t exposure, std::size_t outcome,
                                 const std::vector<std::size_t>& given) const {
    const auto mask = conditioning(size(), exposure, outcome, given);
    const auto downstream = descendants(*this, exposure);
    for (const auto i : given)
        if (downstream[i]) return false;
    return separated(*this, exposure, outcome, mask, true);
}

std::optional<std::vector<std::size_t>> DAG::backdoor_adjustment(
    std::size_t exposure, std::size_t outcome, std::size_t max_size, std::size_t max_sets) const {
    (void)conditioning(size(), exposure, outcome, {});
    if (max_sets == 0) throw std::invalid_argument("DAG: adjustment search budget is zero");
    const auto downstream = descendants(*this, exposure);
    std::vector<std::size_t> candidates;
    for (std::size_t i = 0; i < size(); ++i)
        if (i != exposure && i != outcome && !downstream[i]) candidates.push_back(i);
    std::optional<std::vector<std::size_t>> result;
    for (std::size_t k = 0; k <= std::min(max_size, candidates.size()); ++k)
        if (detail::subsets(candidates, k, [&](const auto& set) {
            if (max_sets == 0) throw std::length_error("DAG: adjustment search budget exhausted");
            --max_sets;
            if (!separated(*this, exposure, outcome, conditioning(size(), exposure, outcome, set), true))
                return false;
            result = set;
            return true;
        })) break;
    return result;
}

graphics::Diagram DAG::diagram(const std::vector<graphics::DiagramPoint>& positions,
                                const std::vector<std::string>& labels) const {
    if (positions.size() != size() || (!labels.empty() && labels.size() != size()))
        throw std::invalid_argument("DAG diagram: node count mismatch");
    graphics::Diagram result;
    for (std::size_t i = 0; i < size(); ++i) {
        const auto id = std::to_string(i);
        result.node(id, positions[i], labels.empty() ? id : labels[i]);
    }
    for (std::size_t i = 0; i < size(); ++i)
        for (const auto p : parents_[i]) result.edge(std::to_string(p), std::to_string(i));
    return result;
}

} // namespace datamunge::causal
