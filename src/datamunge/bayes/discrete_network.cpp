/// @file discrete_network.cpp
/// @brief Categorical Bayesian networks and bounded exact inference.
#include <datamunge/bayes/discrete_network.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::bayes {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief State counts or joint categorical assignments.
using Counts = std::vector<std::size_t>;
/// @brief Complete cases in rows with node states in columns.
using Cases = linalg::DenseMatrix<std::size_t>;

/// @brief Multiplies cardinalities with a bound on addressable elements.
std::size_t product(std::size_t a, std::size_t b, std::size_t limit = std::vector<double>().max_size()) {
    if (b != 0 && a > limit / b) throw std::length_error("Discrete network size or inference budget exceeded");
    return a * b;
}

/// @brief Checks the number of states supplied for every column.
void check_cardinalities(std::size_t nodes, const Counts& counts) {
    if (counts.size() != nodes || std::ranges::find(counts, 0U) != counts.end())
        throw std::invalid_argument("Discrete network needs one positive cardinality per node");
}

/// @brief Validates complete categorical observations.
void check_cases(const Cases& data, const Counts& counts) {
    check_cardinalities(data.cols(), counts);
    for (std::size_t r = 0; r < data.rows(); ++r)
        for (std::size_t j = 0; j < counts.size(); ++j)
            if (data(r, j) >= counts[j]) throw std::invalid_argument("Discrete state exceeds its cardinality");
}

/// @brief Returns the number of child probabilities across all parent configurations.
std::size_t table_size(const causal::DAG& graph, const Counts& counts, std::size_t node) {
    auto size = counts[node];
    for (const auto parent : graph.parents(node)) size = product(size, counts[parent]);
    return size;
}

/// @brief Converts sparse assignments to optional states indexed by node.
auto assigned(const Counts& cardinalities, const DiscreteAssignments& values) {
    std::vector<std::optional<std::size_t>> result(cardinalities.size());
    for (const auto& [node, state] : values) {
        if (node >= cardinalities.size() || result[node] || state >= cardinalities[node])
            throw std::invalid_argument("Discrete assignments need distinct nodes and valid states");
        result[node] = state;
    }
    return result;
}

/// @brief Returns the offset of a child probability row by mixed-radix indexing.
template<class State>
std::size_t row_offset(const causal::DAG& graph, const Counts& counts, std::size_t node, State state) {
    std::size_t offset = 0;
    for (const auto parent : graph.parents(node)) offset = offset * counts[parent] + state(parent);
    return offset * counts[node];
}

/// @brief Adds masses represented as logarithms without underflow.
double log_add(double a, double b) {
    const auto largest = std::max(a, b);
    if (!std::isfinite(largest)) return largest;
    return largest + std::log1p(std::exp(std::min(a, b) - largest));
}

/// @brief Computes unsmoothed pairwise mutual information from observed counts.
double mutual_information(const Cases& data, const Counts& counts, std::size_t a, std::size_t b) {
    Counts joint(product(counts[a], counts[b])), left(counts[a]), right(counts[b]);
    for (std::size_t i = 0; i < data.rows(); ++i) {
        const auto x = data(i, a), y = data(i, b);
        ++joint[x * counts[b] + y];
        ++left[x];
        ++right[y];
    }
    const auto n = static_cast<double>(data.rows());
    double information = 0;
    for (std::size_t x = 0; x < counts[a]; ++x)
        for (std::size_t y = 0; y < counts[b]; ++y) {
            const auto count = joint[x * counts[b] + y];
            if (count == 0) continue;
            const auto p = static_cast<double>(count) / n;
            information += p * std::log(p / ((static_cast<double>(left[x]) / n) * (static_cast<double>(right[y]) / n)));
        }
    return information;
}
} // namespace

DiscreteBayesianNetwork::DiscreteBayesianNetwork(const causal::DAG& graph, const Counts& counts,
    const std::vector<std::vector<double>>& probabilities)
    : graph_(graph), cardinalities_(counts), tables_(probabilities) {
    check_cardinalities(graph.size(), counts);
    if (tables_.size() != graph.size()) throw std::invalid_argument("Discrete network needs one table per node");
    for (std::size_t node = 0; node < graph.size(); ++node) {
        auto& table = tables_[node];
        if (table.size() != table_size(graph, counts, node))
            throw std::invalid_argument("Discrete network conditional table size mismatch");
        const auto stride = counts[node];
        for (std::size_t offset = 0; offset < table.size(); offset += stride) {
            double sum = 0;
            for (std::size_t j = 0; j < stride; ++j) {
                const auto p = table[offset + j];
                if (!(p >= 0 && p <= 1)) throw std::invalid_argument("Invalid conditional probability");
                sum += p;
            }
            if (std::abs(sum - 1) > 1e-10) throw std::invalid_argument("Conditional probabilities must sum to one");
            for (std::size_t j = 0; j < stride; ++j) table[offset + j] /= sum;
        }
    }
}

std::size_t DiscreteBayesianNetwork::offset(std::size_t node, const Counts& state) const {
    return row_offset(graph_, cardinalities_, node, [&](auto p) { return state[p]; });
}

std::vector<double> DiscreteBayesianNetwork::marginal(std::size_t query, const DiscreteAssignments& evidence,
    const DiscreteAssignments& intervention, std::size_t max_assignments) const {
    if (query >= graph_.size() || max_assignments == 0)
        throw std::invalid_argument("Discrete inference needs a valid query and a positive budget");
    const auto observed = assigned(cardinalities_, evidence);
    const auto intervened = assigned(cardinalities_, intervention);
    Counts states(graph_.size()), free, factors;
    std::vector<bool> active(graph_.size());
    active[query] = true;
    for (const auto& [node, state] : evidence) {
        if (intervened[node] && state != *intervened[node])
            throw std::domain_error("Evidence contradicts intervention");
        active[node] = true;
    }
    for (auto it = graph_.order().rbegin(); it != graph_.order().rend(); ++it)
        if (active[*it] && !intervened[*it])
            for (const auto parent : graph_.parents(*it)) active[parent] = true;
    std::size_t assignments = 1;
    for (const auto node : graph_.order()) {
        if (!active[node]) continue;
        if (intervened[node]) {
            states[node] = *intervened[node];
            continue;
        }
        factors.push_back(node);
        if (observed[node]) states[node] = *observed[node];
        else {
            assignments = product(assignments, cardinalities_[node], max_assignments);
            free.push_back(node);
        }
    }
    const double impossible = -std::numeric_limits<double>::infinity();
    std::vector<double> masses(cardinalities_[query], impossible);
    for (std::size_t i = 0; i < assignments; ++i) {
        double log_mass = 0;
        for (const auto node : factors) {
            const double p = tables_[node][offset(node, states) + states[node]];
            if (p == 0) {
                log_mass = impossible;
                break;
            }
            log_mass += std::log(p);
        }
        auto& mass = masses[states[query]];
        mass = log_add(mass, log_mass);
        for (const auto node : free) {
            if (++states[node] < cardinalities_[node]) break;
            states[node] = 0;
        }
    }
    const double total = std::accumulate(masses.begin(), masses.end(), impossible, log_add);
    if (!std::isfinite(total)) throw std::domain_error("Evidence has zero probability");
    for (auto& mass : masses) mass = std::exp(mass - total);
    return masses;
}

Cases DiscreteBayesianNetwork::sample(std::size_t n, std::uint64_t seed, const DiscreteAssignments& intervention) const {
    const auto fixed = assigned(cardinalities_, intervention);
    Cases result(n, graph_.size());
    Counts states(graph_.size());
    std::mt19937_64 engine(seed);
    for (std::size_t i = 0; i < n; ++i)
        for (const auto node : graph_.order()) {
            if (fixed[node]) states[node] = *fixed[node];
            else {
                const auto begin = tables_[node].begin() + static_cast<std::ptrdiff_t>(offset(node, states));
                std::discrete_distribution<std::size_t> distribution(begin, begin + static_cast<std::ptrdiff_t>(cardinalities_[node]));
                states[node] = distribution(engine);
            }
            result(i, node) = states[node];
        }
    return result;
}

DiscreteBayesianNetwork fit_discrete_network(const causal::DAG& graph, const std::vector<std::size_t>& counts,
                                             const linalg::DenseMatrix<std::size_t>& data,
                                             double alpha) {
    if (data.cols() != graph.size() || !(alpha > 0) || !std::isfinite(alpha))
        throw std::invalid_argument("Discrete network fit needs matching data columns and a finite positive alpha");
    check_cases(data, counts);
    std::vector<std::vector<double>> tables(graph.size());
    for (const auto node : graph.order()) {
        const auto stride = counts[node];
        auto& table = tables[node];
        table.resize(table_size(graph, counts, node));
        for (std::size_t r = 0; r < data.rows(); ++r) {
            const auto offset = row_offset(graph, counts, node, [&](auto p) { return data(r, p); });
            ++table[offset + data(r, node)];
        }
        for (std::size_t offset = 0; offset < table.size(); offset += stride) {
            const auto first = table.begin() + static_cast<std::ptrdiff_t>(offset);
            const auto end = first + static_cast<std::ptrdiff_t>(stride);
            const double scale = std::max(1.0, alpha);
            const double prior = alpha / scale;
            const double total = std::accumulate(first, end, 0.0) / scale + prior * static_cast<double>(stride);
            for (auto it = first; it != end; ++it) *it = (*it / scale + prior) / total;
        }
    }
    return DiscreteBayesianNetwork(graph, counts, tables);
}

DiscreteBayesianNetwork fit_chow_liu(const linalg::DenseMatrix<std::size_t>& data,
    const std::vector<std::size_t>& counts, std::size_t root, double alpha) {
    const auto n = data.cols();
    if (root >= n || data.rows() == 0 || !(alpha > 0) || !std::isfinite(alpha))
        throw std::invalid_argument("Chow-Liu needs observations, a valid root and a finite positive alpha");
    check_cases(data, counts);
    std::vector<double> best(n, -std::numeric_limits<double>::infinity());
    Counts parent(n, root);
    std::vector<bool> added(n);
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    auto next = root;
    for (std::size_t i = 0; i < n; ++i) {
        const auto node = next;
        added[node] = true;
        if (node != root) edges.emplace_back(parent[node], node);
        next = n;
        for (std::size_t peer = 0; peer < n; ++peer) {
            if (added[peer]) continue;
            const auto information = mutual_information(data, counts, node, peer);
            if (information > best[peer]) {
                best[peer] = information;
                parent[peer] = node;
            }
            if (next == n || best[peer] > best[next]) next = peer;
        }
    }
    return fit_discrete_network(causal::DAG(n, edges), counts, data, alpha);
}
} // namespace datamunge::bayes
