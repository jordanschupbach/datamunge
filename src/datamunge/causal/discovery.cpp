/// @file discovery.cpp
/// @brief Gaussian conditional independence and PC-stable graph discovery.
#include <datamunge/causal/discovery.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include "detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Double-valued observational or correlation matrix.
using Matrix = linalg::DenseMatrix<double>;
/// @brief Integer adjacency or orientation proposal matrix.
using Edges = linalg::DenseMatrix<int>;

/// @brief Builds a Pearson correlation matrix using centered, normalized columns.
Matrix correlations(const Matrix& data, const std::vector<std::size_t>& columns) {
    detail::finite_matrix(data);
    Matrix standardized(data.rows(), columns.size());
    for (std::size_t j = 0; j < columns.size(); ++j) {
        long double sum = 0;
        for (std::size_t i = 0; i < data.rows(); ++i) sum += data(i, columns[j]);
        const auto mean = sum / data.rows();
        long double ss = 0;
        for (std::size_t i = 0; i < data.rows(); ++i) {
            const long double centered = data(i, columns[j]) - mean;
            ss += centered * centered;
        }
        if (!(ss > 0) || !std::isfinite(ss)) throw std::invalid_argument("Gaussian CI requires nonconstant columns");
        for (std::size_t i = 0; i < data.rows(); ++i)
            standardized(i, j) = static_cast<double>((data(i, columns[j]) - mean) / std::sqrt(ss));
    }
    Matrix result(columns.size(), columns.size(), 1.0);
    for (std::size_t j = 0; j < columns.size(); ++j)
        for (std::size_t k = 0; k < j; ++k) {
            double dot = 0;
            for (std::size_t i = 0; i < data.rows(); ++i) dot += standardized(i, j) * standardized(i, k);
            result(j, k) = result(k, j) = std::clamp(dot, -1.0, 1.0);
        }
    return result;
}

/// @brief Tests partial correlation by the Schur complement of the conditioning block.
GaussianCITest test(const Matrix& r, std::size_t rows, std::size_t x, std::size_t y,
                    const std::vector<std::size_t>& given) {
    double partial = r(x, y);
    if (!given.empty()) {
        Matrix block(given.size(), given.size());
        std::vector<double> rx(given.size()), ry(given.size());
        for (std::size_t i = 0; i < given.size(); ++i) {
            rx[i] = r(x, given[i]);
            ry[i] = r(y, given[i]);
            for (std::size_t j = 0; j < given.size(); ++j) block(i, j) = r(given[i], given[j]);
        }
        const auto factor = linalg::cholesky(block);
        const auto bx = factor.solve(rx);
        const auto by = factor.solve(ry);
        const double vx = 1 - std::inner_product(rx.begin(), rx.end(), bx.begin(), 0.0);
        const double vy = 1 - std::inner_product(ry.begin(), ry.end(), by.begin(), 0.0);
        if (std::min(vx, vy) <= 32 * std::numeric_limits<double>::epsilon())
            throw std::runtime_error("Gaussian CI has singular conditional variance");
        partial = (partial - std::inner_product(rx.begin(), rx.end(), by.begin(), 0.0)) / std::sqrt(vx * vy);
    }
    partial = std::clamp(partial, -1.0, 1.0);
    const double z = std::abs(std::atanh(partial)) * std::sqrt(static_cast<double>(rows - given.size() - 3));
    return {partial, std::erfc(z / std::numbers::sqrt2)};
}

/// @brief Checks for an arrow path, including paths created in the current orientation pass.
bool directed_path(const Edges& edges, std::size_t start, std::size_t end) {
    std::vector<bool> seen(edges.rows());
    std::vector<std::size_t> stack{start};
    seen[start] = true;
    while (!stack.empty()) {
        const auto u = stack.back();
        stack.pop_back();
        if (u == end) return true;
        for (std::size_t v = 0; v < edges.rows(); ++v)
            if (edges(u, v) && !edges(v, u) && !seen[v]) {
                seen[v] = true;
                stack.push_back(v);
            }
    }
    return false;
}

/// @brief Applies compatible orientation proposals and preserves conflicted edges as undirected.
bool apply_arrows(Edges& edges, const Edges& proposals, Edges& conflicts, PCResult& result) {
    bool changed = false;
    for (std::size_t a = 0; a < edges.rows(); ++a)
        for (std::size_t b = a + 1; b < edges.rows(); ++b) {
            if (!edges(a, b) || !edges(b, a) || conflicts(a, b)) continue;
            const bool ab = proposals(a, b), ba = proposals(b, a);
            if (!ab && !ba) continue;
            if ((ab && ba) || directed_path(edges, ab ? b : a, ab ? a : b)) {
                conflicts(a, b) = 1;
                ++result.orientation_conflicts;
                continue;
            }
            if (ab) edges(b, a) = 0;
            else edges(a, b) = 0;
            changed = true;
        }
    return changed;
}

/// @brief Proposes Meek R1, directed-path R2 and two-parent R3 orientations.
bool meek_arrow(const Edges& edges, std::size_t a, std::size_t b) {
    if (directed_path(edges, a, b)) return true;
    std::vector<std::size_t> parents;
    for (std::size_t c = 0; c < edges.rows(); ++c) {
        if (edges(c, a) && !edges(a, c) && !edges(c, b) && !edges(b, c)) return true;
        if (edges(a, c) && edges(c, a) && edges(c, b) && !edges(b, c)) {
            for (const auto d : parents)
                if (!edges(c, d) && !edges(d, c)) return true;
            parents.push_back(c);
        }
    }
    return false;
}
} // namespace

GaussianCITest gaussian_ci_test(const linalg::DenseMatrix<double>& data, std::size_t x, std::size_t y,
                                 const std::vector<std::size_t>& given) {
    if (data.rows() < 4 || given.size() >= data.rows() - 3)
        throw std::invalid_argument("Gaussian CI needs more rows than the conditioning size plus three");
    std::vector<std::size_t> columns{x, y};
    columns.insert(columns.end(), given.begin(), given.end());
    auto sorted = columns;
    std::ranges::sort(sorted);
    if (sorted.back() >= data.cols() || std::ranges::adjacent_find(sorted) != sorted.end())
        throw std::invalid_argument("Gaussian CI needs distinct valid columns");
    std::vector<std::size_t> conditional(given.size());
    std::iota(conditional.begin(), conditional.end(), 2);
    return test(correlations(data, columns), data.rows(), 0, 1, conditional);
}

PCResult pc_stable(const linalg::DenseMatrix<double>& data, PCOptions options) {
    if (data.rows() < 4 || data.cols() == 0 || !(options.alpha > 0 && options.alpha < 1) || options.max_tests == 0)
        throw std::invalid_argument("PC-stable needs at least four rows, variables and valid search options");
    const auto n = data.cols();
    std::vector<std::size_t> columns(n);
    std::iota(columns.begin(), columns.end(), 0);
    const auto r = correlations(data, columns);
    Edges edges(n, n, 1);
    for (std::size_t i = 0; i < n; ++i) edges(i, i) = 0;
    std::vector<std::vector<std::size_t>> separation(n * n);
    PCResult result{};
    result.nodes = n;
    const auto depth = std::min({options.max_conditioning, data.rows() - 4, n - 1});
    for (std::size_t k = 0; k <= depth; ++k) {
        const auto frozen = edges;
        bool eligible = false;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) {
                if (!frozen(i, j)) continue;
                for (const auto endpoint : {i, j}) {
                    if (endpoint == j && k == 0) break;
                    std::vector<std::size_t> neighbors;
                    for (std::size_t c = 0; c < n; ++c)
                        if (c != i && c != j && frozen(endpoint, c)) neighbors.push_back(c);
                    eligible |= neighbors.size() >= k;
                    if (detail::subsets(neighbors, k, [&](const auto& subset) {
                        if (result.tests == options.max_tests) throw std::length_error("PC-stable test budget exhausted");
                        ++result.tests;
                        if (test(r, data.rows(), i, j, subset).p_value <= options.alpha) return false;
                        edges(i, j) = edges(j, i) = 0;
                        separation[i * n + j] = subset;
                        return true;
                    })) break;
                }
            }
        if (!eligible) break;
    }

    Edges proposals(n, n, 0), conflicts(n, n, 0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            if (edges(i, j)) continue;
            const auto& set = separation[i * n + j];
            for (std::size_t middle = 0; middle < n; ++middle)
                if (edges(i, middle) && edges(j, middle) && !std::ranges::binary_search(set, middle))
                    proposals(i, middle) = proposals(j, middle) = 1;
        }
    apply_arrows(edges, proposals, conflicts, result);
    do {
        proposals = Edges(n, n, 0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (edges(i, j) && edges(j, i) && !conflicts(std::min(i, j), std::max(i, j)))
                    proposals(i, j) = meek_arrow(edges, i, j);
    } while (apply_arrows(edges, proposals, conflicts, result));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (edges(i, j)) {
                if (!edges(j, i)) result.directed.emplace_back(i, j);
                else if (i < j) result.undirected.emplace_back(i, j);
            }
    return result;
}

graphics::Diagram PCResult::diagram(const std::vector<graphics::DiagramPoint>& positions,
                                     const std::vector<std::string>& labels) const {
    auto result = DAG(nodes, directed).diagram(positions, labels);
    for (const auto& [a, b] : undirected) result.link(std::to_string(a), std::to_string(b));
    return result;
}
} // namespace datamunge::causal
