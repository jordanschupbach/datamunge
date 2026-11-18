/// @file sem.cpp
/// @brief Observed linear Gaussian structural equation fitting and inference.
#include <datamunge/causal/sem.hpp>
#include <datamunge/linalg/least_squares.hpp>
#include "detail.hpp"

#include <cmath>
#include <random>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {

/// @brief Validates and indexes sparse interventions.
auto fixed_values(std::size_t nodes, const Intervention& intervention) {
    std::vector<std::optional<double>> fixed(nodes);
    for (const auto& [node, value] : intervention) {
        if (node >= nodes || fixed[node] || !std::isfinite(value))
            throw std::invalid_argument("SEM interventions must specify distinct nodes and finite values");
        fixed[node] = value;
    }
    return fixed;
}

/// @brief Evaluates one structural equation without its disturbance.
template<class Value>
double equation(const LinearSEM& model, std::size_t node, Value value) {
    double result = model.intercepts()[node];
    for (const auto parent : model.graph().parents(node))
        result += model.coefficients()(node, parent) * value(parent);
    return result;
}
} // namespace

LinearSEM::LinearSEM(const DAG& graph)
    : graph_(graph), coefficients_(graph.size(), graph.size(), 0.0),
      intercepts_(graph.size()), variances_(graph.size()) {}

LinearSEM fit_linear_sem(const DAG& graph, const linalg::DenseMatrix<double>& data) {
    if (data.cols() != graph.size() || data.rows() == 0)
        throw std::invalid_argument("SEM data must have one column per node and at least one row");
    detail::finite_matrix(data);
    LinearSEM model(graph);
    for (const auto node : graph.order()) {
        const auto& parents = graph.parents(node);
        const auto columns = parents.size() + 1;
        if (data.rows() <= columns) throw std::invalid_argument("SEM equations require residual degrees of freedom");
        linalg::DenseMatrix<double> design(data.rows(), columns, 1.0);
        std::vector<double> response(data.rows());
        for (std::size_t row = 0; row < data.rows(); ++row) {
            response[row] = data(row, node);
            for (std::size_t col = 0; col < parents.size(); ++col)
                design(row, col + 1) = data(row, parents[col]);
        }
        const auto fit = linalg::least_squares(design, response, linalg::LeastSquaresMethod::SVD);
        if (fit.rank != columns) throw std::runtime_error("Rank-deficient SEM equation");
        model.intercepts_[node] = fit.coefficients[0];
        for (std::size_t col = 0; col < parents.size(); ++col)
            model.coefficients_(node, parents[col]) = fit.coefficients[col + 1];
        model.variances_[node] = fit.residual_norm * fit.residual_norm / static_cast<double>(data.rows());
    }
    return model;
}

SEMMoments LinearSEM::moments(const Intervention& intervention) const {
    const auto nodes = graph_.size();
    const auto fixed = fixed_values(nodes, intervention);
    SEMMoments result{std::vector<double>(nodes), linalg::DenseMatrix<double>(nodes, nodes, 0.0)};
    const auto& order = graph_.order();
    for (std::size_t i = 0; i < nodes; ++i) {
        const auto u = order[i];
        if (fixed[u]) {
            result.mean[u] = *fixed[u];
            continue;
        }
        result.mean[u] = equation(*this, u, [&](auto parent) { return result.mean[parent]; });
        for (std::size_t j = 0; j < i; ++j) {
            const auto v = order[j];
            double covariance = 0.0;
            for (const auto p : graph_.parents(u)) covariance += coefficients_(u, p) * result.covariance(p, v);
            result.covariance(u, v) = result.covariance(v, u) = covariance;
        }
        result.covariance(u, u) = variances_[u];
        for (const auto p : graph_.parents(u))
            result.covariance(u, u) += coefficients_(u, p) * result.covariance(u, p);
    }
    return result;
}

linalg::DenseMatrix<double> LinearSEM::sample(std::size_t n, std::uint64_t seed,
                                             const Intervention& intervention) const {
    const auto fixed = fixed_values(graph_.size(), intervention);
    linalg::DenseMatrix<double> draws(n, graph_.size());
    std::vector<double> deviations(graph_.size());
    for (const auto node : graph_.order()) deviations[node] = std::sqrt(variances_[node]);
    std::mt19937_64 engine(seed);
    std::normal_distribution<double> normal;
    for (std::size_t row = 0; row < n; ++row)
        for (const auto node : graph_.order())
            draws(row, node) = fixed[node] ? *fixed[node]
                : equation(*this, node, [&](auto p) { return draws(row, p); }) + deviations[node] * normal(engine);
    return draws;
}

double LinearSEM::total_effect(std::size_t exposure, std::size_t outcome) const {
    if (exposure >= graph_.size() || outcome >= graph_.size() || exposure == outcome)
        throw std::invalid_argument("SEM effect requires two distinct valid nodes");
    std::vector<double> derivative(graph_.size());
    for (const auto node : graph_.order()) {
        if (node == exposure) derivative[node] = 1.0;
        else for (const auto p : graph_.parents(node)) derivative[node] += coefficients_(node, p) * derivative[p];
    }
    return derivative[outcome];
}

std::vector<double> LinearSEM::counterfactual(const std::vector<double>& observed,
                                               const Intervention& intervention) const {
    if (observed.size() != graph_.size()) throw std::invalid_argument("SEM observation size mismatch");
    for (const double value : observed)
        if (!std::isfinite(value)) throw std::invalid_argument("SEM observation must be finite");
    const auto fixed = fixed_values(graph_.size(), intervention);
    std::vector<double> result(graph_.size());
    for (const auto node : graph_.order()) {
        if (fixed[node]) result[node] = *fixed[node];
        else {
            const auto factual_mean = equation(*this, node, [&](auto p) { return observed[p]; });
            const auto changed_mean = equation(*this, node, [&](auto p) { return result[p]; });
            result[node] = observed[node] + (changed_mean - factual_mean);
        }
    }
    return result;
}
} // namespace datamunge::causal
