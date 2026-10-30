/// @file lle.cpp
/// @brief Locally Linear Embedding: neighbour search, reconstruction weights, the sparse
///        assembly of \f$M=(I-W)^\top(I-W)\f$ and its bottom eigenvectors.

#include <datamunge/stats/lle.hpp>

#include <datamunge/linalg/eigen.hpp>
#include <datamunge/linalg/lu.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Helpers private to this file.
namespace {

/// @brief Colours of the groups in LLE::plot_embedding(), cycled.
const graphics::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

/// @brief Sorted distinct labels.
/// @param labels  Labels, with repeats.
/// @return Each label once, sorted.
std::vector<std::string> unique_sorted(const std::vector<std::string>& labels) {
    std::vector<std::string> result(labels.begin(), labels.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

/// @brief Distance between rows @p i and @p j of @p X.
/// @param X  Data, one row per point.
/// @param i  First row.
/// @param j  Second row.
/// @param metric  Euclidean or Manhattan.
/// @return The distance.
double row_distance(const linalg::DenseMatrix<double>& X, const std::size_t i, const std::size_t j,
                    const DistanceMetric metric) {
    double sum = 0.0;
    if (metric == DistanceMetric::Euclidean) {
        for (std::size_t k = 0; k < X.cols(); ++k) {
            const double d = X(i, k) - X(j, k);
            sum += d * d;
        }
        return std::sqrt(sum);
    }
    for (std::size_t k = 0; k < X.cols(); ++k) sum += std::abs(X(i, k) - X(j, k));
    return sum;
}

/// @brief Union-find root of @p x, with path halving.
/// @param parent  Parent links, updated.
/// @param x  Element.
/// @return Its root.
std::size_t find_root(std::vector<std::size_t>& parent, std::size_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x         = parent[x];
    }
    return x;
}

} // namespace

std::vector<double> lle_reconstruction_weights(const std::vector<double>&              point,
                                               const std::vector<std::vector<double>>& neighbors,
                                               const double                            regularization) {
    if (neighbors.empty()) throw std::invalid_argument("lle_reconstruction_weights: no neighbours");
    if (regularization < 0.0) throw std::invalid_argument("lle_reconstruction_weights: regularization must be non-negative");
    const std::size_t k = neighbors.size();
    const std::size_t p = point.size();
    for (const auto& nb : neighbors)
        if (nb.size() != p) throw std::invalid_argument("lle_reconstruction_weights: dimensions differ");

    linalg::DenseMatrix<double> C(k, k, 0.0);
    for (std::size_t a = 0; a < k; ++a) {
        for (std::size_t b = a; b < k; ++b) {
            double dot = 0.0;
            for (std::size_t f = 0; f < p; ++f) dot += (point[f] - neighbors[a][f]) * (point[f] - neighbors[b][f]);
            C(a, b) = dot;
            C(b, a) = dot;
        }
    }
    double trace = 0.0;
    for (std::size_t a = 0; a < k; ++a) trace += C(a, a);
    if (trace == 0.0) return std::vector<double>(k, 1.0 / static_cast<double>(k));
    const double ridge = regularization * trace;
    for (std::size_t a = 0; a < k; ++a) C(a, a) += ridge;

    const auto          decomposition = linalg::lu(C);
    std::vector<double> w = decomposition.singular ? std::vector<double>(k, 1.0)
                                                   : decomposition.solve(std::vector<double>(k, 1.0));
    double sum = std::accumulate(w.begin(), w.end(), 0.0);
    if (!std::isfinite(sum) || sum == 0.0) {
        w.assign(k, 1.0);
        sum = static_cast<double>(k);
    }
    for (auto& v : w) v /= sum;
    return w;
}

LLE::LLE(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, LLEOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void LLE::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("LLE: feature_columns must not be empty");
    if (options_.n_components == 0) throw std::invalid_argument("LLE: n_components must be at least 1");
    if (options_.n_neighbors < 2) throw std::invalid_argument("LLE: n_neighbors must be at least 2");
    if (options_.regularization < 0.0) throw std::invalid_argument("LLE: regularization must be non-negative");
    if (options_.n_components >= options_.n_neighbors)
        throw std::invalid_argument("LLE: n_components must be less than n_neighbors");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("LLE: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("LLE: column '" + col + "' must be numeric");
    }

    kept_row_indices_.clear();
    for (std::size_t i = 0; i < data.nrows(); ++i) {
        const bool complete = std::none_of(feature_columns_.begin(), feature_columns_.end(),
                                           [&](const std::string& col) { return data.is_null(col, i); });
        if (complete) kept_row_indices_.push_back(i);
    }

    const std::size_t n = kept_row_indices_.size();
    const std::size_t p = feature_columns_.size();
    const std::size_t k = options_.n_neighbors;
    const std::size_t d = options_.n_components;
    if (n < k + 1) throw std::invalid_argument("LLE: need at least n_neighbors + 1 complete observations");

    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]);

    // Step 1a: the k nearest neighbours of each point (ties broken by index).
    neighbors_.assign(n, {});
    std::vector<double>      dist(n);
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) dist[j] = (j == i) ? 0.0 : row_distance(X, i, j, options_.metric);
        idx.resize(n);
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        idx.erase(idx.begin() + static_cast<std::ptrdiff_t>(i));
        std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                          [&](std::size_t a, std::size_t b) { return dist[a] < dist[b] || (dist[a] == dist[b] && a < b); });
        neighbors_[i].assign(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k));
    }

    // Connected components of the neighbour graph: each gives M a zero eigenvalue.
    std::vector<std::size_t> parent(n);
    std::iota(parent.begin(), parent.end(), std::size_t{0});
    for (std::size_t i = 0; i < n; ++i)
        for (const auto j : neighbors_[i]) parent[find_root(parent, i)] = find_root(parent, j);
    component_labels_.assign(n, -1);
    std::vector<int> label_of_root(n, -1);
    int              next_label = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t r = find_root(parent, i);
        if (label_of_root[r] < 0) label_of_root[r] = next_label++;
        component_labels_[i] = label_of_root[r];
    }
    component_count_ = static_cast<std::size_t>(next_label);

    // Step 1b: reconstruction weights and errors.
    weights_.assign(n, {});
    reconstruction_errors_.assign(n, 0.0);
    std::vector<double>              point(p);
    std::vector<std::vector<double>> nbrs(k, std::vector<double>(p));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t f = 0; f < p; ++f) point[f] = X(i, f);
        for (std::size_t a = 0; a < k; ++a)
            for (std::size_t f = 0; f < p; ++f) nbrs[a][f] = X(neighbors_[i][a], f);
        weights_[i] = lle_reconstruction_weights(point, nbrs, options_.regularization);
        double err  = 0.0;
        for (std::size_t f = 0; f < p; ++f) {
            double r = point[f];
            for (std::size_t a = 0; a < k; ++a) r -= weights_[i][a] * nbrs[a][f];
            err += r * r;
        }
        reconstruction_errors_[i] = err;
    }

    // Step 2: M = (I - W)^T (I - W) = I - W - W^T + W^T W, assembled in O(n k^2).
    linalg::DenseMatrix<double> M(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        M(i, i) += 1.0;
        const auto& nb = neighbors_[i];
        const auto& w  = weights_[i];
        for (std::size_t a = 0; a < k; ++a) {
            M(i, nb[a]) -= w[a];
            M(nb[a], i) -= w[a];
            for (std::size_t b = 0; b < k; ++b) M(nb[a], nb[b]) += w[a] * w[b];
        }
    }

    // Each component's indicator vector e is an exact null vector of M (rows of W sum to 1 and
    // never cross components). When an embedding eigenvalue is also near zero, a solver may
    // return any rotation of the two, so add shift * e e^T / |e| with shift above the whole
    // spectrum (Gershgorin bound): the trivial vectors become the top eigenvectors and every
    // other eigenpair is unchanged, so the embedding is exactly orthogonal to them.
    if (n < component_count_ + d)
        throw std::invalid_argument("LLE: too few observations for n_components after discarding trivial eigenvalues");
    double bound = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double row = 0.0;
        for (std::size_t j = 0; j < n; ++j) row += std::abs(M(i, j));
        bound = std::max(bound, row);
    }
    const double             shift = bound + 1.0;
    std::vector<std::size_t> component_size(component_count_, 0);
    for (const int c : component_labels_) ++component_size[static_cast<std::size_t>(c)];
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (component_labels_[i] == component_labels_[j])
                M(i, j) += shift / static_cast<double>(component_size[static_cast<std::size_t>(component_labels_[i])]);

    const auto eig = linalg::symmetric_eigen(M);
    // Eigenvalues are decreasing: the first component_count_ are the shifted trivial ones (0 in
    // M itself, reported last); the last d, read backwards, are the embedding.
    eigenvalues_.assign(eig.eigenvalues.begin() + static_cast<std::ptrdiff_t>(component_count_), eig.eigenvalues.end());
    eigenvalues_.insert(eigenvalues_.end(), component_count_, 0.0);
    embedding_ = linalg::DenseMatrix<double>(n, d, 0.0);
    for (std::size_t c = 0; c < d; ++c) {
        const std::size_t col  = n - 1 - c;
        std::size_t       peak = 0;
        for (std::size_t i = 1; i < n; ++i)
            if (std::abs(eig.eigenvectors(i, col)) > std::abs(eig.eigenvectors(peak, col))) peak = i;
        const double sign = eig.eigenvectors(peak, col) < 0.0 ? -1.0 : 1.0;
        for (std::size_t i = 0; i < n; ++i) embedding_(i, c) = sign * eig.eigenvectors(i, col);
    }
    observations_ = n;
}

double LLE::cost(const std::vector<double>& values) const {
    if (values.size() != observations_) throw std::invalid_argument("LLE::cost: values must have one entry per fitted observation");
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double       norm = 0.0;
    for (const double v : values) norm += (v - mean) * (v - mean);
    if (!(norm > 0.0)) throw std::invalid_argument("LLE::cost: values must not be constant");
    norm = std::sqrt(norm);
    double total = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        double r = values[i] - mean;
        for (std::size_t a = 0; a < neighbors_[i].size(); ++a) r -= weights_[i][a] * (values[neighbors_[i][a]] - mean);
        total += (r / norm) * (r / norm);
    }
    return total;
}

std::vector<double> LLE::dimension(const std::size_t index) const {
    if (index >= embedding_.cols()) throw std::out_of_range("LLE::dimension index out of range");
    return embedding_.col(index);
}

std::string LLE::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void LLE::print_summary() const { print_summary(std::cout); }

void LLE::print_summary(std::ostream& os) const {
    os << "Locally Linear Embedding\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Distance metric: " << (options_.metric == DistanceMetric::Euclidean ? "euclidean" : "manhattan") << "\n";
    os << "n_neighbors: " << options_.n_neighbors << ", regularization: " << options_.regularization << "\n";
    os << "Number of obs: " << observations_ << ", dimensions: " << n_components() << "\n";
    os << "Neighbour-graph components: " << component_count_
       << (component_count_ > 1 ? " (each embedded separately; consider more neighbours)" : "") << "\n";
    os << std::scientific << std::setprecision(4);
    os << "Embedding eigenvalues (costs):\n";
    const std::size_t n = eigenvalues_.size();
    for (std::size_t c = 0; c < n_components(); ++c)
        os << "  Dim" << (c + 1) << ": " << eigenvalues_[n - component_count_ - 1 - c] << "\n";
    os << std::defaultfloat;
}

graphics::RPlot LLE::plot_embedding(const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("LLE::plot_embedding dimension index out of range");

    auto plot = graphics::RPlot::create();
    plot.points(embedding_.col(dimension_x), embedding_.col(dimension_y), "", kSeriesColors[0]);
    plot.title("Locally Linear Embedding")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

graphics::RPlot LLE::plot_embedding(const std::vector<std::string>& group_labels, const std::size_t dimension_x,
                                    const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("LLE::plot_embedding dimension index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument("LLE::plot_embedding: group_labels must have one entry per fitted observation");

    auto       plot   = graphics::RPlot::create();
    const auto groups = unique_sorted(group_labels);
    for (std::size_t g = 0; g < groups.size(); ++g) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < observations_; ++i) {
            if (group_labels[i] != groups[g]) continue;
            xs.push_back(embedding_(i, dimension_x));
            ys.push_back(embedding_(i, dimension_y));
        }
        plot.points(xs, ys, groups[g], kSeriesColors[g % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
    }
    plot.title("Locally Linear Embedding")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

} // namespace datamunge::stats
