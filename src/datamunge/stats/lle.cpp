#include <datamunge/stats/lle.hpp>

#include <datamunge/linalg/eigen.hpp>
#include <datamunge/linalg/lu.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

namespace {

const plot::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

std::vector<std::string> unique_sorted(const std::vector<std::string>& labels) {
    std::vector<std::string> result(labels.begin(), labels.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

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

} // namespace

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
        bool ok = true;
        for (const auto& col : feature_columns_) {
            if (data.is_null(col, i)) {
                ok = false;
                break;
            }
        }
        if (ok) kept_row_indices_.push_back(i);
    }

    const std::size_t n = kept_row_indices_.size();
    const std::size_t p = feature_columns_.size();
    const std::size_t k = options_.n_neighbors;
    if (n < k + 1)
        throw std::invalid_argument("LLE: need at least n_neighbors + 1 complete observations");

    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]);

    // Pairwise distance matrix.
    linalg::DenseMatrix<double> D(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double d = row_distance(X, i, j, options_.metric);
            D(i, j) = d;
            D(j, i) = d;
        }
    }

    // For each point, find its k nearest neighbors (excluding itself).
    std::vector<std::vector<std::size_t>> neighbors(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<std::size_t> idx;
        idx.reserve(n - 1);
        for (std::size_t j = 0; j < n; ++j)
            if (j != i) idx.push_back(j);
        std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                          [&](std::size_t a, std::size_t b) { return D(i, a) < D(i, b); });
        idx.resize(k);
        neighbors[i] = std::move(idx);
    }

    // The classical LLE derivation assumes the k-NN neighbor graph is connected, in which case
    // M = (I-W)^T(I-W) has exactly one trivial (~0) eigenvalue, the constant vector. If the
    // neighbor graph is actually disconnected into multiple components (e.g. a tightly separated
    // cluster whose k nearest neighbors are all within the cluster itself, common on real data --
    // observed empirically on the iris dataset, where the setosa species is isolated from the
    // other two regardless of n_neighbors), EACH component contributes its own trivial ~0
    // eigenvalue (constant on that component, zero elsewhere), not just one overall. Failing to
    // discard all of them leaks a degenerate eigenvector into the embedding that pins an entire
    // component's points to the same coordinate (verified: without this fix, all 50 iris setosa
    // rows collapsed to exactly (0, 0)). Count connected components of the (undirected closure of
    // the directed) k-NN graph via union-find and discard that many trailing eigenvalues instead
    // of always exactly one.
    std::vector<std::size_t> uf_parent(n);
    for (std::size_t i = 0; i < n; ++i) uf_parent[i] = i;
    std::function<std::size_t(std::size_t)> uf_find = [&](std::size_t x) {
        while (uf_parent[x] != x) {
            uf_parent[x] = uf_parent[uf_parent[x]];
            x = uf_parent[x];
        }
        return x;
    };
    for (std::size_t i = 0; i < n; ++i) {
        for (const auto j : neighbors[i]) {
            const std::size_t ri = uf_find(i);
            const std::size_t rj = uf_find(j);
            if (ri != rj) uf_parent[ri] = rj;
        }
    }
    std::vector<bool> is_root(n, false);
    std::size_t num_components = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t r = uf_find(i);
        if (!is_root[r]) {
            is_root[r] = true;
            ++num_components;
        }
    }

    // Build the n x n reconstruction weight matrix W.
    linalg::DenseMatrix<double> W(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& nbrs = neighbors[i];

        linalg::DenseMatrix<double> C(k, k, 0.0);
        for (std::size_t a = 0; a < k; ++a) {
            for (std::size_t b = a; b < k; ++b) {
                double dot = 0.0;
                for (std::size_t f = 0; f < p; ++f) {
                    const double da = X(i, f) - X(nbrs[a], f);
                    const double db = X(i, f) - X(nbrs[b], f);
                    dot += da * db;
                }
                C(a, b) = dot;
                C(b, a) = dot;
            }
        }

        double trace = 0.0;
        for (std::size_t a = 0; a < k; ++a) trace += C(a, a);
        const double reg = (trace > 0.0) ? (options_.regularization * trace / static_cast<double>(k)) : 1e-10;
        for (std::size_t a = 0; a < k; ++a) C(a, a) += reg;

        const std::vector<double> ones(k, 1.0);
        const auto lu_dec = linalg::lu(C);
        std::vector<double> w = lu_dec.singular ? std::vector<double>(k, 1.0) : lu_dec.solve(ones);

        double wsum = 0.0;
        for (const auto v : w) wsum += v;
        if (wsum == 0.0) wsum = 1.0;
        for (auto& v : w) v /= wsum;

        for (std::size_t a = 0; a < k; ++a) W(i, nbrs[a]) = w[a];
    }

    // M = (I - W)^T (I - W)
    const auto I = linalg::DenseMatrix<double>::identity(n);
    const auto IW = I - W;
    const auto M = IW.transpose() * IW;

    const auto eig = linalg::jacobi_eigen(M);
    eigenvalues_ = eig.eigenvalues;

    // Eigenvalues are descending; the trivial (near-zero) eigenvalue(s) are the smallest, i.e.
    // the trailing columns -- one per connected component of the neighbor graph (see above; for
    // the common case of a fully-connected neighbor graph, num_components == 1 and this reduces
    // to exactly "discard the very last column"). Take the d eigenvectors just before that block.
    const std::size_t d = options_.n_components;
    const std::size_t discard = std::min(num_components, n > d ? n - d : std::size_t{1});
    embedding_ = linalg::DenseMatrix<double>(n, d, 0.0);
    for (std::size_t c = 0; c < d; ++c) {
        const std::size_t col = n - discard - d + c;
        for (std::size_t i = 0; i < n; ++i) embedding_(i, c) = eig.eigenvectors(i, col);
    }
    discarded_trivial_count_ = discard;

    observations_ = n;
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
    os << std::fixed << std::setprecision(6);
    os << "\nEigenvalues used for the embedding:\n";
    const std::size_t n = eigenvalues_.size();
    const std::size_t d = n_components();
    if (n >= discarded_trivial_count_ + d) {
        for (std::size_t c = 0; c < d; ++c)
            os << "  Dim" << (c + 1) << ": " << eigenvalues_[n - discarded_trivial_count_ - d + c] << "\n";
    }
    os << "Discarded trivial eigenvalue(s): " << discarded_trivial_count_
       << (discarded_trivial_count_ > 1 ? " (neighbor graph has multiple connected components)" : "") << "\n";
}

plot::RPlot LLE::plot_embedding(const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("LLE::plot_embedding dimension index out of range");

    auto plot = plot::RPlot::create();
    plot.points(embedding_.col(dimension_x), embedding_.col(dimension_y), "", kSeriesColors[0]);
    plot.title("Locally Linear Embedding")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

plot::RPlot LLE::plot_embedding(const std::vector<std::string>& group_labels, const std::size_t dimension_x,
                                const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("LLE::plot_embedding dimension index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument("LLE::plot_embedding: group_labels must have one entry per fitted observation");

    auto plot = plot::RPlot::create();
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
