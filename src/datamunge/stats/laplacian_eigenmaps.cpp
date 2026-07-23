#include <datamunge/stats/laplacian_eigenmaps.hpp>

#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
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

} // namespace

LaplacianEigenmaps::LaplacianEigenmaps(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
                                       LaplacianEigenmapsOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void LaplacianEigenmaps::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("LaplacianEigenmaps: feature_columns must not be empty");
    if (options_.n_components == 0) throw std::invalid_argument("LaplacianEigenmaps: n_components must be at least 1");
    if (options_.n_neighbors < 2) throw std::invalid_argument("LaplacianEigenmaps: n_neighbors must be at least 2");
    if (options_.heat_kernel_t <= 0.0) throw std::invalid_argument("LaplacianEigenmaps: heat_kernel_t must be positive");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("LaplacianEigenmaps: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("LaplacianEigenmaps: column '" + col + "' must be numeric");
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
    if (n <= options_.n_neighbors)
        throw std::invalid_argument(
            "LaplacianEigenmaps: not enough complete observations for the requested n_neighbors");
    if (options_.n_components >= n)
        throw std::invalid_argument(
            "LaplacianEigenmaps: n_components must be less than the number of complete observations");

    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]);

    // Full pairwise squared Euclidean distance matrix -- Laplacian Eigenmaps' heat kernel is
    // standardly defined with Euclidean distance regardless of any general DistanceMetric option.
    linalg::DenseMatrix<double> D2(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < p; ++k) {
                const double d = X(i, k) - X(j, k);
                sum += d * d;
            }
            D2(i, j) = sum;
            D2(j, i) = sum;
        }
    }

    // Symmetric k-NN adjacency: an edge (i, j) exists if j is among i's n_neighbors nearest
    // neighbors OR i is among j's n_neighbors nearest neighbors.
    std::vector<std::vector<bool>> adjacent(n, std::vector<bool>(n, false));
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<std::size_t> others;
        others.reserve(n - 1);
        for (std::size_t j = 0; j < n; ++j)
            if (j != i) others.push_back(j);
        std::partial_sort(others.begin(), others.begin() + static_cast<std::ptrdiff_t>(options_.n_neighbors),
                          others.end(),
                          [&](std::size_t a, std::size_t b) { return D2(i, a) < D2(i, b); });
        for (std::size_t k = 0; k < options_.n_neighbors; ++k) {
            const std::size_t j = others[k];
            adjacent[i][j] = true;
            adjacent[j][i] = true;
        }
    }

    // Heat-kernel-weighted graph weight matrix.
    linalg::DenseMatrix<double> W(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            if (!adjacent[i][j]) continue;
            const double w = std::exp(-D2(i, j) / options_.heat_kernel_t);
            W(i, j) = w;
            W(j, i) = w;
        }

    // Degree matrix (diagonal, stored as a vector) -- an all-zero degree means an isolated point
    // in the neighbor graph, which would divide by zero in the D^{-1/2} normalization below.
    std::vector<double> degree(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) sum += W(i, j);
        degree[i] = sum;
        if (degree[i] <= 0.0)
            throw std::invalid_argument(
                "LaplacianEigenmaps: isolated point in neighbor graph; increase n_neighbors or heat_kernel_t");
    }

    std::vector<double> inv_sqrt_degree(n);
    for (std::size_t i = 0; i < n; ++i) inv_sqrt_degree[i] = 1.0 / std::sqrt(degree[i]);

    // Symmetric normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}.
    linalg::DenseMatrix<double> Lsym(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        Lsym(i, i) = 1.0;
        for (std::size_t j = i + 1; j < n; ++j) {
            const double v = -W(i, j) * inv_sqrt_degree[i] * inv_sqrt_degree[j];
            Lsym(i, j) = v;
            Lsym(j, i) = v;
        }
    }

    const auto eig = linalg::jacobi_eigen(Lsym);

    // eig.eigenvalues is descending; the smallest (index n-1) is the trivial ~0 eigenvalue paired
    // with the (non-constant, because of the D^{-1/2} normalization) eigenvector D^{1/2} * 1, and
    // must be discarded. Take the n_components eigenvalues/eigenvectors just before it.
    embedding_ = linalg::DenseMatrix<double>(n, options_.n_components, 0.0);
    eigenvalues_.assign(options_.n_components, 0.0);
    for (std::size_t c = 0; c < options_.n_components; ++c) {
        const std::size_t idx = n - 2 - c;
        eigenvalues_[c] = eig.eigenvalues[idx];
        for (std::size_t i = 0; i < n; ++i) embedding_(i, c) = eig.eigenvectors(i, idx) * inv_sqrt_degree[i];
    }

    observations_ = n;
}

std::vector<double> LaplacianEigenmaps::dimension(const std::size_t index) const {
    if (index >= embedding_.cols()) throw std::out_of_range("LaplacianEigenmaps::dimension index out of range");
    return embedding_.col(index);
}

std::string LaplacianEigenmaps::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void LaplacianEigenmaps::print_summary() const { print_summary(std::cout); }

void LaplacianEigenmaps::print_summary(std::ostream& os) const {
    os << "Laplacian Eigenmaps\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "n_neighbors: " << options_.n_neighbors << ", heat_kernel_t: " << options_.heat_kernel_t << "\n";
    os << "Number of obs: " << observations_ << ", dimensions: " << n_components() << "\n";
    os << std::fixed << std::setprecision(6);
    os << "Non-trivial eigenvalues used (ascending):\n";
    for (std::size_t c = 0; c < n_components(); ++c) os << "  Dim" << (c + 1) << ": " << eigenvalues_[c] << "\n";
}

plot::RPlot LaplacianEigenmaps::plot_embedding(const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("LaplacianEigenmaps::plot_embedding dimension index out of range");

    auto plot = plot::RPlot::create();
    plot.points(embedding_.col(dimension_x), embedding_.col(dimension_y), "", kSeriesColors[0]);
    plot.title("Laplacian Eigenmaps")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

plot::RPlot LaplacianEigenmaps::plot_embedding(const std::vector<std::string>& group_labels,
                                               const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("LaplacianEigenmaps::plot_embedding dimension index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument(
            "LaplacianEigenmaps::plot_embedding: group_labels must have one entry per fitted observation");

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
    plot.title("Laplacian Eigenmaps")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

} // namespace datamunge::stats
