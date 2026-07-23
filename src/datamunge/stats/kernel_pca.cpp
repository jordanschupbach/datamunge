#include <datamunge/stats/kernel_pca.hpp>

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

double dot_product(const linalg::DenseMatrix<double>& X, const std::size_t i, const std::size_t j) {
    double sum = 0.0;
    for (std::size_t k = 0; k < X.cols(); ++k) sum += X(i, k) * X(j, k);
    return sum;
}

double squared_euclidean(const linalg::DenseMatrix<double>& X, const std::size_t i, const std::size_t j) {
    double sum = 0.0;
    for (std::size_t k = 0; k < X.cols(); ++k) {
        const double d = X(i, k) - X(j, k);
        sum += d * d;
    }
    return sum;
}

} // namespace

KernelPCA::KernelPCA(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
                     KernelPCAOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void KernelPCA::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("KernelPCA: feature_columns must not be empty");
    if (options_.n_components == 0) throw std::invalid_argument("KernelPCA: n_components must be at least 1");
    if (options_.kernel != "linear" && options_.kernel != "rbf" && options_.kernel != "polynomial")
        throw std::invalid_argument("KernelPCA: kernel must be 'linear', 'rbf', or 'polynomial'");
    if ((options_.kernel == "rbf" || options_.kernel == "polynomial") && options_.gamma <= 0.0)
        throw std::invalid_argument("KernelPCA: gamma must be positive for the '" + options_.kernel + "' kernel");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("KernelPCA: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("KernelPCA: column '" + col + "' must be numeric");
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
    if (options_.n_components >= n)
        throw std::invalid_argument("KernelPCA: n_components must be less than the number of complete observations");

    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]);

    linalg::DenseMatrix<double> K(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            double value;
            if (options_.kernel == "linear") {
                value = dot_product(X, i, j);
            } else if (options_.kernel == "rbf") {
                value = std::exp(-options_.gamma * squared_euclidean(X, i, j));
            } else {
                value = std::pow(options_.gamma * dot_product(X, i, j) + options_.coef0, options_.degree);
            }
            K(i, j) = value;
            K(j, i) = value;
        }
    }

    std::vector<double> row_mean(n, 0.0);
    double grand_mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) sum += K(i, j);
        row_mean[i] = sum / static_cast<double>(n);
        grand_mean += sum;
    }
    grand_mean /= static_cast<double>(n) * static_cast<double>(n);

    linalg::DenseMatrix<double> Kc(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) Kc(i, j) = K(i, j) - row_mean[i] - row_mean[j] + grand_mean;

    const auto eig = linalg::jacobi_eigen(Kc);
    std::vector<double> all_eigenvalues = eig.eigenvalues;
    for (auto& v : all_eigenvalues) v = std::max(v, 0.0);

    for (std::size_t c = 0; c < options_.n_components; ++c) {
        if (all_eigenvalues[c] <= 0.0)
            throw std::invalid_argument("KernelPCA: fewer than n_components positive eigenvalues available");
    }

    eigenvalues_.assign(options_.n_components, 0.0);
    embedding_ = linalg::DenseMatrix<double>(n, options_.n_components, 0.0);
    for (std::size_t c = 0; c < options_.n_components; ++c) {
        const double lambda = all_eigenvalues[c];
        eigenvalues_[c] = lambda;
        const double scale = std::sqrt(lambda);
        for (std::size_t i = 0; i < n; ++i) embedding_(i, c) = eig.eigenvectors(i, c) * scale;
    }
    observations_ = n;
}

std::vector<double> KernelPCA::dimension(const std::size_t index) const {
    if (index >= embedding_.cols()) throw std::out_of_range("KernelPCA::dimension index out of range");
    return embedding_.col(index);
}

std::string KernelPCA::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void KernelPCA::print_summary() const { print_summary(std::cout); }

void KernelPCA::print_summary(std::ostream& os) const {
    os << "Kernel Principal Component Analysis\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Kernel: " << options_.kernel << "\n";
    os << "Number of obs: " << observations_ << ", components: " << n_components() << "\n";
    os << std::fixed << std::setprecision(4);
    os << "\nEigenvalues used:\n";
    for (std::size_t c = 0; c < n_components(); ++c) os << "  KPC" << (c + 1) << ": " << eigenvalues_[c] << "\n";
}

plot::RPlot KernelPCA::plot_embedding(const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("KernelPCA::plot_embedding dimension index out of range");

    auto plot = plot::RPlot::create();
    plot.points(embedding_.col(dimension_x), embedding_.col(dimension_y), "", kSeriesColors[0]);
    plot.title("Kernel Principal Component Analysis")
        .x_label("KPC" + std::to_string(dimension_x + 1))
        .y_label("KPC" + std::to_string(dimension_y + 1));
    return plot;
}

plot::RPlot KernelPCA::plot_embedding(const std::vector<std::string>& group_labels, const std::size_t dimension_x,
                                      const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("KernelPCA::plot_embedding dimension index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument("KernelPCA::plot_embedding: group_labels must have one entry per fitted observation");

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
    plot.title("Kernel Principal Component Analysis")
        .x_label("KPC" + std::to_string(dimension_x + 1))
        .y_label("KPC" + std::to_string(dimension_y + 1));
    return plot;
}

} // namespace datamunge::stats
