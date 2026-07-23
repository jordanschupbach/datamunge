#include <datamunge/stats/mds.hpp>

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

MDS::MDS(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, MDSOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void MDS::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("MDS: feature_columns must not be empty");
    if (options_.n_components == 0) throw std::invalid_argument("MDS: n_components must be at least 1");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("MDS: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("MDS: column '" + col + "' must be numeric");
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
        throw std::invalid_argument("MDS: n_components must be less than the number of complete observations");

    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]);

    linalg::DenseMatrix<double> D2(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double d = row_distance(X, i, j, options_.metric);
            D2(i, j) = d * d;
            D2(j, i) = d * d;
        }
    }

    std::vector<double> row_mean(n, 0.0);
    double grand_mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) sum += D2(i, j);
        row_mean[i] = sum / static_cast<double>(n);
        grand_mean += sum;
    }
    grand_mean /= static_cast<double>(n) * static_cast<double>(n);

    linalg::DenseMatrix<double> B(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) B(i, j) = -0.5 * (D2(i, j) - row_mean[i] - row_mean[j] + grand_mean);

    const auto eig = linalg::jacobi_eigen(B);
    eigenvalues_ = eig.eigenvalues;

    double positive_total = 0.0;
    for (const auto v : eigenvalues_)
        if (v > 0.0) positive_total += v;

    embedding_ = linalg::DenseMatrix<double>(n, options_.n_components, 0.0);
    double used_total = 0.0;
    for (std::size_t c = 0; c < options_.n_components; ++c) {
        const double value = std::max(eigenvalues_[c], 0.0);
        used_total += value;
        const double scale = std::sqrt(value);
        for (std::size_t i = 0; i < n; ++i) embedding_(i, c) = eig.eigenvectors(i, c) * scale;
    }
    goodness_of_fit_ = positive_total > 0.0 ? used_total / positive_total : 0.0;
    observations_ = n;
}

std::vector<double> MDS::dimension(const std::size_t index) const {
    if (index >= embedding_.cols()) throw std::out_of_range("MDS::dimension index out of range");
    return embedding_.col(index);
}

std::string MDS::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void MDS::print_summary() const { print_summary(std::cout); }

void MDS::print_summary(std::ostream& os) const {
    os << "Classical (metric) Multidimensional Scaling\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Distance metric: " << (options_.metric == DistanceMetric::Euclidean ? "euclidean" : "manhattan") << "\n";
    os << "Number of obs: " << observations_ << ", dimensions: " << n_components() << "\n";
    os << std::fixed << std::setprecision(4);
    os << "Goodness of fit: " << goodness_of_fit_ << "\n\n";

    os << "Eigenvalues used:\n";
    for (std::size_t c = 0; c < n_components(); ++c) os << "  Dim" << (c + 1) << ": " << eigenvalues_[c] << "\n";
}

plot::RPlot MDS::plot_embedding(const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("MDS::plot_embedding dimension index out of range");

    auto plot = plot::RPlot::create();
    plot.points(embedding_.col(dimension_x), embedding_.col(dimension_y), "", kSeriesColors[0]);
    plot.title("Multidimensional Scaling")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

plot::RPlot MDS::plot_embedding(const std::vector<std::string>& group_labels, const std::size_t dimension_x,
                                const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("MDS::plot_embedding dimension index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument("MDS::plot_embedding: group_labels must have one entry per fitted observation");

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
    plot.title("Multidimensional Scaling")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

} // namespace datamunge::stats
