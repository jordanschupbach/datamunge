#include <datamunge/stats/pca.hpp>

#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
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

PCA::PCA(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, PCAOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void PCA::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("PCA: feature_columns must not be empty");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("PCA: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("PCA: column '" + col + "' must be numeric");
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
    if (n < 2) throw std::invalid_argument("PCA: fewer than 2 complete observations");

    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]);

    mean_.assign(p, 0.0);
    scale_.assign(p, 1.0);
    for (std::size_t j = 0; j < p; ++j) {
        if (!options_.center) continue;
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) sum += X(i, j);
        mean_[j] = sum / static_cast<double>(n);
    }

    linalg::DenseMatrix<double> Xc(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) Xc(i, j) = X(i, j) - mean_[j];

    if (options_.scale) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum_sq = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum_sq += Xc(i, j) * Xc(i, j);
            const double variance = sum_sq / static_cast<double>(n - 1);
            const double sd = std::sqrt(variance);
            scale_[j] = sd > 0.0 ? sd : 1.0;
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < p; ++j) Xc(i, j) /= scale_[j];
    }

    linalg::DenseMatrix<double> covariance(p, p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t k = j; k < p; ++k) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += Xc(i, j) * Xc(i, k);
            const double value = sum / static_cast<double>(n - 1);
            covariance(j, k) = value;
            covariance(k, j) = value;
        }
    }

    const auto eig = linalg::jacobi_eigen(covariance);
    explained_variance_ = eig.eigenvalues;
    for (auto& v : explained_variance_) v = std::max(v, 0.0);
    loadings_ = eig.eigenvectors;
    scores_ = Xc * loadings_;
    observations_ = n;
}

std::vector<double> PCA::explained_variance_ratio() const {
    const double total = std::accumulate(explained_variance_.begin(), explained_variance_.end(), 0.0);
    std::vector<double> ratios(explained_variance_.size(), 0.0);
    if (total <= 0.0) return ratios;
    for (std::size_t i = 0; i < explained_variance_.size(); ++i) ratios[i] = explained_variance_[i] / total;
    return ratios;
}

std::vector<double> PCA::cumulative_explained_variance_ratio() const {
    auto ratios = explained_variance_ratio();
    double running = 0.0;
    for (auto& v : ratios) {
        running += v;
        v = running;
    }
    return ratios;
}

std::vector<double> PCA::component_loadings(const std::size_t component_index) const {
    if (component_index >= loadings_.cols()) throw std::out_of_range("PCA::component_loadings index out of range");
    return loadings_.col(component_index);
}

std::vector<double> PCA::component_scores(const std::size_t component_index) const {
    if (component_index >= scores_.cols()) throw std::out_of_range("PCA::component_scores index out of range");
    return scores_.col(component_index);
}

linalg::DenseMatrix<double> PCA::transform(const dstruct::DataFrame& newdata) const {
    for (const auto& col : feature_columns_)
        if (!newdata.has_column(col)) throw std::invalid_argument("PCA::transform: column '" + col + "' not found");

    const std::size_t m = newdata.nrows();
    const std::size_t p = feature_columns_.size();
    linalg::DenseMatrix<double> Xc(m, p, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            if (newdata.is_null(feature_columns_[j], i))
                throw std::invalid_argument("PCA::transform: null value in feature column '" + feature_columns_[j] + "'");
            Xc(i, j) = (newdata.double_at(feature_columns_[j], i) - mean_[j]) / scale_[j];
        }
    }
    return Xc * loadings_;
}

std::string PCA::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void PCA::print_summary() const { print_summary(std::cout); }

void PCA::print_summary(std::ostream& os) const {
    os << "Principal Component Analysis\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Number of obs: " << observations_ << ", components: " << num_components() << "\n";
    os << (options_.scale ? "Scaled (correlation matrix)\n" : "Unscaled (covariance matrix)\n");

    const auto ratio = explained_variance_ratio();
    const auto cumulative = cumulative_explained_variance_ratio();
    os << std::fixed << std::setprecision(4);
    os << "\n" << std::left << std::setw(12) << "Component" << std::setw(16) << "Std.Dev." << std::setw(20)
       << "Prop.Variance" << "Cum.Variance\n";
    for (std::size_t c = 0; c < num_components(); ++c) {
        os << std::left << std::setw(12) << ("PC" + std::to_string(c + 1)) << std::setw(16)
           << std::sqrt(explained_variance_[c]) << std::setw(20) << ratio[c] << cumulative[c] << "\n";
    }
}

plot::RPlot PCA::plot_scores(const std::size_t component_x, const std::size_t component_y) const {
    if (component_x >= num_components() || component_y >= num_components())
        throw std::out_of_range("PCA::plot_scores component index out of range");

    auto plot = plot::RPlot::create();
    plot.points(scores_.col(component_x), scores_.col(component_y), "", kSeriesColors[0]);
    const auto ratio = explained_variance_ratio();
    std::ostringstream x_label, y_label;
    x_label << "PC" << (component_x + 1) << " (" << std::fixed << std::setprecision(1) << ratio[component_x] * 100.0
            << "%)";
    y_label << "PC" << (component_y + 1) << " (" << std::fixed << std::setprecision(1) << ratio[component_y] * 100.0
            << "%)";
    plot.title("Principal Component Analysis").x_label(x_label.str()).y_label(y_label.str());
    return plot;
}

plot::RPlot PCA::plot_scores(const std::vector<std::string>& group_labels, const std::size_t component_x,
                             const std::size_t component_y) const {
    if (component_x >= num_components() || component_y >= num_components())
        throw std::out_of_range("PCA::plot_scores component index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument("PCA::plot_scores: group_labels must have one entry per fitted observation");

    auto plot = plot::RPlot::create();
    const auto groups = unique_sorted(group_labels);
    for (std::size_t g = 0; g < groups.size(); ++g) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < observations_; ++i) {
            if (group_labels[i] != groups[g]) continue;
            xs.push_back(scores_(i, component_x));
            ys.push_back(scores_(i, component_y));
        }
        plot.points(xs, ys, groups[g], kSeriesColors[g % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
    }

    const auto ratio = explained_variance_ratio();
    std::ostringstream x_label, y_label;
    x_label << "PC" << (component_x + 1) << " (" << std::fixed << std::setprecision(1) << ratio[component_x] * 100.0
            << "%)";
    y_label << "PC" << (component_y + 1) << " (" << std::fixed << std::setprecision(1) << ratio[component_y] * 100.0
            << "%)";
    plot.title("Principal Component Analysis").x_label(x_label.str()).y_label(y_label.str());
    return plot;
}

plot::RPlot PCA::plot_scree() const {
    const auto ratio = explained_variance_ratio();
    std::vector<double> percentages;
    std::vector<std::string> names;
    for (std::size_t c = 0; c < ratio.size(); ++c) {
        percentages.push_back(ratio[c] * 100.0);
        names.push_back("PC" + std::to_string(c + 1));
    }
    auto plot = plot::RPlot::barplot(percentages, names, "", kSeriesColors[0]);
    plot.title("Scree Plot").x_label("Component").y_label("% Variance Explained");
    return plot;
}

} // namespace datamunge::stats
