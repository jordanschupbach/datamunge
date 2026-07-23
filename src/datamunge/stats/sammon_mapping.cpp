#include <datamunge/stats/sammon_mapping.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

namespace {

const plot::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

constexpr double kZeroFloor = 1e-10;
constexpr double kHessianFloor = 1e-12;

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

double embedded_distance(const linalg::DenseMatrix<double>& Y, const std::size_t i, const std::size_t j) {
    double sum = 0.0;
    for (std::size_t k = 0; k < Y.cols(); ++k) {
        const double d = Y(i, k) - Y(j, k);
        sum += d * d;
    }
    return std::sqrt(sum);
}

double sammon_stress(const linalg::DenseMatrix<double>& Dstar, const linalg::DenseMatrix<double>& Y,
                     const double c) {
    const std::size_t n = Y.rows();
    double e = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double dstar = Dstar(i, j);
            double dij = embedded_distance(Y, i, j);
            if (dij < kZeroFloor) dij = kZeroFloor;
            const double diff = dstar - dij;
            e += (diff * diff) / dstar;
        }
    }
    return e / c;
}

} // namespace

SammonMapping::SammonMapping(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
                             SammonMappingOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void SammonMapping::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("SammonMapping: feature_columns must not be empty");
    if (options_.n_components == 0) throw std::invalid_argument("SammonMapping: n_components must be at least 1");
    if (options_.learning_rate <= 0.0) throw std::invalid_argument("SammonMapping: learning_rate must be positive");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("SammonMapping: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("SammonMapping: column '" + col + "' must be numeric");
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
        throw std::invalid_argument(
            "SammonMapping: n_components must be less than the number of complete observations");

    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]);

    // Step 1: full pairwise high-dimensional distance matrix, floored to avoid division by zero.
    linalg::DenseMatrix<double> Dstar(n, n, 0.0);
    double c = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double d = row_distance(X, i, j, options_.metric);
            if (d < kZeroFloor) d = kZeroFloor;
            Dstar(i, j) = d;
            Dstar(j, i) = d;
            c += d;
        }
    }

    // Step 2: seeded small-jitter initialization.
    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> jitter(0.0, 0.01);
    linalg::DenseMatrix<double> Y(n, options_.n_components, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < options_.n_components; ++k) Y(i, k) = jitter(rng);

    double prev_stress = sammon_stress(Dstar, Y, c);
    std::size_t iterations_run = 0;

    // Step 4: pseudo-Newton sweeps -- all updates for a sweep are computed from a fixed snapshot
    // of Y and applied together at the end of the sweep (same-sweep consistency requirement).
    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        const linalg::DenseMatrix<double> Y_snapshot = Y;
        linalg::DenseMatrix<double> Y_next = Y_snapshot;

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < options_.n_components; ++k) {
                double grad = 0.0;
                double hess = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    if (j == i) continue;
                    double dij = embedded_distance(Y_snapshot, i, j);
                    if (dij < kZeroFloor) dij = kZeroFloor;
                    const double dstar = Dstar(i, j);
                    const double diff = dstar - dij;
                    const double delta_k = Y_snapshot(i, k) - Y_snapshot(j, k);
                    grad += (diff / (dstar * dij)) * delta_k;
                    hess += (1.0 / (dstar * dij)) * (diff - ((delta_k * delta_k) / dij) * (1.0 + diff / dij));
                }
                grad *= -2.0 / c;
                hess *= -2.0 / c;

                if (std::abs(hess) < kHessianFloor) continue; // near-zero Hessian guard: skip this coordinate

                const double step = options_.learning_rate * grad / std::abs(hess);
                Y_next(i, k) = Y_snapshot(i, k) - step;
            }
        }

        Y = Y_next;
        ++iterations_run;

        const double new_stress = sammon_stress(Dstar, Y, c);
        const bool decreasing = new_stress < prev_stress;
        const double relative_improvement =
            prev_stress > 0.0 ? (prev_stress - new_stress) / prev_stress : 0.0;
        prev_stress = new_stress;
        if (decreasing && relative_improvement < options_.tolerance) break;
    }

    embedding_ = Y;
    stress_ = prev_stress;
    iterations_run_ = iterations_run;
    observations_ = n;
}

std::vector<double> SammonMapping::dimension(const std::size_t index) const {
    if (index >= embedding_.cols()) throw std::out_of_range("SammonMapping::dimension index out of range");
    return embedding_.col(index);
}

std::string SammonMapping::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void SammonMapping::print_summary() const { print_summary(std::cout); }

void SammonMapping::print_summary(std::ostream& os) const {
    os << "Sammon Mapping\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Distance metric: " << (options_.metric == DistanceMetric::Euclidean ? "euclidean" : "manhattan") << "\n";
    os << "Number of obs: " << observations_ << ", dimensions: " << n_components() << "\n";
    os << "Iterations run: " << iterations_run_ << " / " << options_.max_iterations << "\n";
    os << std::fixed << std::setprecision(6);
    os << "Final stress: " << stress_ << "\n";
}

plot::RPlot SammonMapping::plot_embedding(const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("SammonMapping::plot_embedding dimension index out of range");

    auto plot = plot::RPlot::create();
    plot.points(embedding_.col(dimension_x), embedding_.col(dimension_y), "", kSeriesColors[0]);
    plot.title("Sammon Mapping")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

plot::RPlot SammonMapping::plot_embedding(const std::vector<std::string>& group_labels, const std::size_t dimension_x,
                                          const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("SammonMapping::plot_embedding dimension index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument(
            "SammonMapping::plot_embedding: group_labels must have one entry per fitted observation");

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
    plot.title("Sammon Mapping")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

} // namespace datamunge::stats
