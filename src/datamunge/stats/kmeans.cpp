#include <datamunge/stats/kmeans.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

namespace {

double squared_distance(const linalg::DenseMatrix<double>& X, const std::size_t row,
                         const linalg::DenseMatrix<double>& centers, const std::size_t center) {
    double s = 0.0;
    for (std::size_t j = 0; j < X.cols(); ++j) {
        const double d = X(row, j) - centers(center, j);
        s += d * d;
    }
    return s;
}

std::size_t nearest_center(const linalg::DenseMatrix<double>& X, const std::size_t row,
                            const linalg::DenseMatrix<double>& centers, double& out_dist_sq) {
    std::size_t best = 0;
    double best_dist = squared_distance(X, row, centers, 0);
    for (std::size_t c = 1; c < centers.rows(); ++c) {
        const double d = squared_distance(X, row, centers, c);
        if (d < best_dist) {
            best_dist = d;
            best = c;
        }
    }
    out_dist_sq = best_dist;
    return best;
}

linalg::DenseMatrix<double> kmeans_plus_plus_init(const linalg::DenseMatrix<double>& X, const std::size_t k,
                                                   std::mt19937_64& rng) {
    const std::size_t n = X.rows(), d = X.cols();
    linalg::DenseMatrix<double> centers(k, d, 0.0);
    std::uniform_int_distribution<std::size_t> first_dist(0, n - 1);
    const std::size_t first = first_dist(rng);
    for (std::size_t j = 0; j < d; ++j) centers(0, j) = X(first, j);

    std::vector<double> min_dist_sq(n, std::numeric_limits<double>::max());
    for (std::size_t c = 1; c < k; ++c) {
        for (std::size_t i = 0; i < n; ++i) {
            double d2 = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                const double diff = X(i, j) - centers(c - 1, j);
                d2 += diff * diff;
            }
            if (d2 < min_dist_sq[i]) min_dist_sq[i] = d2;
        }
        double total = 0.0;
        for (const double v : min_dist_sq) total += v;
        if (total <= 0.0) {
            std::uniform_int_distribution<std::size_t> fallback(0, n - 1);
            const std::size_t next = fallback(rng);
            for (std::size_t j = 0; j < d; ++j) centers(c, j) = X(next, j);
            continue;
        }
        std::uniform_real_distribution<double> unif(0.0, total);
        double target = unif(rng);
        std::size_t chosen = n - 1;
        for (std::size_t i = 0; i < n; ++i) {
            if (target <= min_dist_sq[i]) {
                chosen = i;
                break;
            }
            target -= min_dist_sq[i];
        }
        for (std::size_t j = 0; j < d; ++j) centers(c, j) = X(chosen, j);
    }
    return centers;
}

struct LloydResult {
    linalg::DenseMatrix<double> centers;
    std::vector<std::size_t> labels;
    double inertia{0.0};
    std::size_t iterations{0};
};

LloydResult run_lloyd(const linalg::DenseMatrix<double>& X, linalg::DenseMatrix<double> centers,
                       const std::size_t max_iterations, const double tolerance) {
    const std::size_t n = X.rows(), d = X.cols(), k = centers.rows();
    std::vector<std::size_t> labels(n, 0);
    std::size_t iter = 0;
    for (; iter < max_iterations; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            double dist_sq = 0.0;
            labels[i] = nearest_center(X, i, centers, dist_sq);
        }

        linalg::DenseMatrix<double> new_centers(k, d, 0.0);
        std::vector<std::size_t> counts(k, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t c = labels[i];
            ++counts[c];
            for (std::size_t j = 0; j < d; ++j) new_centers(c, j) += X(i, j);
        }
        for (std::size_t c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                std::size_t farthest = 0;
                double farthest_dist = -1.0;
                for (std::size_t i = 0; i < n; ++i) {
                    double dist_sq = 0.0;
                    nearest_center(X, i, centers, dist_sq);
                    if (dist_sq > farthest_dist) {
                        farthest_dist = dist_sq;
                        farthest = i;
                    }
                }
                for (std::size_t j = 0; j < d; ++j) new_centers(c, j) = X(farthest, j);
            } else {
                for (std::size_t j = 0; j < d; ++j) new_centers(c, j) /= static_cast<double>(counts[c]);
            }
        }

        double max_shift = 0.0;
        for (std::size_t c = 0; c < k; ++c) {
            double shift_sq = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                const double diff = new_centers(c, j) - centers(c, j);
                shift_sq += diff * diff;
            }
            max_shift = std::max(max_shift, std::sqrt(shift_sq));
        }
        centers = new_centers;
        if (max_shift < tolerance) {
            ++iter;
            break;
        }
    }

    double inertia = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double dist_sq = 0.0;
        labels[i] = nearest_center(X, i, centers, dist_sq);
        inertia += dist_sq;
    }
    return LloydResult{std::move(centers), std::move(labels), inertia, iter};
}

} // namespace

KMeans::KMeans(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, KMeansOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void KMeans::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("KMeans: feature_columns must not be empty");
    if (options_.n_clusters == 0) throw std::invalid_argument("KMeans: n_clusters must be at least 1");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("KMeans: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("KMeans: column '" + col + "' must be numeric");
    }

    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < data.nrows(); ++i) {
        bool ok = true;
        for (const auto& col : feature_columns_) {
            if (data.is_null(col, i)) {
                ok = false;
                break;
            }
        }
        if (ok) rows.push_back(i);
    }
    const std::size_t n = rows.size();
    const std::size_t d = feature_columns_.size();
    if (n < options_.n_clusters) throw std::invalid_argument("KMeans: fewer complete observations than n_clusters");

    linalg::DenseMatrix<double> X(n, d, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j) X(i, j) = data.double_at(feature_columns_[j], rows[i]);

    std::mt19937_64 rng(options_.seed);
    bool have_best = false;
    LloydResult best;
    for (std::size_t init = 0; init < options_.n_init; ++init) {
        auto init_centers = kmeans_plus_plus_init(X, options_.n_clusters, rng);
        auto result = run_lloyd(X, std::move(init_centers), options_.max_iterations, options_.tolerance);
        if (!have_best || result.inertia < best.inertia) {
            best = std::move(result);
            have_best = true;
        }
    }

    centers_ = std::move(best.centers);
    labels_ = std::move(best.labels);
    inertia_ = best.inertia;
    iterations_used_ = best.iterations;
    observations_ = n;
}

std::vector<std::size_t> KMeans::predict(const dstruct::DataFrame& newdata) const {
    for (const auto& col : feature_columns_)
        if (!newdata.has_column(col)) throw std::invalid_argument("KMeans::predict: column '" + col + "' not found");

    const std::size_t n = newdata.nrows();
    const std::size_t d = feature_columns_.size();
    linalg::DenseMatrix<double> X(n, d, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            if (newdata.is_null(feature_columns_[j], i))
                throw std::invalid_argument("KMeans::predict: null value in feature column '" + feature_columns_[j] + "'");
            X(i, j) = newdata.double_at(feature_columns_[j], i);
        }
    }

    std::vector<std::size_t> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        double dist_sq = 0.0;
        result[i] = nearest_center(X, i, centers_, dist_sq);
    }
    return result;
}

std::string KMeans::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void KMeans::print_summary() const { print_summary(std::cout); }

void KMeans::print_summary(std::ostream& os) const {
    os << "K-means clustering with " << options_.n_clusters << " clusters\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Number of obs: " << observations_ << "\n";
    os << std::fixed << std::setprecision(4);
    os << "Inertia (within-cluster sum of squares): " << inertia_ << "\n";
    os << "Iterations used (best run): " << iterations_used_ << "\n\n";

    std::vector<std::size_t> counts(options_.n_clusters, 0);
    for (const auto l : labels_) ++counts[l];
    os << "Cluster sizes:\n";
    for (std::size_t c = 0; c < options_.n_clusters; ++c) os << "  cluster " << c << ": " << counts[c] << " points\n";

    os << "\nCluster centers:\n";
    for (std::size_t c = 0; c < options_.n_clusters; ++c) {
        os << "  " << c << ": [";
        for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << centers_(c, j);
        os << "]\n";
    }
}

} // namespace datamunge::stats
