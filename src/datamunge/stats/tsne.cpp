#include <datamunge/stats/tsne.hpp>

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

TSNE::TSNE(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, TSNEOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void TSNE::fit(const dstruct::DataFrame& data) {
    if (feature_columns_.empty()) throw std::invalid_argument("TSNE: feature_columns must not be empty");
    if (options_.n_components == 0) throw std::invalid_argument("TSNE: n_components must be at least 1");
    if (!(options_.perplexity > 0.0)) throw std::invalid_argument("TSNE: perplexity must be positive");
    if (!(options_.learning_rate > 0.0)) throw std::invalid_argument("TSNE: learning_rate must be positive");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("TSNE: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("TSNE: column '" + col + "' must be numeric");
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
    if (n < 2) throw std::invalid_argument("TSNE: fewer than 2 complete observations");
    if (!(options_.perplexity < static_cast<double>(n)))
        throw std::invalid_argument("TSNE: perplexity must be less than the number of complete observations");

    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]);

    // Pairwise squared distances.
    linalg::DenseMatrix<double> D2(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double d = row_distance(X, i, j, options_.metric);
            D2(i, j) = d * d;
            D2(j, i) = d * d;
        }
    }

    // Per-point binary search on beta_i = 1/(2*sigma_i^2) to hit the target perplexity.
    const double target_entropy = std::log2(options_.perplexity);
    const double tol = 1e-5;
    constexpr int kMaxBinarySearchSteps = 50;

    linalg::DenseMatrix<double> p_cond(n, n, 0.0); // p_cond(i, j) = p_{j|i}
    achieved_perplexity_.assign(n, 0.0);

    std::vector<double> unnorm(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double beta_min = -std::numeric_limits<double>::infinity();
        double beta_max = std::numeric_limits<double>::infinity();
        double beta = 1.0;
        double sum_p = 0.0;
        double H = 0.0;

        for (int step = 0; step < kMaxBinarySearchSteps; ++step) {
            sum_p = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                if (j == i) {
                    unnorm[j] = 0.0;
                    continue;
                }
                unnorm[j] = std::exp(-beta * D2(i, j));
                sum_p += unnorm[j];
            }

            if (sum_p <= 0.0 || !std::isfinite(sum_p)) {
                // Guard: beta is far too large (all weights underflowed to 0), treat as
                // zero-entropy (fully peaked) distribution rather than dividing by zero.
                H = 0.0;
            } else {
                double entropy = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    if (j == i) continue;
                    const double pj = unnorm[j] / sum_p;
                    if (pj > 1e-12) entropy -= pj * std::log2(pj);
                }
                H = entropy;
            }

            const double diff = H - target_entropy;
            if (std::abs(diff) < tol) break;

            if (diff > 0.0) {
                // Entropy (perplexity) too high -> distribution too spread out -> increase beta.
                beta_min = beta;
                beta = std::isinf(beta_max) ? beta * 2.0 : (beta + beta_max) / 2.0;
            } else {
                // Entropy too low -> increase spread -> decrease beta.
                beta_max = beta;
                beta = std::isinf(beta_min) ? beta / 2.0 : (beta + beta_min) / 2.0;
            }
        }

        achieved_perplexity_[i] = std::pow(2.0, H);
        if (sum_p > 0.0 && std::isfinite(sum_p)) {
            for (std::size_t j = 0; j < n; ++j) {
                if (j == i) continue;
                p_cond(i, j) = unnorm[j] / sum_p;
            }
        }
        // else: sum_p degenerate (all-zero row); leave p_cond(i, .) as zeros for this point.
    }

    // Symmetrize into the joint distribution P.
    linalg::DenseMatrix<double> P(n, n, 0.0);
    const double two_n = 2.0 * static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (i != j) P(i, j) = (p_cond(i, j) + p_cond(j, i)) / two_n;

    // Initialize the low-dimensional embedding from N(0, 1e-4).
    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> init_dist(0.0, 0.01); // stddev = sqrt(1e-4)

    const std::size_t k = options_.n_components;
    linalg::DenseMatrix<double> Y(n, k, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = 0; c < k; ++c) Y(i, c) = init_dist(rng);

    linalg::DenseMatrix<double> dY(n, k, 0.0);
    linalg::DenseMatrix<double> num(n, n, 0.0);
    linalg::DenseMatrix<double> grad(n, k, 0.0);

    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        // Low-dimensional Student-t affinities.
        double sum_num = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            num(i, i) = 0.0;
            for (std::size_t j = i + 1; j < n; ++j) {
                double dist_sq = 0.0;
                for (std::size_t c = 0; c < k; ++c) {
                    const double d = Y(i, c) - Y(j, c);
                    dist_sq += d * d;
                }
                const double value = 1.0 / (1.0 + dist_sq);
                num(i, j) = value;
                num(j, i) = value;
                sum_num += 2.0 * value;
            }
        }
        // Guard against a degenerate (zero) normalizer -- cannot happen for n >= 2 since every
        // num(i, j) > 0, but stay defensive against a divide-by-zero/NaN cascade regardless.
        const bool exaggerate = iteration < options_.early_exaggeration_iterations;
        const double exaggeration = exaggerate ? options_.early_exaggeration : 1.0;

        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t c = 0; c < k; ++c) grad(i, c) = 0.0;

        if (sum_num > 0.0) {
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    if (i == j) continue;
                    const double q_ij = num(i, j) / sum_num;
                    const double p_ij = exaggeration * P(i, j);
                    const double coeff = 4.0 * (p_ij - q_ij) * num(i, j);
                    for (std::size_t c = 0; c < k; ++c) grad(i, c) += coeff * (Y(i, c) - Y(j, c));
                }
            }
        }

        const double momentum =
            iteration < options_.momentum_switch_iteration ? options_.initial_momentum : options_.final_momentum;

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t c = 0; c < k; ++c) {
                dY(i, c) = momentum * dY(i, c) - options_.learning_rate * grad(i, c);
                Y(i, c) += dY(i, c);
            }
        }

        // Re-center each column.
        for (std::size_t c = 0; c < k; ++c) {
            double mean = 0.0;
            for (std::size_t i = 0; i < n; ++i) mean += Y(i, c);
            mean /= static_cast<double>(n);
            for (std::size_t i = 0; i < n; ++i) Y(i, c) -= mean;
        }
    }

    embedding_ = std::move(Y);
    observations_ = n;
}

std::vector<double> TSNE::dimension(const std::size_t index) const {
    if (index >= embedding_.cols()) throw std::out_of_range("TSNE::dimension index out of range");
    return embedding_.col(index);
}

std::string TSNE::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void TSNE::print_summary() const { print_summary(std::cout); }

void TSNE::print_summary(std::ostream& os) const {
    os << "t-Distributed Stochastic Neighbor Embedding\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Distance metric: " << (options_.metric == DistanceMetric::Euclidean ? "euclidean" : "manhattan") << "\n";
    os << "Number of obs: " << observations_ << ", dimensions: " << n_components() << "\n";
    os << "Target perplexity: " << options_.perplexity << ", iterations: " << options_.max_iterations << "\n";

    if (!achieved_perplexity_.empty()) {
        double mean_perp = 0.0;
        for (const auto v : achieved_perplexity_) mean_perp += v;
        mean_perp /= static_cast<double>(achieved_perplexity_.size());
        os << std::fixed << std::setprecision(4);
        os << "Mean achieved perplexity: " << mean_perp << "\n";
    }
}

plot::RPlot TSNE::plot_embedding(const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("TSNE::plot_embedding dimension index out of range");

    auto plot = plot::RPlot::create();
    plot.points(embedding_.col(dimension_x), embedding_.col(dimension_y), "", kSeriesColors[0]);
    plot.title("t-SNE Embedding")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

plot::RPlot TSNE::plot_embedding(const std::vector<std::string>& group_labels, const std::size_t dimension_x,
                                 const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("TSNE::plot_embedding dimension index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument("TSNE::plot_embedding: group_labels must have one entry per fitted observation");

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
    plot.title("t-SNE Embedding")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

} // namespace datamunge::stats
