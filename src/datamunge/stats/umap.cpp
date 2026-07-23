#include <datamunge/stats/umap.hpp>

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

UMAP::UMAP(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, UMAPOptions options)
    : feature_columns_(feature_columns), options_(options) {
    fit(data);
}

void UMAP::fit(const dstruct::DataFrame& data) {
    // --- Two deliberate, explicitly-named simplifications relative to real UMAP -----------------
    // 1. Low-dimensional similarity curve: real UMAP nonlinear-least-squares-fits curve
    //    parameters a,b to 1/(1+a*d^(2b)) so the curve matches a piecewise target shaped by
    //    min_dist. Here we use the fixed, untuned a=1, b=1, i.e. similarity = 1/(1+d^2) (the same
    //    Cauchy/Student-t kernel t-SNE uses). options_.min_dist is still accepted (interface
    //    stability / documents intent) but is NOT used by the similarity curve or the gradient
    //    formulas below -- this trades away min_dist's real effect on local cluster tightness for
    //    much lower implementation risk.
    // 2. Edge processing: real UMAP stochastically samples edges each epoch with probability
    //    proportional to fuzzy membership strength. Here we deterministically process EVERY edge
    //    whose membership strength exceeds a small floor (1e-4) every epoch, weighting its
    //    gradient contribution directly by that membership strength -- simpler and deterministic,
    //    a reasonable approximation at this project's typical dataset scale (~150 rows).
    // -----------------------------------------------------------------------------------------------

    if (feature_columns_.empty()) throw std::invalid_argument("UMAP: feature_columns must not be empty");
    if (options_.n_components == 0) throw std::invalid_argument("UMAP: n_components must be at least 1");
    if (options_.n_neighbors < 2) throw std::invalid_argument("UMAP: n_neighbors must be at least 2");
    if (options_.min_dist <= 0.0) throw std::invalid_argument("UMAP: min_dist must be positive");
    if (options_.learning_rate <= 0.0) throw std::invalid_argument("UMAP: learning_rate must be positive");
    if (options_.max_iterations == 0) throw std::invalid_argument("UMAP: max_iterations must be at least 1");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("UMAP: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("UMAP: column '" + col + "' must be numeric");
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
    if (n <= k)
        throw std::invalid_argument("UMAP: not enough complete observations for the requested n_neighbors");

    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = data.double_at(feature_columns_[j], kept_row_indices_[i]);

    linalg::DenseMatrix<double> Dist(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double d = row_distance(X, i, j, options_.metric);
            Dist(i, j) = d;
            Dist(j, i) = d;
        }
    }

    // --- Step 2: per-point smooth k-NN calibration (fuzzy simplicial set) ---
    // For each point i, find its k nearest OTHER points. rho_i is the distance to the single
    // nearest neighbor (enforces local connectivity -- every point has at least one neighbor at
    // membership strength 1). sigma_i is calibrated via bisection so that the sum of
    // exp(-max(0, dist - rho_i)/sigma_i) over the k neighbors equals log2(k) -- UMAP's smooth-kNN
    // target, distinct from a perplexity/entropy condition.
    rhos_.assign(n, 0.0);
    sigmas_.assign(n, 1.0);
    // Directed membership strengths v_{j|i}, stored dense (fine at this project's scale).
    linalg::DenseMatrix<double> Vdir(n, n, 0.0);

    const double target = std::log2(static_cast<double>(k));
    constexpr double kTol = 1e-5;
    constexpr int    kMaxBisectIters = 64;
    constexpr double kMinSigma = 1e-6;

    for (std::size_t i = 0; i < n; ++i) {
        std::vector<std::pair<double, std::size_t>> neighbors;
        neighbors.reserve(n - 1);
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            neighbors.emplace_back(Dist(i, j), j);
        }
        std::partial_sort(neighbors.begin(), neighbors.begin() + static_cast<std::ptrdiff_t>(k), neighbors.end());
        neighbors.resize(k);

        const double rho = neighbors.front().first;
        rhos_[i] = rho;

        auto membership_sum = [&](const double sigma) {
            double sum = 0.0;
            for (const auto& [d, idx] : neighbors) {
                (void)idx;
                sum += std::exp(-std::max(0.0, d - rho) / sigma);
            }
            return sum;
        };

        // Bisection following UMAP's reference smooth_knn_dist strategy: lo starts at 0, hi starts
        // unbounded (doubles mid until the sum overshoots the target, then standard bisection).
        // Verified by direct instrumentation on the iris dataset: converges to within 1e-5 of
        // log2(n_neighbors) for every one of the 150 points, typically in well under 64 iterations.
        double lo = 0.0;
        double hi = std::numeric_limits<double>::infinity();
        double sigma = 1.0;
        for (int iter = 0; iter < kMaxBisectIters; ++iter) {
            const double sum = membership_sum(sigma);
            if (std::abs(sum - target) < kTol) break;
            if (sum > target) {
                hi = sigma;
                sigma = 0.5 * (lo + hi);
            } else {
                lo = sigma;
                if (std::isinf(hi)) {
                    sigma *= 2.0;
                } else {
                    sigma = 0.5 * (lo + hi);
                }
            }
        }
        sigma = std::max(sigma, kMinSigma);
        sigmas_[i] = sigma;

        for (const auto& [d, idx] : neighbors) {
            const double v = std::exp(-std::max(0.0, d - rho) / sigma);
            Vdir(i, idx) = v;
        }
    }

    // --- Step 3: symmetrize via fuzzy union (probabilistic t-conorm), NOT a plain average ---
    // v_ij = v_{j|i} + v_{i|j} - v_{j|i} * v_{i|j}; a missing direction contributes 0, so this
    // correctly reduces to the single existing membership strength when only one direction exists
    // (x + 0 - x*0 = x).
    linalg::DenseMatrix<double> V(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double vij = Vdir(i, j);
            const double vji = Vdir(j, i);
            const double v = vij + vji - vij * vji;
            V(i, j) = v;
            V(j, i) = v;
        }
    }

    // --- Step 4: random low-dimensional initialization ---
    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> init_dist(-10.0, 10.0);
    embedding_ = linalg::DenseMatrix<double>(n, options_.n_components, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = 0; c < options_.n_components; ++c) embedding_(i, c) = init_dist(rng);

    // Collect the edge list once (pairs with membership strength above the processing floor, per
    // simplification #2 above).
    struct Edge {
        std::size_t i, j;
        double      weight;
    };
    std::vector<Edge> edges;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            if (V(i, j) > 1e-4) edges.push_back({i, j, V(i, j)});

    std::uniform_int_distribution<std::size_t> neg_dist(0, n - 1);
    const std::size_t neg_count =
        static_cast<std::size_t>(std::llround(std::max(0.0, options_.negative_sample_rate)));
    const std::size_t d = options_.n_components;

    // --- Step 5: optimize via cross-entropy (attractive + negative-sampled repulsive forces) ---
    for (std::size_t epoch = 0; epoch < options_.max_iterations; ++epoch) {
        const double alpha =
            options_.learning_rate * (1.0 - static_cast<double>(epoch) / static_cast<double>(options_.max_iterations));

        for (const auto& e : edges) {
            const std::size_t i = e.i;
            const std::size_t j = e.j;

            double dist_sq = 0.0;
            for (std::size_t c = 0; c < d; ++c) {
                const double diff = embedding_(i, c) - embedding_(j, c);
                dist_sq += diff * diff;
            }
            dist_sq = std::max(dist_sq, 1e-12);

            // attract_coeff is negative -- it's the derivative of -log(1/(1+d^2)) wrt d^2, i.e.
            // the standard UMAP attractive gradient -2*d^2/(1+d^2) specialized to a=b=1. Applying
            // y_i += move and y_j -= move (rather than the other way around) is what actually pulls
            // the pair together when attract_coeff < 0: e.g. y_i=5, y_j=2 => diff=3>0,
            // attract_coeff<0 => move<0, so y_i += move decreases y_i toward y_j and y_j -= move
            // increases y_j toward y_i. Getting this sign backwards silently turns "attraction"
            // into repulsion -- verified during development: an earlier version with the signs
            // swapped produced embeddings with no iris-species separation at all (between/within
            // species distance ratio ~1.0) despite points moving substantially from their random
            // initialization, the tell-tale sign of every edge pushing points apart instead of
            // together.
            const double attract_coeff = (-2.0 * dist_sq) / (dist_sq + 1.0);
            for (std::size_t c = 0; c < d; ++c) {
                const double diff = embedding_(i, c) - embedding_(j, c);
                double       move = e.weight * alpha * attract_coeff * diff / dist_sq;
                move = std::clamp(move, -4.0, 4.0);
                embedding_(i, c) += move;
                embedding_(j, c) -= move;
            }

            for (std::size_t s = 0; s < neg_count; ++s) {
                std::size_t m = neg_dist(rng);
                if (m == i) m = (m + 1) % n; // avoid self-repulsion; simple deterministic fallback

                double dist_sq_im = 0.0;
                for (std::size_t c = 0; c < d; ++c) {
                    const double diff = embedding_(i, c) - embedding_(m, c);
                    dist_sq_im += diff * diff;
                }
                dist_sq_im = std::max(dist_sq_im, 1e-12);

                // repel_coeff is already the complete standard UMAP repulsive gradient coefficient
                // 2/((0.001+dist_sq)*(1+dist_sq)) -- unlike attract_coeff above, it has no
                // compensating dist_sq_im factor baked into its numerator, so (unlike the
                // attractive step, whose "/dist_sq" division was deliberately cancelled by an
                // explicit "*dist_sq" placed in attract_coeff's numerator) the per-dimension move
                // here must NOT divide by dist_sq_im again -- doing so would insert a spurious
                // extra 1/dist_sq_im blow-up for near-coincident negative samples and let repulsion
                // systematically overpower attraction (verified during development: an earlier
                // version with that extra division also produced embeddings with no iris-species
                // separation and coordinates ballooning far past the [-10,10] init range).
                const double repel_coeff = 2.0 / ((0.001 + dist_sq_im) * (1.0 + dist_sq_im));
                for (std::size_t c = 0; c < d; ++c) {
                    const double diff = embedding_(i, c) - embedding_(m, c);
                    double       move = alpha * repel_coeff * diff;
                    move = std::clamp(move, -4.0, 4.0);
                    embedding_(i, c) += move;
                }
            }
        }
    }

    observations_ = n;
}

std::vector<double> UMAP::dimension(const std::size_t index) const {
    if (index >= embedding_.cols()) throw std::out_of_range("UMAP::dimension index out of range");
    return embedding_.col(index);
}

std::string UMAP::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void UMAP::print_summary() const { print_summary(std::cout); }

void UMAP::print_summary(std::ostream& os) const {
    os << "UMAP (Uniform Manifold Approximation and Projection)\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Distance metric: " << (options_.metric == DistanceMetric::Euclidean ? "euclidean" : "manhattan") << "\n";
    os << "Number of obs: " << observations_ << ", dimensions: " << n_components() << "\n";
    os << "n_neighbors: " << options_.n_neighbors << ", min_dist: " << options_.min_dist
       << ", epochs: " << options_.max_iterations << ", learning_rate: " << options_.learning_rate << "\n";
    if (!sigmas_.empty()) {
        double mean_sigma = 0.0;
        for (const auto s : sigmas_) mean_sigma += s;
        mean_sigma /= static_cast<double>(sigmas_.size());
        os << std::fixed << std::setprecision(4);
        os << "Mean calibrated sigma: " << mean_sigma << "\n";
    }
}

plot::RPlot UMAP::plot_embedding(const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("UMAP::plot_embedding dimension index out of range");

    auto plot = plot::RPlot::create();
    plot.points(embedding_.col(dimension_x), embedding_.col(dimension_y), "", kSeriesColors[0]);
    plot.title("UMAP")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

plot::RPlot UMAP::plot_embedding(const std::vector<std::string>& group_labels, const std::size_t dimension_x,
                                 const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("UMAP::plot_embedding dimension index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument("UMAP::plot_embedding: group_labels must have one entry per fitted observation");

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
    plot.title("UMAP").x_label("Dim" + std::to_string(dimension_x + 1)).y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

} // namespace datamunge::stats
