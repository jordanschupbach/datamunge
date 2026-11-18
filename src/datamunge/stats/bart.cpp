/// @file bart.cpp
/// @brief Grow/prune Bayesian backfitting and Gaussian BART prediction.

#include <datamunge/stats/bart.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Private BART proposal, leaf update and prediction machinery.
namespace {

/// @brief Rejects an invalid size, sampler setting or prior parameter.
/// @param options User-supplied BART settings.
/// @throws std::invalid_argument For invalid settings.
void validate_options(const BARTOptions& options) {
    if (options.num_trees == 0 || options.draws == 0 || options.min_leaf == 0 ||
        options.max_depth > 12 || options.max_cutpoints == 0 ||
        options.warmup > std::numeric_limits<std::size_t>::max() - options.draws ||
        !(options.split_base > 0 && options.split_base < 1) ||
        !(options.split_power >= 0) || !std::isfinite(options.split_power) ||
        !(options.leaf_shrinkage > 0) || !std::isfinite(options.leaf_shrinkage) ||
        !(options.noise_prior_df > 0) || !std::isfinite(options.noise_prior_df) ||
        !(options.noise_prior_quantile > 0 && options.noise_prior_quantile < 1) ||
        (options.fixed_noise_sd && (!std::isfinite(*options.fixed_noise_sd) || *options.fixed_noise_sd <= 0)))
        throw std::invalid_argument("BART: invalid sampler or prior options");
}

/// @brief Validates a finite predictor matrix against the expected feature count.
/// @param x Training or query matrix.
/// @param features Expected positive feature count.
/// @throws std::invalid_argument For an invalid feature count or nonfinite predictors.
void validate_design(const linalg::DenseMatrix<double>& x, std::size_t features) {
    if (features == 0 || x.cols() != features)
        throw std::invalid_argument("BART: invalid predictor feature count");
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < x.cols(); ++j)
            if (!std::isfinite(x(i, j))) throw std::invalid_argument("BART: predictors must be finite");
}

/// @brief Converts a widened calculation into a finite double.
/// @param value Candidate value.
/// @return Finite converted value.
/// @throws std::overflow_error If conversion would yield a nonfinite result.
double finite_double(long double value) {
    constexpr long double max = std::numeric_limits<double>::max();
    if (!(value >= -max && value <= max)) throw std::overflow_error("BART: nonfinite calculation");
    return static_cast<double>(value);
}

/// @brief A split rule indexed into the training cut grid.
struct Rule {
    std::size_t feature; ///< Predictor column.
    std::size_t cut; ///< Cut position in that column's candidate grid.
};

/// @brief Training cut thresholds and a compact bin lookup for each training value.
struct CutGrid {
    std::vector<std::vector<double>> cuts; ///< Strictly increasing thresholds per predictor.
    linalg::DenseMatrix<std::size_t> bins; ///< Number of cut thresholds strictly below each input value.
};

/// @brief Builds sample-derived cut grids and bins with stable split comparisons.
/// @param x Validated training design.
/// @param limit Positive cut grid capacity per predictor.
/// @return Owned cutpoints and indexed training bins.
CutGrid training_grid(const linalg::DenseMatrix<double>& x, std::size_t limit) {
    CutGrid grid{{}, linalg::DenseMatrix<std::size_t>(x.rows(), x.cols())};
    std::vector<double> values;
    for (std::size_t j = 0; j < x.cols(); ++j) {
        values.clear();
        for (std::size_t i = 0; i < x.rows(); ++i) values.push_back(x(i, j));
        std::ranges::sort(values);
        values.erase(std::unique(values.begin(), values.end()), values.end());
        const auto gaps = values.size() - 1;
        const auto count = std::min(gaps, limit);
        auto& cuts = grid.cuts.emplace_back();
        for (std::size_t k = 0; k < count; ++k) {
            const auto gap = count == 1 ? (gaps - 1) / 2
                : static_cast<std::size_t>(static_cast<long double>(k) * (gaps - 1) / (count - 1));
            const auto mid = std::midpoint(values[gap], values[gap + 1]);
            cuts.push_back(mid < values[gap + 1] ? mid : values[gap]);
        }
        for (std::size_t i = 0; i < x.rows(); ++i)
            grid.bins(i, j) = std::ranges::lower_bound(cuts, x(i, j)) - cuts.begin();
    }
    return grid;
}

/// @brief A training node attached to a binary heap index.
struct Node {
    std::vector<std::size_t> rows; ///< All training rows reaching this node.
    double split_probability; ///< Depth prior, or zero when no rule can split this node.
    std::optional<Rule> rule; ///< Split rule when internal, otherwise empty.
    double mean{0}; ///< Gibbs leaf value, unused in internal nodes.
};

/// @brief Sparse binary heap of live tree nodes, rooted at index one.
using Tree = std::map<std::size_t, Node>;

/// @brief Enumerates available splits that preserve minimum child sizes.
/// @param grid Training cut lookup.
/// @param rows Training rows at the node.
/// @param min_leaf Minimum required child size.
/// @param first_only Stop after finding a single feasible split.
/// @return Available feature/cut pairs.
std::vector<Rule> valid_rules(const CutGrid& grid, const std::vector<std::size_t>& rows,
                              std::size_t min_leaf, bool first_only = false) {
    std::vector<Rule> rules;
    if (rows.size() / 2 < min_leaf) return rules;
    std::vector<std::size_t> histogram;
    for (std::size_t j = 0; j < grid.cuts.size(); ++j) {
        const auto count = grid.cuts[j].size();
        if (count == 0) continue;
        histogram.assign(count + 1, 0);
        for (const auto row : rows) ++histogram[grid.bins(row, j)];
        std::size_t left = 0;
        for (std::size_t k = 0; k < count; ++k) {
            left += histogram[k];
            if (left >= min_leaf && rows.size() - left >= min_leaf) {
                rules.push_back({j, k});
                if (first_only) return rules;
            }
        }
    }
    return rules;
}

/// @brief Computes a node's calibrated probability of being internal.
/// @param grid Training cut lookup.
/// @param rows Training rows at the node.
/// @param depth Node depth, with root at zero.
/// @param options Tree structure controls.
/// @return Split probability, including forced termination.
double split_probability(const CutGrid& grid, const std::vector<std::size_t>& rows,
                          std::size_t depth, const BARTOptions& options) {
    if (depth >= options.max_depth || valid_rules(grid, rows, options.min_leaf, true).empty()) return 0;
    return options.split_base * std::pow(1. + depth, -options.split_power);
}

/// @brief Computes the conjugate Gaussian leaf log integral up to common constants.
/// @param node Subset of rows at the leaf.
/// @param residual Partial residual with all other trees removed.
/// @param variance Residual variance.
/// @param leaf_var Prior leaf variance.
/// @return Integrated leaf log likelihood, dropping row-wise terms common to all partitions.
long double log_leaf(const Node& node, const std::vector<double>& residual, double variance, double leaf_var) {
    long double sum = 0;
    for (auto i : node.rows) sum += residual[i];
    const long double ratio = static_cast<long double>(leaf_var) / variance;
    const long double precision_factor = 1 + node.rows.size() * ratio;
    return 0.5L * (sum * sum / variance * (ratio / precision_factor) - std::log(precision_factor));
}

/// @brief Computes the local log posterior and Hastings ratio for a grow.
/// @param parent Existing terminal node to split.
/// @param left Proposed left child.
/// @param right Proposed right child.
/// @param leaves Number of terminal nodes in the pre-grow tree.
/// @param reverse_prunes Number of prunable nodes in the post-grow tree.
/// @param residual Current partial residual.
/// @param variance Residual variance.
/// @param leaf_var Leaf prior variance.
/// @return Grow log acceptance ratio. A matched prune uses its negative.
long double grow_log_ratio(const Node& parent, const Node& left, const Node& right,
                           std::size_t leaves, std::size_t reverse_prunes,
                           const std::vector<double>& residual, double variance, double leaf_var) {
    // Uniform feasible-rule probabilities cancel between prior and proposal.
    return std::log(static_cast<long double>(leaves) / reverse_prunes)
        + std::log(parent.split_probability) - std::log1p(-parent.split_probability)
        + std::log1p(-left.split_probability) + std::log1p(-right.split_probability)
        + log_leaf(left, residual, variance, leaf_var) + log_leaf(right, residual, variance, leaf_var)
        - log_leaf(parent, residual, variance, leaf_var);
}

/// @brief Chooses a grow or prune and performs a reversible structure update.
/// @param tree Mutable current tree.
/// @param grid Training cut lookup.
/// @param residual Partial residual for the current tree.
/// @param variance Gaussian residual variance.
/// @param leaf_var Gaussian prior leaf variance.
/// @param options Tree structure controls.
/// @param rng Local chain engine.
/// @return True if the structure changed.
bool update_structure(Tree& tree, const CutGrid& grid, const std::vector<double>& residual,
                      double variance, double leaf_var, const BARTOptions& options, std::mt19937_64& rng) {
    std::vector<std::size_t> leaves, prunes;
    for (const auto& [id, node] : tree) {
        if (!node.rule) {
            leaves.push_back(id);
        } else if (!tree.at(2 * id).rule && !tree.at(2 * id + 1).rule) {
            prunes.push_back(id);
        }
    }
    if (std::bernoulli_distribution(0.5)(rng)) {
        const auto id = leaves[std::uniform_int_distribution<std::size_t>(0, leaves.size() - 1)(rng)];
        auto& parent = tree.at(id);
        if (parent.split_probability == 0) return false;
        const auto rules = valid_rules(grid, parent.rows, options.min_leaf);
        const auto rule = rules[std::uniform_int_distribution<std::size_t>(0, rules.size() - 1)(rng)];
        Node left{{}, 0, std::nullopt}, right{{}, 0, std::nullopt};
        for (const auto i : parent.rows)
            (grid.bins(i, rule.feature) <= rule.cut ? left.rows : right.rows).push_back(i);
        std::size_t depth = 0;
        for (auto key = id; key != 0; key /= 2) ++depth;
        left.split_probability = split_probability(grid, left.rows, depth, options);
        right.split_probability = split_probability(grid, right.rows, depth, options);
        const auto reverse_prunes = prunes.size() + 1 - (id > 1 && !tree.at(id ^ 1).rule);
        const auto ratio = grow_log_ratio(parent, left, right, leaves.size(), reverse_prunes,
                                          residual, variance, leaf_var);
        if (std::isnan(ratio)) throw std::overflow_error("BART: invalid grow acceptance ratio");
        if (std::log(std::generate_canonical<double, 53>(rng)) >= ratio) return false;
        parent.rule = rule;
        tree.emplace(2 * id, std::move(left));
        tree.emplace(2 * id + 1, std::move(right));
        return true;
    }
    if (prunes.empty()) return false;
    const auto id = prunes[std::uniform_int_distribution<std::size_t>(0, prunes.size() - 1)(rng)];
    auto& parent = tree.at(id);
    const auto ratio = -grow_log_ratio(parent, tree.at(2 * id), tree.at(2 * id + 1),
                                      leaves.size() - 1, prunes.size(), residual, variance, leaf_var);
    if (std::isnan(ratio)) throw std::overflow_error("BART: invalid prune acceptance ratio");
    if (std::log(std::generate_canonical<double, 53>(rng)) >= ratio) return false;
    parent.rule.reset();
    tree.erase(2 * id);
    tree.erase(2 * id + 1);
    return true;
}

/// @brief A compact immutable node used by posterior predictions.
struct DrawNode {
    std::size_t left{0}; ///< Left child index, or zero for a leaf.
    std::size_t right{0}; ///< Right child index.
    std::size_t feature{0}; ///< Split column for an internal node.
    double cut_or_mean{0}; ///< Split threshold or terminal mean.
};

/// @brief Compact preorder representation of one fitted tree.
using DrawTree = std::vector<DrawNode>;

/// @brief List of trees in one retained MCMC sweep.
using DrawForest = std::vector<DrawTree>;

/// @brief Copies a training subtree into compact prediction storage.
/// @param tree Current training topology.
/// @param grid Training cut lookup.
/// @param id Binary heap index of the subtree root.
/// @param out Destination tree.
/// @return Compact index of the copied root node.
std::size_t snapshot(const Tree& tree, const CutGrid& grid, std::size_t id, DrawTree& out) {
    const auto index = out.size();
    out.emplace_back();
    const auto& node = tree.at(id);
    if (node.rule) {
        const auto left = snapshot(tree, grid, 2 * id, out);
        const auto right = snapshot(tree, grid, 2 * id + 1, out);
        const auto [feature, cut] = *node.rule;
        out[index] = {left, right, feature, grid.cuts[feature][cut]};
    } else {
        out[index].cut_or_mean = node.mean;
    }
    return index;
}

/// @brief Evaluates one stored forest at one query row.
/// @param forest Retained regression trees.
/// @param x Validated query matrix.
/// @param row Selected query row.
/// @return Sum of scaled leaf means.
long double forest_prediction(const DrawForest& forest, const linalg::DenseMatrix<double>& x, std::size_t row) {
    long double value = 0;
    for (const auto& tree : forest) {
        std::size_t index = 0;
        while (tree[index].left != 0) {
            const auto& node = tree[index];
            index = x(row, node.feature) <= node.cut_or_mean ? node.left : node.right;
        }
        value += tree[index].cut_or_mean;
    }
    return value;
}

/// @brief Interpolated type-7 quantile of sorted draws.
/// @param values Nonempty sorted draws.
/// @param p Closed-unit-interval probability.
/// @return Draw quantile.
double quantile(const std::vector<double>& values, double p) {
    const auto index = (values.size() - 1) * p;
    const auto lo = static_cast<std::size_t>(index);
    return std::lerp(values[lo], values[std::min(lo + 1, values.size() - 1)], index - lo);
}
} // namespace

/// @brief Owned immutable posterior forests, scaling and residual variation.
struct BART::Impl {
    std::size_t features; ///< Training predictor count.
    long double center; ///< Original response midrange.
    long double scale; ///< Original response range, or one for a constant response.
    std::vector<DrawForest> forests; ///< Posterior ensembles in chain order.
    std::vector<double> noise_sd; ///< Residual SD per draw in response units.
    double acceptance{0}; ///< Fraction of attempted topology updates accepted.

    /// @brief Runs Bayesian backfitting and owns the retained ensembles.
    /// @param x Validated training predictors.
    /// @param y Validated training response.
    /// @param options Validated prior and chain settings.
    Impl(const linalg::DenseMatrix<double>& x, const std::vector<double>& y, const BARTOptions& options)
        : features(x.cols()) {
        const auto [lo, hi] = std::ranges::minmax_element(y);
        center = std::midpoint(static_cast<long double>(*lo), static_cast<long double>(*hi));
        scale = *lo == *hi ? 1.L : static_cast<long double>(*hi) - *lo;
        if (!std::isfinite(scale)) throw std::overflow_error("BART: response range overflows");
        std::vector<double> scaled(y.size()), total(y.size()), residual(y.size());
        long double mean = 0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            scaled[i] = static_cast<double>((y[i] - center) / scale);
            mean += static_cast<long double>(scaled[i]) / y.size();
        }
        double sample_variance = 0;
        for (auto value : scaled) sample_variance += (value - mean) * (value - mean) / (y.size() - 1);
        if (sample_variance == 0) sample_variance = 0.01;

        const auto leaf_sd = 0.5L / options.leaf_shrinkage / std::sqrt(static_cast<long double>(options.num_trees));
        const auto leaf_var = finite_double(leaf_sd * leaf_sd);
        if (!(leaf_var > 0)) throw std::overflow_error("BART: invalid scaled leaf variance");
        double variance = sample_variance, noise_scale = 0;
        if (options.fixed_noise_sd) {
            const auto sd = *options.fixed_noise_sd / scale;
            variance = finite_double(sd * sd);
        } else {
            noise_scale = 0.5 * sample_variance * random::chi_squared_quantile(
                1 - options.noise_prior_quantile, options.noise_prior_df);
        }
        if (!(variance > 0) || (!options.fixed_noise_sd && (!(noise_scale > 0) || !std::isfinite(noise_scale))))
            throw std::overflow_error("BART: invalid scaled noise prior");

        const auto grid = training_grid(x, options.max_cutpoints);
        Node root{std::vector<std::size_t>(x.rows()), 0, std::nullopt};
        std::iota(root.rows.begin(), root.rows.end(), 0);
        root.split_probability = split_probability(grid, root.rows, 0, options);
        const Tree initial{{1, root}};
        std::vector<Tree> trees(options.num_trees, initial);
        const auto sweeps = options.warmup + options.draws;
        std::mt19937_64 rng(options.seed);
        std::normal_distribution<double> normal;
        long double accepted = 0;
        forests.reserve(options.draws);
        noise_sd.reserve(options.draws);
        for (std::size_t iter = 0; iter < sweeps; ++iter) {
            for (auto& tree : trees) {
                for (const auto& [id, node] : tree)
                    if (!node.rule)
                        for (auto row : node.rows) residual[row] = scaled[row] - total[row] + node.mean;
                accepted += update_structure(tree, grid, residual, variance, leaf_var, options, rng);
                for (auto& [id, node] : tree) {
                    if (node.rule) continue;
                    long double sum = 0;
                    for (auto row : node.rows) sum += residual[row];
                    const long double denominator = variance + node.rows.size() * static_cast<long double>(leaf_var);
                    const auto fitted = sum * leaf_var / denominator;
                    const auto sd = std::sqrt(static_cast<long double>(variance) * leaf_var / denominator);
                    node.mean = finite_double(fitted + sd * normal(rng));
                    for (auto row : node.rows) total[row] = scaled[row] - residual[row] + node.mean;
                }
            }
            if (!options.fixed_noise_sd) {
                long double rss = 0;
                for (std::size_t i = 0; i < y.size(); ++i) {
                    const long double error = scaled[i] - total[i];
                    rss += error * error;
                }
                const auto gamma = std::gamma_distribution<double>(0.5 * options.noise_prior_df + 0.5 * y.size())(rng);
                variance = finite_double((noise_scale + rss / 2) / gamma);
                if (!(variance > 0)) throw std::overflow_error("BART: nonpositive sampled noise variance");
            }
            if (iter >= options.warmup) {
                auto& forest = forests.emplace_back();
                forest.reserve(trees.size());
                for (const auto& tree : trees) snapshot(tree, grid, 1, forest.emplace_back());
                noise_sd.push_back(options.fixed_noise_sd
                    ? *options.fixed_noise_sd : finite_double(scale * std::sqrt(variance)));
            }
        }
        acceptance = static_cast<double>(accepted / sweeps / options.num_trees);
    }
};

BART::BART(const linalg::DenseMatrix<double>& x, const std::vector<double>& y, BARTOptions options) {
    validate_options(options);
    validate_design(x, x.cols());
    if (x.rows() < 2 || y.size() != x.rows() || !std::ranges::all_of(y, [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("BART: training response must be finite and match the predictor rows");
    state_ = std::make_shared<Impl>(x, y, options);
}

std::vector<double> BART::predict(const linalg::DenseMatrix<double>& x) const {
    validate_design(x, state_->features);
    std::vector<double> result(x.rows());
    for (std::size_t i = 0; i < x.rows(); ++i) {
        long double total = 0;
        for (const auto& forest : state_->forests) total += forest_prediction(forest, x, i);
        result[i] = finite_double(state_->center + state_->scale * (total / state_->forests.size()));
    }
    return result;
}

linalg::DenseMatrix<double> BART::posterior_draws(const linalg::DenseMatrix<double>& x) const {
    validate_design(x, state_->features);
    if (x.rows() > std::vector<double>{}.max_size() / state_->forests.size())
        throw std::overflow_error("BART: prediction draw matrix too large");
    linalg::DenseMatrix<double> result(state_->forests.size(), x.rows());
    for (std::size_t draw = 0; draw < state_->forests.size(); ++draw)
        for (std::size_t i = 0; i < x.rows(); ++i)
            result(draw, i) = finite_double(state_->center + state_->scale * forest_prediction(state_->forests[draw], x, i));
    return result;
}

BARTPrediction BART::predict_interval(const linalg::DenseMatrix<double>& x, double level,
                                     BARTPredictionTarget target, std::uint64_t seed) const {
    validate_design(x, state_->features);
    if (!(level > 0 && level < 1) ||
        (target != BARTPredictionTarget::MeanResponse && target != BARTPredictionTarget::NewObservation))
        throw std::invalid_argument("BART: invalid interval probability or target");
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal;
    BARTPrediction result;
    std::vector<double> sampled(state_->forests.size());
    const auto tail = (1 - level) / 2;
    for (std::size_t i = 0; i < x.rows(); ++i) {
        long double total = 0;
        for (std::size_t d = 0; d < sampled.size(); ++d) {
            const auto latent = forest_prediction(state_->forests[d], x, i);
            total += latent;
            const long double noise = target == BARTPredictionTarget::NewObservation
                ? static_cast<long double>(state_->noise_sd[d]) * normal(rng) : 0.L;
            sampled[d] = finite_double(state_->center + state_->scale * latent + noise);
        }
        result.mean.push_back(finite_double(state_->center + state_->scale * (total / sampled.size())));
        std::ranges::sort(sampled);
        result.lower.push_back(quantile(sampled, tail));
        result.upper.push_back(quantile(sampled, 1 - tail));
    }
    return result;
}

const std::vector<double>& BART::noise_sd_draws() const noexcept { return state_->noise_sd; }

double BART::structure_acceptance_rate() const noexcept { return state_->acceptance; }

} // namespace datamunge::stats
