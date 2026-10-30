/// @file random_forest_classifier.cpp
/// @brief Random forest classifier: fitting, out-of-bag summaries, importance and plots.

#include <datamunge/stats/random_forest_classifier.hpp>

#include <datamunge/plot/class_regions.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Formatting, colour and argmax helpers for the forest.
namespace {

/// @brief Formats @p v with fixed @p precision ("NaN" for NaN).
/// @param v          Value.
/// @param precision  Digits after the point.
/// @return The text.
std::string format_stat(double v, int precision = 4) {
    if (std::isnan(v)) return "NaN";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

/// @brief Index of the largest entry; ties go to the first.
/// @param v  Values (not empty).
/// @return The index.
std::size_t argmax_index(const std::vector<double>& v) {
    return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

/// @brief Series colours for the plots, one per class.
const graphics::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

} // namespace

RandomForestClassifier::RandomForestClassifier(const dstruct::DataFrame& data, const std::string& formula,
                                               RandomForestClassifierOptions options)
    : formula_(formula) {
    fit(data, std::move(options));
}

void RandomForestClassifier::fit(const dstruct::DataFrame& data, RandomForestClassifierOptions options) {
    options_ = options;
    if (options_.n_trees == 0) throw std::invalid_argument("RandomForestClassifier: n_trees must be at least 1");
    if (options_.sample_fraction <= 0.0)
        throw std::invalid_argument("RandomForestClassifier: sample_fraction must be positive");

    design_ = formula_.resolve(data, ResponseKind::Categorical);
    auto              cd = design_.build_classification_matrix(data);
    const std::size_t n  = cd.X.rows();
    const std::size_t p  = cd.X.cols();
    // Rows with a missing value are dropped; every row index below refers to this frame.
    const auto complete = data.take_rows(cd.used_row_indices);

    predictor_names_ = design_.coefficient_names;
    observations_    = n;
    training_labels_ = cd.labels;
    training_x_      = cd.X;

    classes_.clear();
    for (const auto& label : cd.labels)
        if (std::find(classes_.begin(), classes_.end(), label) == classes_.end()) classes_.push_back(label);
    std::sort(classes_.begin(), classes_.end());
    if (classes_.size() < 2)
        throw std::invalid_argument("RandomForestClassifier: response must have at least 2 distinct classes, found "
                                    + std::to_string(classes_.size()));
    training_class_.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        training_class_[i] = static_cast<std::size_t>(std::lower_bound(classes_.begin(), classes_.end(), cd.labels[i])
                                                      - classes_.begin());

    feature_min_.assign(p, std::numeric_limits<double>::infinity());
    feature_max_.assign(p, -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            feature_min_[j] = std::min(feature_min_[j], cd.X(i, j));
            feature_max_[j] = std::max(feature_max_[j], cd.X(i, j));
        }

    max_features_used_ = options_.max_features;
    if (max_features_used_ == 0)
        max_features_used_ = std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(static_cast<double>(p))));
    max_features_used_ = std::min(max_features_used_, p);

    const auto sample_size = static_cast<std::size_t>(
        std::max(1.0, std::round(options_.sample_fraction * static_cast<double>(n))));

    random::SplitMix64 sampler(options_.seed);

    trees_.clear();
    trees_.reserve(options_.n_trees);
    in_bag_counts_.assign(options_.n_trees, std::vector<int>(n, 0));
    tree_columns_.assign(options_.n_trees, {});
    tree_classes_.assign(options_.n_trees, {});

    DecisionTreeClassifierOptions tree_options;
    tree_options.max_depth          = options_.max_depth;
    tree_options.min_samples_split  = options_.min_samples_split;
    tree_options.min_samples_leaf   = options_.min_samples_leaf;
    tree_options.criterion          = options_.criterion;
    tree_options.max_features       = max_features_used_;

    for (std::size_t t = 0; t < options_.n_trees; ++t) {
        // A bootstrap sample can, in rare cases (small n, imbalanced
        // classes), miss a class entirely -- retry with fresh draws rather
        // than letting the whole forest fail to fit.
        constexpr int kMaxAttempts = 25;
        bool          built        = false;
        for (int attempt = 0; attempt < kMaxAttempts && !built; ++attempt) {
            std::vector<std::size_t> indices;
            indices.reserve(sample_size);
            std::vector<int> counts(n, 0);

            if (options_.bootstrap) {
                for (std::size_t i = 0; i < sample_size; ++i) {
                    const auto idx = static_cast<std::size_t>(sampler.next_u64() % n);
                    indices.push_back(idx);
                    ++counts[idx];
                }
            } else {
                std::vector<std::size_t> all(n);
                std::iota(all.begin(), all.end(), 0);
                const auto take = std::min(sample_size, n);
                for (std::size_t i = 0; i < take; ++i) {
                    const std::size_t j = i + static_cast<std::size_t>(sampler.next_u64() % (n - i));
                    std::swap(all[i], all[j]);
                    indices.push_back(all[i]);
                    ++counts[all[i]];
                }
            }

            const auto bootstrap_frame = complete.take_rows(indices);
            tree_options.random_seed   = options_.seed * 1000003ULL + t * 7919ULL + static_cast<std::uint64_t>(attempt) + 1;

            try {
                trees_.emplace_back(bootstrap_frame, formula_.text(), tree_options);
                in_bag_counts_[t] = std::move(counts);
                built             = true;
            } catch (const std::invalid_argument&) {
                // Likely a single-class bootstrap sample; retry.
            }
        }
        if (!built)
            throw std::runtime_error("RandomForestClassifier: could not draw a bootstrap sample containing at "
                                     "least 2 classes after repeated attempts");

        // A tree fit to a sample that lacks a class or a categorical level has fewer classes or
        // dummy columns than the forest: map its indices onto the forest's.
        const auto& tree = trees_.back();
        for (const auto& name : tree.predictor_names()) {
            const auto it = std::find(predictor_names_.begin(), predictor_names_.end(), name);
            if (it == predictor_names_.end())
                throw std::runtime_error("RandomForestClassifier: tree predictor '" + name + "' is not a forest predictor");
            tree_columns_[t].push_back(static_cast<std::size_t>(it - predictor_names_.begin()));
        }
        for (const auto& label : tree.classes())
            tree_classes_[t].push_back(
                static_cast<std::size_t>(std::lower_bound(classes_.begin(), classes_.end(), label) - classes_.begin()));
    }

    // Every tree's vote on every training row, for the resubstitution and out-of-bag summaries.
    training_votes_.assign(trees_.size(), std::vector<std::size_t>(n, 0));
    std::vector<double> row(p), buffer;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) row[j] = training_x_(i, j);
        for (std::size_t t = 0; t < trees_.size(); ++t) training_votes_[t][i] = tree_vote(t, row, buffer);
    }

    const std::size_t k = classes_.size();
    fitted_classes_.assign(n, "");
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> votes(k, 0.0);
        for (std::size_t t = 0; t < trees_.size(); ++t) votes[training_votes_[t][i]] += 1.0;
        fitted_classes_[i] = classes_[argmax_index(votes)];
    }

    const auto  oob     = oob_vote_counts();
    std::size_t correct = 0, considered = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::accumulate(oob[i].begin(), oob[i].end(), 0.0) == 0.0) continue;
        ++considered;
        if (argmax_index(oob[i]) == training_class_[i]) ++correct;
    }
    oob_accuracy_ = considered > 0 ? static_cast<double>(correct) / static_cast<double>(considered)
                                    : std::numeric_limits<double>::quiet_NaN();
}

const DecisionTreeClassifier& RandomForestClassifier::tree(std::size_t t) const {
    if (t >= trees_.size()) throw std::out_of_range("RandomForestClassifier::tree: index out of range");
    return trees_[t];
}

std::vector<int> RandomForestClassifier::in_bag_counts(std::size_t t) const {
    if (t >= trees_.size()) throw std::out_of_range("RandomForestClassifier::in_bag_counts: index out of range");
    return in_bag_counts_[t];
}

std::size_t RandomForestClassifier::tree_vote(std::size_t t, const std::vector<double>& x,
                                              std::vector<double>& buffer) const {
    const auto& columns = tree_columns_[t];
    buffer.resize(columns.size());
    for (std::size_t j = 0; j < columns.size(); ++j) buffer[j] = x[columns[j]];
    return tree_classes_[t][trees_[t].predict_row(buffer)];
}

std::vector<std::size_t> RandomForestClassifier::tree_votes(const std::vector<double>& x) const {
    if (x.size() != predictor_names_.size())
        throw std::invalid_argument("RandomForestClassifier::tree_votes: expected " + std::to_string(predictor_names_.size())
                                    + " predictor values, got " + std::to_string(x.size()));
    std::vector<std::size_t> votes(trees_.size());
    std::vector<double>      buffer;
    for (std::size_t t = 0; t < trees_.size(); ++t) votes[t] = tree_vote(t, x, buffer);
    return votes;
}

std::vector<double> RandomForestClassifier::vote_share(const std::vector<double>& x) const {
    std::vector<double> share(classes_.size(), 0.0);
    for (const auto v : tree_votes(x)) share[v] += 1.0 / static_cast<double>(trees_.size());
    return share;
}

std::vector<std::vector<double>> RandomForestClassifier::oob_vote_counts() const {
    std::vector<std::vector<double>> counts(observations_, std::vector<double>(classes_.size(), 0.0));
    for (std::size_t t = 0; t < trees_.size(); ++t)
        for (std::size_t i = 0; i < observations_; ++i)
            if (in_bag_counts_[t][i] == 0) counts[i][training_votes_[t][i]] += 1.0;
    return counts;
}

std::vector<int> RandomForestClassifier::oob_tree_counts() const {
    std::vector<int> counts(observations_, 0);
    for (const auto& bag : in_bag_counts_)
        for (std::size_t i = 0; i < observations_; ++i) counts[i] += bag[i] == 0;
    return counts;
}

std::vector<std::vector<double>> RandomForestClassifier::oob_vote_share() const {
    auto counts = oob_vote_counts();
    for (auto& row : counts) {
        const double total = std::accumulate(row.begin(), row.end(), 0.0);
        if (total == 0.0) row.clear();
        else
            for (double& v : row) v /= total;
    }
    return counts;
}

std::vector<std::string> RandomForestClassifier::oob_predictions() const {
    const auto               share = oob_vote_share();
    std::vector<std::string> out(observations_);
    for (std::size_t i = 0; i < observations_; ++i)
        if (!share[i].empty()) out[i] = classes_[argmax_index(share[i])];
    return out;
}

std::vector<double> RandomForestClassifier::oob_margins() const {
    const auto          share = oob_vote_share();
    std::vector<double> margins(observations_, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < observations_; ++i) {
        if (share[i].empty()) continue;
        double other = 0.0;
        for (std::size_t c = 0; c < classes_.size(); ++c)
            if (c != training_class_[i]) other = std::max(other, share[i][c]);
        margins[i] = share[i][training_class_[i]] - other;
    }
    return margins;
}

std::vector<double> RandomForestClassifier::oob_accuracy_path() const {
    std::vector<std::vector<double>> counts(observations_, std::vector<double>(classes_.size(), 0.0));
    std::vector<char>                seen(observations_, 0), right(observations_, 0);
    std::size_t                      n_seen = 0, n_right = 0;
    std::vector<double>              path;
    path.reserve(trees_.size());
    for (std::size_t t = 0; t < trees_.size(); ++t) {
        for (std::size_t i = 0; i < observations_; ++i) {
            if (in_bag_counts_[t][i] != 0) continue;
            counts[i][training_votes_[t][i]] += 1.0;
            if (!seen[i]) seen[i] = 1, ++n_seen;
            const char now = argmax_index(counts[i]) == training_class_[i];
            n_right += static_cast<std::size_t>(now) - static_cast<std::size_t>(right[i]);
            right[i] = now;
        }
        path.push_back(n_seen > 0 ? static_cast<double>(n_right) / static_cast<double>(n_seen)
                                  : std::numeric_limits<double>::quiet_NaN());
    }
    return path;
}

std::vector<double> RandomForestClassifier::oob_tree_accuracies() const {
    std::vector<double> acc(trees_.size(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t t = 0; t < trees_.size(); ++t) {
        std::size_t right = 0, total = 0;
        for (std::size_t i = 0; i < observations_; ++i) {
            if (in_bag_counts_[t][i] != 0) continue;
            ++total;
            right += training_votes_[t][i] == training_class_[i];
        }
        if (total > 0) acc[t] = static_cast<double>(right) / static_cast<double>(total);
    }
    return acc;
}

RandomForestStrength RandomForestClassifier::strength_correlation() const {
    const auto share = oob_vote_share();
    // The strongest wrong class of each row, from its out-of-bag votes.
    std::vector<std::size_t> rival(observations_, 0);
    std::vector<double>      margin;
    std::vector<char>        used(observations_, 0);
    for (std::size_t i = 0; i < observations_; ++i) {
        if (share[i].empty()) continue;
        used[i]     = 1;
        double best = -1.0;
        for (std::size_t c = 0; c < classes_.size(); ++c)
            if (c != training_class_[i] && share[i][c] > best) best = share[i][c], rival[i] = c;
        margin.push_back(share[i][training_class_[i]] - best);
    }
    RandomForestStrength out;
    out.rows = margin.size();
    if (margin.empty()) return out;
    const double m = static_cast<double>(margin.size());
    out.strength   = std::accumulate(margin.begin(), margin.end(), 0.0) / m;
    for (const double v : margin) out.margin_variance += (v - out.strength) * (v - out.strength) / m;

    double      sd_sum = 0.0;
    std::size_t trees  = 0;
    for (std::size_t t = 0; t < trees_.size(); ++t) {
        double      p1 = 0.0, p2 = 0.0;
        std::size_t total = 0;
        for (std::size_t i = 0; i < observations_; ++i) {
            if (in_bag_counts_[t][i] != 0 || !used[i]) continue;
            ++total;
            p1 += training_votes_[t][i] == training_class_[i];
            p2 += training_votes_[t][i] == rival[i];
        }
        if (total == 0) continue;
        p1 /= static_cast<double>(total);
        p2 /= static_cast<double>(total);
        sd_sum += std::sqrt(std::max(0.0, p1 + p2 - (p1 - p2) * (p1 - p2)));
        ++trees;
    }
    out.mean_tree_sd = trees > 0 ? sd_sum / static_cast<double>(trees) : 0.0;
    out.correlation  = out.mean_tree_sd > 0.0 ? out.margin_variance / (out.mean_tree_sd * out.mean_tree_sd)
                                              : std::numeric_limits<double>::quiet_NaN();
    out.error_bound  = out.strength > 0.0 ? out.correlation * (1.0 - out.strength * out.strength)
                                               / (out.strength * out.strength)
                                          : std::numeric_limits<double>::infinity();
    return out;
}

RandomForestPermutationImportance RandomForestClassifier::permutation_importance(std::uint64_t seed) const {
    const std::size_t p = predictor_names_.size();
    RandomForestPermutationImportance out;
    out.predictor_names = predictor_names_;
    out.importance.assign(p, 0.0);
    out.standard_error.assign(p, 0.0);
    random::SplitMix64               rng(seed);
    std::vector<std::vector<double>> drops(p);
    std::vector<double>              row(p), buffer;
    for (std::size_t t = 0; t < trees_.size(); ++t) {
        std::vector<std::size_t> oob;
        for (std::size_t i = 0; i < observations_; ++i)
            if (in_bag_counts_[t][i] == 0) oob.push_back(i);
        if (oob.empty()) continue;
        std::size_t base = 0;
        for (const auto i : oob) base += training_votes_[t][i] == training_class_[i];
        for (std::size_t j = 0; j < p; ++j) {
            std::vector<std::size_t> shuffled = oob; // row whose value of predictor j each OOB row receives
            for (std::size_t a = shuffled.size() - 1; a > 0; --a)
                std::swap(shuffled[a], shuffled[static_cast<std::size_t>(rng.next_u64() % (a + 1))]);
            std::size_t right = 0;
            for (std::size_t r = 0; r < oob.size(); ++r) {
                for (std::size_t c = 0; c < p; ++c) row[c] = training_x_(oob[r], c);
                row[j] = training_x_(shuffled[r], j);
                right += tree_vote(t, row, buffer) == training_class_[oob[r]];
            }
            drops[j].push_back((static_cast<double>(base) - static_cast<double>(right)) / static_cast<double>(oob.size()));
        }
    }
    for (std::size_t j = 0; j < p; ++j) {
        const double b = static_cast<double>(drops[j].size());
        if (drops[j].empty()) continue;
        out.importance[j] = std::accumulate(drops[j].begin(), drops[j].end(), 0.0) / b;
        double ss         = 0.0;
        for (const double d : drops[j]) ss += (d - out.importance[j]) * (d - out.importance[j]);
        out.standard_error[j] = drops[j].size() > 1 ? std::sqrt(ss / (b - 1.0)) / std::sqrt(b) : 0.0;
    }
    return out;
}

std::vector<double> RandomForestClassifier::feature_importance() const {
    std::vector<double>      importance(predictor_names_.size(), 0.0);
    std::vector<std::size_t> counts(predictor_names_.size(), 0);
    for (const auto& tree : trees_) {
        const auto& tree_predictors = tree.predictor_names();
        const auto  tree_importance = tree.feature_importance();
        for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
            const auto it = std::find(tree_predictors.begin(), tree_predictors.end(), predictor_names_[j]);
            if (it == tree_predictors.end()) continue;
            importance[j] += tree_importance[static_cast<std::size_t>(it - tree_predictors.begin())];
            ++counts[j];
        }
    }
    for (std::size_t j = 0; j < importance.size(); ++j)
        if (counts[j] > 0) importance[j] /= static_cast<double>(counts[j]);
    const double sum = std::accumulate(importance.begin(), importance.end(), 0.0);
    if (sum > 0.0)
        for (double& v : importance) v /= sum;
    return importance;
}

double RandomForestClassifier::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> RandomForestClassifier::confusion_matrix() const {
    const std::size_t k = classes_.size();
    linalg::DenseMatrix<double> matrix(k, k, 0.0);
    for (std::size_t i = 0; i < training_labels_.size(); ++i) {
        const auto actual_it    = std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]);
        const auto predicted_it = std::lower_bound(classes_.begin(), classes_.end(), fitted_classes_[i]);
        matrix(static_cast<std::size_t>(actual_it - classes_.begin()),
               static_cast<std::size_t>(predicted_it - classes_.begin())) += 1.0;
    }
    return matrix;
}

std::string RandomForestClassifier::summary() const {
    std::ostringstream out;
    out << "Call:\nrandomForest(formula = " << formula_.text() << ", n_trees = " << options_.n_trees
        << ", max_features = " << max_features_used_
        << ", criterion = " << (options_.criterion == SplitCriterion::Gini ? "gini" : "entropy")
        << ", max_depth = " << options_.max_depth << ")\n\n";

    out << "Number of Classes: " << classes_.size() << "\n";
    out << "Levels: ";
    for (const auto& c : classes_) out << c << " ";
    out << "\n\n";

    out << "Trees: " << trees_.size() << "  (max_features per split: " << max_features_used_ << " of "
        << predictor_names_.size() << ")\n\n";

    out << "Feature importance:\n";
    const auto importance = feature_importance();
    for (std::size_t j = 0; j < predictor_names_.size(); ++j)
        out << "  " << predictor_names_[j] << ": " << format_stat(importance[j], 4) << "\n";

    out << "\nTraining accuracy: " << format_stat(training_accuracy() * 100.0, 2) << "%\n";
    out << "Out-of-bag accuracy: " << format_stat(oob_accuracy_ * 100.0, 2) << "%  (" << observations_
        << " observations)\n";

    return out.str();
}

void RandomForestClassifier::print_summary(std::ostream& os) const { os << summary(); }

void RandomForestClassifier::print_summary() const { print_summary(std::cout); }

std::vector<std::string> RandomForestClassifier::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).class_label;
}

RandomForestClassifierPrediction RandomForestClassifier::predict_detail(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "RandomForestClassifier::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);
    if (dm.X.cols() != predictor_names_.size())
        throw std::logic_error("RandomForestClassifier::predict: design has an unexpected number of columns");

    RandomForestClassifierPrediction result;
    result.class_label.assign(newdata.nrows(), "");
    result.vote_share.assign(newdata.nrows(), {});
    std::vector<double> row(dm.X.cols()), buffer;
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        for (std::size_t j = 0; j < dm.X.cols(); ++j) row[j] = dm.X(i, j);
        std::vector<double> share(classes_.size(), 0.0);
        for (std::size_t t = 0; t < trees_.size(); ++t) share[tree_vote(t, row, buffer)] += 1.0;
        for (double& v : share) v /= static_cast<double>(trees_.size());
        const auto original               = dm.used_row_indices[i];
        result.class_label[original]      = classes_[argmax_index(share)];
        result.vote_share[original]       = std::move(share);
    }
    return result;
}

graphics::RPlot RandomForestClassifier::plot_classification(const dstruct::DataFrame& data,
                                                               const std::string& x_feature,
                                                               const std::string& y_feature) const {
    const auto  predictions = predict(data);
    const auto& response    = design_.response_name;

    auto plot = graphics::RPlot::create();
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < data.nrows(); ++i) {
            const auto actual = data.optional_string_at(response, i);
            if (!actual || *actual != classes_[c]) continue;
            const auto x = data.optional_double_at(x_feature, i);
            const auto y = data.optional_double_at(y_feature, i);
            if (!x || !y) continue;
            xs.push_back(*x);
            ys.push_back(*y);
        }
        plot.points(xs, ys, classes_[c], kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
    }

    std::vector<double> mis_x, mis_y;
    for (std::size_t i = 0; i < data.nrows(); ++i) {
        const auto actual = data.optional_string_at(response, i);
        if (!actual || i >= predictions.size() || predictions[i].empty() || predictions[i] == *actual) continue;
        const auto x = data.optional_double_at(x_feature, i);
        const auto y = data.optional_double_at(y_feature, i);
        if (!x || !y) continue;
        mis_x.push_back(*x);
        mis_y.push_back(*y);
    }
    if (!mis_x.empty()) plot.points(mis_x, mis_y, "misclassified", {15, 15, 15}, 9.0);

    plot.title("Random Forest Classification").x_label(x_feature).y_label(y_feature);
    return plot;
}

graphics::RPlot RandomForestClassifier::plot_decision_regions(const std::string& x_feature,
                                                              const std::string& y_feature, std::size_t grid_resolution,
                                                              std::optional<std::array<double, 4>> limits) const {
    if (predictor_names_.size() != 2)
        throw std::invalid_argument("RandomForestClassifier::plot_decision_regions: model must have exactly 2 "
                                    "predictors");
    std::size_t xi = 2, yi = 2;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_names_[j] == x_feature) xi = j;
        if (predictor_names_[j] == y_feature) yi = j;
    }
    if (xi == 2 || yi == 2 || xi == yi)
        throw std::invalid_argument("RandomForestClassifier::plot_decision_regions: x_feature/y_feature must match "
                                    "the model's two predictors (" + predictor_names_[0] + ", " + predictor_names_[1]
                                    + ")");
    const double x_pad = std::max(1e-6, (feature_max_[xi] - feature_min_[xi]) * 0.05);
    const double y_pad = std::max(1e-6, (feature_max_[yi] - feature_min_[yi]) * 0.10);
    const auto   box   = limits.value_or(std::array<double, 4>{feature_min_[xi] - x_pad, feature_max_[xi] + x_pad,
                                                               feature_min_[yi] - y_pad, feature_max_[yi] + y_pad});

    const std::size_t          n_colors = sizeof(kSeriesColors) / sizeof(kSeriesColors[0]);
    std::vector<graphics::RGB> fills;
    for (std::size_t c = 0; c < classes_.size(); ++c)
        for (const double white : {0.55, 0.75, 0.9}) fills.push_back(graphics::tint(kSeriesColors[c % n_colors], white));

    auto                plot = graphics::RPlot::create();
    std::vector<double> x(2);
    graphics::add_class_regions(
        plot,
        [&](double a, double b) {
            x[xi]             = a;
            x[yi]             = b;
            const auto  share = vote_share(x);
            const auto  c     = argmax_index(share);
            const std::size_t level = share[c] >= 0.9 ? 0 : share[c] >= 0.6 ? 1 : 2;
            return 3 * c + level;
        },
        box, grid_resolution, fills);
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < observations_; ++i)
            if (training_class_[i] == c) xs.push_back(training_x_(i, xi)), ys.push_back(training_x_(i, yi));
        plot.points(xs, ys, classes_[c], kSeriesColors[c % n_colors]);
    }
    plot.title("Random forest decision regions").x_label(x_feature).y_label(y_feature);
    return plot;
}

} // namespace datamunge::stats
