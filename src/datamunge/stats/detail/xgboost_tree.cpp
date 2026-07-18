#include <datamunge/stats/detail/xgboost_tree.hpp>

#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace datamunge::stats::detail {

namespace {

double soft_threshold(double g, double alpha) {
    if (g > alpha) return g - alpha;
    if (g < -alpha) return g + alpha;
    return 0.0;
}

double leaf_score(double G, double H, double lambda, double alpha) {
    const double s = soft_threshold(G, alpha);
    return (s * s) / (H + lambda);
}

double leaf_weight(double G, double H, double lambda, double alpha) {
    return -soft_threshold(G, alpha) / (H + lambda);
}

} // namespace

void XGBoostTree::fit(const linalg::DenseMatrix<double>& X, const std::vector<double>& gradient,
                      const std::vector<double>& hessian, XGBoostTreeOptions options) {
    options_ = options;
    nodes_.clear();

    const std::size_t n = X.rows();
    const std::size_t p = X.cols();

    std::vector<std::size_t> active_columns;
    if (options_.colsample_bytree >= 1.0 || p == 0) {
        active_columns.resize(p);
        std::iota(active_columns.begin(), active_columns.end(), 0);
    } else {
        const std::size_t k =
            std::max<std::size_t>(1, static_cast<std::size_t>(std::round(options_.colsample_bytree
                                                                         * static_cast<double>(p))));
        std::vector<std::size_t> all(p);
        std::iota(all.begin(), all.end(), 0);
        random::SplitMix64 rng(options_.seed);
        for (std::size_t i = 0; i < k; ++i) {
            const std::size_t j = i + static_cast<std::size_t>(rng.next_u64() % (p - i));
            std::swap(all[i], all[j]);
        }
        active_columns.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k));
        std::sort(active_columns.begin(), active_columns.end());
    }

    std::vector<std::size_t> all_rows(n);
    std::iota(all_rows.begin(), all_rows.end(), 0);
    build_node(X, gradient, hessian, active_columns, std::move(all_rows), 0);
}

std::size_t XGBoostTree::build_node(const linalg::DenseMatrix<double>& X, const std::vector<double>& gradient,
                                    const std::vector<double>& hessian,
                                    const std::vector<std::size_t>& active_columns,
                                    std::vector<std::size_t> rows, std::size_t depth) {
    double G = 0.0, H = 0.0;
    for (const auto r : rows) {
        G += gradient[r];
        H += hessian[r];
    }

    Node node;
    node.weight = leaf_weight(G, H, options_.lambda, options_.alpha);
    const std::size_t my_index = nodes_.size();
    nodes_.push_back(node);

    if (depth >= options_.max_depth || rows.size() < 2 * options_.min_samples_leaf
        || H < 2.0 * options_.min_child_weight)
        return my_index;

    const double parent_score = leaf_score(G, H, options_.lambda, options_.alpha);

    double      best_gain      = 0.0;
    std::size_t best_feature   = 0;
    double      best_threshold = 0.0;
    bool        found          = false;

    for (const auto feature : active_columns) {
        std::set<double> unique_values;
        for (const auto r : rows) unique_values.insert(X(r, feature));
        if (unique_values.size() < 2) continue;

        std::vector<double> sorted_values(unique_values.begin(), unique_values.end());
        for (std::size_t v = 0; v + 1 < sorted_values.size(); ++v) {
            const double threshold = 0.5 * (sorted_values[v] + sorted_values[v + 1]);

            double      G_L = 0.0, H_L = 0.0;
            std::size_t n_L = 0;
            for (const auto r : rows) {
                if (X(r, feature) <= threshold) {
                    G_L += gradient[r];
                    H_L += hessian[r];
                    ++n_L;
                }
            }
            const std::size_t n_R = rows.size() - n_L;
            if (n_L < options_.min_samples_leaf || n_R < options_.min_samples_leaf) continue;
            if (H_L < options_.min_child_weight || (H - H_L) < options_.min_child_weight) continue;

            const double G_R  = G - G_L;
            const double H_R  = H - H_L;
            const double gain = 0.5 * (leaf_score(G_L, H_L, options_.lambda, options_.alpha)
                                       + leaf_score(G_R, H_R, options_.lambda, options_.alpha) - parent_score)
                               - options_.gamma;
            if (gain > best_gain + 1e-12) {
                best_gain      = gain;
                best_feature   = feature;
                best_threshold = threshold;
                found          = true;
            }
        }
    }

    if (!found) return my_index;

    std::vector<std::size_t> left_rows, right_rows;
    for (const auto r : rows) (X(r, best_feature) <= best_threshold ? left_rows : right_rows).push_back(r);

    const std::size_t left_index  = build_node(X, gradient, hessian, active_columns, std::move(left_rows), depth + 1);
    const std::size_t right_index = build_node(X, gradient, hessian, active_columns, std::move(right_rows), depth + 1);

    nodes_[my_index].is_leaf       = false;
    nodes_[my_index].feature_index = best_feature;
    nodes_[my_index].threshold     = best_threshold;
    nodes_[my_index].left          = left_index;
    nodes_[my_index].right         = right_index;
    nodes_[my_index].gain          = best_gain;

    return my_index;
}

std::size_t XGBoostTree::leaf_for(const linalg::DenseMatrix<double>& X, std::size_t row) const {
    std::size_t index = 0;
    while (!nodes_[index].is_leaf)
        index = (X(row, nodes_[index].feature_index) <= nodes_[index].threshold) ? nodes_[index].left
                                                                                  : nodes_[index].right;
    return index;
}

std::vector<double> XGBoostTree::predict(const linalg::DenseMatrix<double>& X) const {
    std::vector<double> result(X.rows());
    for (std::size_t i = 0; i < X.rows(); ++i) result[i] = nodes_[leaf_for(X, i)].weight;
    return result;
}

std::vector<double> XGBoostTree::feature_gain(std::size_t p) const {
    std::vector<double> gain(p, 0.0);
    for (const auto& node : nodes_)
        if (!node.is_leaf) gain[node.feature_index] += node.gain;
    return gain;
}

} // namespace datamunge::stats::detail
