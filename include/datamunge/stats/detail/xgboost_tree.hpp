#pragma once

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats::detail {

// Not a general-purpose supervised learner (unlike DecisionTreeRegressor):
// an XGBoost tree only ever makes sense fit to a (gradient, hessian) pair
// supplied by an outer boosting loop, so it takes raw matrices/vectors
// rather than a DataFrame + formula, and lives in `detail` rather than the
// public `stats` API used by XGBoostClassifier/XGBoostRegressor.
struct XGBoostTreeOptions {
    std::size_t   max_depth{6};
    double        lambda{1.0};       // L2 regularization on leaf weights
    double        alpha{0.0};        // L1 regularization on leaf weights
    double        gamma{0.0};        // minimum gain required to keep a split
    double        min_child_weight{1.0}; // minimum sum of hessian in a child
    std::size_t   min_samples_leaf{1};
    // Fraction of columns considered for this tree (chosen once, at the
    // root -- "colsample_bytree" rather than XGBoost's finer-grained
    // per-node option).
    double        colsample_bytree{1.0};
    std::uint64_t seed{0};
};

// Regularized, second-order ("Newton boosting") regression tree, grown by
// maximizing XGBoost's exact regularized gain at each split:
//   Gain = 0.5*(score(G_L,H_L) + score(G_R,H_R) - score(G,H)) - gamma
//   score(G,H) = softthresh(G,alpha)^2 / (H + lambda)
// with leaf weight w* = -softthresh(G,alpha) / (H + lambda) (the Newton
// step minimizing the node's regularized quadratic loss approximation).
class XGBoostTree {
 public:
    void fit(const linalg::DenseMatrix<double>& X, const std::vector<double>& gradient,
            const std::vector<double>& hessian, XGBoostTreeOptions options);

    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& X) const;

    // Total gain contributed by splits on each of the p features, not yet normalized.
    [[nodiscard]] std::vector<double> feature_gain(std::size_t p) const;

 private:
    struct Node {
        bool        is_leaf{true};
        std::size_t feature_index{0};
        double      threshold{0.0};
        std::size_t left{0};
        std::size_t right{0};
        double      weight{0.0};
        double      gain{0.0}; // gain of the split at this node (0 for leaves)
    };

    std::size_t build_node(const linalg::DenseMatrix<double>& X, const std::vector<double>& gradient,
                           const std::vector<double>& hessian, const std::vector<std::size_t>& active_columns,
                           std::vector<std::size_t> rows, std::size_t depth);
    std::size_t leaf_for(const linalg::DenseMatrix<double>& X, std::size_t row) const;

    XGBoostTreeOptions options_;
    std::vector<Node>  nodes_;
};

} // namespace datamunge::stats::detail
