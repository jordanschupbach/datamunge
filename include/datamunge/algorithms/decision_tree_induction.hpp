#pragma once

/// \file decision_tree_induction.hpp
/// \brief Classic top-down decision-tree induction: ID3 (categorical attributes,
///        information gain) and C4.5 (continuous attributes, gain ratio).
///
/// A decision tree classifies by asking a sequence of attribute tests, one per node,
/// until it reaches a leaf carrying a class label. The classic induction algorithms
/// grow the tree *greedily and top-down*, at each node choosing the attribute test
/// that best separates the classes by an information-theoretic criterion:
///
///   - *ID3* (Quinlan 1986) handles *categorical* attributes and splits a node into
///     one branch per attribute value, choosing the attribute of maximum
///     *information gain* \f$\mathrm{Gain}(S,A)=H(S)-\sum_v \frac{|S_v|}{|S|}H(S_v)\f$,
///     where \f$H\f$ is the class entropy. It recurses until a node is pure or no
///     attributes remain.
///   - *C4.5* (Quinlan 1993) extends ID3 to *continuous* attributes via binary
///     threshold splits \f$x_a \le t\f$, and replaces information gain with *gain
///     ratio* -- gain normalized by the split's own entropy -- to remove ID3's bias
///     toward many-valued attributes.
///
/// Both are implemented over a flat node array so the returned tree is copyable and
/// self-contained.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace datamunge::algorithms {

/// One node of an induced decision tree (flat-array representation).
struct DecisionTreeNode {
    bool               leaf       = true;   ///< Leaf vs internal node.
    int                prediction = 0;      ///< Class label (leaf) or majority fallback.
    std::size_t        feature    = 0;      ///< Split attribute (internal).
    double             threshold  = 0.0;    ///< C4.5 continuous split point (x <= threshold -> left).
    std::map<int, int> children;            ///< ID3: attribute value -> child node index.
    int                left = -1, right = -1;  ///< C4.5: <= threshold / > threshold child indices.
};

/// A decision tree; the root is node 0.
struct DecisionTree {
    std::vector<DecisionTreeNode> nodes;
};

namespace detail {

/// Shannon entropy (in bits) of the class distribution of \p labels restricted to \p rows.
inline double class_entropy(const std::vector<int>& labels, const std::vector<std::size_t>& rows) {
    std::map<int, std::size_t> counts;
    for (std::size_t i : rows) ++counts[labels[i]];
    double h = 0.0;
    const double n = static_cast<double>(rows.size());
    for (const auto& [cls, c] : counts) {
        const double p = c / n;
        h -= p * std::log2(p);
    }
    return h;
}

/// Majority class among \p rows.
inline int majority_class(const std::vector<int>& labels, const std::vector<std::size_t>& rows) {
    std::map<int, std::size_t> counts;
    for (std::size_t i : rows) ++counts[labels[i]];
    int         best = 0;
    std::size_t bc   = 0;
    for (const auto& [cls, c] : counts)
        if (c > bc) { bc = c; best = cls; }
    return best;
}

inline bool pure(const std::vector<int>& labels, const std::vector<std::size_t>& rows) {
    for (std::size_t i = 1; i < rows.size(); ++i)
        if (labels[rows[i]] != labels[rows[0]]) return false;
    return true;
}

}  // namespace detail

/// \brief ID3: induce a decision tree over categorical attributes by information gain.
///
/// \param features   Categorical (integer-coded) design matrix.
/// \param labels     Integer class labels.
/// \param max_depth  Depth cap (0 = unlimited).
inline DecisionTree id3_train(const std::vector<std::vector<int>>& features, const std::vector<int>& labels,
                              std::size_t max_depth = 0) {
    DecisionTree      tree;
    const std::size_t dim = features.empty() ? 0 : features.front().size();

    // Recursive builder returning the created node index.
    std::function<int(const std::vector<std::size_t>&, std::set<std::size_t>, std::size_t)> build =
        [&](const std::vector<std::size_t>& rows, std::set<std::size_t> used, std::size_t depth) -> int {
        const int idx = static_cast<int>(tree.nodes.size());
        tree.nodes.push_back({});
        tree.nodes[idx].prediction = detail::majority_class(labels, rows);

        if (detail::pure(labels, rows) || used.size() == dim || (max_depth && depth >= max_depth)) {
            tree.nodes[idx].leaf = true;
            return idx;
        }

        // Choose the attribute of maximum information gain.
        const double base_h = detail::class_entropy(labels, rows);
        double       best_gain = -1.0;
        std::size_t  best_feat = 0;
        std::map<int, std::vector<std::size_t>> best_partition;
        for (std::size_t f = 0; f < dim; ++f) {
            if (used.count(f)) continue;
            std::map<int, std::vector<std::size_t>> part;
            for (std::size_t i : rows) part[features[i][f]].push_back(i);
            double cond_h = 0.0;
            for (const auto& [val, sub] : part)
                cond_h += (static_cast<double>(sub.size()) / rows.size()) * detail::class_entropy(labels, sub);
            const double gain = base_h - cond_h;
            if (gain > best_gain) {
                best_gain      = gain;
                best_feat      = f;
                best_partition = part;
            }
        }

        if (best_gain <= 0.0 || best_partition.size() < 2) {  // no useful split
            tree.nodes[idx].leaf = true;
            return idx;
        }

        tree.nodes[idx].leaf    = false;
        tree.nodes[idx].feature = best_feat;
        std::set<std::size_t> used2 = used;
        used2.insert(best_feat);
        for (const auto& [val, sub] : best_partition) {
            const int child = build(sub, used2, depth + 1);
            tree.nodes[idx].children[val] = child;
        }
        return idx;
    };

    std::vector<std::size_t> all(features.size());
    for (std::size_t i = 0; i < features.size(); ++i) all[i] = i;
    if (!all.empty()) build(all, {}, 0);
    return tree;
}

/// Classify a categorical vector with an ID3 tree.
inline int id3_predict(const DecisionTree& tree, const std::vector<int>& x) {
    int idx = 0;
    while (!tree.nodes[idx].leaf) {
        const auto it = tree.nodes[idx].children.find(x[tree.nodes[idx].feature]);
        if (it == tree.nodes[idx].children.end()) return tree.nodes[idx].prediction;  // unseen value
        idx = it->second;
    }
    return tree.nodes[idx].prediction;
}

/// \brief C4.5: induce a decision tree over continuous attributes by gain ratio with
///        binary threshold splits.
///
/// \param features   Continuous design matrix.
/// \param labels     Integer class labels.
/// \param max_depth  Depth cap (0 = unlimited).
/// \param min_samples Minimum rows required to split a node.
inline DecisionTree c45_train(const std::vector<std::vector<double>>& features, const std::vector<int>& labels,
                              std::size_t max_depth = 0, std::size_t min_samples = 2) {
    DecisionTree      tree;
    const std::size_t dim = features.empty() ? 0 : features.front().size();

    std::function<int(const std::vector<std::size_t>&, std::size_t)> build =
        [&](const std::vector<std::size_t>& rows, std::size_t depth) -> int {
        const int idx = static_cast<int>(tree.nodes.size());
        tree.nodes.push_back({});
        tree.nodes[idx].prediction = detail::majority_class(labels, rows);
        tree.nodes[idx].leaf       = true;

        if (detail::pure(labels, rows) || rows.size() < min_samples || (max_depth && depth >= max_depth))
            return idx;

        const double base_h    = detail::class_entropy(labels, rows);
        double       best_ratio = 0.0, best_thr = 0.0;
        std::size_t  best_feat = 0;
        bool         found     = false;

        for (std::size_t f = 0; f < dim; ++f) {
            // Candidate thresholds: midpoints of sorted unique values of feature f.
            std::vector<double> vals;
            vals.reserve(rows.size());
            for (std::size_t i : rows) vals.push_back(features[i][f]);
            std::sort(vals.begin(), vals.end());
            vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
            for (std::size_t k = 0; k + 1 < vals.size(); ++k) {
                const double thr = 0.5 * (vals[k] + vals[k + 1]);
                std::vector<std::size_t> lo, hi;
                for (std::size_t i : rows) (features[i][f] <= thr ? lo : hi).push_back(i);
                if (lo.empty() || hi.empty()) continue;
                const double pl = static_cast<double>(lo.size()) / rows.size();
                const double ph = static_cast<double>(hi.size()) / rows.size();
                const double cond_h = pl * detail::class_entropy(labels, lo) + ph * detail::class_entropy(labels, hi);
                const double gain   = base_h - cond_h;
                const double split_info = -pl * std::log2(pl) - ph * std::log2(ph);
                const double ratio  = split_info > 0.0 ? gain / split_info : 0.0;
                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_feat  = f;
                    best_thr   = thr;
                    found      = true;
                }
            }
        }

        if (!found || best_ratio <= 0.0) return idx;

        std::vector<std::size_t> lo, hi;
        for (std::size_t i : rows) (features[i][best_feat] <= best_thr ? lo : hi).push_back(i);
        tree.nodes[idx].leaf      = false;
        tree.nodes[idx].feature   = best_feat;
        tree.nodes[idx].threshold = best_thr;
        const int l = build(lo, depth + 1);
        const int r = build(hi, depth + 1);
        tree.nodes[idx].left  = l;
        tree.nodes[idx].right = r;
        return idx;
    };

    std::vector<std::size_t> all(features.size());
    for (std::size_t i = 0; i < features.size(); ++i) all[i] = i;
    if (!all.empty()) build(all, 0);
    return tree;
}

/// Classify a continuous vector with a C4.5 tree.
inline int c45_predict(const DecisionTree& tree, const std::vector<double>& x) {
    int idx = 0;
    while (!tree.nodes[idx].leaf)
        idx = (x[tree.nodes[idx].feature] <= tree.nodes[idx].threshold) ? tree.nodes[idx].left
                                                                        : tree.nodes[idx].right;
    return tree.nodes[idx].prediction;
}

}  // namespace datamunge::algorithms
