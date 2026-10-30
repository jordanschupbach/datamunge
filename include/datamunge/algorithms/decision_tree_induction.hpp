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
///     attributes remain; this implementation also stops at nonpositive gain.
///   - *C4.5* (Quinlan 1993) extends ID3 to *continuous* attributes via binary
///     threshold splits \f$x_a \le t\f$, and replaces information gain with *gain
///     ratio* -- gain normalized by the split's own entropy -- to reduce ID3's bias
///     toward many-valued attributes.
///
/// Both are implemented over a flat node array so the returned tree is copyable and
/// self-contained.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// @brief One owned node of a categorical ID3 or continuous C4.5-style tree.
/// Internal nodes keep the local majority label for unseen categorical values.
/// Class ties choose the smallest integer label. Indices refer to DecisionTree::nodes.
/// @par Example
/// @code
/// dm::algorithms::DecisionTreeNode n;
/// n.prediction = 7; // a leaf predicting class 7
/// @endcode
/// @see DecisionTree, id3_train(), c45_train()
struct DecisionTreeNode {
  bool               leaf       = true; ///< Leaf vs internal node.
  int                prediction = 0;    ///< Class label (leaf) or majority fallback.
  std::size_t        feature    = 0;    ///< Split attribute (internal).
  double             threshold  = 0.0;  ///< C4.5 continuous split point (x <= threshold -> left).
  std::map<int, int> children;          ///< ID3: attribute value -> child node index.
  int                left  = -1;        ///< C4.5 child index for values at or below threshold.
  int                right = -1;        ///< C4.5 child index for values above threshold.
};

/// @brief Owned flat node array, with root zero and an empty array for empty training data.
/// Copies are independent; prediction borrows the tree without mutation. Do not mix
/// categorical and continuous prediction functions: no runtime model-kind tag is stored.
/// @par Example
/// @code
/// const auto t=dm::algorithms::id3_train({{0},{1}}, {3,7});
/// std::print("{}\n",t.nodes.size()); // 3
/// @endcode
/// @see DecisionTreeNode, id3_predict(), c45_predict()
struct DecisionTree {
  std::vector<DecisionTreeNode> nodes; ///< Root-first nodes; children hold array indices.
};

/// @brief Shared entropy and majority primitives for tree induction.
namespace detail {

/// @brief Shannon entropy in bits on selected class labels; empty rows give zero.
/// @param labels Borrowed class labels.
/// @param rows Valid row indices into labels.
/// @return Nonnegative empirical entropy, with zero-count classes omitted.
/// @par References
/// - J. R. Quinlan (1986), Induction of decision trees. https://doi.org/10.1007/BF00116251
inline double class_entropy(const std::vector<int>& labels, const std::vector<std::size_t>& rows) {
  std::map<int, std::size_t> counts;
  for (std::size_t i : rows)
    ++counts[labels[i]];
  double       h = 0.0;
  const double n = static_cast<double>(rows.size());
  for (const auto& [cls, c] : counts) {
    const double p = c / n;
    h -= p * std::log2(p);
  }
  return h;
}

/// @brief Most frequent label, with the smallest integer breaking ties.
/// @param labels Borrowed class labels.
/// @param rows Valid indices; empty rows give the default label zero.
/// @return Majority class or zero for no rows.
inline int majority_class(const std::vector<int>& labels, const std::vector<std::size_t>& rows) {
  std::map<int, std::size_t> counts;
  for (std::size_t i : rows)
    ++counts[labels[i]];
  int         best = 0;
  std::size_t bc   = 0;
  for (const auto& [cls, c] : counts)
    if (c > bc) {
      bc   = c;
      best = cls;
    }
  return best;
}

/// @brief Whether all selected labels agree (also true for zero or one row).
/// @param labels Borrowed class labels.
/// @param rows Valid indices into labels.
/// @return True for a homogeneous or empty selection.
inline bool pure(const std::vector<int>& labels, const std::vector<std::size_t>& rows) {
  for (std::size_t i = 1; i < rows.size(); ++i)
    if (labels[rows[i]] != labels[rows[0]])
      return false;
  return true;
}

} // namespace detail

/// @brief Greedy multiway ID3 induction over integer-coded categorical attributes.
/// Features are categories, never ordered thresholds. Each path uses a feature at
/// most once. Stop at purity, exhaustion of features, the depth cap, or nonpositive
/// best information gain. Root depth is zero; max_depth=1 permits only a root split.
/// Equal computed gains choose the lowest feature index; majority ties choose the
/// smallest class label. No pruning, missing-value inference or gain-ratio correction
/// is performed. Empty aligned inputs return an empty tree; rows of width zero give
/// a majority leaf. Borrows inputs, owns result, no shared mutable state.
/// Work is O(n*d*h*log(n)) as an upper bound with ordered maps, h<=d; recursive row
/// partitions can use O(n*h) storage in addition to the returned nodes.
/// @param features Rectangular n-by-d matrix of arbitrary integer categories.
/// @param labels Exactly n arbitrary integer class labels.
/// @param max_depth Maximum split depth; zero (default) means unlimited.
/// @return Owned categorical tree, with local majority fallbacks at all nodes.
/// @throws std::invalid_argument Ragged rows or labels/features length mismatch.
/// @throws std::length_error Node count cannot fit the integer child-index type.
/// @par Example
/// @code
/// const auto t=dm::algorithms::id3_train({{0},{0},{1},{1}}, {0,0,1,1});
/// std::print("{}\n",dm::algorithms::id3_predict(t,{1})); // 1
/// @endcode
/// @par References
/// - J. R. Quinlan (1986), Induction of decision trees, Machine Learning 1, 81-106.
///   https://doi.org/10.1007/BF00116251
/// @see id3_predict(), c45_train()
inline DecisionTree id3_train(
    const std::vector<std::vector<int>>& features,
    const std::vector<int>&              labels,
    std::size_t                          max_depth = 0) {
  if (features.size() != labels.size())
    throw std::invalid_argument("id3_train: label count mismatch");
  const std::size_t dim = features.empty() ? 0 : features.front().size();
  for (const auto& row : features)
    if (row.size() != dim)
      throw std::invalid_argument("id3_train: ragged features");
  DecisionTree tree;

  // Recursive builder owns its row selection and used-feature set; indices, rather
  // than node references, remain valid when child construction reallocates nodes.
  std::function<int(const std::vector<std::size_t>&, std::set<std::size_t>, std::size_t)> build =
      [&](const std::vector<std::size_t>& rows,
          std::set<std::size_t>           used,
          std::size_t                     depth) -> int {
    if (tree.nodes.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::length_error("id3_train: too many nodes");
    const int idx = static_cast<int>(tree.nodes.size());
    tree.nodes.push_back({});
    tree.nodes[idx].prediction = detail::majority_class(labels, rows);

    if (detail::pure(labels, rows) || used.size() == dim || (max_depth && depth >= max_depth)) {
      tree.nodes[idx].leaf = true;
      return idx;
    }

    // Choose the attribute of maximum information gain.
    const double                            base_h    = detail::class_entropy(labels, rows);
    double                                  best_gain = -1.0;
    std::size_t                             best_feat = 0;
    std::map<int, std::vector<std::size_t>> best_partition;
    for (std::size_t f = 0; f < dim; ++f) {
      if (used.count(f))
        continue;
      std::map<int, std::vector<std::size_t>> part;
      for (std::size_t i : rows)
        part[features[i][f]].push_back(i);
      double cond_h = 0.0;
      for (const auto& [val, sub] : part)
        cond_h +=
            (static_cast<double>(sub.size()) / rows.size()) * detail::class_entropy(labels, sub);
      const double gain = base_h - cond_h;
      if (gain > best_gain) {
        best_gain      = gain;
        best_feat      = f;
        best_partition = part;
      }
    }

    if (best_gain <= 0.0 || best_partition.size() < 2) { // no useful split
      tree.nodes[idx].leaf = true;
      return idx;
    }

    tree.nodes[idx].leaf        = false;
    tree.nodes[idx].feature     = best_feat;
    std::set<std::size_t> used2 = used;
    used2.insert(best_feat);
    for (const auto& [val, sub] : best_partition) {
      const int child               = build(sub, used2, depth + 1);
      tree.nodes[idx].children[val] = child;
    }
    return idx;
  };

  std::vector<std::size_t> all(features.size());
  for (std::size_t i = 0; i < features.size(); ++i)
    all[i] = i;
  if (!all.empty())
    build(all, {}, 0);
  return tree;
}

/// @brief Predict a class by categorical traversal, falling back on an unseen value.
/// Reads only visited features; extra query values are ignored. A leaf-only tree
/// accepts an empty query. Checks visited indices and limits visits to the node
/// count, rejecting cycles; unvisited malformed branches are not inspected.
/// Work O(h*log(b)), b the largest visited branch count, with O(1) extra storage.
/// @param tree Borrowed categorical tree from id3_train(), nonempty.
/// @param x Integer categories with every visited feature present.
/// @return Leaf label, or current node majority when the category has no branch.
/// @throws std::invalid_argument Empty tree, invalid visited index, cycle or short query.
/// @par Example
/// @code
/// const auto t=dm::algorithms::id3_train({{0},{0},{1}}, {2,2,7});
/// std::print("{}\n",dm::algorithms::id3_predict(t,{99})); // 2
/// @endcode
/// @see id3_train(), DecisionTree
inline int id3_predict(const DecisionTree& tree, const std::vector<int>& x) {
  int idx = 0;
  for (std::size_t steps = 0; steps < tree.nodes.size(); ++steps) {
    if (idx < 0 || static_cast<std::size_t>(idx) >= tree.nodes.size())
      throw std::invalid_argument("id3_predict: invalid node index");
    const auto& node = tree.nodes[idx];
    if (node.leaf)
      return node.prediction;
    if (node.feature >= x.size())
      throw std::invalid_argument("id3_predict: short query");
    const auto it = node.children.find(x[node.feature]);
    if (it == node.children.end())
      return node.prediction;
    idx = it->second;
  }
  throw std::invalid_argument("id3_predict: empty tree or cycle");
}

/// @brief Continuous binary tree using information-gain ratio, inspired by C4.5.
/// This is a restricted threshold-search variant, without C4.5 pruning, missing
/// values, categorical tests or its average-gain eligibility filter. Thresholds
/// are midpoints of adjacent sorted distinct values; the first strict maximum
/// ratio wins. Pure/small/depth-limited nodes or no positive ratio stop growth.
/// Inputs must be rectangular and finite with matching labels; these preconditions
/// are not checked by this legacy continuous routine. Empty inputs give no nodes.
/// Returned nodes own their storage; inputs are unchanged, no shared state.
/// @param features Rectangular n-by-d finite continuous matrix.
/// @param labels Exactly n integer labels; ties choose the smallest label.
/// @param max_depth Maximum split depth; zero (default) means unlimited, root depth zero.
/// @param min_samples Minimum node rows before a split is searched; default two.
/// @return Owned binary threshold tree; not a complete C4.5 implementation.
/// @par Example
/// @code
/// const auto t=dm::algorithms::c45_train({{0.0},{1.0},{8.0},{9.0}}, {0,0,1,1});
/// std::print("{}\n",dm::algorithms::c45_predict(t,{7.0})); // 1
/// @endcode
/// @par References
/// - J. R. Quinlan (1986), Induction of decision trees. https://doi.org/10.1007/BF00116251
/// @see c45_predict(), id3_train()
inline DecisionTree c45_train(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    std::size_t                             max_depth   = 0,
    std::size_t                             min_samples = 2) {
  DecisionTree      tree;
  const std::size_t dim = features.empty() ? 0 : features.front().size();

  std::function<int(const std::vector<std::size_t>&, std::size_t)> build =
      [&](const std::vector<std::size_t>& rows, std::size_t depth) -> int {
    const int idx = static_cast<int>(tree.nodes.size());
    tree.nodes.push_back({});
    tree.nodes[idx].prediction = detail::majority_class(labels, rows);
    tree.nodes[idx].leaf       = true;

    if (detail::pure(labels, rows) || rows.size() < min_samples ||
        (max_depth && depth >= max_depth))
      return idx;

    const double base_h     = detail::class_entropy(labels, rows);
    double       best_ratio = 0.0, best_thr = 0.0;
    std::size_t  best_feat = 0;
    bool         found     = false;

    for (std::size_t f = 0; f < dim; ++f) {
      // Candidate thresholds: midpoints of sorted unique values of feature f.
      std::vector<double> vals;
      vals.reserve(rows.size());
      for (std::size_t i : rows)
        vals.push_back(features[i][f]);
      std::sort(vals.begin(), vals.end());
      vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
      for (std::size_t k = 0; k + 1 < vals.size(); ++k) {
        const double             thr = 0.5 * (vals[k] + vals[k + 1]);
        std::vector<std::size_t> lo, hi;
        for (std::size_t i : rows)
          (features[i][f] <= thr ? lo : hi).push_back(i);
        if (lo.empty() || hi.empty())
          continue;
        const double pl = static_cast<double>(lo.size()) / rows.size();
        const double ph = static_cast<double>(hi.size()) / rows.size();
        const double cond_h =
            pl * detail::class_entropy(labels, lo) + ph * detail::class_entropy(labels, hi);
        const double gain       = base_h - cond_h;
        const double split_info = -pl * std::log2(pl) - ph * std::log2(ph);
        const double ratio      = split_info > 0.0 ? gain / split_info : 0.0;
        if (ratio > best_ratio) {
          best_ratio = ratio;
          best_feat  = f;
          best_thr   = thr;
          found      = true;
        }
      }
    }

    if (!found || best_ratio <= 0.0)
      return idx;

    std::vector<std::size_t> lo, hi;
    for (std::size_t i : rows)
      (features[i][best_feat] <= best_thr ? lo : hi).push_back(i);
    tree.nodes[idx].leaf      = false;
    tree.nodes[idx].feature   = best_feat;
    tree.nodes[idx].threshold = best_thr;
    const int l               = build(lo, depth + 1);
    const int r               = build(hi, depth + 1);
    tree.nodes[idx].left      = l;
    tree.nodes[idx].right     = r;
    return idx;
  };

  std::vector<std::size_t> all(features.size());
  for (std::size_t i = 0; i < features.size(); ++i)
    all[i] = i;
  if (!all.empty())
    build(all, 0);
  return tree;
}

/// @brief Follow continuous threshold tests to a leaf in a valid binary tree.
/// Legacy unchecked traversal: requires a nonempty acyclic continuous tree and all
/// tested finite query values. Inputs are unchanged; O(depth) work and O(1) storage.
/// @param tree Valid tree returned by c45_train(), borrowed.
/// @param x Finite continuous values with every tested feature present.
/// @return Reached leaf's class label.
/// @par Example
/// @code
/// const auto t=dm::algorithms::c45_train({{0.0},{9.0}}, {3,8});
/// std::print("{}\n",dm::algorithms::c45_predict(t,{1.0})); // 3
/// @endcode
/// @see c45_train()
inline int c45_predict(const DecisionTree& tree, const std::vector<double>& x) {
  int idx = 0;
  while (!tree.nodes[idx].leaf)
    idx = (x[tree.nodes[idx].feature] <= tree.nodes[idx].threshold) ? tree.nodes[idx].left
                                                                    : tree.nodes[idx].right;
  return tree.nodes[idx].prediction;
}

} // namespace datamunge::algorithms
