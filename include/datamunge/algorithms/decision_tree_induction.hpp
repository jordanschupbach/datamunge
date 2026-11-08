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
/// The continuous learner is a restricted unpruned C4.5-inspired variant; see
/// c45_train() for the omitted selection rules and missing-value machinery.
/// Both use a flat node array so the returned tree is copyable and
/// self-contained.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>
#include <utility>

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

/// @brief Growth controls for the restricted continuous gain-ratio learner.
/// Root depth is zero. A zero depth cap is unlimited; min_samples is a node-size
/// stopping rule, not a minimum child size. Zero/one both allow any impure node.
/// max_candidates bounds the whole call before an excess candidate is partitioned;
/// zero means unlimited. No partial model is returned on any exception.
/// @par Example
/// @code
/// dm::algorithms::C45Options options;
/// options.max_depth=1;
/// std::print("{}\n",options.min_samples); // 2
/// @endcode
/// @see c45_train_trace(), c45_train()
struct C45Options {
  std::size_t max_depth=0; ///< Maximum split depth; zero is unlimited.
  std::size_t min_samples=2; ///< Search only nodes with at least this many rows.
  std::size_t max_candidates=0; ///< Whole-call candidate budget; zero is unlimited.
};

/// @brief Reason a continuous-tree node stopped, or confirmation that it split.
/// Reasons are checked in declaration order; no-positive-gain includes no features
/// or no distinct feature values. This is training provenance, not a prediction.
/// @par Example
/// @code
/// const auto t=dm::algorithms::c45_train_trace({{0.0},{1.0}}, {7,7});
/// std::print("{}\n",t.nodes[0].reason==dm::algorithms::C45StopReason::pure); // true
/// @endcode
/// @see C45NodeTrace
enum class C45StopReason {
  pure, ///< All node labels agree.
  too_small, ///< Impure node has fewer than min_samples rows.
  depth_limit, ///< Impure, sufficiently large node has reached max_depth.
  no_positive_gain, ///< No candidate has a strictly positive computed ratio.
  split ///< The selected candidate produced two nonempty child nodes.
};

/// @brief One actually evaluated continuous threshold, with entropy in bits.
/// Candidates appear in node preorder, then increasing feature and distinct-value
/// boundary order. Threshold is a finite separating midpoint, falling back to the
/// lower value if rounding reaches the upper value. selected marks the first strict
/// positive maximum ratio at this node. The records belong to their owning trace.
/// @par Example
/// @code
/// const auto t=dm::algorithms::c45_train_trace({{0.0},{2.0}}, {0,1});
/// std::print("{} {}\n",t.candidates[0].threshold,t.candidates[0].gain); // 1 1
/// @endcode
/// @par References
/// - J. R. Quinlan (1996), Improved Use of Continuous Attributes in C4.5.
///   https://arxiv.org/abs/cs/9603103 (the unpenalized threshold variant only).
/// @see C45Trace
struct C45Candidate {
  std::size_t node=0; ///< Owning tree/trace node index.
  std::size_t feature=0; ///< Zero-based numeric feature.
  double lower=0; ///< Lower distinct observed value of the boundary.
  double upper=0; ///< Upper distinct observed value of the boundary.
  double threshold=0; ///< Finite test cutoff satisfying lower <= threshold < upper.
  std::size_t left_count=0; ///< Rows at or below threshold.
  std::size_t right_count=0; ///< Rows above threshold.
  double conditional_entropy=0; ///< Child-size-weighted class entropy, in bits.
  double gain=0; ///< Parent entropy minus conditional entropy, in bits.
  double split_info=0; ///< Binary partition entropy, in bits, strictly positive.
  double ratio=0; ///< Computed gain divided by split_info; not clamped.
  bool selected=false; ///< Whether this candidate supplied the actual split.
};

/// @brief Actual selected rows, depth, entropy and stopping reason of one node.
/// Vector position equals the corresponding DecisionTree node id. rows preserves
/// original input order, with owned indices; labels and feature values are not
/// copied. candidate_begin/count delimit this node's records in C45Trace::candidates.
/// @par Example
/// @code
/// const auto t=dm::algorithms::c45_train_trace({{0.0},{2.0}}, {0,1});
/// std::print("{} {}\n",t.nodes[0].rows,t.nodes[0].entropy); // [0, 1] 1
/// @endcode
/// @see C45Trace, C45StopReason
struct C45NodeTrace {
  std::size_t depth=0; ///< Root-zero depth of the actual node.
  std::vector<std::size_t> rows; ///< Owned original row indices reaching the node.
  double entropy=0; ///< Class entropy of rows, in bits, including stopped nodes.
  C45StopReason reason=C45StopReason::pure; ///< First applicable stopping rule, or split.
  std::size_t candidate_begin=0; ///< First candidate index for this node.
  std::size_t candidate_count=0; ///< Number of actually evaluated candidates here.
};

/// @brief Owned continuous model and actual training provenance from a shared kernel.
/// No training is replayed. Ordinary c45_train() omits row/candidate records. Input
/// values/labels are borrowed only during the call and are never mutated. Empty
/// aligned inputs give empty vectors. Trace storage includes O(n*h) row indices and
/// one record per evaluated candidate, h the largest reached depth.
/// @par Example
/// @code
/// const auto t=dm::algorithms::c45_train_trace({{0.0},{2.0}}, {0,1});
/// std::print("{} {}\n",t.tree.nodes.size(),t.candidate_count); // 3 1
/// @endcode
/// @see c45_train_trace(), C45Candidate
struct C45Trace {
  DecisionTree tree; ///< Owned model; same result as ordinary training with these controls.
  C45Options options; ///< Effective growth and whole-call budget controls.
  std::vector<C45NodeTrace> nodes; ///< Actual node-preorder provenance.
  std::vector<C45Candidate> candidates; ///< Actual feature/boundary evaluations.
  std::size_t candidate_count=0; ///< Number of threshold evaluations across all nodes.
  std::size_t candidate_row_tests=0; ///< Actual row/threshold tests during candidate evaluation only.
};

/// @brief Private continuous-tree implementation using explicit pending partitions.
namespace c45_detail {

/// @brief One pending child partition, with owned rows and stable parent index.
struct Work {
  std::vector<std::size_t> rows; ///< Selected original rows, in input order.
  std::size_t depth=0; ///< Root-zero depth.
  int parent=-1; ///< Parent index, or -1 for root.
  bool left=false; ///< Whether this is the parent's left child.
};

/// @brief Checked addition for actual continuous-training work counters.
/// @param value Counter to update, unchanged on overflow.
/// @param amount Nonnegative increment.
/// @throws std::overflow_error Sum exceeds size_t.
inline void add_work(std::size_t& value,std::size_t amount) {
  if (amount>std::numeric_limits<std::size_t>::max()-value)
    throw std::overflow_error("c45_train: work count overflow");
  value+=amount;
}

/// @brief Build the legacy maximum-ratio variant, optionally retaining actual records.
/// No full C4.5 average-gain filter, MDL penalty, categorical tests, pruning or missing
/// values are implemented. Flat model ownership and a DFS work stack avoid recursive
/// construction/destruction. Both modes use identical split selection and numbering.
/// @tparam Record Whether to retain node rows and candidate records.
/// @param features Borrowed rectangular finite numeric matrix.
/// @param labels Borrowed aligned arbitrary integer labels.
/// @param options Growth controls and whole-call candidate budget.
/// @return Owned model and counters; provenance vectors are empty in ordinary mode.
/// @throws std::invalid_argument Ragged/nonfinite features or mismatched label count.
/// @throws std::length_error Candidate budget exceeded or node index exceeds int.
/// @throws std::overflow_error Actual work counters exceed size_t.
template<bool Record>
inline C45Trace train(const std::vector<std::vector<double>>& features,
                      const std::vector<int>& labels,C45Options options) {
  if (features.size()!=labels.size()) throw std::invalid_argument("c45_train: label count mismatch");
  const std::size_t dim=features.empty()?0:features.front().size();
  for (const auto& row:features) {
    if (row.size()!=dim) throw std::invalid_argument("c45_train: ragged features");
    for (double value:row) if (!std::isfinite(value))
      throw std::invalid_argument("c45_train: nonfinite feature");
  }
  C45Trace out;out.options=options;
  if (features.empty()) return out;
  std::vector<std::size_t> all(features.size());
  std::iota(all.begin(),all.end(),std::size_t{0});
  std::vector<Work> pending;pending.push_back({std::move(all),0,-1,false});
  while (!pending.empty()) {
    Work work=std::move(pending.back());pending.pop_back();
    const auto& rows=work.rows;
    if (out.tree.nodes.size()>=static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::length_error("c45_train: too many nodes");
    const int id=static_cast<int>(out.tree.nodes.size());
    out.tree.nodes.push_back({});
    out.tree.nodes[id].prediction=detail::majority_class(labels,rows);
    if (work.parent>=0) {
      if (work.left) out.tree.nodes[work.parent].left=id;
      else out.tree.nodes[work.parent].right=id;
    }
    C45NodeTrace provenance;provenance.depth=work.depth;
    if constexpr(Record) {
      provenance.rows=rows;provenance.entropy=detail::class_entropy(labels,rows);
      provenance.candidate_begin=out.candidates.size();
    }
    if (detail::pure(labels,rows)) provenance.reason=C45StopReason::pure;
    else if (rows.size()<options.min_samples) provenance.reason=C45StopReason::too_small;
    else if (options.max_depth && work.depth>=options.max_depth) provenance.reason=C45StopReason::depth_limit;
    else {
      const double base_h=detail::class_entropy(labels,rows);
      double best_ratio=0,best_threshold=0;
      std::size_t best_feature=0,best_candidate=0;
      bool found=false;
      for (std::size_t f=0;f<dim;++f) {
        std::vector<double> values;values.reserve(rows.size());
        for (auto row:rows) values.push_back(features[row][f]);
        std::sort(values.begin(),values.end());
        values.erase(std::unique(values.begin(),values.end()),values.end());
        for (std::size_t k=0;k+1<values.size();++k) {
          if (options.max_candidates && out.candidate_count>=options.max_candidates)
            throw std::length_error("c45_train: candidate budget exceeded");
          add_work(out.candidate_count,1);add_work(out.candidate_row_tests,rows.size());
          double threshold=std::midpoint(values[k],values[k+1]);
          if (threshold>=values[k+1]) threshold=values[k];
          std::vector<std::size_t> lo,hi;
          for (auto row:rows) (features[row][f]<=threshold?lo:hi).push_back(row);
          const double pl=static_cast<double>(lo.size())/rows.size();
          const double ph=static_cast<double>(hi.size())/rows.size();
          const double conditional=pl*detail::class_entropy(labels,lo)+ph*detail::class_entropy(labels,hi);
          const double gain=base_h-conditional;
          const double split=-pl*std::log2(pl)-ph*std::log2(ph);
          const double ratio=gain/split;
          const auto candidate=out.candidates.size();
          if constexpr(Record) out.candidates.push_back({static_cast<std::size_t>(id),f,values[k],values[k+1],threshold,
              lo.size(),hi.size(),conditional,gain,split,ratio,false});
          if (ratio>best_ratio) {
            best_ratio=ratio;best_threshold=threshold;best_feature=f;best_candidate=candidate;found=true;
          }
        }
      }
      if (!found) provenance.reason=C45StopReason::no_positive_gain;
      else {
        provenance.reason=C45StopReason::split;
        if constexpr(Record) out.candidates[best_candidate].selected=true;
        std::vector<std::size_t> lo,hi;
        for (auto row:rows) (features[row][best_feature]<=best_threshold?lo:hi).push_back(row);
        out.tree.nodes[id].leaf=false;
        out.tree.nodes[id].feature=best_feature;out.tree.nodes[id].threshold=best_threshold;
        pending.push_back({std::move(hi),work.depth+1,id,false});
        pending.push_back({std::move(lo),work.depth+1,id,true});
      }
    }
    if constexpr(Record) {
      provenance.candidate_count=out.candidates.size()-provenance.candidate_begin;
      out.nodes.push_back(std::move(provenance));
    }
  }
  return out;
}
} // namespace c45_detail

/// @brief Continuous binary tree using maximum information-gain ratio, inspired by C4.5.
/// Restricted unpruned threshold variant: no average-gain filter, MDL penalty,
/// categorical tests or missing-value inference. Each feature considers all adjacent
/// sorted distinct boundaries. Midpoints avoid overflow; if rounding reaches the upper
/// value, the lower value is used so both children are nonempty. First strict positive
/// maximum wins (lowest feature, then earliest boundary); no tolerance is applied.
/// Label ties choose the smallest integer. Purity, min_samples, max_depth or no positive
/// computed ratio stop growth. Root depth is zero; min_samples limits the parent size,
/// not child size. Empty aligned inputs return no nodes; width-zero rows give a majority
/// leaf. Inputs are checked, borrowed and unchanged; result owns storage, no shared state.
/// Construction uses an explicit stack. Work at a searched node of m rows and d features
/// is O(d*m*m*log(m)) with current ordered-map entropy calculations; sum this over nodes.
/// Temporary partition storage is O(n) excluding sorting/maps and the O(n)-node model.
/// @param features Rectangular n-by-d finite continuous matrix, possibly empty or width zero.
/// @param labels Exactly n arbitrary integer labels.
/// @param max_depth Maximum split depth; zero (default) means unlimited.
/// @param min_samples Minimum node rows before search; default two; zero/one are allowed.
/// @return Owned binary threshold tree, not a complete C4.5 implementation.
/// @throws std::invalid_argument Ragged/nonfinite features or mismatched labels.
/// @throws std::length_error Node count cannot fit the integer child-index type.
/// @throws std::overflow_error Actual work counters exceed size_t.
/// @par Example
/// @code
/// const auto t=dm::algorithms::c45_train({{0.0},{1.0},{8.0},{9.0}}, {0,0,1,1});
/// std::print("{}\n",dm::algorithms::c45_predict(t,{7.0})); // 1
/// @endcode
/// @par References
/// - J. R. Quinlan (1996), Improved Use of Continuous Attributes in C4.5,
///   Journal of Artificial Intelligence Research 4, 77-90. https://arxiv.org/abs/cs/9603103
/// @see c45_predict(), c45_train_trace(), id3_train()
inline DecisionTree c45_train(
    const std::vector<std::vector<double>>& features,const std::vector<int>& labels,
    std::size_t max_depth=0,std::size_t min_samples=2) {
  return c45_detail::train<false>(features,labels,{max_depth,min_samples,0}).tree;
}

/// @brief Train the restricted gain-ratio variant and retain actual candidates and rows.
/// Uses the same kernel as c45_train(); no candidate search or model is replayed.
/// The whole-call budget counts only thresholds actually evaluated, including zero-
/// gain candidates. It is checked before excess partition work; exceptions return no
/// partial model. Trace records own indices and scalar statistics, not copies of inputs.
/// Recording adds O(n*h+c) storage for row memberships and c candidates, with h reached
/// depth; training complexity and finite-input rules match c45_train(). No pruning,
/// average-gain eligibility filter or MDL threshold penalty is added by tracing.
/// @param features Borrowed rectangular finite numeric matrix, empty/width zero allowed.
/// @param labels Borrowed aligned integer class labels.
/// @param options Growth controls and optional whole-call candidate budget; defaults unrestricted.
/// @return Owned model, actual node/candidate records and checked work counters.
/// @throws std::invalid_argument Ragged/nonfinite features or mismatched labels.
/// @throws std::length_error Candidate budget exceeded or node indices cannot fit int.
/// @throws std::overflow_error Actual work counters exceed size_t.
/// @par Example
/// @code
/// const auto t=dm::algorithms::c45_train_trace({{0.0},{2.0}}, {0,1}, {1,2,1});
/// std::print("{} {}\n",t.candidate_count,t.candidate_row_tests); // 1 2
/// @endcode
/// @par References
/// - J. R. Quinlan (1996), Improved Use of Continuous Attributes in C4.5.
///   https://arxiv.org/abs/cs/9603103 (restricted unpenalized variant).
/// @see c45_train(), C45Trace, C45Options
inline C45Trace c45_train_trace(
    const std::vector<std::vector<double>>& features,const std::vector<int>& labels,C45Options options={}) {
  return c45_detail::train<true>(features,labels,options);
}

/// @brief Follow checked continuous threshold tests to a leaf without mutating inputs.
/// O(visited depth) work, O(1) extra storage. Validate only the visited path: bad indices,
/// nonfinite thresholds/tested query values, categorical internal nodes, short queries
/// and cycles throw. Unvisited malformed branches and unused query values are ignored;
/// a leaf accepts an empty query. No runtime model-kind tag exists for leaf-only models.
/// @param tree Borrowed nonempty continuous tree, normally from c45_train().
/// @param x Numeric query containing every visited feature; extra values are ignored.
/// @return Reached leaf label.
/// @throws std::invalid_argument Empty tree, invalid visited node/test/query, or cycle.
/// @par Example
/// @code
/// const auto t=dm::algorithms::c45_train({{0.0},{9.0}}, {3,8});
/// std::print("{}\n",dm::algorithms::c45_predict(t,{1.0})); // 3
/// @endcode
/// @see c45_train(), c45_train_trace()
inline int c45_predict(const DecisionTree& tree,const std::vector<double>& x) {
  int id=0;
  for (std::size_t steps=0;steps<tree.nodes.size();++steps) {
    if (id<0 || static_cast<std::size_t>(id)>=tree.nodes.size())
      throw std::invalid_argument("c45_predict: invalid node index");
    const auto& node=tree.nodes[id];
    if (node.leaf) return node.prediction;
    if (node.feature>=x.size() || !std::isfinite(x[node.feature]) || !std::isfinite(node.threshold) || !node.children.empty())
      throw std::invalid_argument("c45_predict: invalid continuous test/query");
    id=x[node.feature]<=node.threshold?node.left:node.right;
  }
  throw std::invalid_argument("c45_predict: empty tree or cycle");
}

} // namespace datamunge::algorithms
