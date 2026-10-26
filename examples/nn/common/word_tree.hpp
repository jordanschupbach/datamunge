// WordTree conditional sibling softmaxes and ancestor-path supervision (port
// of tflow's common/nn/word_tree.h).
// https://arxiv.org/abs/1612.08242
//
// The root is implicit. Each node's parent is -1 or an earlier node; nodes
// sharing a parent form a softmax group. A marginal probability is the
// product of conditional probabilities along the path to the root (computed
// here as the exponential of the summed log-conditionals along the padded
// ancestor path). Supervision of an internal node updates its ancestors and
// that node's sibling group, leaving probabilities further down the tree
// unsupervised. Logits are a matrix [examples, nodes].
#pragma once

#include <algorithm>
#include <datamunge/nn/nn.hpp>
#include <map>
#include <stdexcept>
#include <vector>

namespace wordtree {

namespace nn = datamunge::nn;

struct WordTree {
  std::vector<int>                parents;
  std::map<int, std::vector<int>> groups;
  int                             maximum_depth = 0;
  std::vector<std::int32_t>       path;    // [nodes * maximum_depth] ancestor indices (0 padded)
  std::vector<float>              active;  // [nodes * maximum_depth]
  explicit WordTree(std::vector<int> parent) : parents(std::move(parent)) {
    if (parents.empty()) throw std::invalid_argument("WordTree requires nodes");
    std::vector<int> depth(parents.size());
    for (std::size_t i = 0; i < parents.size(); ++i) {
      if (parents[i] < -1 || parents[i] >= int(i)) throw std::invalid_argument("WordTree parents must precede their children");
      groups[parents[i]].push_back(int(i));
      depth[i]      = parents[i] < 0 ? 1 : depth[std::size_t(parents[i])] + 1;
      maximum_depth = std::max(maximum_depth, depth[i]);
    }
    const std::size_t d = std::size_t(maximum_depth);
    path.assign(parents.size() * d, 0);
    active.assign(parents.size() * d, 0);
    for (std::size_t i = 0; i < parents.size(); ++i) {
      std::size_t column = 0;
      for (int node = int(i); node >= 0; node = parents[std::size_t(node)], ++column) {
        path[i * d + column]   = node;
        active[i * d + column] = 1;
      }
    }
  }
  [[nodiscard]] int nodes() const { return int(parents.size()); }
};

struct WordTreeOutputs {
  nn::Tensor log_conditional, conditional, marginal;  // [M, nodes] each
};
inline WordTreeOutputs WordTreeProbabilities(const nn::Tensor& logits, const WordTree& tree) {
  const std::int64_t        n = tree.nodes(), d = tree.maximum_depth;
  std::vector<nn::Tensor>   parts;
  std::vector<std::int32_t> order;
  for (const auto& [parent, nodes] : tree.groups) {
    const std::vector<std::int32_t> columns(nodes.begin(), nodes.end());
    parts.push_back(nn::log_softmax(nn::gather(logits, nn::from_vector(columns), 1), -1));
    order.insert(order.end(), columns.begin(), columns.end());
  }
  std::vector<std::int32_t> inverse(order.size());
  for (std::size_t j = 0; j < order.size(); ++j) inverse[std::size_t(order[j])] = std::int32_t(j);
  const nn::Tensor grouped = parts.size() == 1 ? parts[0] : nn::concat(parts, 1);
  WordTreeOutputs  out;
  out.log_conditional      = nn::gather(grouped, nn::from_vector(inverse), 1);
  out.conditional          = nn::exp(out.log_conditional);
  const auto along_path    = nn::reshape(nn::gather(out.log_conditional, nn::from_vector(tree.path), 1), {-1, n, d});
  out.marginal             = nn::exp(nn::sum(along_path * nn::from_vector(tree.active, {n, d}), {2}));
  return out;
}

/// Negative log-likelihood of each example's label path [M].
inline nn::Tensor WordTreeLoss(const WordTreeOutputs& output, const std::vector<int>& labels, const WordTree& tree) {
  const std::int64_t        n = tree.nodes(), d = tree.maximum_depth, m = std::int64_t(labels.size());
  std::vector<std::int32_t> flat(std::size_t(m * d));
  std::vector<float>        mask(std::size_t(m * d));
  for (std::int64_t i = 0; i < m; ++i) {
    const int label = labels[std::size_t(i)];
    if (label < 0 || label >= n) throw std::invalid_argument("WordTree label outside the hierarchy");
    for (std::int64_t k = 0; k < d; ++k) {
      flat[std::size_t(i * d + k)] = std::int32_t(i * n + tree.path[std::size_t(label * d + k)]);
      mask[std::size_t(i * d + k)] = tree.active[std::size_t(label * d + k)];
    }
  }
  const auto selected = nn::reshape(nn::gather(nn::reshape(output.log_conditional, {-1}), nn::from_vector(flat), 0), {m, d});
  return -nn::sum(selected * nn::from_vector(mask, {m, d}), {1});
}

}  // namespace wordtree
