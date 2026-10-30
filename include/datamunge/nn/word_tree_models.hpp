#pragma once
/// @file word_tree_models.hpp
/// @brief Tensor sibling-softmax WordTree probabilities and ancestor-path supervision for YOLO9000.
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own validated acyclic parent topology for sibling-softmax WordTree classes.
///
/// The root is implicit. Every parent is-1 or an earlier node; siblings form
/// one conditional softmax. Paths run from node to root with zero-index padding
/// and an active metadata mask. Do not mutate public topology after construction.
/// No model weights, state, tensors, random draws or fitting are created.
///
/// @par Example
/// @code
/// dm::nn::models::WordTree tree({-1, -1, 0, 0, 1, 1});
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see WordTreeProbabilities(), WordTreeLoss()
struct WordTree {
  std::vector<int>
      parents; ///< Owned parent IDs, -1 for implicit-root children, otherwise earlier nodes.
  std::map<int, std::vector<int>>
      groups;            ///< Parent-to-sibling IDs in increasing original node order.
  int maximum_depth = 0; ///< Longest number of nodes on any node-to-root path.
  std::vector<std::int32_t>
      path;                  ///< Flattened[nodes,maximum_depth] topology indices, zero-padded.
  std::vector<float> active; ///< Flattened path0/1 metadata; padding contributes zero.

  /// @brief Validate and retain a parent-before-child WordTree hierarchy.
  ///
  /// Moves the supplied topology and builds deterministic sibling/path metadata.
  /// No numerical probability computation, learned model state or fitting.
  ///
  /// @param parent Nonempty owned vector of parent IDs; values-1 or less than their node index.
  /// @throws std::invalid_argument For empty topology, too many nodes for int32 indices or invalid parent ordering.
  /// @par Example
  /// @code
  /// dm::nn::models::WordTree tree({-1, -1, 0, 0, 1, 1});
  /// @endcode
  /// @par References
  /// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
  /// @see WordTreeProbabilities()
  explicit WordTree(std::vector<int> parent) : parents(std::move(parent)) {
    if (parents.empty() ||
        parents.size() > std::size_t(std::numeric_limits<std::int32_t>::max()))
      throw std::invalid_argument("WordTree requires nodes");
    std::vector<int> depth(parents.size());
    for (std::size_t i = 0; i < parents.size(); ++i) {
      if (parents[i] < -1 || parents[i] >= int(i))
        throw std::invalid_argument("WordTree parents must precede their children");
      groups[parents[i]].push_back(int(i));
      depth[i] = parents[i] < 0 ? 1 : depth[std::size_t(parents[i])] + 1;
      maximum_depth = std::max(maximum_depth, depth[i]);
    }
    const std::size_t d = std::size_t(maximum_depth);
    path.assign(parents.size() * d, 0);
    active.assign(parents.size() * d, 0);
    for (std::size_t i = 0; i < parents.size(); ++i) {
      std::size_t column = 0;
      for (int node = int(i); node >= 0; node = parents[std::size_t(node)], ++column) {
        path[i * d + column] = node;
        active[i * d + column] = 1;
      }
    }
  }
  /// @brief Return the validated number of hierarchy nodes.
  /// @return Positive node count represented by the parent metadata.
  [[nodiscard]] int nodes() const {
    return int(parents.size());
  }
};
/// @brief Retain sibling log-conditionals, conditionals and full-path marginal probabilities.
///
/// Tensor graphs retain logits. Topology is borrowed only during the forward
/// call; output tensors do not own/mutate the WordTree. No fitted state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::WordTree tree({-1, -1, 0, 0, 1, 1});
/// const auto p = n::models::WordTreeProbabilities(n::zeros({2, 6}), tree);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see WordTreeProbabilities(), WordTreeLoss()
struct WordTreeOutputs {
  nn::Tensor log_conditional; ///< Float32/64 [M,nodes] per-sibling-group log-softmax.
  nn::Tensor conditional; ///< Same-shape exp(log_conditional), each sibling group sums to1.
  nn::Tensor marginal;    ///< Same-shape exp(sum active ancestor-path log-conditionals).
};
/// @brief Normalize sibling groups and multiply conditional probabilities along each ancestor path.
///
/// Root sibling probabilities sum to1. Internal sibling conditionals sum to1,
/// while their marginals sum to their parent marginal. Computes path products
/// as exp(sum log-conditionals) to preserve long-path stability. Host loops
/// visit only discrete sibling metadata; every numerical operation uses tensors.
/// No fitting, parameter/state mutation or random draws.
///
/// @param logits Float32/64 [M,tree.nodes()] finite logits, M may be zero.
/// @param tree Valid constructed topology; do not mutate its public metadata.
/// @return Differentiable same-dtype [M,nodes] log-conditionals, conditionals and path marginals.
/// @throws std::invalid_argument For invalid logits rank/dtype/node count or malformed mutated metadata in core gather.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::WordTree tree({-1, -1, 0, 0, 1, 1});
/// const auto p = n::models::WordTreeProbabilities(n::zeros({2, 6}), tree);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see WordTreeLoss()
inline WordTreeOutputs WordTreeProbabilities(const nn::Tensor &logits, const WordTree &tree) {
  if (logits.dim() != 2 || logits.shape()[1] != tree.nodes() ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid WordTree logits");
  const std::int64_t n = tree.nodes(), d = tree.maximum_depth;
  std::vector<nn::Tensor> parts;
  std::vector<std::int32_t> order;
  for (const auto &[parent, nodes] : tree.groups) {
    const std::vector<std::int32_t> columns(nodes.begin(), nodes.end());
    parts.push_back(nn::log_softmax(nn::gather(logits, nn::from_vector(columns), 1), -1));
    order.insert(order.end(), columns.begin(), columns.end());
  }
  std::vector<std::int32_t> inverse(order.size());
  for (std::size_t j = 0; j < order.size(); ++j)
    inverse[std::size_t(order[j])] = std::int32_t(j);
  const nn::Tensor grouped = parts.size() == 1 ? parts[0] : nn::concat(parts, 1);
  WordTreeOutputs out;
  out.log_conditional = nn::gather(grouped, nn::from_vector(inverse), 1);
  out.conditional = nn::exp(out.log_conditional);
  const auto along_path =
      nn::reshape(nn::gather(out.log_conditional, nn::from_vector(tree.path), 1), {-1, n, d});
  out.marginal = nn::exp(
      nn::sum(along_path * nn::from_vector(tree.active, {n, d}).to(logits.dtype()), {2}));
  return out;
}

/// @brief Compute negative log-likelihood of each supplied node ancestor path.
///
/// Internal labels supervise only their ancestors and current sibling group,
/// leaving descendants unsupervised. Flattened gathers and masks use tensors;
/// no fitted state, parameter mutation, optimizer or host numerical reduction.
///
/// @param output Log-conditional Float32/64 [M,tree.nodes()] tensor; other output fields are not read.
/// @param labels Int64 [M] node IDs in[0,nodes), including internal/coarse nodes.
/// @param tree Valid constructed topology, unchanged since probability evaluation.
/// @return Differentiable [M] summed negative active-path log-conditionals.
/// @throws std::invalid_argument For inconsistent shapes/dtypes or out-of-range labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::WordTree tree({-1, -1, 0, 0, 1, 1});
/// const auto p = n::models::WordTreeProbabilities(n::zeros({2, 6}), tree);
/// const auto loss =
///     n::models::WordTreeLoss(p, n::from_vector(std::vector<std::int64_t>{0, 3}), tree);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see WordTreeProbabilities()
inline nn::Tensor WordTreeLoss(const WordTreeOutputs &output, const nn::Tensor &labels,
                               const WordTree &tree) {
  const auto &x = output.log_conditional;
  const auto N = std::int64_t(tree.nodes()), D = std::int64_t(tree.maximum_depth);
  if (x.dim() != 2 || x.shape()[1] != N || labels.shape() != nn::Shape{x.shape()[0]} ||
      labels.dtype() != nn::DType::Int64 ||
      (x.dtype() != nn::DType::Float32 && x.dtype() != nn::DType::Float64) ||
      !nn::all(nn::logical_and(nn::greater_equal(labels, 0.), nn::less(labels, double(N))))
           .item<bool>())
    throw std::invalid_argument("invalid WordTree loss labels/shapes");
  const auto M = x.shape()[0];
  const auto path =
                 nn::gather(nn::from_vector(tree.path, {N, D}).to(nn::DType::Int64), labels),
             mask = nn::gather(nn::from_vector(tree.active, {N, D}).to(x.dtype()), labels);
  const auto offset = (nn::arange(0, double(M), 1, nn::DType::Int64) *
                       nn::full({}, double(N), nn::DType::Int64))
                          .reshape({M, 1});
  const auto selected =
      nn::gather(x.reshape({M * N}), (path + offset).reshape({M * D})).reshape({M, D});
  return -nn::sum(selected * mask, {1});
}
} // namespace datamunge::nn::models
