/// @file
/// @brief Soft and hard dense graph pooling with explicit node assignments.
#pragma once

#include <datamunge/nn/ops.hpp>

#include <stdexcept>

namespace datamunge::nn {

/// @brief Own pooled features, coarsened adjacency and node assignments.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto pool = n::graph_topk_pool(n::ones({1,2,1}), n::ones({1,2,2}),
///                                     n::tensor({1.,2.}, {1,2}), 1);
/// // pool.adjacency.shape() == {1,1,1}.
/// @endcode
/// @see coarsen_graph(), graph_topk_pool()
struct GraphPool {
  Tensor features; ///< [B,K,F] pooled node features.
  Tensor adjacency; ///< [B,K,K] coarsened edge weights.
  Tensor assignment; ///< [B,K,N] assignment of original nodes to pooled nodes.
};

/// @brief Pool a batch of dense graphs by a shared feature and adjacency assignment.
///
/// Computes SX and SAS^T. A soft assignment supports DiffPool and retains
/// gradients. A one-hot assignment selects the induced subgraph.
/// @param features [B,N,F] Float32/64 node features.
/// @param adjacency [B,N,N] original adjacency, same dtype as features.
/// @param assignment [B,K,N] assignment, same dtype as features.
/// @return New pooled features and adjacency, and a handle to the supplied assignment.
/// @throws std::invalid_argument For incompatible shapes or dtypes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto pool = n::coarsen_graph(n::ones({1,2,1}), n::ones({1,2,2}), n::ones({1,1,2}));
/// // pool.features.item() == 2 and pool.adjacency.item() == 4.
/// @endcode
/// @par References
/// - Ying et al. (2018), Hierarchical Graph Representation Learning with
///   Differentiable Pooling, https://arxiv.org/abs/1806.08804
/// @see graph_topk_pool()
inline GraphPool coarsen_graph(const Tensor &features, const Tensor &adjacency,
                               const Tensor &assignment) {
  if (features.dim() != 3 || adjacency.dim() != 3 || assignment.dim() != 3 ||
      features.size(0) <= 0 || features.size(1) <= 0 || features.size(2) <= 0 ||
      features.size(0) != adjacency.size(0) || features.size(0) != assignment.size(0) ||
      features.size(1) != adjacency.size(1) || features.size(1) != adjacency.size(2) ||
      features.size(1) != assignment.size(2) || assignment.size(1) <= 0 ||
      (features.dtype() != DType::Float32 && features.dtype() != DType::Float64) ||
      features.dtype() != adjacency.dtype() || features.dtype() != assignment.dtype())
    throw std::invalid_argument("coarsen_graph: incompatible features, adjacency or assignment");
  return {matmul(assignment, features),
          matmul(matmul(assignment, adjacency), assignment, false, true), assignment};
}

/// @brief Retain the highest-scoring nodes and gate their features using tanh.
///
/// Retains entries in descending score order, breaking ties by node index.
/// The hard assignment is discrete. Score gradients flow through the selected
/// feature gates. Supply normalized projection scores for TopK or
/// neighborhood-aware scores for SAGPool.
/// @param features [B,N,F] Float32/64 node features.
/// @param adjacency [B,N,N] original adjacency, same dtype as features.
/// @param scores [B,N] finite node scores, same dtype as features.
/// @param keep Number of nodes to keep in each graph, between 1 and N.
/// @return Gated features, induced adjacency, and one-hot selection.
/// @throws std::invalid_argument For incompatible shapes, dtypes or count.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto pool = n::graph_topk_pool(n::ones({1,2,1}), n::ones({1,2,2}),
///                                     n::tensor({1.,2.}, {1,2}), 1);
/// // pool.assignment contains [0,1].
/// @endcode
/// @par References
/// - Gao and Ji (2019), Graph U-Nets, https://arxiv.org/abs/1905.05178
/// - Lee, Lee, and Kang (2019), Self-Attention Graph Pooling, https://arxiv.org/abs/1904.08082
/// @see top_k(), coarsen_graph()
inline GraphPool graph_topk_pool(const Tensor &features, const Tensor &adjacency,
                                 const Tensor &scores, std::int64_t keep) {
  if (features.dim() != 3 || scores.dim() != 2 || scores.size(0) != features.size(0) ||
      scores.size(1) != features.size(1) || scores.dtype() != features.dtype() ||
      keep <= 0 || keep > features.size(1))
    throw std::invalid_argument("graph_topk_pool: incompatible features, scores or count");
  const auto indices = top_k(scores, keep).second;
  const auto assignment = one_hot(indices, features.size(1), 1., 0., features.dtype());
  return coarsen_graph(features * expand_dims(tanh(scores), 2), adjacency, assignment);
}

} // namespace datamunge::nn
