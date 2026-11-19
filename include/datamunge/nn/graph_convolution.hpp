/// @file
/// @brief Normalization for graph convolutional propagation.
#pragma once

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <limits>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Add unit self-loops and symmetrically degree-normalize a dense adjacency.
///
/// Returns D^(-1/2) (A + I) D^(-1/2), where D is the row sum of A + I.
/// Existing diagonal weights are incremented by one. Isolated nodes retain
/// their own features under propagation. For undirected graphs, supply a
/// symmetric adjacency. No symmetry check is applied. The result retains
/// its gradient with respect to the input.
///
/// @param adjacency Nonempty square Float32/Float64 matrix of finite nonnegative weights.
/// @return Normalized adjacency with the same shape, dtype and device.
/// @throws std::invalid_argument For invalid shape, dtype or weights.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto a = n::tensor({0., 1., 1., 0.}, {2, 2});
/// const auto normalized = n::gcn_normalize(a); // all entries are 0.5 up to rounding.
/// @endcode
/// @par References
/// - Kipf and Welling (2017), Semi-Supervised Classification with Graph Convolutional Networks,
///   https://arxiv.org/abs/1609.02907
/// @see matmul(), coo_matmul()
inline Tensor gcn_normalize(const Tensor &adjacency) {
  if (adjacency.dim() != 2 || adjacency.size(0) <= 0 || adjacency.size(1) != adjacency.size(0) ||
      (adjacency.dtype() != DType::Float32 && adjacency.dtype() != DType::Float64) ||
      !all(is_finite(adjacency)).item<bool>() || any(less(adjacency, 0.)).item<bool>())
    throw std::invalid_argument("gcn_normalize: expected a square matrix of finite nonnegative weights");
  const auto loop = adjacency + eye(adjacency.size(0), adjacency.dtype()).to(adjacency.context());
  const auto inverse_root = rsqrt(sum(loop, {1}));
  return (expand_dims(inverse_root, 1) * loop) * inverse_root;
}

/// @brief Concatenate Chebyshev filter terms applied to graph node features.
///
/// Computes T0(S)X = X, T1(S)X = SX, and Tk(S)X = 2 S T(k-1)(S)X - T(k-2)(S)X.
/// Concatenates terms from degree zero through the requested degree along the
/// feature axis. For spectral filtering, supply a Laplacian rescaled to
/// spectrum [-1,1]. Both inputs retain their gradients.
///
/// @param graph_operator Float32/Float64 square [N,N] or batched [B,N,N] operator.
/// @param features Same-dtype node features [N,F] or [B,N,F], with positive F.
/// @param degree Highest polynomial degree, zero or greater.
/// @return [N,(degree+1)*F] or [B,N,(degree+1)*F] features in increasing degree order.
/// @throws std::invalid_argument If rank, dimensions, dtype or degree are incompatible.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::tensor({1., 2.}, {2, 1});
/// const auto basis = n::chebyshev_graph_basis(n::eye(2), x, 2); // rows [1,1,1] and [2,2,2].
/// @endcode
/// @par References
/// - Defferrard, Bresson, and Vandergheynst (2016), Convolutional Neural Networks
///   on Graphs with Fast Localized Spectral Filtering, https://arxiv.org/abs/1606.09375
/// @see matmul(), gcn_normalize()
inline Tensor chebyshev_graph_basis(const Tensor &graph_operator, const Tensor &features,
                                    int degree) {
  const auto rank = features.dim();
  if (degree < 0 || (rank != 2 && rank != 3) || graph_operator.dim() != rank ||
      graph_operator.size(rank - 2) <= 0 ||
      graph_operator.size(rank - 1) != graph_operator.size(rank - 2) ||
      features.size(rank - 2) != graph_operator.size(rank - 1) || features.size(rank - 1) <= 0 ||
      (rank == 3 && features.size(0) != graph_operator.size(0)) ||
      (features.dtype() != DType::Float32 && features.dtype() != DType::Float64) ||
      features.dtype() != graph_operator.dtype())
    throw std::invalid_argument("chebyshev_graph_basis: incompatible operator, features or degree");
  if (degree == 0)
    return features;

  std::vector<Tensor> terms{features, matmul(graph_operator, features)};
  for (int k = 1; k < degree; ++k)
    terms.push_back(2. * matmul(graph_operator, terms[k]) - terms[k - 1]);
  return concat(terms, rank - 1);
}

/// @brief Apply a graph isomorphism network layer with learned self weight.
///
/// Adds the neighbor sum to (1 + epsilon) times each node's features, then
/// applies two dense ReLU layers. Epsilon starts at zero. A zero mask entry
/// clears the output at a padded vertex. For a padded graph, clear its
/// incident edges and input features as well. Graphs may be batched.
/// The store owns the learned tensors. Synchronize shared-store mutation.
///
/// @param store Parameter store with the same dtype as the graph tensors.
/// @param name Layer path within the store.
/// @param adjacency [N,N] or [B,N,N] matrix without added self-loops.
/// @param features [N,F] or [B,N,F] node features, matching the adjacency.
/// @param mask [N,1] or [B,N,1] real-node indicators.
/// @param channels Positive width of both dense layers.
/// @return Node features [N,channels] or [B,N,channels] with live gradients.
/// @throws std::invalid_argument For incompatible graph, mask or parameter geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore parameters(7);
/// const auto mask = n::ones({2, 1});
/// const auto hidden = n::gin_layer(parameters, "gin", n::zeros({2, 2}), mask, mask, 4);
/// // hidden.shape() == {2, 4}.
/// @endcode
/// @par References
/// - Xu, Hu, Leskovec and Jegelka (2019), How Powerful are Graph Neural Networks?,
///   https://arxiv.org/abs/1810.00826
/// @see dense(), matmul()
inline Tensor gin_layer(ParameterStore &store, const std::string &name, const Tensor &adjacency,
                        const Tensor &features, const Tensor &mask, std::int64_t channels) {
  const auto rank = features.dim();
  if ((rank != 2 && rank != 3) || adjacency.dim() != rank || mask.dim() != rank ||
      channels <= 0 || features.size(rank - 2) <= 0 || features.size(rank - 1) <= 0 ||
      adjacency.size(rank - 2) != features.size(rank - 2) ||
      adjacency.size(rank - 1) != features.size(rank - 2) ||
      mask.size(rank - 2) != features.size(rank - 2) || mask.size(rank - 1) != 1 ||
      (rank == 3 && (adjacency.size(0) != features.size(0) || mask.size(0) != features.size(0))) ||
      adjacency.dtype() != store.dtype() || features.dtype() != store.dtype() ||
      mask.dtype() != store.dtype())
    throw std::invalid_argument("gin_layer: incompatible adjacency, features, mask or channels");

  const auto epsilon = store.get(name + "/epsilon", {1}, init::zeros());
  const auto aggregate = matmul(adjacency, features) + (1. + epsilon) * features;
  const auto hidden = dense(store, name + "/hidden", aggregate, channels, Activation::Relu);
  return mask * dense(store, name + "/out", hidden, channels, Activation::Relu);
}

/// @brief Apply a sampled mean GraphSAGE layer to features on one graph.
///
/// Gathers each node's requested neighbors, averages them, and concatenates
/// the result with that node's features. A dense ReLU projection follows.
/// Rows are divided by the square root of their squared norm floored at
/// 1e-8. Zero rows stay zero. The caller owns the sample, including the
/// handling of isolated nodes. The store owns the learned parameters.
/// Synchronize calls that mutate the same store.
///
/// @param store Parameter store with the same dtype as features.
/// @param name Layer path within the store.
/// @param features Float32/64 [N,F] node features with positive N and F.
/// @param neighbors Int32/64 [N,K] valid zero-based neighbor indices with positive K.
/// @param channels Positive output width.
/// @return Live [N,channels] normalized embeddings.
/// @throws std::invalid_argument For incompatible shapes, dtypes or index values.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore parameters(7);
/// const auto neighbors = n::from_vector(std::vector<int>{1,0}, {2,1});
/// const auto h = n::graphsage_mean_layer(parameters, "sage", n::ones({2,3}), neighbors, 4);
/// // h.shape() == {2,4}.
/// @endcode
/// @par References
/// - Hamilton, Ying, and Leskovec (2017), Inductive Representation Learning
///   on Large Graphs, https://arxiv.org/abs/1706.02216
/// @see gather(), dense()
inline Tensor graphsage_mean_layer(ParameterStore &store, const std::string &name,
                                   const Tensor &features, const Tensor &neighbors,
                                   std::int64_t channels) {
  if (features.dim() != 2 || features.size(0) <= 0 || features.size(1) <= 0 ||
      features.dtype() != store.dtype() || neighbors.dim() != 2 ||
      neighbors.size(0) != features.size(0) || neighbors.size(1) <= 0 ||
      (neighbors.dtype() != DType::Int32 && neighbors.dtype() != DType::Int64) || channels <= 0)
    throw std::invalid_argument("graphsage_mean_layer: incompatible features or neighbors");
  const auto ids = neighbors.to_vector<std::int64_t>();
  for (const auto id : ids)
    if (id < 0 || id >= features.size(0))
      throw std::invalid_argument("graphsage_mean_layer: invalid neighbor index");

  const auto neighbor_mean = mean(gather(features, neighbors), {1});
  const auto projected = dense(store, name, concat({features, neighbor_mean}, 1), channels,
                               Activation::Relu);
  const auto floor = full({}, 1e-8, projected.dtype(), projected.context());
  return projected * rsqrt(maximum(sum(square(projected), {1}, true), floor));
}

/// @brief Own the score, attention, and output tensors of a graph attention head.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(7);
/// const auto head = n::graph_attention_head(p, "gat", n::ones({1,2,3}),
///                                          n::ones({2,2}, n::DType::Bool), 4);
/// // head.output.shape() == {1,2,4}.
/// @endcode
/// @see graph_attention_head()
struct GraphAttentionHead {
  Tensor scores; ///< Unmasked [B,N,N] logits, receiver axis before sender axis.
  Tensor attention; ///< [B,N,N] weights, zero at disabled edges.
  Tensor output; ///< [B,N,channels] weighted sender projections.
};

/// @brief Apply a single GAT or GATv2 attention head to a batch of graphs.
///
/// GAT applies LeakyReLU to the sum of scalar sender and receiver
/// projections. GATv2 applies it to the sum of vector projections before
/// scoring. Each receiver softmaxes over its allowed senders. A receiver
/// with no allowed senders has zero attention and output. The caller
/// specifies self-edges in the mask. The store owns all learned projections.
/// Synchronize concurrent mutation of a shared store.
///
/// @param store Parameter store with the same dtype as features.
/// @param name Layer path for learned projections.
/// @param features Finite Float32/64 [B,N,F] node features with positive dimensions.
/// @param allowed Bool [N,N] shared mask or [B,N,N] per-graph mask.
/// @param channels Positive projection width.
/// @param dynamic Use GATv2 scoring if true. Default false uses GAT.
/// @return Unmasked scores, masked attention, and mixed features with live gradients.
/// @throws std::invalid_argument For incompatible feature or mask geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(7);
/// const auto head = n::graph_attention_head(p, "gatv2", n::ones({1,2,3}),
///                                          n::ones({2,2}, n::DType::Bool), 4, true);
/// // Each attention row sums to one.
/// @endcode
/// @par References
/// - Veličković et al. (2018), Graph Attention Networks, https://arxiv.org/abs/1710.10903
/// - Brody, Alon, and Yahav (2022), How Attentive are Graph Attention Networks?,
///   https://arxiv.org/abs/2105.14491
/// @see softmax(), matmul()
inline GraphAttentionHead graph_attention_head(ParameterStore &store, const std::string &name,
                                                const Tensor &features, const Tensor &allowed,
                                                std::int64_t channels, bool dynamic = false) {
  if (features.dim() != 3 || features.size(0) <= 0 || features.size(1) <= 0 ||
      features.size(2) <= 0 || features.dtype() != store.dtype() || channels <= 0 ||
      allowed.dtype() != DType::Bool || (allowed.dim() != 2 && allowed.dim() != 3) ||
      allowed.size(allowed.dim() - 1) != features.size(1) ||
      allowed.size(allowed.dim() - 2) != features.size(1) ||
      (allowed.dim() == 3 && allowed.size(0) != features.size(0)))
    throw std::invalid_argument("graph_attention_head: incompatible features or edge mask");

  const auto project = [&](const Tensor &x, const std::string &id, std::int64_t width) {
    return matmul(x, store.get(name + "/" + id, {x.shape().back(), width}, init::glorot_uniform()));
  };
  const auto values = project(features, "value", channels);
  Tensor scores;
  if (dynamic) {
    const auto query = project(features, "query", channels);
    const auto pairs = leaky_relu(expand_dims(query, 2) + expand_dims(values, 1), .2);
    scores = squeeze(project(pairs, "attention_vector", 1), {-1});
  } else {
    const auto left = project(values, "left", 1);
    const auto right = transpose(project(values, "right", 1), {0, 2, 1});
    scores = leaky_relu(left + right, .2);
  }
  const auto masked = where(allowed, scores, -std::numeric_limits<double>::infinity());
  const auto safe = where(any(allowed, {-1}, true), masked, 0.);
  const auto attention = where(allowed, softmax(safe, -1), 0.);
  return {scores, attention, matmul(attention, values)};
}

/// @brief Apply a basis-decomposed relational graph convolution.
///
/// Each relation has its own weighted incoming sum. Row weights are divided
/// by their sum floored at one, so empty neighborhoods stay zero. Learned
/// coefficients combine shared bases into one transform per relation. A
/// separate biased dense transform supplies the self-node contribution.
/// The caller chooses any output activation. Synchronize store mutation.
///
/// @param store Parameter store owning the learned tensors.
/// @param name Layer path within the store.
/// @param features Float32/64 [B,N,F] node features matching the store dtype.
/// @param edges [B,R,N,N] nonnegative adjacency weights in receiver/sender order.
/// @param channels Positive output width.
/// @param bases Positive number of shared bases.
/// @return Live [B,N,channels] sum of self and relation contributions.
/// @throws std::invalid_argument For incompatible shapes, dtypes or widths.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(7);
/// const auto y = n::rgcn_layer(p, "relation", n::ones({1,2,3}),
///                             n::zeros({1,2,2,2}), 4, 2);
/// // y.shape() == {1,2,4}.
/// @endcode
/// @par References
/// - Schlichtkrull et al. (2018), Modeling Relational Data with Graph
///   Convolutional Networks, https://arxiv.org/abs/1703.06103
/// @see gcn_normalize(), dense()
inline Tensor rgcn_layer(ParameterStore &store, const std::string &name,
                         const Tensor &features, const Tensor &edges,
                         std::int64_t channels, std::int64_t bases) {
  if (features.dim() != 3 || edges.dim() != 4 || features.size(0) <= 0 ||
      features.size(1) <= 0 || features.size(2) <= 0 || edges.size(1) <= 0 ||
      features.size(0) != edges.size(0) || features.size(1) != edges.size(2) ||
      features.size(1) != edges.size(3) || features.dtype() != edges.dtype() ||
      features.dtype() != store.dtype() || channels <= 0 || bases <= 0)
    throw std::invalid_argument("rgcn_layer: incompatible features, relations or widths");

  const auto in = features.size(2), relations = edges.size(1);
  const auto basis = store.get(name + "/bases", {bases, in * channels}, init::uniform(.35));
  const auto coefficients = store.get(name + "/coefficients", {relations, bases},
                                      init::uniform(.65));
  const auto transforms = matmul(coefficients, basis).reshape({relations, in, channels});
  const auto unit = full({}, 1., edges.dtype(), edges.context());
  const auto normalized = edges / maximum(sum(edges, {3}, true), unit);
  auto output = dense(store, name + "/self", features, channels);
  for (std::int64_t r = 0; r < relations; ++r) {
    const auto transformed = matmul(features, select(transforms, 0, r));
    output = output + matmul(select(normalized, 1, r), transformed);
  }
  return output;
}

/// @brief Sum signed-edge messages and update node states with a shared GRU.
///
/// Positive and negative edges select separate learned linear maps. Edge
/// magnitudes are ignored. Zero entries contribute no messages, and a node
/// with no incoming edges updates from a zero message and its old state.
/// Reuse the same name and store to share weights across rounds.
///
/// @param store Parameter store owning the message and GRU parameters.
/// @param name Layer path within the store.
/// @param state Float32/64 [B,N,H] current node states matching the store dtype.
/// @param edges Same-dtype [B,N,N] signed adjacency in receiver/sender order.
/// @return Pair of live [B,N,H] node states and aggregate messages, in that order.
/// @throws std::invalid_argument For incompatible shapes or dtypes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(7);
/// const auto [state, messages] = n::signed_graph_gru_step(
///     p, "signed", n::ones({1,2,3}), n::zeros({1,2,2}));
/// // messages is all zero.
/// @endcode
/// @par References
/// - Gilmer et al. (2017), Neural Message Passing for Quantum Chemistry,
///   https://proceedings.mlr.press/v70/gilmer17a.html
/// @see gru_step(), rgcn_layer()
inline std::pair<Tensor, Tensor> signed_graph_gru_step(ParameterStore &store,
                                                       const std::string &name,
                                                       const Tensor &state,
                                                       const Tensor &edges) {
  if (state.dim() != 3 || edges.dim() != 3 || state.size(0) <= 0 ||
      state.size(1) <= 0 || state.size(2) <= 0 || state.size(0) != edges.size(0) ||
      state.size(1) != edges.size(1) || state.size(1) != edges.size(2) ||
      state.dtype() != store.dtype() || state.dtype() != edges.dtype())
    throw std::invalid_argument("signed_graph_gru_step: incompatible states or edges");

  const auto width = state.size(2);
  const auto positive = store.get(name + "/positive", {width, width}, init::uniform(.35));
  const auto negative = store.get(name + "/negative", {width, width}, init::uniform(.35));
  const auto message = matmul(greater(edges, 0.).to(state.dtype()), matmul(state, positive)) +
                       matmul(less(edges, 0.).to(state.dtype()), matmul(state, negative));
  const auto next = gru_step(store, name + "/update", message.reshape({-1, width}),
                             state.reshape({-1, width}), 0.);
  return {next.reshape(state.shape()), message};
}

} // namespace datamunge::nn
