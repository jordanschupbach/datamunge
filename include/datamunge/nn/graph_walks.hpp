/// @file
/// @brief Seeded graph walks and skip-gram context pairs.
#pragma once

#include <datamunge/nn/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Own a corpus of node walks, sampling draws, and ordered context pairs.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::mt19937 rng(7);
/// const auto graph = n::tensor({0.,1.,1.,0.}, {2,2}, n::DType::Int32);
/// const auto corpus = n::node_walk_corpus(graph, 1, 3, 1, rng);
/// // corpus.walks contains [0,1,0] and [1,0,1].
/// @endcode
/// @see node_walk_corpus()
struct NodeWalkCorpus {
  Tensor walks; ///< Int32 [walks,length], including the starting node.
  Tensor draws; ///< Float64 [walks,length-1] uniforms consumed in sampling order.
  Tensor pairs; ///< Int32 [pairs,2], with the center ID before the context ID.
};

/// @brief Sample DeepWalk or node2vec walks and extract skip-gram pairs.
///
/// Uses a binary adjacency in source/destination order. Starts each node
/// once per round in ascending node order. The first transition is uniform.
/// Subsequent transitions weight a return by 1/p, an edge to a neighbor of
/// the previous node by 1, and an outward edge by 1/q. The supplied RNG
/// advances once per transition. The context window stays inside each walk.
/// The result owns CPU tensors.
///
/// @param adjacency Square Int32/64 binary adjacency with at least one outgoing edge per node.
/// @param per_node Positive number of walks per start node.
/// @param length Length of each walk, including its start, at least two.
/// @param window Positive context radius.
/// @param rng Caller-owned MT19937 engine, advanced in place.
/// @param p Positive finite return parameter. Default one.
/// @param q Positive finite outward parameter. Default one.
/// @return Walks, uniforms, and ordered skip-gram pairs.
/// @throws std::invalid_argument For invalid dimensions, graph entries or transition weights.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::mt19937 rng(7);
/// const auto graph = n::tensor({0.,1.,1.,0.}, {2,2}, n::DType::Int32);
/// const auto corpus = n::node_walk_corpus(graph, 1, 3, 1, rng, .5, 2.);
/// // corpus.pairs.shape() == {8,2}.
/// @endcode
/// @par References
/// - Perozzi, Al-Rfou, and Skiena (2014), DeepWalk, https://arxiv.org/abs/1403.6652
/// - Grover and Leskovec (2016), node2vec, https://arxiv.org/abs/1607.00653
/// @see NodeWalkCorpus, gather()
inline NodeWalkCorpus node_walk_corpus(const Tensor &adjacency, int per_node, int length,
                                        int window, std::mt19937 &rng, double p = 1.,
                                        double q = 1.) {
  if (adjacency.dim() != 2 || adjacency.size(0) <= 0 ||
      adjacency.size(0) != adjacency.size(1) ||
      (adjacency.dtype() != DType::Int32 && adjacency.dtype() != DType::Int64) ||
      adjacency.size(0) > std::numeric_limits<int>::max() ||
      per_node <= 0 || length < 2 || window <= 0 || !std::isfinite(p) ||
      !std::isfinite(q) || p <= 0 || q <= 0)
    throw std::invalid_argument("node_walk_corpus: invalid adjacency or walk parameters");
  const auto nodes = adjacency.size(0);
  const auto graph = adjacency.to_vector<std::int64_t>();
  std::vector<std::vector<int>> neighbors(nodes);
  for (int a = 0; a < nodes; ++a) {
    for (int b = 0; b < nodes; ++b) {
      const auto edge = graph[a * nodes + b];
      if (edge < 0 || edge > 1)
        throw std::invalid_argument("node_walk_corpus: adjacency must be binary");
      if (edge)
        neighbors[a].push_back(b);
    }
    if (neighbors[a].empty())
      throw std::invalid_argument("node_walk_corpus: a node has no outgoing edges");
  }
  const double back = 1. / p, outward = 1. / q;
  if (!std::isfinite((back + outward + 1.) * nodes))
    throw std::invalid_argument("node_walk_corpus: transition weights can overflow");

  std::vector<int> walks, pairs, walk(length);
  std::vector<double> draws, weights;
  for (int repeat = 0; repeat < per_node; ++repeat)
    for (int start = 0; start < nodes; ++start) {
      walk[0] = start;
      for (int i = 1; i < length; ++i) {
        const auto &next = neighbors[walk[i - 1]];
        const int previous = i > 1 ? walk[i - 2] : -1;
        weights.clear();
        double total = 0.;
        for (const int node : next) {
          const double weight = previous < 0 ? 1. : node == previous ? back
                                : graph[previous * nodes + node] ? 1. : outward;
          weights.push_back(weight);
          total += weight;
        }
        const double uniform = (double(rng()) + .5) / 4294967296.;
        draws.push_back(uniform);
        double remaining = uniform * total;
        walk[i] = next.back();
        for (std::size_t j = 0; j < next.size(); ++j) {
          remaining -= weights[j];
          if (remaining < 0.) {
            walk[i] = next[j];
            break;
          }
        }
      }
      walks.insert(walks.end(), walk.begin(), walk.end());
      for (std::int64_t center = 0; center < length; ++center)
        for (auto context = std::max<std::int64_t>(0, center - window);
             context <= std::min<std::int64_t>(length - 1, center + window); ++context)
          if (context != center)
            pairs.insert(pairs.end(), {walk[center], walk[context]});
    }
  return {from_vector(walks, {nodes * per_node, length}),
          from_vector(draws, {nodes * per_node, length - 1}),
          from_vector(pairs, {static_cast<std::int64_t>(pairs.size() / 2), 2})};
}

} // namespace datamunge::nn
