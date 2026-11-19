/// @file
/// @brief Embedding scores and filtered ranking for knowledge graph triples.
#pragma once

#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/params.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::nn {

/// @brief Select a relation embedding and triple scoring rule.
enum class TripleScorer {
  TransE, ///< Real additive translation with L1 distance.
  DistMult, ///< Symmetric real trilinear product.
  ComplEx, ///< Complex trilinear product with conjugated tail.
  RotatE ///< Complex distance after a unit-modulus rotation.
};

/// @brief Score triples with one of four learned embedding models.
///
/// Larger scores indicate more plausible triples. The two distance
/// models use a margin of two. RotatE sums complex magnitudes with
/// 1e-8 added under each square root. The store owns all embeddings.
/// Synchronize shared-store mutation.
///
/// @param store Parameter store owning the embeddings.
/// @param name Parameter path within the store.
/// @param triples Int32/64 [T,3] head, relation, and tail IDs.
/// @param entities Positive entity vocabulary size.
/// @param relations Positive relation vocabulary size.
/// @param width Positive embedding width.
/// @param kind Scoring rule.
/// @return Live [T] triple scores.
/// @throws std::invalid_argument For invalid IDs, shapes, vocabulary sizes or scoring rule.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore parameters(7);
/// const auto triples = n::from_vector(std::vector<int>{0,0,1}, {1,3});
/// const auto scores = n::knowledge_graph_scores(parameters, "kge", triples, 2, 1, 4,
///                                                n::TripleScorer::DistMult);
/// // scores.shape() == {1}.
/// @endcode
/// @par References
/// - Bordes et al. (2013), Translating Embeddings for Modeling Multi-relational Data.
/// - Yang et al. (2015), Embedding Entities and Relations for Learning and Inference in Knowledge Bases.
/// - Trouillon et al. (2016), Complex Embeddings for Simple Link Prediction.
/// - Sun et al. (2019), RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.
/// @see filtered_triple_ranks()
inline Tensor knowledge_graph_scores(ParameterStore &store, const std::string &name,
                                     const Tensor &triples, std::int64_t entities,
                                     std::int64_t relations, std::int64_t width,
                                     TripleScorer kind) {
  if (triples.dim() != 2 || triples.size(0) <= 0 || triples.size(1) != 3 ||
      (triples.dtype() != DType::Int32 && triples.dtype() != DType::Int64) ||
      entities <= 0 || relations <= 0 || width <= 0 ||
      (kind != TripleScorer::TransE && kind != TripleScorer::DistMult &&
       kind != TripleScorer::ComplEx && kind != TripleScorer::RotatE))
    throw std::invalid_argument("knowledge_graph_scores: invalid triple geometry");
  const auto ids = triples.to_vector<std::int64_t>();
  for (std::size_t i = 0; i < ids.size(); ++i)
    if (ids[i] < 0 || ids[i] >= (i % 3 == 1 ? relations : entities))
      throw std::invalid_argument("knowledge_graph_scores: invalid triple ID");

  const auto entity = store.get(name + "/entity", {entities, width}, init::uniform(.35));
  const auto relation = store.get(name + "/relation", {relations, width}, init::uniform(.35));
  const auto head_ids = select(triples, 1, 0), relation_ids = select(triples, 1, 1),
             tail_ids = select(triples, 1, 2);
  const auto h = gather(entity, head_ids), r = gather(relation, relation_ids),
             t = gather(entity, tail_ids);
  if (kind == TripleScorer::TransE)
    return 2. - sum(abs(h + r - t), {1});
  if (kind == TripleScorer::DistMult)
    return sum(h * r * t, {1});

  const auto imaginary = store.get(name + "/entity_imaginary", {entities, width}, init::uniform(.35));
  const auto hi = gather(imaginary, head_ids), ti = gather(imaginary, tail_ids);
  const bool rotation = kind == TripleScorer::RotatE;
  const auto rr = rotation ? cos(r) : r;
  const auto ri = rotation ? sin(r) : gather(
      store.get(name + "/relation_imaginary", {relations, width}, init::uniform(.35)), relation_ids);
  const auto real = h * rr - hi * ri, imag = h * ri + hi * rr;
  if (rotation)
    return 2. - sum(sqrt(square(real - t) + square(imag - ti) + 1e-8), {1});
  return sum(real * t + imag * ti, {1});
}

/// @brief Rank held-out heads and tails after filtering other known positives.
///
/// For each target, starts rank at one and adds one per higher-scoring
/// candidate and one half per tie. Candidates marked as known truths
/// are filtered. Ranks are computed on the host and detached.
///
/// @param scores Finite Float32/64 [E,R,E] table in head, relation, tail order.
/// @param known Bool [E,R,E] mask of all known positives, including evaluation targets.
/// @param queries Int32/64 [Q,3] known positive triples to rank.
/// @return Float64 [Q,2] head rank followed by tail rank.
/// @throws std::invalid_argument For incompatible shapes, dtypes, nonfinite scores or unknown queries.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto score = n::tensor({0.,2.,1.,0.}, {2,1,2});
/// const auto truth = n::tensor({0.,1.,0.,0.}, {2,1,2}, n::DType::Bool);
/// const auto query = n::from_vector(std::vector<int>{0,0,1}, {1,3});
/// const auto ranks = n::filtered_triple_ranks(score, truth, query);
/// // ranks contains [1,1].
/// @endcode
/// @par References
/// - Bordes et al. (2013), Translating Embeddings for Modeling Multi-relational Data.
/// @see knowledge_graph_scores()
inline Tensor filtered_triple_ranks(const Tensor &scores, const Tensor &known,
                                    const Tensor &queries) {
  if (scores.dim() != 3 || scores.size(0) <= 0 || scores.size(1) <= 0 ||
      scores.size(0) != scores.size(2) ||
      (scores.dtype() != DType::Float32 && scores.dtype() != DType::Float64) ||
      known.dtype() != DType::Bool || known.shape() != scores.shape() ||
      queries.dim() != 2 || queries.size(1) != 3 ||
      (queries.dtype() != DType::Int32 && queries.dtype() != DType::Int64))
    throw std::invalid_argument("filtered_triple_ranks: incompatible score, mask or queries");
  const auto entities = scores.size(0), relations = scores.size(1);
  const auto values = scores.to_vector<double>();
  const auto truth = known.to_vector<int>();
  const auto ids = queries.to_vector<std::int64_t>();
  for (const auto x : values)
    if (!std::isfinite(x))
      throw std::invalid_argument("filtered_triple_ranks: nonfinite score");
  const auto index = [&](std::int64_t h, std::int64_t r, std::int64_t t) {
    return (h * relations + r) * entities + t;
  };
  std::vector<double> ranks;
  for (std::size_t i = 0; i < ids.size(); i += 3) {
    const auto h = ids[i], r = ids[i + 1], t = ids[i + 2];
    if (h < 0 || h >= entities || r < 0 || r >= relations || t < 0 || t >= entities ||
        !truth[index(h, r, t)])
      throw std::invalid_argument("filtered_triple_ranks: expected known positive query");
    const auto target = values[index(h, r, t)];
    for (const bool tail : {false, true}) {
      double rank = 1.;
      for (std::int64_t c = 0; c < entities; ++c) {
        const auto addressed = index(tail ? h : c, r, tail ? c : t);
        if (!truth[addressed]) {
          const auto alternative = values[addressed];
          rank += alternative > target ? 1. : alternative == target ? .5 : 0.;
        }
      }
      ranks.push_back(rank);
    }
  }
  return from_vector(ranks, {queries.size(0), 2});
}

} // namespace datamunge::nn
