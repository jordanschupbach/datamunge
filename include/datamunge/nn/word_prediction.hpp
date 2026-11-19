/// @file
/// @brief Fixed word windows and sampled CBOW and skip-gram objectives.
#pragma once

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/word_embeddings.hpp>

#include <cmath>
#include <limits>
#include <random>

namespace datamunge::nn {

/// @brief Generate synthetic sentences from three disjoint four-word topics.
///
/// Sentence row modulo three selects the topic. Words within a topic have
/// relative frequencies 4, 3, 2, 1 and are sampled with replacement.
///
/// @param count Positive number of sentences.
/// @param seed Seed for the local Mersenne Twister stream.
/// @return Int32 sentences [count,8] with token IDs in [0,12).
/// @throws std::invalid_argument If count is nonpositive.
/// @par Example
/// @code
/// const auto corpus = dm::nn::synthetic_topic_corpus(6, 7);
/// @endcode
/// @see sampled_word_windows()
inline Tensor synthetic_topic_corpus(int count, unsigned seed) {
  if (count <= 0)
    throw std::invalid_argument("synthetic_topic_corpus: positive count required");
  std::mt19937 rng(seed);
  std::discrete_distribution<int> word{4., 3., 2., 1.};
  std::vector<int> tokens(std::size_t(count) * 8);
  for (int row = 0; row < count; ++row)
    for (int i = 0; i < 8; ++i)
      tokens[std::size_t(row) * 8 + i] = 4 * (row % 3) + word(rng);
  return from_vector(tokens, {count, 8});
}

/// @brief Own the inputs and sampled candidates for a word prediction objective.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto batch = n::sampled_word_windows(n::synthetic_topic_corpus(3,7), n::ones({12}), true, 8);
/// @endcode
/// @see sampled_word_windows()
struct SampledWordWindows {
  Tensor source;     ///< Int32 source IDs [N,4] for CBOW or [N,1] for skip-gram.
  Tensor candidates; ///< Int32 target then four negative IDs per row [N,5].
};

/// @brief Extract complete radius-two windows and sample four negatives per target.
///
/// Windows never cross sentence boundaries. CBOW predicts the center from
/// four surrounding words. Skip-gram predicts each surrounding word from
/// the center. Negatives exclude the positive ID and may repeat one another.
///
/// @param corpus Integral [S,L] token IDs with S positive and L >= 5.
/// @param weights Floating [V] nonnegative finite sampling weights with at least two positive entries.
/// @param cbow True for CBOW, false for skip-gram.
/// @param seed Local Mersenne Twister seed.
/// @return Detached source IDs and candidate IDs in sentence and window order.
/// @throws std::invalid_argument For invalid shapes, token IDs or weights.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto batch = n::sampled_word_windows(n::synthetic_topic_corpus(3,7), n::ones({12}), true, 8);
/// @endcode
/// @par References
/// - Mikolov et al. (2013), https://arxiv.org/abs/1310.4546
/// @see synthetic_topic_corpus(), sampled_word_logits()
inline SampledWordWindows sampled_word_windows(const Tensor &corpus, const Tensor &weights,
                                               bool cbow, unsigned seed) {
  if (corpus.dim() != 2 || !is_integral(corpus.dtype()) || corpus.shape()[0] < 1 ||
      corpus.shape()[1] < 5 || weights.dim() != 1 || !is_floating(weights.dtype()) ||
      weights.shape()[0] > std::numeric_limits<int>::max())
    throw std::invalid_argument("sampled_word_windows: invalid corpus or weights");
  const auto mass = weights.to_vector<double>();
  const auto tokens = corpus.to_vector<std::int64_t>();
  int positive = 0;
  double total = 0.;
  for (double w : mass) {
    if (!std::isfinite(w) || w < 0.)
      throw std::invalid_argument("sampled_word_windows: invalid weight");
    positive += w > 0.;
    total += w;
  }
  if (positive < 2 || !std::isfinite(total))
    throw std::invalid_argument("sampled_word_windows: requires finite mass on at least two words");
  for (auto token : tokens)
    if (token < 0 || token >= weights.shape()[0])
      throw std::invalid_argument("sampled_word_windows: token outside vocabulary");

  std::mt19937 rng(seed);
  std::discrete_distribution<int> draw(mass.begin(), mass.end());
  std::vector<int> source, candidate;
  const auto add = [&](std::initializer_list<int> ids, int target) {
    source.insert(source.end(), ids.begin(), ids.end());
    candidate.push_back(target);
    for (int i = 0; i < 4; ++i) {
      int sampled = target;
      for (int retry = 0; retry < 128 && sampled == target; ++retry)
        sampled = draw(rng);
      if (sampled == target) {
        auto filtered = mass;
        filtered[target] = 0.;
        std::discrete_distribution<int> conditional(filtered.begin(), filtered.end());
        sampled = conditional(rng);
      }
      candidate.push_back(sampled);
    }
  };
  const auto length = corpus.shape()[1];
  for (std::int64_t row = 0; row < corpus.shape()[0]; ++row)
    for (std::int64_t col = 2; col < length - 2; ++col) {
      const auto i = row * length + col;
      const int l2 = tokens[i-2], l1 = tokens[i-1], center = tokens[i];
      const int r1 = tokens[i+1], r2 = tokens[i+2];
      if (cbow)
        add({l2, l1, r1, r2}, center);
      else
        for (int nearby : {l2, l1, r1, r2})
          add({center}, nearby);
    }
  const auto rows = static_cast<std::int64_t>(candidate.size() / 5);
  return {from_vector(source, {rows, cbow ? 4 : 1}), from_vector(candidate, {rows, 5})};
}

/// @brief Score word candidates against a mean-pooled source embedding.
///
/// Single source IDs give skip-gram. Multiple source IDs give CBOW.
/// Repeated IDs contribute once per occurrence and accumulate gradients.
///
/// @param input Floating source embedding table [V_in,D].
/// @param output Floating candidate embedding table [V_out,D].
/// @param source Integral source IDs [N,S], all in [0,V_in).
/// @param candidates Integral candidate IDs [N,K], all in [0,V_out).
/// @return Differentiable candidate logits [N,K].
/// @throws std::invalid_argument If table or ID geometry is incompatible.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto b = n::sampled_word_windows(n::synthetic_topic_corpus(3,7), n::ones({12}), false, 8);
/// const auto logits = n::sampled_word_logits(n::ones({12,8}), n::ones({12,8}), b.source, b.candidates);
/// @endcode
/// @see sampled_word_windows()
inline Tensor sampled_word_logits(const Tensor &input, const Tensor &output,
                                  const Tensor &source, const Tensor &candidates) {
  if (input.dim() != 2 || output.dim() != 2 || input.shape()[1] != output.shape()[1] ||
      input.shape()[0] < 1 || output.shape()[0] < 1 ||
      input.shape()[1] < 1 || !is_floating(input.dtype()) || !is_floating(output.dtype()) ||
      source.dim() != 2 || candidates.dim() != 2 || source.shape()[0] != candidates.shape()[0] ||
      source.shape()[1] < 1 || candidates.shape()[1] < 1 ||
      !is_integral(source.dtype()) || !is_integral(candidates.dtype()))
    throw std::invalid_argument("sampled_word_logits: incompatible tables or IDs");
  const auto width = input.shape()[1], count = source.shape()[0];
  const auto context = mean(gather(input, source.reshape({count * source.shape()[1]}))
                            .reshape({count, source.shape()[1], width}), {1}, true);
  const auto words = gather(output, candidates.reshape({count * candidates.shape()[1]}))
                         .reshape({count, candidates.shape()[1], width});
  return sum(context * words, {2});
}

/// @brief Own the nonzero directed pairs and harmonic co-occurrence counts.
/// @par Example
/// @code
/// const auto data = dm::nn::glove_pairs(dm::nn::synthetic_topic_corpus(6,7), 12);
/// @endcode
/// @see glove_pairs()
struct GloVePairs {
  Tensor ids;    ///< Int64 [N,2] center and context token IDs in lexicographic order.
  Tensor counts; ///< Float64 [N] strictly positive weighted counts.
};

/// @brief Count directed within-sentence co-occurrences weighted by inverse distance.
///
/// Boundary windows are truncated. A pair of distinct positions contributes
/// in both directions, including two diagonal contributions when the token
/// IDs match. Counting uses a dense vocabulary-square workspace.
///
/// @param corpus Integral token IDs [S,L], each in [0,vocab).
/// @param vocab Positive vocabulary size.
/// @param window Positive context radius, default 2.
/// @return Only positive-count rows, including valid empty arrays for no pairs.
/// @throws std::invalid_argument For invalid shapes, vocabulary, radius or token IDs.
/// @par Example
/// @code
/// const auto data = dm::nn::glove_pairs(dm::nn::synthetic_topic_corpus(6,7), 12);
/// @endcode
/// @par References
/// - Pennington et al. (2014), https://aclanthology.org/D14-1162/
/// @see cooccurrence_matrix(), glove()
inline GloVePairs glove_pairs(const Tensor &corpus, std::int64_t vocab, std::int64_t window = 2) {
  if (corpus.dim() != 2 || !is_integral(corpus.dtype()) || vocab < 1 ||
      vocab > std::numeric_limits<std::int64_t>::max() / vocab || window < 1)
    throw std::invalid_argument("glove_pairs: invalid corpus, vocabulary or radius");
  const auto flat = corpus.to_vector<std::int64_t>();
  const auto length = corpus.shape()[1];
  std::vector<std::vector<std::int64_t>> sentences;
  for (std::int64_t row = 0; row < corpus.shape()[0] && length; ++row)
    sentences.emplace_back(flat.begin() + row * length, flat.begin() + (row + 1) * length);
  const auto matrix = cooccurrence_matrix(sentences, vocab, std::min(window, std::max(length, std::int64_t{1})), true)
                          .to_vector<double>();
  std::vector<std::int64_t> ids;
  std::vector<double> counts;
  for (std::int64_t i = 0; i < vocab; ++i)
    for (std::int64_t j = 0; j < vocab; ++j)
      if (const auto count = matrix[i * vocab + j]; count > 0.) {
        ids.insert(ids.end(), {i, j});
        counts.push_back(count);
      }
  return {from_vector(ids, {std::int64_t(counts.size()), 2}), from_vector(counts)};
}

/// @brief Own GloVe log-count predictions, weights and embeddings.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto data = dm::nn::glove_pairs(dm::nn::synthetic_topic_corpus(6,7), 12);
/// const auto fit = dm::nn::glove(model, "glove", data, 12);
/// @endcode
/// @see glove()
struct GloVeResult {
  Tensor prediction; ///< Predicted log counts [N].
  Tensor weight;     ///< Count-dependent regression weights [N].
  Tensor embedding;  ///< Sum of word and context embeddings [V,D].
  Tensor loss;       ///< Scalar mean weighted squared log-count error.
};

/// @brief Evaluate weighted log-count regression with word and context factors.
///
/// Predictions add a word-context dot product and two bias terms. Each row
/// receives weight min(1, (count / 100)^0.75).
///
/// @param model Owns all four parameter tensors.
/// @param name Prefix reused on subsequent calls.
/// @param pairs Nonempty valid word-context IDs and finite positive counts.
/// @param vocab Positive vocabulary size.
/// @param width Embedding dimension, default 8.
/// @return Predictions, weights, combined embedding and differentiable loss.
/// @throws std::invalid_argument For incompatible inputs or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto data = dm::nn::glove_pairs(dm::nn::synthetic_topic_corpus(6,7), 12);
/// const auto fit = dm::nn::glove(model, "glove", data, 12);
/// @endcode
/// @par References
/// - Pennington et al. (2014), https://aclanthology.org/D14-1162/
/// @see glove_pairs()
inline GloVeResult glove(ParameterStore &model, const std::string &name, const GloVePairs &pairs,
                         std::int64_t vocab, int width = 8) {
  if (pairs.ids.dim() != 2 || pairs.ids.shape()[1] != 2 || pairs.ids.shape()[0] < 1 ||
      pairs.counts.shape() != Shape{pairs.ids.shape()[0]} || !is_floating(pairs.counts.dtype()) ||
      !is_integral(pairs.ids.dtype()) || vocab < 1 || width < 1)
    throw std::invalid_argument("glove: invalid pair geometry or vocabulary");
  if (!all(is_finite(pairs.counts)).item<bool>() || min(pairs.counts).item() <= 0. ||
      min(pairs.ids).item<std::int64_t>() < 0 || max(pairs.ids).item<std::int64_t>() >= vocab)
    throw std::invalid_argument("glove: invalid counts or token IDs");
  const auto word = model.get(name + "/word", {vocab, width}, init::uniform(.25));
  const auto context = model.get(name + "/context", {vocab, width}, init::uniform(.25));
  const auto word_bias = model.get(name + "/word_bias", {vocab, 1}, init::zeros());
  const auto context_bias = model.get(name + "/context_bias", {vocab, 1}, init::zeros());
  const auto i = select(pairs.ids, 1, 0), j = select(pairs.ids, 1, 1);
  const auto prediction = sum(gather(word, i) * gather(context, j), {1}) +
                          (gather(word_bias, i) + gather(context_bias, j)).reshape({-1});
  const auto &count = pairs.counts;
  const auto weight = minimum(pow(count / 100., .75), scalar(1., count.dtype()));
  return {prediction, weight, word + context, mean(weight * square(prediction - log(count)))};
}

/// @brief Count hashed fastText subwords for an ordered word list.
///
/// Add boundary markers and hash every 3- to 6-byte substring with
/// unsigned-byte FNV-1a. Collisions and repeated occurrences add counts.
/// The first known words have a dedicated identity feature each. Later
/// words use bucket features only. This compact fixture uses byte lengths,
/// which coincide with character lengths for ASCII words.
///
/// @param words Ordered byte strings, with known words preceding unseen words.
/// @param known Number of known words, in [0,words.size()].
/// @param buckets Positive count of hashed feature columns, default 128.
/// @return Dense Float32 count matrix [words.size(),known+buckets].
/// @throws std::invalid_argument For invalid counts or matrix dimensions.
/// @par Example
/// @code
/// const auto features = dm::nn::fasttext_subword_features({"walk","walker"}, 1);
/// @endcode
/// @par References
/// - Bojanowski et al. (2017), https://aclanthology.org/Q17-1010/
/// @see sampled_word_logits()
inline Tensor fasttext_subword_features(const std::vector<std::string> &words,
                                        std::int64_t known, int buckets = 128) {
  constexpr auto max_dim = std::numeric_limits<std::int64_t>::max();
  if (known < 0 || std::uint64_t(known) > words.size() || buckets < 1 ||
      known > max_dim - buckets)
    throw std::invalid_argument("fasttext_subword_features: invalid vocabulary or bucket count");
  const auto width = known + buckets;
  if (words.size() > std::uint64_t(max_dim / width))
    throw std::invalid_argument("fasttext_subword_features: matrix dimensions overflow");
  const auto count = static_cast<std::int64_t>(words.size());
  std::vector<float> features(count * width, 0.f);
  for (std::int64_t row = 0; row < count; ++row) {
    if (row < known)
      features[row * width + row] = 1.f;
    const auto wrapped = "<" + words[row] + ">";
    for (std::size_t n = 3; n <= 6 && n <= wrapped.size(); ++n)
      for (std::size_t start = 0; start <= wrapped.size() - n; ++start) {
        std::uint32_t hash = 2166136261u;
        for (std::size_t j = start; j < start + n; ++j) {
          hash ^= static_cast<unsigned char>(wrapped[j]);
          hash *= 16777619u;
        }
        features[row * width + known + hash % buckets] += 1.f;
      }
  }
  return from_vector(features, {count, width});
}

}  // namespace datamunge::nn
