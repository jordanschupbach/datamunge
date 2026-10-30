#pragma once
/// @file sentence_data.hpp
/// @brief Original movie-review token tensors,checked provenance and tensor count-reference oracles.
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/kim_cnn_models.hpp"
#include "datamunge/nn/sequence_classification.hpp"
#include <filesystem>
#include <map>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Own observed integer sequences,lengths,labels and original source-line identities.
///
/// Generic[N,T] sequence container also reused for raw-character metadata.
/// Word observations use right PAD0/UNK1;character observations may use-1
/// structural zero slots. Only tokens and observed lengths enter encoders.
/// Copying shares tensor storage;metadata does not create new source records.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SentenceData d{n::tensor({2, 3, 0, 4, 0, 0}, {2, 3}).to(n::DType::Int32),
///                   n::tensor({2, 1}).to(n::DType::Int32), n::tensor({0, 1}).to(n::DType::Int32),
///                   n::tensor({10, 11}).to(n::DType::Int32)};
/// @endcode
/// @par References
/// - Pang and Lee(2005),Seeing stars:Exploiting class relationships for sentiment categorization with respect to rating scales;official MR v1.0 source, https://www.cs.cornell.edu/people/pabo/movie-review-data/. Source preparation uses ASCII word/apostrophe tokenization,first64words,fitting-only vocabulary,duplicate-token-group holdout and custom1000/200/200 balanced rows,not a published cross-validation score.
/// @see SentenceRows(), LoadSentences()
struct SentenceData {
  Tensor tokens;     ///< Int32[N,T] original observed token/character identities.
  Tensor lengths;    ///< Int32[N] observed sequence lengths before right padding.
  Tensor labels;     ///< Int32[N] original binary sentiment labels0negative/1positive.
  Tensor source_ids; ///< Int32[N] exact original line IDs,negative then positive source order.

  /// @brief Return the number of aligned source-label observations.
  /// @return Leading label dimension,requiring initialized labels.
  [[nodiscard]] std::int64_t size() const {
    return labels.size(0);
  }
};
/// @brief Own original MR word partitions and fitting-only compact vocabulary configuration.
///
/// Pool order fitting,validation,test. Fitting-only vocabulary/time metadata
/// select compact embedding32/filters32 while retaining all3/4/5branches.
/// Raw-character companions reuse exactly the same source IDs/labels.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::LoadSentences("build/cnn-datasets/sentence");
/// @endcode
/// @par References
/// - Pang and Lee(2005),Seeing stars:Exploiting class relationships for sentiment categorization with respect to rating scales;official MR v1.0 source, https://www.cs.cornell.edu/people/pabo/movie-review-data/. Source preparation uses ASCII word/apostrophe tokenization,first64words,fitting-only vocabulary,duplicate-token-group holdout and custom1000/200/200 balanced rows,not a published cross-validation score.
/// @see LoadSentences()
struct SentenceSplits {
  std::vector<SentenceData> pools; ///< Ordered original fitting/validation/test observations.
  KimConfig config; ///< Fitting vocabulary count,time64 and compactE32/F32 architecture.
};
/// @brief Gather observed sequence IDs,lengths,labels and source provenance together.
///
/// No numerical host sequence copies,source reassignment or target fitting.
///
/// @param d Aligned generic integer-sequence observations.
/// @param rows Bounded Int64 leading-row metadata,requested order preserved.
/// @return Tensor-gathered sequences and aligned metadata,retaining dtype and graph.
/// @throws std::invalid_argument For missing axes,misaligned metadata or out-of-range indices.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SentenceData d{n::tensor({2, 3, 0, 4, 0, 0}, {2, 3}).to(n::DType::Int32),
///                   n::tensor({2, 1}).to(n::DType::Int32), n::tensor({0, 1}).to(n::DType::Int32),
///                   n::tensor({10, 11}).to(n::DType::Int32)};
/// const auto row = m::SentenceRows(d, {1, 0});
/// @endcode
/// @see SentenceData

inline SentenceData SentenceRows(const SentenceData &d,
                                 const std::vector<std::int64_t> &rows) {
  if (d.tokens.dim() != 2 || d.lengths.shape() != nn::Shape{d.size()} ||
      d.source_ids.shape() != d.labels.shape() || d.tokens.size(0) != d.size())
    throw std::invalid_argument("aligned observed sequence rows required");
  for (auto row : rows)
    if (row < 0 || row >= d.size())
      throw std::invalid_argument("bounded source sequence index required");
  return {nn::take_rows(d.tokens, rows), nn::take_rows(d.lengths, rows),
          nn::take_rows(d.labels, rows), nn::take_rows(d.source_ids, rows)};
}
/// @brief Load prepared genuine MR word sequences with fitting-only vocabulary and disjoint original IDs.
///
/// No auto-download or hidden fitting. Require Int32[N,64] tokens,positive
/// observed lengths<=64,binary labels,original source IDs0..10661 unique
/// across pools,observed IDs1..V-1,right PAD0 and vocabulary2..8000. Numeric
/// contract checks use tensors;host sets manage only exact source identities.
/// Duplicate-text grouping and fitting-only vocabulary are reproducible in
/// preparation metadata,not inferred from held-out inputs by this loader.
///
/// @param root Directory containing sentence.npz and pinned original-source manifest.
/// @return Three checked nonempty source pools and compact vocabulary/time metadata.
/// @throws std::invalid_argument For invalid file arrays,word/padding bounds or original source leakage.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::LoadSentences("build/cnn-datasets/sentence");
/// @endcode
/// @par References
/// - Pang and Lee(2005),Seeing stars:Exploiting class relationships for sentiment categorization with respect to rating scales;official MR v1.0 source, https://www.cs.cornell.edu/people/pabo/movie-review-data/. Source preparation uses ASCII word/apostrophe tokenization,first64words,fitting-only vocabulary,duplicate-token-group holdout and custom1000/200/200 balanced rows,not a published cross-validation score.
/// @see SentenceRows(), KimCnn()

inline SentenceSplits LoadSentences(const std::filesystem::path &root) {
  nn::NoGradGuard guard;
  std::map<std::string, Tensor> arrays;
  for (const auto &[name, value] : nn::load_npz((root / "sentence.npz").string()))
    if (!arrays.emplace(name, value).second)
      throw std::invalid_argument("unique MR array names required");
  const auto &cfg = arrays.at("config");
  if (cfg.dtype() != nn::DType::Int32 || cfg.shape() != nn::Shape{2})
    throw std::invalid_argument("fitting MR vocabulary metadata required");
  const auto c = cfg.to_vector<int>();
  SentenceSplits out;
  out.config = {c[0], c[1], 32, 32};
  if (c[0] < 2 || c[0] > 8000 || c[1] != 64)
    throw std::invalid_argument("bounded fitting word inventory and time64 required");
  std::set<int> ids;
  for (const std::string name : {"fit", "validation", "test"}) {
    SentenceData d{arrays.at(name + "_tokens"), arrays.at(name + "_lengths"),
                   arrays.at(name + "_labels"), arrays.at(name + "_source_ids")};
    if (d.tokens.dim() != 2 || d.size() < 1 || d.tokens.dtype() != nn::DType::Int32 ||
        d.tokens.shape() != nn::Shape{d.size(), 64} || d.lengths.dtype() != nn::DType::Int32 ||
        d.lengths.shape() != nn::Shape{d.size()} || d.labels.dtype() != nn::DType::Int32 ||
        d.labels.shape() != d.lengths.shape() || d.source_ids.dtype() != nn::DType::Int32 ||
        d.source_ids.shape() != d.labels.shape())
      throw std::invalid_argument("aligned original MR integer arrays required");
    const auto active = nn::less(nn::arange(0, 64).reshape({1, 64}),
                                 d.lengths.reshape({d.size(), 1})),
               valid_tokens = nn::logical_and(nn::greater_equal(d.tokens, 0.),
                                              nn::less(d.tokens, double(c[0]))),
               padding = nn::where(active, nn::greater(d.tokens, 0.), nn::equal(d.tokens, 0.));
    if (!nn::all(valid_tokens).item<bool>() || !nn::all(padding).item<bool>() ||
        !nn::all(nn::logical_and(nn::greater(d.lengths, 0.), nn::less_equal(d.lengths, 64.)))
             .item<bool>() ||
        !nn::all(nn::logical_and(nn::greater_equal(d.labels, 0.), nn::less(d.labels, 2.)))
             .item<bool>())
      throw std::invalid_argument(
          "bounded original MR labels/lengths/IDs and right padding required");
    for (int id : d.source_ids.to_vector<int>())
      if (id < 0 || id >= 10662 || !ids.insert(id).second)
        throw std::invalid_argument("unique disjoint original MR source-line IDs required");
    out.pools.push_back(d);
  }
  return out;
}
/// @brief Count observed word IDs with tensor segment reduction instead of a dense onehot time cube.
///
/// O(N*vocabulary+N*T) storage,avoiding[N,T,V] onehot expansion. Every
/// observed ID,includingUNK1,counts once;normal prepared PAD0 lies outside
/// length and contributes0. Length0 gives zero counts. No learned leaf,
/// class label or held-out frequency enters this numerical feature transform.
///
/// @param d Int32/64[N,T] bounded token IDs and integer observed lengths0..T.
/// @param vocabulary Positive inventory>=2,includingPAD0/UNK1.
/// @return Double[N,vocabulary] observed word frequencies,zero outside observed lengths.
/// @throws std::invalid_argument For invalid geometry,token/length dtype or bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SentenceData d{n::tensor({2, 3, 0, 4, 0, 0}, {2, 3}).to(n::DType::Int32),
///                   n::tensor({2, 1}).to(n::DType::Int32), n::tensor({0, 1}).to(n::DType::Int32),
///                   n::tensor({10, 11}).to(n::DType::Int32)};
/// const auto counts = m::SentenceWordCounts(d, 6);
/// @endcode
/// @par References
/// - Pang and Lee(2005),Seeing stars:Exploiting class relationships for sentiment categorization with respect to rating scales;official MR v1.0 source, https://www.cs.cornell.edu/people/pabo/movie-review-data/. Source preparation uses ASCII word/apostrophe tokenization,first64words,fitting-only vocabulary,duplicate-token-group holdout and custom1000/200/200 balanced rows,not a published cross-validation score.
/// @see SentenceNB()

inline Tensor SentenceWordCounts(const SentenceData &d, int vocabulary) {
  nn::NoGradGuard guard;
  if (d.tokens.dim() != 2 || d.tokens.size(0) < 1 || d.tokens.size(1) < 1 || vocabulary < 2 ||
      (d.tokens.dtype() != nn::DType::Int32 && d.tokens.dtype() != nn::DType::Int64) ||
      d.lengths.shape() != nn::Shape{d.tokens.size(0)} ||
      !nn::all(nn::logical_and(nn::greater_equal(d.tokens, 0.),
                               nn::less(d.tokens, double(vocabulary))))
           .item<bool>())
    throw std::invalid_argument("bounded observed word count inputs required");
  const auto n = d.tokens.size(0), time = d.tokens.size(1);
  const auto active = LengthMask(d.lengths, time, nn::DType::Float64),
             segments =
                 (nn::arange(0, n, 1, nn::DType::Int64).reshape({n, 1}) * double(vocabulary) +
                  d.tokens.to(nn::DType::Int64))
                     .reshape({n * time});
  return nn::unsorted_segment_sum(active.reshape({n * time}), segments,
                                  n * std::int64_t(vocabulary))
      .reshape({n, vocabulary});
}
/// @brief Compute the source fitting unigram NB rule solely as a visible-optimization parity oracle.
///
/// Double tensor count features,class-token sums and class priors each add1.
/// Conditional denominators include all vocabulary slots as in source. Report
/// loops explicitly fit count fields with tensor SGD,normalize and score them;
/// this closed-form helper only verifies those scores. No test labels or token
/// frequencies estimate fitting counts,and no host numerical token loops occur.
///
/// @param fit Fitting-only observed word sequences,lengths and bounded binary labels.
/// @param test Observed test word sequences/lengths;labels/source IDs are ignored.
/// @param vocabulary Positive fitting-only inventory>=2,including PAD/UNK slots.
/// @return Float32[M,2] add-one multinomial NB log scores,not normalized probabilities.
/// @throws std::invalid_argument For invalid observed sequences or fitting binary-label contract.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SentenceData d{n::tensor({2, 3, 0, 4, 0, 0}, {2, 3}).to(n::DType::Int32),
///                   n::tensor({2, 1}).to(n::DType::Int32), n::tensor({0, 1}).to(n::DType::Int32),
///                   n::tensor({10, 11}).to(n::DType::Int32)};
/// const auto z = m::SentenceNB(d, d, 6);
/// @endcode
/// @par References
/// - Pang and Lee(2005),Seeing stars:Exploiting class relationships for sentiment categorization with respect to rating scales;official MR v1.0 source, https://www.cs.cornell.edu/people/pabo/movie-review-data/. Source preparation uses ASCII word/apostrophe tokenization,first64words,fitting-only vocabulary,duplicate-token-group holdout and custom1000/200/200 balanced rows,not a published cross-validation score.
/// @see SentenceWordCounts(), ScoreSequenceClassifier()

inline Tensor SentenceNB(const SentenceData &fit, const SentenceData &test, int vocabulary) {
  nn::NoGradGuard guard;
  if (fit.labels.shape() != nn::Shape{fit.tokens.size(0)} ||
      (fit.labels.dtype() != nn::DType::Int32 && fit.labels.dtype() != nn::DType::Int64) ||
      !nn::all(nn::logical_and(nn::greater_equal(fit.labels, 0.), nn::less(fit.labels, 2.)))
           .item<bool>())
    throw std::invalid_argument("bounded fitting sentiment labels required");
  const auto x = SentenceWordCounts(fit, vocabulary), y = SentenceWordCounts(test, vocabulary),
             class_rows = nn::one_hot(fit.labels, 2, 1., 0., nn::DType::Float64),
             counts = nn::matmul(nn::transpose(class_rows, {1, 0}), x) + 1.,
             prior = nn::sum(class_rows, {0}) + 1.,
             log_word = nn::log(counts / nn::sum(counts, {1}, true)),
             log_prior = nn::log(prior / nn::sum(prior));
  return (nn::matmul(y, nn::transpose(log_word, {1, 0})) + log_prior).to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
