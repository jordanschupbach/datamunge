#pragma once
/// @file character_data.hpp
/// @brief Genuine raw-character MR metadata and tensor-only character-count parity references.
#include "datamunge/nn/character_cnn_models.hpp"
#include "datamunge/nn/sentence_data.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Reuse the aligned sequence container for original raw-character categorical identities.
///
/// Unlike words,character tokens may be-1 for unknown bytes/space/PAD.
/// Only categorical IDs/observed lengths enter forward;source labels and exact
/// line identities stay aligned with the companion word-CNN partitions.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::CharacterData d{n::tensor({0, -1, 1, 2, 3, -1}, {2, 3}).to(n::DType::Int32),
///                    n::tensor({2, 3}).to(n::DType::Int32),
///                    n::tensor({0, 1}).to(n::DType::Int32),
///                    n::tensor({10, 11}).to(n::DType::Int32)};
/// @endcode
/// @par References
/// - Zhang,Zhao and LeCun(2015),Character-level Convolutional Networks for Text Classification, https://arxiv.org/abs/1509.01626;official Pang/Lee MR v1.0 corpus, https://www.cs.cornell.edu/people/pabo/movie-review-data/. Source binary teaching adaptation uses reversed last256raw Latin1bytes and same source rows as word CNN,not paper-length full training or benchmark accuracy.
/// @see SentenceData, LoadCharacters()
using CharacterData = SentenceData;
/// @brief Own genuine raw-character source partitions and compact six-convolution configuration.
///
/// Pools are fitting,validation,test,matching original word-CNN source IDs/labels.
/// Compact time256/channels16/dense64/alphabet69 retains all18learned leaves.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::LoadCharacters("build/cnn-datasets/character");
/// @endcode
/// @par References
/// - Zhang,Zhao and LeCun(2015),Character-level Convolutional Networks for Text Classification, https://arxiv.org/abs/1509.01626;official Pang/Lee MR v1.0 corpus, https://www.cs.cornell.edu/people/pabo/movie-review-data/. Source binary teaching adaptation uses reversed last256raw Latin1bytes and same source rows as word CNN,not paper-length full training or benchmark accuracy.
/// @see LoadCharacters()
struct CharacterSplits {
  std::vector<CharacterData>
      pools; ///< Ordered raw-character fitting/validation/test observations.
  CharacterConfig
      config; ///< Compact256/16/64/69 shape metadata from original prepared inventory.
};
/// @brief Load original reversed-character MR tensors with exact source identity and padding checks.
///
/// Require Int32[N,256] raw byte IDs-1..68,positive observed lengths<=256,
/// right PAD=-1,binary labels,unique source IDs0..10661 across partitions and
/// fixed69-slot metadata. Known bytes and unknown holes inside length remain
/// unchanged. Checks use tensors except source-ID sets. No auto-download,
/// character synthesis,statistic estimation or fitting;paired word-source
/// identity checks are performed by preparation/native/report validation.
///
/// @param root Directory containing character.npz and original-source manifest.
/// @return Three nonempty source-disjoint pools with compact configuration.
/// @throws std::invalid_argument For invalid shape,dtype,ID/length/padding metadata or leaked source identity.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::LoadCharacters("build/cnn-datasets/character");
/// @endcode
/// @par References
/// - Zhang,Zhao and LeCun(2015),Character-level Convolutional Networks for Text Classification, https://arxiv.org/abs/1509.01626;official Pang/Lee MR v1.0 corpus, https://www.cs.cornell.edu/people/pabo/movie-review-data/. Source binary teaching adaptation uses reversed last256raw Latin1bytes and same source rows as word CNN,not paper-length full training or benchmark accuracy.
/// @see CharacterEncode(), CharacterCnn()

inline CharacterSplits LoadCharacters(const std::filesystem::path &root) {
  nn::NoGradGuard guard;
  std::map<std::string, Tensor> arrays;
  for (const auto &[name, value] : nn::load_npz((root / "character.npz").string()))
    if (!arrays.emplace(name, value).second)
      throw std::invalid_argument("unique character array names required");
  const auto &cfg = arrays.at("config");
  if (cfg.dtype() != nn::DType::Int32 || cfg.shape() != nn::Shape{2})
    throw std::invalid_argument("original character inventory/time metadata required");
  const auto c = cfg.to_vector<int>();
  CharacterSplits out;
  out.config = {c[1], 16, 64, c[0]};
  if (c[1] != 256 || c[0] != int(CharacterAlphabet.size()))
    throw std::invalid_argument("fixed source69slot/time256 character metadata required");
  std::set<int> seen;
  for (const std::string name : {"fit", "validation", "test"}) {
    CharacterData d{arrays.at(name + "_ids"), arrays.at(name + "_lengths"),
                    arrays.at(name + "_labels"), arrays.at(name + "_source_ids")};
    if (d.tokens.dim() != 2 || d.size() < 1 || d.tokens.dtype() != nn::DType::Int32 ||
        d.tokens.shape() != nn::Shape{d.size(), 256} ||
        d.lengths.dtype() != nn::DType::Int32 || d.lengths.shape() != nn::Shape{d.size()} ||
        d.labels.dtype() != nn::DType::Int32 || d.labels.shape() != d.lengths.shape() ||
        d.source_ids.dtype() != nn::DType::Int32 || d.source_ids.shape() != d.lengths.shape())
      throw std::invalid_argument("aligned original raw-character metadata arrays required");
    const auto active =
        nn::less(nn::arange(0, 256).reshape({1, 256}), d.lengths.reshape({d.size(), 1}));
    if (!nn::all(nn::logical_and(nn::greater_equal(d.tokens, -1.), nn::less(d.tokens, 69.)))
             .item<bool>() ||
        !nn::all(nn::where(active, nn::ones_like(active), nn::equal(d.tokens, -1.)))
             .item<bool>() ||
        !nn::all(nn::logical_and(nn::greater(d.lengths, 0.), nn::less_equal(d.lengths, 256.)))
             .item<bool>() ||
        !nn::all(nn::logical_and(nn::greater_equal(d.labels, 0.), nn::less(d.labels, 2.)))
             .item<bool>())
      throw std::invalid_argument(
          "bounded original character labels/lengths/slots and right-1padding required");
    for (int id : d.source_ids.to_vector<int>())
      if (id < 0 || id >= 10662 || !seen.insert(id).second)
        throw std::invalid_argument("unique original character source lines required");
    out.pools.push_back(d);
  }
  return out;
}
/// @brief Count known observed raw-character IDs with tensor segment reduction.
///
/// Length masking and ID>=0 exclude original unknown bytes before counting.
/// Earlier duplicate-hyphen slot stays unused by the source encoder. No class
/// label or learned field enters this parameter-free feature transform.
///
/// @param d Integer[N,T] source character IDs-1..68 and observed lengths0..T.
/// @return Double[N,69] known-character counts;unknown holes/right PAD contribute0.
/// @throws std::invalid_argument For invalid token/length geometry,dtype or ID bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::CharacterData d{n::tensor({0, -1, 1, 2, 3, -1}, {2, 3}).to(n::DType::Int32),
///                    n::tensor({2, 3}).to(n::DType::Int32),
///                    n::tensor({0, 1}).to(n::DType::Int32),
///                    n::tensor({10, 11}).to(n::DType::Int32)};
/// const auto counts = m::CharacterCounts(d);
/// @endcode
/// @par References
/// - Zhang,Zhao and LeCun(2015),Character-level Convolutional Networks for Text Classification, https://arxiv.org/abs/1509.01626;official Pang/Lee MR v1.0 corpus, https://www.cs.cornell.edu/people/pabo/movie-review-data/. Source binary teaching adaptation uses reversed last256raw Latin1bytes and same source rows as word CNN,not paper-length full training or benchmark accuracy.
/// @see CharacterNB()

inline Tensor CharacterCounts(const CharacterData &d) {
  nn::NoGradGuard guard;
  if (d.tokens.dim() != 2 || d.tokens.size(0) < 1 || d.tokens.size(1) < 1 ||
      (d.tokens.dtype() != nn::DType::Int32 && d.tokens.dtype() != nn::DType::Int64) ||
      d.lengths.shape() != nn::Shape{d.tokens.size(0)} ||
      !nn::all(nn::logical_and(nn::greater_equal(d.tokens, -1.), nn::less(d.tokens, 69.)))
           .item<bool>())
    throw std::invalid_argument("bounded observed raw character IDs required");
  const auto n = d.tokens.size(0), time = d.tokens.size(1);
  const auto mask = nn::logical_and(LengthMask(d.lengths, time, nn::DType::Bool),
                                    nn::greater_equal(d.tokens, 0.)),
             safe = nn::where(mask, d.tokens.to(nn::DType::Int64), 0.),
             ids = (nn::arange(0, n, 1, nn::DType::Int64).reshape({n, 1}) * 69. + safe)
                       .reshape({n * time});
  return nn::unsorted_segment_sum(mask.to(nn::DType::Float64).reshape({n * time}), ids, n * 69)
      .reshape({n, 69});
}
/// @brief Compute a fitting-only character unigram reference solely as an SGD-fit parity oracle.
///
/// All known observed character counts/conditional normalization/class priors
/// use Double tensors. Unknown-1 positions contribute nothing. All69slots,
/// including the unused earlier-hyphen slot,receive source add-one smoothing.
/// Reports explicitly fit count fields with SGD;this closed-form helper only
/// checks parity and never replaces the displayed fitting loop.
///
/// @param fit Fitting raw-character observations and bounded binary sentiment labels.
/// @param test Observed raw-character IDs/lengths;test labels and source IDs are ignored.
/// @return Float32[M,2] add-one character multinomial NB log scores.
/// @throws std::invalid_argument For invalid observed characters or fitting binary-label contract.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::CharacterData d{n::tensor({0, -1, 1, 2, 3, -1}, {2, 3}).to(n::DType::Int32),
///                    n::tensor({2, 3}).to(n::DType::Int32),
///                    n::tensor({0, 1}).to(n::DType::Int32),
///                    n::tensor({10, 11}).to(n::DType::Int32)};
/// const auto scores = m::CharacterNB(d, d);
/// @endcode
/// @par References
/// - Zhang,Zhao and LeCun(2015),Character-level Convolutional Networks for Text Classification, https://arxiv.org/abs/1509.01626;official Pang/Lee MR v1.0 corpus, https://www.cs.cornell.edu/people/pabo/movie-review-data/. Source binary teaching adaptation uses reversed last256raw Latin1bytes and same source rows as word CNN,not paper-length full training or benchmark accuracy.
/// @see CharacterCounts(), SentenceNB()

inline Tensor CharacterNB(const CharacterData &fit, const CharacterData &test) {
  nn::NoGradGuard guard;
  if (fit.labels.shape() != nn::Shape{fit.tokens.size(0)} ||
      (fit.labels.dtype() != nn::DType::Int32 && fit.labels.dtype() != nn::DType::Int64) ||
      !nn::all(nn::logical_and(nn::greater_equal(fit.labels, 0.), nn::less(fit.labels, 2.)))
           .item<bool>())
    throw std::invalid_argument("bounded fitting character labels required");
  const auto x = CharacterCounts(fit), y = CharacterCounts(test),
             labels = nn::one_hot(fit.labels, 2, 1., 0., nn::DType::Float64),
             counts = nn::matmul(nn::transpose(labels, {1, 0}), x) + 1.,
             prior = nn::sum(labels, {0}) + 1.;
  return (nn::matmul(y, nn::transpose(nn::log(counts / nn::sum(counts, {1}, true)), {1, 0})) +
          nn::log(prior / nn::sum(prior)))
      .to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
