// Genuine raw-character classification, short runs and independently
// restored inference (port of tflow's common/ml/character.h). Reuses the
// generic integer-sequence container/row/CE/metric helpers of sentence.hpp;
// here tokens holds literal character indices, including -1 zero vectors.
#pragma once

#include "common/character_cnn.hpp"
#include "common/sentence.hpp"

namespace charcnn {

using sentence::Range;
using sentence::Rows;
using CharacterData = sentence::SentenceData;

struct CharacterSplits {
  std::vector<CharacterData> pools;
  CharacterConfig            config;
};

inline CharacterSplits LoadCharacters(const std::filesystem::path& root) {
  const auto arrays = nn::load_npz((root / "character.npz").string());
  const auto cfg    = sentence::Npz(arrays, "config");
  check::Require(cfg.dtype() == nn::DType::Int32 && cfg.shape() == nn::Shape{2}, "character configuration");
  const auto      c = cfg.to_vector<int>();
  CharacterSplits result;
  result.config = {c[1], 16, 64, c[0]};
  check::Require(result.config.time == 256 && result.config.alphabet == static_cast<int>(CharacterAlphabet.size()),
                 "fixed raw-character alphabet/time");
  std::set<int> ids;
  for (const std::string split : {"fit", "validation", "test"}) {
    CharacterData d{sentence::Npz(arrays, split + "_ids"), sentence::Npz(arrays, split + "_lengths"),
                    sentence::Npz(arrays, split + "_labels"), sentence::Npz(arrays, split + "_source_ids")};
    const std::int64_t n = d.tokens.size(0);
    check::Require(n > 0 && d.tokens.dtype() == nn::DType::Int32 && d.tokens.shape() == nn::Shape{n, 256}, "character index matrix");
    for (const auto* t : {&d.lengths, &d.labels, &d.source_ids})
      check::Require(t->dtype() == nn::DType::Int32 && t->shape() == nn::Shape{n}, "character aligned metadata");
    const auto tokens = d.tokens.to_vector<int>(), lengths = d.lengths.to_vector<int>(), labels = d.labels.to_vector<int>(),
               sources = d.source_ids.to_vector<int>();
    int positive = 0;
    for (std::int64_t i = 0; i < n; ++i) {
      const int length = lengths[i], label = labels[i], id = sources[i];
      check::Require(length > 0 && length <= 256 && label >= 0 && label < 2 && id >= 0 && id < 10662 && ids.insert(id).second,
                     "valid character lengths/labels and disjoint unique source IDs");
      positive += label;
      for (int t = 0; t < 256; ++t) {
        const int ch = tokens[i * 256 + t];
        check::Require(t < length ? (ch >= -1 && ch < 69) : ch == -1, "literal character indices/zero PAD");
      }
    }
    std::cout << " character " << split << ":" << n << " samples,positive=" << positive << std::endl;
    result.pools.push_back(d);
  }
  return result;
}

class CharacterModel {
 public:
  nn::ParameterStore             parameters;
  CharacterConfig                config;
  std::unique_ptr<nn::Optimizer> optimizer;
  nn::Generator                  dropout{367};

  explicit CharacterModel(CharacterConfig c = {256, 16, 64, 69}, bool training = true, std::uint64_t seed = 379)
      : parameters(seed), config(c) {
    {
      nn::NoGradGuard guard;
      CharacterCnn(parameters, nn::full({1, c.time}, -1, nn::DType::Int32), nn::full({1}, 1, nn::DType::Int32), c);
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind     = nn::OptimizerKind::Momentum;
      opt.momentum = .9;
      optimizer    = std::make_unique<nn::Optimizer>(parameters, .01, opt);
    }
  }
  CharacterOutput Forward(const CharacterData& d, bool training) {
    return CharacterCnn(parameters, d.tokens, d.lengths, config, training, .5, &dropout);
  }
  /// Inference (dropout-free, no gradient) CE.
  double Loss(const CharacterData& d) {
    nn::NoGradGuard guard;
    const double    value = sentence::SentenceCE(Forward(d, false).logits, d.labels).item<double>();
    check::Require(std::isfinite(value) && value >= 0, "finite character CE");
    return value;
  }
  Tensor Predict(const CharacterData& d) {
    nn::NoGradGuard guard;
    return Forward(d, false).logits.detach();
  }
  replay::Map Snapshot() const { return sentence::Snapshot({&parameters}); }
  void        Save(const std::string& path) const { sentence::SaveStores(path, parameters); }
  void        Restore(const std::string& path) { sentence::RestoreStores(path, parameters); }
};

inline Tensor CharacterPredict(CharacterModel& model, const CharacterData& d) {
  std::vector<Tensor> parts;
  for (std::int64_t start = 0; start < d.size(); start += 50)
    parts.push_back(model.Predict(sentence::SentenceRows(d, Range(start, std::min<std::int64_t>(start + 50, d.size())))));
  return nn::concat(parts, 0);
}

inline Tensor CharacterNB(const CharacterData& fit, const CharacterData& test) {
  std::vector<std::vector<double>> counts(2, std::vector<double>(69, 1));
  double                           totals[2] = {69, 69}, prior[2] = {1, 1};
  const auto ft = fit.tokens.to_vector<int>(), fl = fit.lengths.to_vector<int>(), fy = fit.labels.to_vector<int>();
  const std::int64_t T = fit.tokens.size(1);
  for (std::int64_t n = 0; n < fit.size(); ++n) {
    const int label = fy[n];
    ++prior[label];
    for (int t = 0; t < fl[n]; ++t) {
      const int c = ft[n * T + t];
      if (c >= 0) {
        ++counts[label][c];
        ++totals[label];
      }
    }
  }
  const auto         tt = test.tokens.to_vector<int>(), tl = test.lengths.to_vector<int>();
  std::vector<float> result(static_cast<std::size_t>(2 * test.size()));
  for (std::int64_t n = 0; n < test.size(); ++n)
    for (int label = 0; label < 2; ++label) {
      double score = std::log(prior[label] / (prior[0] + prior[1]));
      for (int t = 0; t < tl[n]; ++t) {
        const int c = tt[n * T + t];
        if (c >= 0) score += std::log(counts[label][c] / totals[label]);
      }
      result[2 * n + label] = static_cast<float>(score);
    }
  return nn::from_vector(result, {test.size(), 2});
}

}  // namespace charcnn
