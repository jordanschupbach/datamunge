#pragma once
#include "common/character_cnn.hpp"
#include "common/sentence.hpp"
#include <datamunge/nn/character_data.hpp>
namespace charcnn {using sentence::Range;using sentence::Rows;
using datamunge::nn::models::CharacterData;
using datamunge::nn::models::CharacterSplits;
using datamunge::nn::models::LoadCharacters;
using datamunge::nn::models::CharacterNB;
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

}
