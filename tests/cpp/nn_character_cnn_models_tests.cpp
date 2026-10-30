
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_character_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.9_1d_convolutions/character_cnn.cpp"
#undef main
#include "../fixtures/nn_character_reference.hpp"
TEST(CharacterCnn, OriginalRawEncodingUnknownZeroAndMinimumGeometry) {
  EncodingChecks();
}
TEST(CharacterCnn, IndependentEvery62ConvolutionDenseParameterDerivative) {
  HostChecks();
}
TEST(CharacterCnn, FullEveryStageAndAll18LearnedGradients) {
  nn::set_num_threads(2);
  FullChecks();
}
TEST(CharacterCnn, EveryRawSourceByteAndExactWordSourceAlignment) {
  namespace m = dm::nn::models;
  const auto chars = m::LoadCharacters("build/cnn-datasets/character");
  const auto words = m::LoadSentences("build/cnn-datasets/sentence");
  const auto archive =
      dataset_assets::detail::Gunzip("build/cnn-datasets/sentence/rt-polaritydata.tar.gz");
  std::vector<std::string> raw;
  for (const std::string kind : {"neg", "pos"}) {
    std::istringstream source(
        dataset_assets::detail::TarText(archive, "rt-polaritydata/rt-polarity." + kind));
    std::string line;
    while (std::getline(source, line))
      raw.push_back(line);
  }
  std::array<int, 256> lookup;
  lookup.fill(-1);
  for (int i = 0; i < 69; ++i)
    lookup[static_cast<unsigned char>(m::CharacterAlphabet[i])] = i;
  std::map<std::string, int> owner;
  int total = 0;
  for (int split = 0; split < 3; ++split) {
    const auto &d = chars.pools[split];
    EXPECT_EQ(d.source_ids.to_vector<int>(), words.pools[split].source_ids.to_vector<int>());
    EXPECT_EQ(d.labels.to_vector<int>(), words.pools[split].labels.to_vector<int>());
    const auto ids = d.source_ids.to_vector<int>(), lengths = d.lengths.to_vector<int>(),
               labels = d.labels.to_vector<int>(), values = d.tokens.to_vector<int>();
    for (int row = 0; row < d.size(); ++row) {
      const auto &text = raw.at(ids[row]);
      const int length = std::min<std::size_t>(256, text.size());
      EXPECT_EQ(lengths[row], length);
      EXPECT_EQ(labels[row], ids[row] >= 5331);
      std::string key;
      for (int t = 0; t < 256; ++t) {
        int expected = -1;
        if (t < length) {
          unsigned char byte = text[text.size() - 1 - t];
          if (byte >= 65 && byte <= 90)
            byte += 32;
          expected = lookup[byte];
          key.push_back(char(byte));
        }
        EXPECT_EQ(values[row * 256 + t], expected);
      }
      if (owner.count(key))
        EXPECT_EQ(owner.at(key), split);
      else
        owner[key] = split;
      ++total;
    }
  }
  EXPECT_EQ(total, 1400);
}
TEST(CharacterCnn, EveryRealCharacterCountNbAndExplicitSgdReference) {
  namespace m = dm::nn::models;
  const auto d = m::LoadCharacters("build/cnn-datasets/character");
  const auto &fit = d.pools[0];
  const auto &test = d.pools[2];
  const auto x = m::CharacterCounts(fit), y = m::CharacterCounts(test),
             labels = nn::one_hot(fit.labels, 2, 1., 0., nn::DType::Float64),
             target = nn::matmul(nn::transpose(labels, {1, 0}), x) + 1.,
             prior_target = nn::sum(labels, {0}) + 1.;
  const auto raw = fit.tokens.to_vector<int>(), lengths = fit.lengths.to_vector<int>();
  const auto counts = x.to_vector<double>();
  for (int row = 0; row < fit.size(); ++row) {
    std::array<int, 69> literal{};
    for (int t = 0; t < lengths[row]; ++t)
      if (raw[row * 256 + t] >= 0)
        ++literal[raw[row * 256 + t]];
    for (int c = 0; c < 69; ++c)
      EXPECT_EQ(counts[row * 69 + c], literal[c]);
  }
  nn::ParameterStore p(383, nn::DType::Float64);
  const auto learned = p.get("counts", {2, 69}, nn::init::zeros()),
             prior = p.get("prior", {2}, nn::init::zeros());
  nn::OptimizerConfig options;
  options.kind = nn::OptimizerKind::Sgd;
  nn::Optimizer sgd(p, .5, options);
  sgd.minimize(nn::sum(nn::square(learned - target)) +
               nn::sum(nn::square(prior - prior_target)));
  const auto logits =
      (nn::matmul(y, nn::transpose(nn::log(learned / nn::sum(learned, {1}, true)), {1, 0})) +
       nn::log(prior / nn::sum(prior)))
          .to(nn::DType::Float32);
  EXPECT_EQ(nn::max(nn::abs(logits - m::CharacterNB(fit, test))).item(), 0.);
  EXPECT_LE(nn::max(nn::abs(logits - character_reference::CharacterNB(fit, test))).item(),
            2e-5);
}
TEST(CharacterCnn, UnknownHolesPaddingControlsAndInvalidDropout) {
  namespace m = dm::nn::models;
  nn::set_num_threads(2);
  const auto data = m::LoadCharacters("build/cnn-datasets/character");
  const auto d = m::SentenceRows(data.pools[2], {0, 1});
  nn::ParameterStore p(2);
  const auto out = m::CharacterCnn(p, d.tokens, d.lengths, data.config);
  nn::NoGradGuard guard;
  const auto active = m::LengthMask(d.lengths, 256, nn::DType::Bool),
             changed = nn::where(active, d.tokens, 0.);
  EXPECT_EQ(
      nn::max(nn::abs(m::CharacterCnn(p, changed, d.lengths, data.config).logits - out.logits))
          .item(),
      0.);
  const auto unknown =
      m::CharacterCnn(p, nn::full({2, 256}, -1., nn::DType::Int32), d.lengths, data.config);
  EXPECT_EQ(nn::sum(nn::abs(unknown.quantized)).item(), 0.);
  EXPECT_TRUE(nn::all(nn::is_finite(unknown.logits)).item<bool>());
  EXPECT_THROW(m::CharacterCnn(p, d.tokens, d.lengths, data.config, true, .5, nullptr),
               std::invalid_argument);
  EXPECT_THROW(m::CharacterFinalLength(-1), std::invalid_argument);
}
