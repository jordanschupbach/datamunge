
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_wavenet_stack_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.9_1d_convolutions/wavenet_stack.cpp"
#undef main
#include "../fixtures/nn_wave_reference.hpp"
TEST(WaveStack, CodecAndEveryStableMaskedLogitDerivative) {
  CodecLossChecks();
}
TEST(WaveStack, IndependentEvery33GateResidualSkipParameterDerivative) {
  HostChecks();
}
TEST(WaveStack, EveryReachableCausalLagAndExactFutureZero) {
  CausalChecks();
}
TEST(WaveStack, FullEveryGateAndAll76LearnedGradients) {
  nn::set_num_threads(2);
  FullChecks();
}
TEST(WaveStack, EveryOriginalPcmSampleCodeShiftWindowIdentityAndPrompt) {
  namespace m = dm::nn::models;
  const auto root = std::filesystem::path("build/cnn-datasets/wavenet");
  const auto data = m::LoadWaveStack(root);
  std::ifstream files(root / "source_files.tsv");
  std::string split, path;
  int source, speaker, total = 0;
  std::map<int, std::vector<int>> codes;
  while (files >> split >> source >> speaker >> path) {
    const auto wav = nn::read_wav((root / path).string());
    EXPECT_EQ(wav.sample_rate, 8000);
    const auto raw = wav.samples.to_vector<double>();
    std::vector<int> encoded(1025);
    for (int t = 0; t < 1025; ++t) {
      const double x = raw[t],
                   mapped = std::copysign(std::log1p(255 * std::abs(x)) / std::log(256.), x);
      encoded[t] = std::clamp(int(std::floor((mapped + 1) * 127.5 + .5)), 0, 255);
    }
    codes.emplace(source, encoded);
    const auto &d = data.pools[split == "fit" ? 0 : split == "validation" ? 1 : 2];
    const auto ids = d.source.to_vector<int>(), speakers = d.speaker.to_vector<int>(),
               starts = d.start.to_vector<int>(), x = d.input.to_vector<int>(),
               y = d.target.to_vector<int>();
    int matched = 0;
    for (int row = 0; row < d.size(); ++row)
      if (ids[row] == source) {
        EXPECT_EQ(speakers[row], speaker);
        for (int t = 0; t < 256; ++t) {
          EXPECT_EQ(x[row * 256 + t], encoded[starts[row] + t]);
          EXPECT_EQ(y[row * 256 + t], encoded[starts[row] + t + 1]);
        }
        ++matched;
        ++total;
      }
    EXPECT_EQ(matched, 4);
  }
  EXPECT_EQ(total, 108);
  const auto prompt = data.prompts.to_vector<int>(),
             ids = data.prompt_source_ids.to_vector<int>();
  for (int row = 0; row < 3; ++row)
    for (int t = 0; t < 384; ++t)
      EXPECT_EQ(prompt[row * 384 + t], codes.at(ids[row])[t]);
}
TEST(WaveStack, EveryOriginalCountMetricAndExplicitSgdReference) {
  namespace m = dm::nn::models;
  const auto data = m::LoadWaveStack("build/cnn-datasets/wavenet");
  const auto &fit = data.pools[0];
  const auto &test = data.pools[2];
  const auto counts = m::WaveCountTargets(fit);
  const auto old = wave_reference::WavePriors(fit);
  EXPECT_LE(nn::max(nn::abs(nn::log(counts.first).to(nn::DType::Float32) - old.first)).item(),
            1e-6);
  EXPECT_LE(
      nn::max(nn::abs(nn::log(counts.second).to(nn::DType::Float32) - old.second)).item(),
      1e-6);
  nn::ParameterStore p(1, nn::DType::Float64);
  const auto u = p.get("unigram", {256}, nn::init::zeros()),
             t = p.get("transition", {256, 256}, nn::init::zeros());
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Sgd;
  nn::Optimizer sgd(p, .5, config);
  sgd.minimize(nn::sum(nn::square(u - counts.first)) + nn::sum(nn::square(t - counts.second)));
  EXPECT_EQ(nn::max(nn::abs(u - counts.first)).item(), 0);
  EXPECT_EQ(nn::max(nn::abs(t - counts.second)).item(), 0);
  for (const auto &prior : std::vector<nn::Tensor>{nn::log(u).to(nn::DType::Float32),
                                                   nn::log(t).to(nn::DType::Float32)}) {
    const auto z = m::WavePriorLogits(test, prior);
    EXPECT_EQ(nn::max(nn::abs(z - wave_reference::WavePriorLogits(test, prior))).item(), 0);
    const auto a = m::ScoreWave(z, test).to_vector<double>();
    const auto expected = wave_reference::WaveMetrics("oracle", z, test);
    EXPECT_NEAR(a[0], expected.nll, 1e-6);
    EXPECT_NEAR(a[1], expected.mse, 1e-12);
    EXPECT_NEAR(a[2], expected.exact, 1e-12);
  }
}
TEST(WaveStack, InvalidWarmupCodeAndArchitectureAreRejected) {
  namespace m = dm::nn::models;
  EXPECT_THROW(m::WaveStackField({1, 1, 13, 1, 3}), std::invalid_argument);
  EXPECT_THROW(m::WaveNll(nn::zeros({1, 4, 3}), nn::zeros({1, 4}, nn::DType::Int32), 3, 4),
               std::invalid_argument);
  EXPECT_THROW(m::WaveNll(nn::zeros({1, 4, 3}), nn::full({1, 4}, 3., nn::DType::Int32), 3, 0),
               std::invalid_argument);
  nn::ParameterStore p;
  EXPECT_THROW(m::WaveStack(p, nn::ones({1, 4, 2}), {1, 1, 2, 1, 3}), std::invalid_argument);
}
