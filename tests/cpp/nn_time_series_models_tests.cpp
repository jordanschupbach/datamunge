
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_series_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.9_1d_convolutions/time_series_cnn.cpp"
#undef main
TEST(TemporalFcn, IndependentEvenKernelBnGapEveryCoordinateFiniteDifference) {
  HostChecks();
}
TEST(TemporalFcn, IndependentNearestTieAndCentroidValues) {
  BaselineChecks();
}
TEST(TemporalFcn, FullAllStagesAndEveryLearnedGradient) {
  nn::set_num_threads(2);
  FullChecks();
}
TEST(TemporalFcn, EveryOriginalSignalAndFitOnlyNormalization) {
  namespace m = dm::nn::models;
  const auto d = m::LoadSeries("build/cnn-datasets/time_series");
  std::map<std::string, m::SeriesData> parts = {
      {"fit", d.pools[0]}, {"validation", d.pools[1]}, {"test", d.pools[2]}};
  std::map<std::string, int> offsets;
  std::ifstream file("build/cnn-datasets/time_series/raw_signals.tsv");
  std::string line;
  int records = 0;
  double sum = 0, squares = 0;
  std::vector<double> fitting;
  while (std::getline(file, line)) {
    std::istringstream row(line);
    std::string split;
    int id, label;
    row >> split >> id >> label;
    const auto &part = parts.at(split);
    const int at = offsets[split]++;
    EXPECT_EQ(part.source_ids.to_vector<int>()[at], id);
    EXPECT_EQ(part.labels.to_vector<int>()[at], label);
    const auto values = nn::select(part.signals, 0, at).to_vector<float>();
    for (int i = 0; i < 150; ++i) {
      float raw;
      row >> raw;
      EXPECT_EQ(values[i], raw);
      if (split == "fit") {
        sum += raw;
        fitting.push_back(raw);
      }
    }
    ++records;
  }
  ASSERT_EQ(records, 200);
  const double mean = sum / fitting.size();
  for (double x : fitting)
    squares += (x - mean) * (x - mean);
  EXPECT_EQ(d.config.mean, float(mean));
  EXPECT_EQ(d.config.standard_deviation, float(std::sqrt(squares / fitting.size())));
}
TEST(TemporalFcn, CalibratedPopulationMomentsFrozenStateAndBatchAgreement) {
  namespace m = dm::nn::models;
  nn::set_num_threads(2);
  const auto fit = m::LoadSeries("build/cnn-datasets/time_series").pools[0];
  nn::ParameterStore p(419), state(421);
  const m::SeriesConfig c{4, 0, 1};
  {
    nn::NoGradGuard guard;
    m::SeriesCnn(p, state, fit.signals, c, m::BnMode::Calibrate);
  }
  nn::NoGradGuard guard;
  auto x = fit.signals;
  for (int layer = 0; layer < 3; ++layer) {
    const auto prefix = "conv" + std::to_string(layer + 1),
               bn = "bn" + std::to_string(layer + 1);
    const auto pre = nn::bias_add(nn::conv1d(x, p.at(prefix + "/W")
                                                    .reshape({layer == 0   ? 8
                                                              : layer == 1 ? 5
                                                                           : 3,
                                                              x.size(2), layer == 1 ? 8 : 4})),
                                  p.at(prefix + "/b"));
    const auto v = pre.to_vector<float>();
    const int width = pre.size(2), count = pre.size(0) * pre.size(1);
    const auto actual_mean = state.at(bn + "/moving_mean").to_vector<float>(),
               actual_var = state.at(bn + "/moving_var").to_vector<float>();
    for (int ch = 0; ch < width; ++ch) {
      double sum = 0, var = 0;
      for (int i = 0; i < count; ++i)
        sum += v[i * width + ch];
      const float mean = float(sum / count);
      for (int i = 0; i < count; ++i)
        var += std::pow(double(v[i * width + ch]) - mean, 2);
      EXPECT_NEAR(actual_mean[ch], mean, 2e-6);
      EXPECT_NEAR(actual_var[ch], var / count, 2e-5);
    }
    nn::BatchNormOptions options;
    options.eps = .001;
    options.momentum = .99;
    options.update_stats = false;
    x = nn::relu(nn::batch_norm(p, state, bn, pre, false, options));
  }
  std::vector<nn::Tensor> before;
  for (const auto &v : state.parameters())
    before.push_back(v.clone());
  const auto batch = m::SeriesCnn(p, state, fit.signals, c).logits,
             single = m::SeriesCnn(p, state, nn::slice_axis(fit.signals, 0, 0, 1), c).logits;
  EXPECT_LE(nn::max(nn::abs(single - nn::slice_axis(batch, 0, 0, 1))).item(), 1e-6);
  for (std::size_t i = 0; i < before.size(); ++i)
    EXPECT_TRUE(nn::all(nn::equal(before[i], state.parameters()[i])).item<bool>());
}
TEST(TemporalFcn, ExplicitSgdCentroidsAndEveryTensorClassificationDiagnostic) {
  namespace m = dm::nn::models;
  const auto data = m::LoadSeries("build/cnn-datasets/time_series");
  const auto fit = data.pools[0], test = data.pools[2];
  const auto weights = nn::one_hot(fit.labels, 2, 1., 0., nn::DType::Float64)
                           .reshape({fit.size(), 2, 1, 1}),
             counts = nn::sum(weights, {0}),
             target = nn::expand_dims(fit.signals.to(nn::DType::Float64), 1);
  nn::ParameterStore p(431, nn::DType::Float64);
  const auto centers = p.get("centers", {2, 150, 1}, nn::init::zeros());
  nn::OptimizerConfig c;
  c.kind = nn::OptimizerKind::Sgd;
  nn::Optimizer sgd(p, .5, c);
  sgd.minimize(nn::sum(nn::sum(weights * nn::square(centers - target), {0}) / counts));
  const auto logits =
      (-nn::mean(nn::square(nn::expand_dims(test.signals.to(nn::DType::Float64), 1) -
                            nn::expand_dims(centers, 0)),
                 {2, 3}))
          .to(nn::DType::Float32);
  EXPECT_EQ(nn::max(nn::abs(logits - m::SeriesReference(fit, test, true))).item(), 0.);
  const auto score = m::ScoreSequenceClassifier(logits, test.labels);
  const auto lv = logits.to_vector<float>();
  const auto labels = test.labels.to_vector<int>();
  const auto predictions = score.predicted.to_vector<std::int64_t>();
  std::vector<std::int64_t> confusion(4);
  int correct = 0;
  double ce = 0;
  for (int i = 0; i < test.size(); ++i) {
    const int predicted = lv[2 * i + 1] > lv[2 * i];
    EXPECT_EQ(predictions[i], predicted);
    ++confusion[labels[i] * 2 + predicted];
    correct += predicted == labels[i];
    const double maximum = std::max(double(lv[2 * i]), double(lv[2 * i + 1]));
    ce += maximum +
          std::log(std::exp(lv[2 * i] - maximum) + std::exp(lv[2 * i + 1] - maximum)) -
          lv[2 * i + labels[i]];
  }
  EXPECT_EQ(score.confusion.to_vector<std::int64_t>(), confusion);
  EXPECT_DOUBLE_EQ(score.accuracy.item(), double(correct) / test.size());
  EXPECT_NEAR(score.nll.item(), ce / test.size(), 1e-14);
}
