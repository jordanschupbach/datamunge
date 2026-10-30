
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_sensor_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.9_1d_convolutions/ecg_sensor_cnn.cpp"
#undef main
#include "../fixtures/nn_sensor_reference.hpp"
TEST(Sensor, IndependentAll171StridedResidualParameterDerivatives) {
  HostChecks();
}
TEST(Sensor, WeightedLossEveryLogitConfusionF1BalancedAndTieAuc) {
  LossMetricChecks();
}
TEST(Sensor, FullBothDatasetsEveryStageAndEveryLearnedGradient) {
  nn::set_num_threads(2);
  FullChecks();
}
TEST(Sensor, EveryOriginalSampleIdLabelAndFitOnlyChannelStatistic) {
  namespace m = dm::nn::models;
  namespace r = sensor_reference;
  for (const std::string name : {"ECG200", "BasicMotions"}) {
    const std::filesystem::path root =
        std::filesystem::path("build/cnn-datasets/ecg_sensor") / name;
    const auto d = m::LoadSensor(root);
    const auto old = r::LoadSensor(root);
    EXPECT_EQ(d.mean.to_vector<float>(), old.mean.to_vector<float>());
    EXPECT_EQ(d.stddev.to_vector<float>(), old.stddev.to_vector<float>());
    EXPECT_EQ(d.weights.to_vector<float>(), old.weights.to_vector<float>());
    std::map<std::string, m::SeriesData> parts = {
        {"fit", d.pools[0]}, {"validation", d.pools[1]}, {"test", d.pools[2]}};
    std::map<std::string, int> offsets;
    std::ifstream file(root / "raw_signals.tsv");
    std::string line;
    int total = 0;
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
      for (int i = 0; i < d.time * d.config.channels; ++i) {
        float raw;
        row >> raw;
        EXPECT_EQ(values[i], raw);
      }
      ++total;
    }
    EXPECT_EQ(total, name == "ECG200" ? 200 : 80);
  }
}
TEST(Sensor, EveryRealReferenceMetricAndSourceIndexDiagnostic) {
  namespace m = dm::nn::models;
  namespace r = sensor_reference;
  for (const std::string name : {"ECG200", "BasicMotions"}) {
    const auto d =
        m::LoadSensor(std::filesystem::path("build/cnn-datasets/ecg_sensor") / name);
    const auto &fit = d.pools[0];
    const auto &test = d.pools[2];
    for (bool centroid : {false, true}) {
      const auto z = m::SensorReference(fit, test, d.stddev, d.config.classes, centroid);
      const auto old = r::SensorReference(fit, test, d.stddev, d.config.classes, centroid);
      EXPECT_LE(nn::max(nn::abs(z - old)).item(), 1e-6);
      const auto a = m::ScoreSensor(z, test.labels, d.config.classes);
      const auto b = r::ScoreSensor(z, test.labels, d.config.classes);
      EXPECT_DOUBLE_EQ(a.accuracy, b.accuracy);
      EXPECT_NEAR(a.nll, b.nll, 2e-14);
      EXPECT_DOUBLE_EQ(a.balanced, b.balanced);
      EXPECT_DOUBLE_EQ(a.macro_f1, b.macro_f1);
      EXPECT_DOUBLE_EQ(a.auc0, b.auc0);
      EXPECT_EQ(a.confusion, b.confusion);
      EXPECT_EQ(a.recall, b.recall);
    }
  }
}
TEST(Sensor, BothExplicitSgdCentroidFieldsMatchEveryOracleDistance) {
  namespace m = dm::nn::models;
  for (const std::string name : {"ECG200", "BasicMotions"}) {
    const auto d =
        m::LoadSensor(std::filesystem::path("build/cnn-datasets/ecg_sensor") / name);
    const auto &fit = d.pools[0];
    const auto &test = d.pools[2];
    const int k = d.config.classes;
    const auto weights = nn::one_hot(fit.labels, k, 1., 0., nn::DType::Float64)
                             .reshape({fit.size(), k, 1, 1}),
               counts = nn::sum(weights, {0});
    nn::ParameterStore p(599, nn::DType::Float64);
    const auto center = p.get("centers", {k, d.time, d.config.channels}, nn::init::zeros());
    nn::OptimizerConfig config;
    config.kind = nn::OptimizerKind::Sgd;
    nn::Optimizer sgd(p, .5, config);
    sgd.minimize(nn::sum(
        nn::sum(weights * nn::square(center -
                                     nn::expand_dims(fit.signals.to(nn::DType::Float64), 1)),
                {0}) /
        counts));
    const auto diff = (nn::expand_dims(test.signals.to(nn::DType::Float64), 1) -
                       nn::expand_dims(center, 0)) /
                      d.stddev.to(nn::DType::Float64);
    const auto z = (-nn::mean(nn::square(diff), {2, 3})).to(nn::DType::Float32);
    EXPECT_EQ(nn::max(nn::abs(z - m::SensorReference(fit, test, d.stddev, k, true))).item(),
              0.);
  }
}
TEST(Sensor, IndependentAllInputGradientsAndEveryChannelReplacement) {
  namespace m = dm::nn::models;
  nn::set_num_threads(2);
  const auto d = m::LoadSensor("build/cnn-datasets/ecg_sensor/BasicMotions");
  nn::ParameterStore p(587), state(593);
  const auto x = nn::slice_axis(d.pools[2].signals, 0, 0, 1).requires_grad_();
  m::SensorCnn(p, state, x, d.config);
  state.set("normalization/mean", d.mean);
  state.set("normalization/std", d.stddev);
  const auto out = m::SensorCnn(p, state, x, d.config);
  const auto g = nn::grad(nn::sum(nn::square(out.logits)), {x})[0];
  EXPECT_TRUE(nn::all(nn::is_finite(g)).item<bool>());
  nn::NoGradGuard guard;
  for (int ch = 0; ch < 6; ++ch) {
    EXPECT_GT(nn::sum(nn::abs(nn::select(g, 2, ch))).item(), 0.);
    const auto mask = nn::equal(nn::arange(0, 6), double(ch)).reshape({1, 1, 6}),
               changed = nn::where(mask, d.mean.reshape({1, 1, 6}), x);
    EXPECT_GT(m::MeanAbsChange(m::SensorCnn(p, state, changed, d.config).logits, out.logits),
              1e-7);
  }
}
TEST(Sensor, RejectInvalidStoredStdMissingClassesAndUndefinedAuc) {
  namespace m = dm::nn::models;
  nn::ParameterStore p(587), state(593);
  m::SensorCnn(p, state, nn::ones({1, 9, 1}), m::SensorConfig{1, 2, 1});
  state.set("normalization/std", nn::zeros({1}));
  EXPECT_THROW(m::SensorCnn(p, state, nn::ones({1, 9, 1}), m::SensorConfig{1, 2, 1}),
               std::invalid_argument);
  EXPECT_THROW(m::ScoreSensor(nn::zeros({2, 2}), nn::zeros({2}, nn::DType::Int32), 2),
               std::invalid_argument);
  EXPECT_THROW(m::TensorSensorAuc0(nn::ones({2}), nn::zeros({2}, nn::DType::Int32)),
               std::invalid_argument);
}
