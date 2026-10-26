// Short real-surface PointNet/PointNet++ demonstrations, ported from
// tflow's common/ml/point_models.h. Native seeded initializers/dropout give
// statistical training parity; learned and moving state share one NPZ checkpoint.
#pragma once
#include <set>

#include "common/ecg_sensor.hpp"
#include "common/point_models.hpp"
#include "common/tf_replay.hpp"

namespace point_training {
namespace nn = datamunge::nn;
using nn::Tensor;
struct PointData {
  Tensor xyz, labels, source;
};
inline PointData Rows(const PointData& d, const std::vector<std::int64_t>& rows) {
  return {nn::take_rows(d.xyz, rows), nn::take_rows(d.labels, rows), nn::take_rows(d.source, rows)};
}
inline std::vector<PointData> LoadPoints(const std::filesystem::path& root) {
  const auto             arrays = replay::ToMap(nn::load_npz((root / "point_clouds.npz").string()));
  std::vector<PointData> pools;
  std::set<int>          ids;
  for (const std::string split : {"fit", "validation", "test"}) {
    auto      full = arrays.at(split + "_xyz");
    const int n    = split == "fit" ? 24 : split == "validation" ? 6 : 12;
    check::Require(
        full.dtype() == nn::DType::Float32 && full.shape() == nn::Shape({n, 1024, 3}),
        "real surface XYZ shape");
    PointData d{
        nn::slice(full, {0, 0, 0}, {n, 256, 3}),
        arrays.at(split + "_label"),
        arrays.at(split + "_source_id")};
    check::Require(
        d.labels.dtype() == nn::DType::Int32 && d.source.dtype() == nn::DType::Int32 &&
            d.labels.shape() == nn::Shape({n}) && d.source.shape() == nn::Shape({n}),
        "aligned labels/source IDs");
    std::vector<int> counts(3);
    const auto       labels = d.labels.to_vector<int>(), source = d.source.to_vector<int>();
    for (int b = 0; b < n; ++b) {
      check::Require(
          labels[b] >= 0 && labels[b] < 3 && ids.insert(source[b]).second,
          "unique disjoint object IDs");
      ++counts[labels[b]];
    }
    for (float x : d.xyz.to_vector<float>())
      check::Require(std::isfinite(x) && std::abs(x) <= 1.00001, "finite unit-ball points");
    for (int count : counts)
      check::Require(count == n / 3, "balanced source classes");
    pools.push_back(d);
  }
  return pools;
}
class PointModel {
 public:
  nn::ParameterStore             parameters{971}, state{977};
  nn::Generator                  dropout{971};
  bool                           plus;
  point_models::PointConfig      config{4, 3, 256, 64, 16, 16, 32};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit PointModel(bool hierarchical, bool train = true) : plus(hierarchical) {
    {
      nn::NoGradGuard guard;
      Forward(nn::zeros({1, 256, 3}), false);
    }
    if (train) {
      nn::OptimizerConfig cfg;
      cfg.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, cfg);
    }
  }
  point_models::PointOutput Forward(const Tensor& xyz, bool training, bool update = false) {
    return plus
               ? point_models::PointNetPlusPlus(
                     parameters, state, xyz, config, training, update, &dropout)
               : point_models::PointNet(parameters, state, xyz, config, training, update, &dropout);
  }
  Tensor Predict(const PointData& d) {
    nn::NoGradGuard guard;
    return Forward(d.xyz, false).logits;
  }
  Tensor Objective(const PointData& d, bool update) {
    auto net = Forward(d.xyz, true, update);
    auto ce  = nn::sparse_softmax_cross_entropy(net.logits, d.labels);
    return plus ? ce : nn::add(ce, point_models::PointTransformPenalty(net.feature_transform, 16));
  }
  double Update(const PointData& d) {
    auto   loss  = Objective(d, true);
    double value = loss.item();
    check::Require(std::isfinite(value) && value >= 0, "finite point objective");
    optimizer->minimize(loss);
    return value;
  }
  replay::Map Values() const {
    auto out = replay::Arrays(parameters), stats = replay::Arrays(state);
    for (const auto& [key, value] : stats)
      check::Require(out.emplace(key, value).second, "distinct learned/state keys");
    return out;
  }
  void Save(const std::filesystem::path& file) const {
    nn::save_npz(file.string(), replay::ToNamed(Values()));
  }
  void Load(const std::filesystem::path& file) {
    const auto arrays = nn::load_npz(file.string());
    check::Require(
        arrays.size() == parameters.size() + state.size(), "complete learned/BN checkpoint");
    for (const auto& [key, value] : arrays) {
      check::Require(
          parameters.contains(key) || state.contains(key), "checkpoint key belongs to model");
      (parameters.contains(key) ? parameters : state).set(key, value);
    }
  }
};
inline Tensor Predict(PointModel& model, const PointData& d) {
  std::vector<Tensor> outputs;
  for (std::int64_t start = 0; start < d.xyz.size(0); start += 4) {
    std::vector<std::int64_t> rows;
    for (auto i = start; i < std::min(start + 4, d.xyz.size(0)); ++i)
      rows.push_back(i);
    outputs.push_back(model.Predict(Rows(d, rows)));
  }
  return nn::concat(outputs, 0);
}
inline void Metrics(const std::string& name, const PointData& d, const Tensor& logits) {
  const auto score = sensor::ScoreSensor(logits, d.labels, 3);
  std::cout << name << ":accuracy=" << score.accuracy * 100 << "%,macroF1=" << score.macro_f1
            << ",NLL=" << score.nll << ",confusion=";
  for (int c : score.confusion)
    std::cout << c << ',';
  std::cout << std::endl;
}

} // namespace point_training
