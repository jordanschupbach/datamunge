// Short real-surface PointNet/PointNet++ demonstrations, ported from
// tflow's common/ml/point_models.h. Native seeded initializers/dropout give
// statistical training parity; learned and moving state share one NPZ checkpoint.
#pragma once
#include <set>
#include <datamunge/nn/point_data.hpp>

#include "common/ecg_sensor.hpp"
#include "common/point_models.hpp"
#include "common/tf_replay.hpp"

namespace point_training {
namespace nn = datamunge::nn;
using nn::Tensor;
using datamunge::nn::models::PointData;
using datamunge::nn::models::LoadPoints;
inline PointData Rows(const PointData&d,const std::vector<std::int64_t>&rows){return datamunge::nn::models::PointRows(d,rows);}
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
