#pragma once
#include "common/time_series.hpp"
#include <datamunge/nn/tcn_models.hpp>
namespace tcn {namespace nn=datamunge::nn;using nn::Tensor;using sentence::Range;using sentence::Rows;using series::SeriesData;using series::SeriesRows;
using datamunge::nn::models::TcnConfig;
using datamunge::nn::models::TcnReceptiveField;
using datamunge::nn::models::TcnCausalConv;
using datamunge::nn::models::TcnOutput;
using datamunge::nn::models::Tcn;
using datamunge::nn::models::TcnPrefix;
class TcnModel {
 public:
  nn::ParameterStore             parameters{461};
  nn::ParameterStore             state{463};
  TcnConfig                      config;
  std::unique_ptr<nn::Optimizer> optimizer;
  nn::Generator                  dropout{457};

  explicit TcnModel(TcnConfig c = {8, 6, 3, 0, 1}, bool training = true) : config(c) {
    {
      nn::NoGradGuard guard;
      Tcn(parameters, state, nn::zeros({1, 4, 1}), c);
    }
    if (training) optimizer = std::make_unique<nn::Optimizer>(parameters, .001, replay::Adam());
  }
  TcnOutput Forward(const SeriesData& d, bool training) { return Tcn(parameters, state, d.signals, config, training, .8, &dropout); }
  /// Inference (dropout-free, no gradient) CE.
  double Loss(const SeriesData& d) {
    nn::NoGradGuard guard;
    const double    value = sentence::SentenceCE(Forward(d, false).logits, d.labels).item<double>();
    check::Require(std::isfinite(value) && value >= 0, "finite TCN CE");
    return value;
  }
  Tensor Predict(const SeriesData& d, bool all_times = false) {
    nn::NoGradGuard guard;
    const auto      out = Forward(d, false);
    return (all_times ? out.sequence : out.logits).detach().contiguous();
  }
  replay::Map Snapshot() const { return sentence::Snapshot({&parameters, &state}); }
  void        Save(const std::string& path) const { sentence::SaveStores(path, parameters, &state); }
  void        Restore(const std::string& path) { sentence::RestoreStores(path, parameters, &state); }
  void        Export(const std::filesystem::path& path) const { sentence::ExportArrays(path, parameters, &state); }
};

}
