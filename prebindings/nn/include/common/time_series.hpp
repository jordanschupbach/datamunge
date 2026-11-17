#pragma once
#include "common/sentence.hpp"
#include <datamunge/nn/time_series_data.hpp>
namespace series {namespace nn=datamunge::nn;using nn::Tensor;using sentence::Range;using sentence::Rows;
using datamunge::nn::models::SeriesConfig;
using datamunge::nn::models::SeriesOutput;
using datamunge::nn::models::BnMode;
using datamunge::nn::models::SeriesCnn;
using datamunge::nn::models::SeriesData;
using datamunge::nn::models::SeriesSplits;
using datamunge::nn::models::SeriesRows;
using datamunge::nn::models::LoadSeries;
using datamunge::nn::models::SeriesReference;
using datamunge::nn::models::Reversed;
class SeriesModel {
 public:
  nn::ParameterStore             parameters{419};
  nn::ParameterStore             state{421};
  SeriesConfig                   config;
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit SeriesModel(SeriesConfig c = {16, 0, 1}, bool training = true) : config(c) {
    {
      nn::NoGradGuard guard;
      SeriesCnn(parameters, state, nn::zeros({1, 150, 1}), c);
    }
    if (training) optimizer = std::make_unique<nn::Optimizer>(parameters, .001, replay::Adam());
  }
  SeriesOutput Forward(const SeriesData& d, BnMode mode) { return SeriesCnn(parameters, state, d.signals, config, mode); }
  /// Sequential fitting-only population BN calibration, without labels or gradients.
  void CalibrateFit(const SeriesData& fitting) {
    nn::NoGradGuard guard;
    Forward(fitting, BnMode::Calibrate);
  }
  /// Frozen-BN evaluation CE (no gradient).
  double Loss(const SeriesData& d) {
    nn::NoGradGuard guard;
    const double    value = sentence::SentenceCE(Forward(d, BnMode::Infer).logits, d.labels).item<double>();
    check::Require(std::isfinite(value) && value >= 0, "finite time-series CE");
    return value;
  }
  Tensor Predict(const SeriesData& d) {
    nn::NoGradGuard guard;
    return Forward(d, BnMode::Infer).logits.detach();
  }
  replay::Map Snapshot() const { return sentence::Snapshot({&parameters, &state}); }
  void        Save(const std::string& path) const { sentence::SaveStores(path, parameters, &state); }
  void        Restore(const std::string& path) { sentence::RestoreStores(path, parameters, &state); }
  void        Export(const std::filesystem::path& path) const { sentence::ExportArrays(path, parameters, &state); }
};

inline void SeriesMetrics(const std::string& name,const SeriesData& d,const Tensor& logits){const auto score=datamunge::nn::models::ScoreSequenceClassifier(logits,d.labels);std::cout<<" "<<name<<":accuracy="<<(score.accuracy*100.).item()<<"%,CE="<<score.nll.item()<<",confusion="<<score.confusion<<'\n';}
}
