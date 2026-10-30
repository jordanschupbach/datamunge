#pragma once
#include <datamunge/nn/medical_volume_models.hpp>
#include <datamunge/nn/medical_volume_data.hpp>
#include "common/c3d.hpp"
namespace medical_volume{namespace nn=datamunge::nn;using nn::Tensor;using nn::Padding;
using datamunge::nn::models::MedicalVolumeOutput;
using datamunge::nn::models::MedicalResidual;
using datamunge::nn::models::MedicalVolume;
using datamunge::nn::models::MedicalData;
using datamunge::nn::models::MedicalRows;
using datamunge::nn::models::LoadMedical;
using datamunge::nn::models::MedicalBaseline;
struct MedicalModel {
  nn::ParameterStore parameters{1049}, state{1051};
  nn::Generator      dropout{1031};
  MedicalModel() {
    nn::NoGradGuard guard;  // create all 20 learned and 2 normalization tensors
    MedicalVolume(parameters, state, nn::zeros({1, 28, 28, 28, 1}));
  }
  /// Fit-only intensity moments.
  void Normalize(double mean, double stddev) {
    check::Require(std::isfinite(mean) && std::isfinite(stddev) && stddev > 0, "fitting-only CT moments");
    state.set("normalization/mean", nn::full({}, mean));
    state.set("normalization/stddev", nn::full({}, stddev));
  }
  Tensor Logits(const Tensor& voxels, bool training) {
    return MedicalVolume(parameters, state, voxels, 4, training, &dropout).logits;
  }
  /// Inference logits [N, 2] in batches of 4.
  Tensor Predict(const MedicalData& d) {
    nn::NoGradGuard     guard;
    std::vector<Tensor> parts;
    for (std::int64_t start = 0; start < d.size(); start += 4) {
      std::vector<std::int64_t> rows;
      for (std::int64_t i = start; i < std::min<std::int64_t>(start + 4, d.size()); ++i) rows.push_back(i);
      parts.push_back(Logits(nn::take_rows(d.voxels, rows), false));
    }
    return nn::concat(parts, 0);
  }
  std::vector<double> Snapshot() const {
    auto       values = parameters.flatten();
    const auto stats  = state.flatten();
    values.insert(values.end(), stats.begin(), stats.end());
    return values;
  }
  void Save(const std::filesystem::path& prefix) const {
    parameters.save(prefix.string() + ".npz");
    state.save(prefix.string() + "_state.npz");
  }
  void Restore(const std::filesystem::path& prefix) {
    parameters.load(prefix.string() + ".npz");
    state.load(prefix.string() + "_state.npz");
  }
};

inline void MedicalMetrics(const std::string& name, const MedicalData& d, const Tensor& logits) {
  const auto score = c3d::ScoreLogits(logits, d.labels, 2);
  std::cout << name << ":accuracy=" << 100 * score.accuracy << "%,macroF1=" << score.macro_f1 << ",NLL=" << score.nll
            << ",confusion=";
  for (int v : score.confusion) std::cout << v << ',';
  std::cout << std::endl;
}

}
