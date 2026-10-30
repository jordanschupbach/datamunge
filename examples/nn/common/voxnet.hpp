#pragma once
#include <datamunge/nn/voxnet_models.hpp>
#include "common/ecg_sensor.hpp"
#include "common/tf_replay.hpp"
namespace vox{namespace nn=datamunge::nn;using nn::Tensor;using replay::Map;
using Data=datamunge::nn::models::VoxnetData;
using Path=datamunge::nn::models::VoxnetOutput;
using Dropout=datamunge::nn::models::VoxnetDropout;
inline auto Rotate(Tensor x,int k){return nn::models::VoxnetRotate(x,k);}
inline auto Regularized(const nn::ParameterStore&p,const std::vector<Tensor>&g,double l=.001){return nn::models::VoxnetRegularized(p,g,l);}
inline auto Forward(nn::ParameterStore&p,const Tensor&x,int classes=40,bool training=false,Dropout*dropout=nullptr){return nn::models::Voxnet(p,x,classes,training,dropout);}
inline auto Rows(const Data&d,const std::vector<std::int64_t>&rows){return nn::models::VoxnetRows(d,rows);}
inline auto Load(const std::filesystem::path&root){return nn::models::LoadVoxnet(root);}
class Model {
 public:
  nn::ParameterStore             parameters{907};
  Dropout                        dropout;
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(parameters, nn::zeros({1, 32, 32, 32, 1}), 3);
    }
    if (training) {
      nn::OptimizerConfig o;
      o.kind     = nn::OptimizerKind::Momentum;
      o.momentum = .9;
      optimizer  = std::make_unique<nn::Optimizer>(parameters, .001, o);
    }
  }
  Tensor Predict(const Data& d, bool vote = true) {
    nn::NoGradGuard guard;
    if (!vote)
      return Forward(parameters, d.voxels, 3).logits;
    std::vector<Tensor> values;
    for (int k = 0; k < 4; ++k)
      values.push_back(Forward(parameters, Rotate(d.voxels, k), 3).logits);
    auto out = values[0];
    for (int k = 1; k < 4; ++k)
      out = out + values[k];
    return out / 4.;
  }
  Tensor Classification(const Data& d) {
    return nn::sparse_softmax_cross_entropy(
        Forward(parameters, d.voxels, 3, true, &dropout).logits, d.labels);
  }
  double Update(const Data& d) {
    auto   ce      = Classification(d);
    double penalty = 0;
    {
      nn::NoGradGuard guard;
      for (const auto& name : parameters.names())
        if (name.find("/W") != std::string::npos)
          penalty += nn::sum(nn::square(parameters.at(name))).item();
    }
    const double value = ce.item() + .001 * penalty;
    auto         g     = nn::grad(ce, parameters.parameters());
    optimizer->step(Regularized(parameters, g));
    check::Require(std::isfinite(value), "finite regularized voxel objective");
    return value;
  }
  Map  Arrays() const { return replay::Arrays(parameters); }
  void Export(const std::filesystem::path& file) const {
    nn::NamedTensors out;
    for (std::size_t i = 0; i < parameters.size(); ++i)
      out.emplace_back("p" + std::to_string(i), parameters.at(parameters.names()[i]));
    nn::save_npz(file.string(), out);
  }
  void LoadExport(const std::filesystem::path& file) {
    const auto a = replay::ToMap(nn::load_npz(file.string()));
    check::Require(a.size() == parameters.size(), "all8 exported parameters");
    for (std::size_t i = 0; i < parameters.size(); ++i)
      parameters.set(parameters.names()[i], a.at("p" + std::to_string(i)));
  }
};
inline Tensor Predict(Model& model, const Data& d, bool vote = true) {
  std::vector<Tensor> out;
  for (std::int64_t start = 0; start < d.voxels.size(0); start += 4) {
    std::vector<std::int64_t> rows;
    for (auto i = start; i < std::min(start + 4, d.voxels.size(0)); ++i)
      rows.push_back(i);
    out.push_back(model.Predict(Rows(d, rows), vote));
  }
  return nn::concat(out, 0);
}
inline auto Features(const Data&d){return nn::models::VoxnetFeatures(d);}
inline auto Nearest(const Data&a,const Data&b){return nn::models::VoxnetNearest(a,b);}
inline void Metrics(const std::string& name, const Data& d, const Tensor& logits) {
  auto s = sensor::ScoreSensor(logits, d.labels, 3);
  std::cout << name << ":accuracy=" << 100 * s.accuracy << "%,macroF1=" << s.macro_f1
            << ",NLL=" << s.nll << ",confusion=";
  for (int c : s.confusion)
    std::cout << c << ',';
  std::cout << std::endl;
}

} // namespace vox
