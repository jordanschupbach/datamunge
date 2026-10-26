// Port of tflow's common/nn/noise_conditional_score.h and
// common/ml/noise_conditional_score.h: noise-conditional score matching for VE
// Gaussian perturbations (16 geometric sigmas 1 -> .05) with the DDPM epsilon MLP
// (examples/nn/common/ddpm.hpp, time features over 15 levels), score =
// -epsilon / sigma, weighted DSM loss .5 mean ||sigma score + eps||^2, and
// annealed Langevin updates alpha = 2e-5 (sigma / .05)^2.
//
// tflow -> datamunge: graph/session -> eager; AddSymbolicGradients -> nn::grad;
// Adam train_op -> nn::Optimizer (Adam .003); SaveV2 "final" -> final.npz.
// Parameter names match tflow's, so TF's npz weights load directly.
#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "common/ddpm.hpp"

namespace ncsn {

namespace nn = datamunge::nn;
using nn::Tensor;
using Map = replay::Map;

/// tflow's ScoreSigmas: .05^(i / 15), i = 0..15 (float64).
inline const std::vector<double>& ScoreSigmas() {
  static const std::vector<double> values = [] {
    std::vector<double> v(16);
    for (int i = 0; i < 16; ++i) v[i] = std::pow(.05, i / 15.);
    return v;
  }();
  return values;
}

/// float(sigma[level]) as [n, 1, 1, 1].
inline Tensor ScoreSigma(const Tensor& level) {
  const auto         t = nn::cast(level, nn::DType::Int64).to_vector<std::int64_t>();
  std::vector<float> out(t.size());
  for (std::size_t i = 0; i < t.size(); ++i) {
    check::Require(t[i] >= 0 && t[i] < 16, "noise level in 0..15");
    out[i] = float(ScoreSigmas()[std::size_t(t[i])]);
  }
  return nn::from_vector(out, {std::int64_t(t.size()), 1, 1, 1});
}

/// tflow's DdpmPredictor with a configurable time normalisation (maximum_time).
inline diffusion::PredictorPath Predictor(const nn::ParameterStore& p, const Tensor& state, const Tensor& time, float maximum_time) {
  const Tensor embedding = diffusion::TimeEmbedding(time, maximum_time);
  const Tensor encoded   = nn::concat({nn::reshape(state, {-1, 64}), embedding}, 1);
  const Tensor h1 = diffusion::Dense(p, "hidden1", encoded, true), h2 = diffusion::Dense(p, "hidden2", h1, true);
  return {nn::reshape(diffusion::Dense(p, "output", h2, false), {-1, 8, 8, 1}), embedding, h1, h2};
}

struct ScorePath {
  Tensor score, normalized_score, epsilon, time_embedding, hidden1, hidden2;
};

inline ScorePath NoiseConditionalScore(const nn::ParameterStore& p, const Tensor& state, const Tensor& level) {
  const auto   predicted  = Predictor(p, state, level, 15.f);
  const Tensor normalized = nn::neg(predicted.epsilon);
  return {nn::div(normalized, ScoreSigma(level)), normalized, predicted.epsilon, predicted.time_embedding, predicted.hidden1, predicted.hidden2};
}

struct LangevinPath {
  Tensor value, mean, alpha;
};

/// x + .5 alpha score + sqrt(alpha) z, alpha = 2e-5 (sigma / .05)^2.
inline LangevinPath AnnealedLangevin(const Tensor& state, const Tensor& level, const Tensor& score, const Tensor& innovation) {
  const Tensor sigma = ScoreSigma(level);
  const Tensor alpha = nn::mul(nn::square(nn::div(sigma, .05)), 2e-5);
  const Tensor mean  = nn::add(state, nn::mul(nn::mul(alpha, .5), score));
  return {nn::add(mean, nn::mul(nn::sqrt(alpha), innovation)), mean, alpha};
}

/// tflow's ml::NoiseConditionalScoreModel (seed 12003, 5504 parameters / 6 tensors).
class NoiseConditionalScoreModel {
 public:
  nn::ParameterStore             store{12003};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit NoiseConditionalScoreModel(bool training = true) {
    diffusion::CreateMlp(store, 72, 64);
    if (training) optimizer = std::make_unique<nn::Optimizer>(store, .003, diffusion::Adam());
  }
  Tensor Loss(const Tensor& x, const Tensor& e, const Tensor& t, Map* fields = nullptr) const {
    const Tensor sigma  = ScoreSigma(t);
    const Tensor noised = nn::add(x, nn::mul(sigma, e));
    const auto   path   = NoiseConditionalScore(store, noised, t);
    const Tensor loss   = nn::mul(nn::mean(nn::square(nn::add(nn::mul(sigma, path.score), e))), .5);
    if (fields)
      *fields = diffusion::Detached({{"noised", noised}, {"score", path.score}, {"epsilon", path.epsilon},
                                     {"time_embedding", path.time_embedding}, {"hidden1", path.hidden1}, {"hidden2", path.hidden2},
                                     {"loss", loss}});
    return loss;
  }
  Map Evaluate(const Tensor& x, const Tensor& e, const Tensor& t) const {
    nn::NoGradGuard guard;
    Map             fields;
    Loss(x, e, t, &fields);
    return fields;
  }
  Map  Gradient(const Tensor& x, const Tensor& e, const Tensor& t) { return diffusion::ParameterGradients(store, Loss(x, e, t)); }
  void Update(const Tensor& x, const Tensor& e, const Tensor& t) { optimizer->step(nn::grad(Loss(x, e, t), store.parameters())); }
  /// State-only annealed Langevin update; input_gradient = d sum(score[:, pixel 0]) / d state.
  Map Reverse(const Tensor& x, const Tensor& t, const Tensor& z) const {
    Map out;
    {
      nn::NoGradGuard guard;
      const auto      path = NoiseConditionalScore(store, x, t);
      const auto      r    = AnnealedLangevin(x, t, path.score, z);
      out = {{"score", path.score}, {"epsilon", path.epsilon}, {"next", r.value}, {"mean", r.mean}, {"alpha", r.alpha}};
    }
    out["input_gradient"] =
        diffusion::FirstPixelInputGradient(x, [&](const Tensor& s) { return NoiseConditionalScore(store, s, t).score; });
    return out;
  }
  Map                      Arrays() const { return replay::Arrays(store); }
  std::vector<std::string> Snapshot() const {
    std::vector<std::string> out;
    for (const auto& [k, v] : Arrays()) out.push_back(diffusion::Bits(v));
    return out;
  }
};

}  // namespace ncsn
