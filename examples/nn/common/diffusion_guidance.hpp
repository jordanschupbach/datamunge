// Port of tflow's common/nn/diffusion_guidance.h and common/ml/diffusion_guidance.h:
// a noisy-image classifier (3722 parameters) and a NULL-dropout class-conditional
// epsilon MLP (5856 parameters, 11-way one-hot with NULL = 10), classifier
// guidance eps - w sqrt(1 - alpha_bar) grad_x log p(y | x_t, t) and
// classifier-free guidance eps_null + w (eps_class - eps_null), both feeding the
// clipped-x0 DDPM posterior update.
#pragma once

#include "common/ddpm.hpp"

namespace diffusion {

struct GuidancePredictionPath {
  Tensor prediction, time_embedding, hidden1, hidden2, class_onehot;
};

inline void CreateGuidance(nn::ParameterStore& p, bool classifier) { CreateMlp(p, classifier ? 72 : 83, classifier ? 10 : 64); }

/// tflow's GuidancePredictor: classifier [state 64, time 8] -> 10 logits;
/// conditional [state 64, time 8, one-hot 11] -> epsilon [n, 8, 8, 1].
inline GuidancePredictionPath GuidancePredictor(const nn::ParameterStore& p, const Tensor& state, const Tensor& time,
                                                const Tensor& label, bool classifier) {
  const Tensor embedding    = TimeEmbedding(time);
  const Tensor class_onehot = nn::one_hot(nn::cast(label, nn::DType::Int64), 11);
  const Tensor flat         = nn::reshape(state, {-1, 64});
  const Tensor encoded      = classifier ? nn::concat({flat, embedding}, 1) : nn::concat({flat, embedding, class_onehot}, 1);
  const Tensor h1 = Dense(p, "hidden1", encoded, true), h2 = Dense(p, "hidden2", h1, true);
  Tensor       prediction = Dense(p, "output", h2, false);
  if (!classifier) prediction = nn::reshape(prediction, {-1, 8, 8, 1});
  return {prediction, embedding, h1, h2, class_onehot};
}

/// sum(log_softmax(logits) * one_hot(label, 10), 1).
inline Tensor ClassifierLogProbability(const Tensor& logits, const Tensor& label) {
  return nn::sum(nn::mul(nn::log_softmax(logits, -1), nn::one_hot(nn::cast(label, nn::DType::Int64), 10)), {1});
}

inline Tensor ClassifierGuidedEpsilon(const Tensor& epsilon, const Tensor& time, const Tensor& gradient, const Tensor& scale) {
  return nn::sub(epsilon, nn::mul(nn::mul(scale, Extract("sqrt_one_minus_alpha_bar", time)), gradient));
}

inline Tensor ClassifierFreeEpsilon(const Tensor& conditional, const Tensor& null_prediction, const Tensor& scale) {
  return nn::add(null_prediction, nn::mul(scale, nn::sub(conditional, null_prediction)));
}

/// tflow's ml::GuidanceTrainingModel: noisy classifier (seed 14003, CE on x_t)
/// or conditional epsilon (seed 14007, epsilon MSE), Adam .003.
class GuidanceTrainingModel {
 public:
  nn::ParameterStore             store;
  bool                           is_classifier;
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit GuidanceTrainingModel(bool classifier, bool training = true)
      : store(classifier ? 14003 : 14007), is_classifier(classifier) {
    CreateGuidance(store, classifier);
    if (training) optimizer = std::make_unique<nn::Optimizer>(store, .003, Adam());
  }
  Tensor Loss(const Tensor& x, const Tensor& e, const Tensor& t, const Tensor& y, Map* fields = nullptr) {
    const Tensor noised = Noised(x, e, t);
    const auto   path   = GuidancePredictor(store, noised, t, y, is_classifier);
    Tensor       loss, probability, log_probability;
    if (is_classifier) {
      probability     = nn::softmax(path.prediction, -1);
      log_probability = ClassifierLogProbability(path.prediction, y);
      loss            = nn::neg(nn::mean(log_probability, {0}));
    } else {
      loss = nn::mean(nn::square(nn::sub(path.prediction, e)));
    }
    if (fields) {
      *fields = Detached({{"noised", noised}, {"prediction", path.prediction}, {"time_embedding", path.time_embedding},
                          {"hidden1", path.hidden1}, {"hidden2", path.hidden2}, {"loss", loss}});
      if (is_classifier) {
        (*fields)["probability"]     = probability.detach();
        (*fields)["log_probability"] = log_probability.detach();
      } else {
        (*fields)["class_onehot"] = path.class_onehot.detach();
      }
    }
    return loss;
  }
  Map Evaluate(const Tensor& x, const Tensor& e, const Tensor& t, const Tensor& y) {
    nn::NoGradGuard guard;
    Map             fields;
    Loss(x, e, t, y, &fields);
    return fields;
  }
  Map  Gradient(const Tensor& x, const Tensor& e, const Tensor& t, const Tensor& y) { return ParameterGradients(store, Loss(x, e, t, y)); }
  void Update(const Tensor& x, const Tensor& e, const Tensor& t, const Tensor& y) {
    optimizer->step(nn::grad(Loss(x, e, t, y), store.parameters()));
  }
  /// State-only prediction and input gradient (classifier: of sum log p(y|x);
  /// conditional: of the first epsilon pixel).
  Map Predict(const Tensor& x, const Tensor& t, const Tensor& y) {
    Tensor s = x.detach().clone();
    s.requires_grad_();
    const auto path = GuidancePredictor(store, s, t, y, is_classifier);
    Map        out;
    if (is_classifier) {
      const Tensor logp     = ClassifierLogProbability(path.prediction, y);
      out["input_gradient"]  = nn::grad(nn::sum(logp), {s})[0];
      out["log_probability"] = logp.detach();
    } else {
      out["input_gradient"] = nn::grad(nn::sum(nn::slice_axis(nn::reshape(path.prediction, {-1, 64}), 1, 0, 1)), {s})[0];
    }
    out["prediction"] = path.prediction.detach();
    return out;
  }
  Map                      Arrays() const { return replay::Arrays(store); }
  std::vector<std::string> Snapshot() const {
    std::vector<std::string> out;
    for (const auto& [k, v] : Arrays()) out.push_back(Bits(v));
    return out;
  }
};

/// tflow's ml::GuidedDiffusionModel: frozen unconditional DDPM, noisy classifier
/// and conditional epsilon; one call evaluates both guided reverse transitions.
class GuidedDiffusionModel {
 public:
  nn::ParameterStore unconditional{11003}, classifier{14003}, conditional{14007};

  GuidedDiffusionModel() {
    CreateMlp(unconditional, 72, 64);
    CreateGuidance(classifier, true);
    CreateGuidance(conditional, false);
  }
  Map Evaluate(const Tensor& x, const Tensor& t, const Tensor& y, float weight, const Tensor& z) {
    const Tensor scale = nn::scalar(weight, nn::DType::Float32);
    Tensor       s     = x.detach().clone();
    s.requires_grad_();
    const auto   cls  = GuidancePredictor(classifier, s, t, y, true);
    const Tensor logp = ClassifierLogProbability(cls.prediction, y);
    const Tensor classifier_gradient = nn::grad(nn::sum(logp), {s})[0];
    nn::NoGradGuard guard;
    const auto   base     = DdpmPredictor(unconditional, x, t);
    const auto   cond     = GuidancePredictor(conditional, x, t, y, false);
    const auto   null     = GuidancePredictor(conditional, x, t, nn::add(nn::mul(nn::cast(y, nn::DType::Int32), nn::scalar(0, nn::DType::Int32)), nn::scalar(10, nn::DType::Int32)), false);
    const Tensor cg_eps   = ClassifierGuidedEpsilon(base.epsilon, t, classifier_gradient, scale);
    const Tensor cfg_eps  = ClassifierFreeEpsilon(cond.prediction, null.prediction, scale);
    const auto   base_rev = DdpmReverse(x, t, base.epsilon, z), cg_rev = DdpmReverse(x, t, cg_eps, z),
               cfg_rev = DdpmReverse(x, t, cfg_eps, z);
    return Detached({{"base_epsilon", base.epsilon},
                     {"classifier_logits", cls.prediction},
                     {"classifier_logp", logp},
                     {"classifier_gradient", classifier_gradient},
                     {"conditional_epsilon", cond.prediction},
                     {"null_epsilon", null.prediction},
                     {"cg_epsilon", cg_eps},
                     {"cfg_epsilon", cfg_eps},
                     {"base_next", base_rev.value},
                     {"cg_next", cg_rev.value},
                     {"cfg_next", cfg_rev.value},
                     {"cg_mean", cg_rev.mean},
                     {"cfg_mean", cfg_rev.mean},
                     {"cg_x0_raw", cg_rev.x0_raw},
                     {"cfg_x0_raw", cfg_rev.x0_raw},
                     {"cg_x0_clipped", cg_rev.x0_clipped},
                     {"cfg_x0_clipped", cfg_rev.x0_clipped}});
  }
  std::vector<std::string> Snapshot() const {
    std::vector<std::string> out;
    for (const auto* p : {&unconditional, &classifier, &conditional})
      for (const auto& [k, v] : replay::Arrays(*p)) out.push_back(Bits(v));
    return out;
  }
};

}  // namespace diffusion
