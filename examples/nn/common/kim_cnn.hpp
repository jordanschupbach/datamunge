#pragma once
#include <algorithm>
#include <datamunge/nn/kim_cnn_models.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/autograd.hpp>
namespace sentence {
namespace nn = datamunge::nn;
using datamunge::nn::models::KimCnn;
using datamunge::nn::models::KimConfig;
using datamunge::nn::models::KimMaxNorm;
using datamunge::nn::models::KimOutput;
using datamunge::nn::models::LengthMask;
using nn::Tensor;

// Legacy spelling adapter;the actual update rule is the canonical tensor optimizer.
class KimAdadelta {
public:
  explicit KimAdadelta(nn::ParameterStore &p) : p_(p), optimizer_(p, 1., Config()) {
    for (std::size_t i = 0; i < p.size(); ++i) {
      slots.get("gradient" + std::to_string(i), p.parameters()[i].shape(), nn::init::zeros());
      slots.get("update" + std::to_string(i), p.parameters()[i].shape(), nn::init::zeros());
    }
  }
  void Step(const std::vector<Tensor> &gradients) {
    optimizer_.step(gradients);
    nn::NoGradGuard guard;
    if (std::find(p_.names().begin(), p_.names().end(), "classifier/W") != p_.names().end())
      p_.at("classifier/W").copy_(KimMaxNorm(p_.at("classifier/W")));
    for (const auto &[name, value] : optimizer_.state()) {
      if (name.starts_with("accum/"))
        slots.set("gradient" + name.substr(6), value);
      else if (name.starts_with("accum_update/"))
        slots.set("update" + name.substr(13), value);
    }
  }
  nn::ParameterStore slots{319};

private:
  static nn::OptimizerConfig Config() {
    nn::OptimizerConfig c;
    c.kind = nn::OptimizerKind::Adadelta;
    c.rho = .95;
    c.epsilon = 1e-6;
    return c;
  }
  nn::ParameterStore &p_;
  nn::Optimizer optimizer_;
};
} // namespace sentence
