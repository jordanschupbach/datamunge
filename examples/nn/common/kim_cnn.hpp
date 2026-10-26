// Kim CNN-rand: learned word embeddings, 3/4/5 filters, max pooling, dropout
// (port of tflow's common/nn/kim_cnn.h), plus Kim's Adadelta (rho .95,
// eps 1e-6) with the softmax-column max-norm 3 constraint.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <datamunge/nn/nn.hpp>

namespace sentence {

namespace nn = datamunge::nn;
using nn::Tensor;

struct KimConfig {
  int vocabulary = 8000, time = 64, embedding = 300, filters = 100;
};
struct KimOutput {
  Tensor              logits, features, embedding;
  std::vector<Tensor> maps;
};

/// [N, T] float mask: t < lengths[n].
inline Tensor LengthMask(const Tensor& lengths, std::int64_t time, nn::DType dtype = nn::DType::Float32) {
  const Tensor positions = nn::arange(0, static_cast<double>(time), 1, nn::DType::Int32);
  return nn::cast(nn::less(nn::expand_dims(positions, 0), nn::expand_dims(nn::cast(lengths, nn::DType::Int32), 1)), dtype);
}

/// Kim's max-norm constraint: columns of w scaled to norm <= 3.
inline Tensor KimMaxNorm(const Tensor& w) {
  const Tensor norm = nn::sqrt(nn::sum(nn::square(w), {0}, true));
  return nn::div(w, nn::maximum(nn::scalar_like(1.0, w), nn::div(norm, 3.0)));
}

/// tokens [N, T] int, lengths [N]. The PAD row 0 of the embedding is a
/// structural zero; "embedding/nonpad" holds rows 1..V-1 (U(-.25, .25)).
inline KimOutput KimCnn(nn::ParameterStore& p, const Tensor& tokens, const Tensor& lengths, const KimConfig& config = {},
                        bool training = false, double keep = 1.0, nn::Generator* gen = nullptr) {
  if (config.vocabulary < 2 || config.time < 1 || config.embedding < 1 || config.filters < 1)
    throw std::invalid_argument("Kim CNN dimensions");
  KimOutput    out;
  const Tensor learned = p.get("embedding/nonpad", {config.vocabulary - 1, config.embedding}, nn::init::uniform(.25));
  out.embedding = nn::concat({nn::zeros({1, config.embedding}, learned.dtype()), learned}, 0);
  const Tensor mask   = LengthMask(lengths, config.time, learned.dtype());
  const Tensor x      = nn::mul(nn::gather(out.embedding, tokens, 0), nn::expand_dims(mask, 2));
  const Tensor padded = nn::pad(x, {{0, 0}, {4, 4}, {0, 0}});
  std::vector<Tensor> pooled;
  for (int kernel : {3, 4, 5}) {
    nn::ConvLayerOptions opt;
    opt.padding    = nn::Padding::Valid;
    opt.activation = nn::Activation::Relu;
    const Tensor map = nn::conv1d(p, "window" + std::to_string(kernel), padded, config.filters, kernel, opt);
    out.maps.push_back(map);
    pooled.push_back(nn::max(map, {1}));
  }
  out.features   = nn::concat(pooled, 1);
  Tensor feature = out.features;
  if (training && keep < 1) feature = nn::dropout(feature, 1 - keep, true, *gen);
  out.logits = nn::dense(p, "classifier", feature, 2);
  return out;
}

/// Adadelta rho=.95, eps=1e-6 and softmax-column norm <= 3 as in Kim. Slots
/// "gradient<i>" / "update<i>" start at zero; every new value is computed
/// before any parameter or slot is written (TF's all-RHS barrier).
class KimAdadelta {
 public:
  explicit KimAdadelta(nn::ParameterStore& p) : p_(p) {
    for (std::size_t i = 0; i < p.size(); ++i) {
      const auto shape = p.parameters()[i].shape();
      slots.get("gradient" + std::to_string(i), shape, nn::init::zeros());
      slots.get("update" + std::to_string(i), shape, nn::init::zeros());
    }
  }
  void Step(const std::vector<Tensor>& gradients) {
    if (gradients.size() != p_.size()) throw std::invalid_argument("Adadelta gradient alignment");
    nn::NoGradGuard                          guard;
    std::vector<std::pair<std::string, Tensor>> params, slot_values;
    for (std::size_t i = 0; i < gradients.size(); ++i) {
      const std::string name = p_.names()[i];
      const Tensor      g    = gradients[i].detach();
      const Tensor      g2 = slots.at("gradient" + std::to_string(i)).detach(), d2 = slots.at("update" + std::to_string(i)).detach();
      const Tensor next_g = nn::add(nn::mul(g2, .95), nn::mul(nn::square(g), .05));
      const Tensor delta  = nn::mul(nn::div(nn::sqrt(nn::add(d2, 1e-6)), nn::sqrt(nn::add(next_g, 1e-6))), g);
      const Tensor next_d = nn::add(nn::mul(d2, .95), nn::mul(nn::square(delta), .05));
      Tensor       next_p = nn::sub(p_.at(name).detach(), delta);
      if (name == "classifier/W") next_p = KimMaxNorm(next_p);
      params.emplace_back(name, next_p);
      slot_values.emplace_back("gradient" + std::to_string(i), next_g);
      slot_values.emplace_back("update" + std::to_string(i), next_d);
    }
    for (const auto& [name, value] : params) p_.set(name, value.clone());
    for (const auto& [name, value] : slot_values) slots.set(name, value.clone());
  }
  nn::ParameterStore slots{319};

 private:
  nn::ParameterStore& p_;
};

}  // namespace sentence
