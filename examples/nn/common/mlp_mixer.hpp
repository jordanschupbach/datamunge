// Attention-free spatial token/channel MLPs, tflow's common/{nn,ml}/mlp_mixer.h.
#pragma once
#include "common/vision_transformer.hpp"
namespace mixer {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
using vit::Loss;
struct Path {
  Tensor              patches, embedded, hidden, logits;
  std::vector<Tensor> token_states, states;
};
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  const auto n = images.size(0);
  Path       out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions opts;
  opts.init   = nn::init::normal(.1);
  auto linear = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, opts);
  };
  auto h       = linear(out.patches, "patch", 8);
  out.embedded = h;
  for (int i = 0; i < 2; ++i) {
    auto name  = "block" + std::to_string(i);
    auto z     = nn::transpose(nn::layer_norm(p, name + "/ln1", h, 1e-6), {0, 2, 1});
    auto token = linear(nn::gelu(linear(z, name + "/token1", 8), true), name + "/token2", 16);
    h          = nn::add(h, nn::transpose(token, {0, 2, 1}));
    out.token_states.push_back(h);
    auto channel = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    channel =
        linear(nn::gelu(linear(channel, name + "/channel1", 16), true), name + "/channel2", 8);
    h = nn::add(h, channel);
    out.states.push_back(h);
  }
  out.hidden = nn::layer_norm(p, "final_ln", h, 1e-6);
  out.logits = linear(nn::mean(out.hidden, {1}), "head", 10);
  return out;
}
class Model {
 public:
  nn::ParameterStore             parameters{49501};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(parameters, nn::zeros({1, 32, 32, 3}));
    }
    if (training) {
      nn::OptimizerConfig c;
      c.kind    = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, c);
    }
  }
  static void Validate(const A& b) {
    const auto& x = b.at("images");
    check::Require(
        x.dtype() == nn::DType::Float32 && x.dim() == 4 && x.size(0) > 0 && x.size(1) == 32 &&
            x.size(2) == 32 && x.size(3) == 3,
        "MLP-Mixer32x32RGB images");
    for (float v : x.to_vector<float>())
      check::Require(std::isfinite(v), "finite MLP-Mixer pixels");
  }
  A Run(const A& b, bool gradients = false, const A& overrides = {}) {
    Validate(b);
    A saved;
    for (const auto& [k, v] : overrides) {
      saved[k] = parameters.at(k).detach().clone();
      parameters.set(k, v);
    }
    struct Restore {
      nn::ParameterStore& p;
      A&                  saved;
      ~Restore() {
        for (const auto& [k, v] : saved)
          p.set(k, v);
      }
    } restore{parameters, saved};
    auto                x    = b.at("images").detach().requires_grad_();
    auto                path = Forward(parameters, x);
    auto                loss = Loss(path.logits, b.at("labels"));
    std::vector<Tensor> inputs{x};
    if (gradients) {
      auto values = parameters.parameters();
      inputs.insert(inputs.end(), values.begin(), values.end());
    }
    auto g = nn::grad(loss, inputs);
    A    out{
           {"logits", path.logits},
           {"patches", path.patches},
           {"embedded", path.embedded},
           {"hidden", path.hidden},
           {"token_state0", path.token_states[0]},
           {"token_state1", path.token_states[1]},
           {"state0", path.states[0]},
           {"state1", path.states[1]},
           {"loss", loss},
           {"input_gradient", g[0]}};
    if (gradients)
      for (std::size_t i = 0; i < parameters.size(); ++i)
        out[parameters.names()[i]] = g[i + 1];
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  Tensor Predict(const A& b) {
    Validate(b);
    nn::NoGradGuard guard;
    return Forward(parameters, b.at("images")).logits;
  }
  void Update(const A& b) {
    Validate(b);
    auto path = Forward(parameters, b.at("images"));
    optimizer->minimize(Loss(path.logits, b.at("labels")));
  }
  A Arrays() const { return replay::Arrays(parameters); }
};
} // namespace mixer
