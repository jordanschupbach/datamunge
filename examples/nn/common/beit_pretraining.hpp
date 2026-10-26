// Separate categorical patch tokenizer and frozen-target BEiT pretraining.
// Port of tflow's common/{nn,ml}/beit_pretraining.h.
#pragma once
#include "common/mae_pretraining.hpp"
namespace beit {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
inline Tensor Patches(const Tensor& images) {
  const auto n = images.size(0);
  return nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
}
inline A Augment(A b, unsigned seed, bool noisy) {
  const auto                            n = b.at("images").size(0);
  std::vector<float>                    mask(n * 16), noise(n * 16 * 16);
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> uniform(.000001f, .999999f);
  for (int row = 0; row < n; ++row) {
    int h = (rng() % 2) ? 2 : 3, w = 5 - h, y = rng() % (5 - h), x = rng() % (5 - w);
    for (int dy = 0; dy < h; ++dy)
      for (int dx = 0; dx < w; ++dx)
        mask[row * 16 + (y + dy) * 4 + x + dx] = 1;
  }
  if (noisy)
    for (auto& value : noise)
      value = -std::log(-std::log(uniform(rng)));
  b["mask"]  = nn::from_vector(mask, {n, 16});
  b["noise"] = nn::from_vector(noise, {n, 16, 16});
  return b;
}
class Model {
 public:
  bool                           tokenizer;
  nn::ParameterStore             parameters;
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool teacher = false, bool training = true)
      : tokenizer(teacher), parameters(teacher ? 49301 : 49303) {
    {
      nn::NoGradGuard guard;
      Fields(nn::zeros({1, 32, 32, 3}), nn::ones({1, 16}), nn::zeros({1, 16, 16}));
    }
    if (training) {
      nn::OptimizerConfig c;
      c.kind    = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, c);
    }
  }
  A Fields(const Tensor& images, const Tensor& mask, const Tensor& noise, const Tensor& hard = {}) {
    auto       patches = Patches(images);
    const auto n       = images.size(0);
    if (tokenizer) {
      auto hidden         = nn::tanh(mae::Linear(parameters, patches, "encoder", 8));
      auto logits         = mae::Linear(parameters, hidden, "categorical", 16),
           clean          = nn::softmax(logits, -1);
      auto sample         = hard.defined() ? hard : nn::softmax((logits + noise) / .7, -1);
      auto reconstruction = nn::sigmoid(mae::Linear(parameters, sample, "decoder", 192));
      auto mse            = nn::mean(nn::square(reconstruction - patches));
      auto kl = nn::mean(nn::sum(clean * (nn::log(clean + 1e-8) + float(std::log(16.))), {2}));
      return {
          {"patches", patches},
          {"features", hidden},
          {"logits", logits},
          {"probabilities", clean},
          {"sample", sample},
          {"reconstruction", reconstruction},
          {"mse", mse},
          {"kl", kl},
          {"loss", mse + kl * .001}};
    }
    auto patch         = mae::Linear(parameters, patches, "patch", 8);
    auto mt            = parameters.get("mask_token", {1, 1, 8}, nn::init::normal(.02)),
         m             = nn::expand_dims(mask, 2);
    auto h             = patch * (1.0 - m) + mt * m;
    auto cls           = parameters.get("cls", {1, 1, 8}, nn::init::normal(.02));
    auto pos           = parameters.get("position", {1, 17, 8}, nn::init::normal(.02));
    h                  = nn::concat({nn::tile(cls, {n, 1, 1}), h}, 1) + pos;
    auto      embedded = h;
    mae::Path blocks;
    for (int i = 0; i < 2; ++i)
      h = mae::Block(parameters, h, "block" + std::to_string(i), 8, 17, 2, blocks);
    auto hidden = nn::layer_norm(parameters, "final_ln", h, 1e-6);
    auto logits =
        mae::Linear(parameters, nn::slice(hidden, {0, 1, 0}, {n, 16, 8}), "token_head", 16);
    return {
        {"patches", patches},
        {"embedded", embedded},
        {"hidden", hidden},
        {"logits", logits},
        {"attention0", blocks.attention[0]},
        {"attention1", blocks.attention[1]},
        {"state0", blocks.states[0]},
        {"state1", blocks.states[1]}};
  }
  A Forward(const A& b, const Tensor& image) {
    const auto n = image.size(0);
    return Fields(
        image,
        tokenizer ? nn::ones({n, 16}) : b.at("mask"),
        b.count("noise") ? b.at("noise") : nn::zeros({n, 16, 16}));
  }
  Tensor Objective(A& fields, const A& b) {
    if (tokenizer)
      return fields.at("loss");
    auto ce = -nn::sum(
        nn::one_hot(b.at("tokens"), 16, 1, 0, nn::DType::Float32) *
            nn::log_softmax(fields.at("logits"), -1),
        {2});
    auto loss      = nn::sum(ce * b.at("mask")) / nn::sum(b.at("mask"));
    fields["ce"]   = ce;
    fields["loss"] = loss;
    return loss;
  }
  A Run(const A& b, bool gradients = false, const A& overrides = {}) {
    vit::Model::Validate(b);
    A saved;
    for (const auto& [k, v] : overrides) {
      saved[k] = parameters.at(k).detach().clone();
      parameters.set(k, v);
    }
    struct Restore {
      nn::ParameterStore& p;
      A&                  a;
      ~Restore() {
        for (const auto& [k, v] : a)
          p.set(k, v);
      }
    } restore{parameters, saved};
    auto                x    = b.at("images").detach().requires_grad_();
    auto                out  = Forward(b, x);
    auto                loss = Objective(out, b);
    std::vector<Tensor> inputs{x};
    if (gradients) {
      auto p = parameters.parameters();
      inputs.insert(inputs.end(), p.begin(), p.end());
    }
    auto g                = nn::grad(loss, inputs);
    out["input_gradient"] = g[0];
    if (gradients)
      for (std::size_t i = 0; i < parameters.size(); ++i)
        out[parameters.names()[i]] = g[i + 1];
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  Tensor Predict(const A& b) {
    vit::Model::Validate(b);
    nn::NoGradGuard guard;
    return Forward(b, b.at("images")).at("logits");
  }
  Tensor TokenIDs(const A& b) { return nn::cast(nn::argmax(Predict(b), 2), nn::DType::Int32); }
  A      HardDecode(const A& b) {
    check::Require(tokenizer, "hard decode only applies to tokenizer");
    auto            ids = TokenIDs(b);
    nn::NoGradGuard guard;
    auto            out = Fields(
        b.at("images"),
        nn::ones({ids.size(0), 16}),
        nn::zeros({ids.size(0), 16, 16}),
        nn::one_hot(ids, 16, 1, 0, nn::DType::Float32));
    return {{"tokens", ids}, {"reconstruction", out.at("reconstruction")}, {"mse", out.at("mse")}};
  }
  void Update(const A& b) {
    auto fields = Forward(b, b.at("images"));
    optimizer->minimize(Objective(fields, b));
  }
  A Arrays() const { return replay::Arrays(parameters); }
};
} // namespace beit
