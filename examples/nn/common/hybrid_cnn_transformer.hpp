// Learned local CNN followed by global relative-position attention.
// Port of tflow's common/{nn,ml}/hybrid_cnn_transformer.h.
#pragma once
#include "common/vision_transformer.hpp"
namespace hybrid {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
using vit::Loss;
struct Path {
  Tensor              conv0, conv1, features, hidden, logits;
  std::vector<Tensor> attention, states;
};
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  Path       out;
  const auto n    = images.size(0);
  auto       conv = [&](const Tensor& x, const std::string& name, int channels) {
    nn::ConvLayerOptions o;
    o.init       = nn::init::normal(.1);
    o.activation = nn::Activation::Relu;
    return nn::avg_pool2d(nn::conv2d(p, name, x, channels, 3, o), 2, 2, nn::Padding::Valid);
  };
  out.conv0          = conv(images, "cnn/conv0", 4);
  out.conv1          = conv(out.conv0, "cnn/conv1", 8);
  out.features       = nn::avg_pool2d(out.conv1, 2, 2, nn::Padding::Valid);
  auto             h = nn::reshape(out.features, {n, 16, 8});
  nn::DenseOptions o;
  o.init      = nn::init::normal(.1);
  auto linear = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, o);
  };
  std::vector<std::int32_t> indices;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
      indices.push_back((i / 4 - j / 4 + 3) * 7 + i % 4 - j % 4 + 3);
  auto index = nn::from_vector(indices, {16, 16});
  for (int i = 0; i < 2; ++i) {
    const auto name  = "block" + std::to_string(i);
    auto       z     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto       parts = nn::split(linear(z, name + "/qkv", 24), 3, 2);
    auto       heads = [&](const Tensor& x) {
      return nn::transpose(nn::reshape(x, {n, 16, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(parts[0]), k = heads(parts[1]), v = heads(parts[2]);
    auto table = p.get(name + "/relative_bias", {49, 2}, nn::init::normal(.02));
    auto bias  = nn::transpose(nn::gather(table, index, 0), {2, 0, 1});
    auto att   = nn::softmax(nn::matmul(q, k, false, true) * .5 + bias, -1);
    out.attention.push_back(att);
    auto mixed = nn::reshape(nn::transpose(nn::matmul(att, v), {0, 2, 1, 3}), {n, 16, 8});
    h          = h + linear(mixed, name + "/out", 8);
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    h          = h + linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", 8);
    out.states.push_back(h);
  }
  out.hidden = nn::layer_norm(p, "final_ln", h, 1e-6);
  out.logits = linear(nn::mean(out.hidden, {1}), "head", 10);
  return out;
}
class Model {
 public:
  nn::ParameterStore             parameters{49901};
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
        "ViT32x32RGB images");
    for (float v : x.to_vector<float>())
      check::Require(std::isfinite(v), "finite ViT pixels");
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
           {"conv0", path.conv0},
           {"conv1", path.conv1},
           {"features", path.features},
           {"hidden", path.hidden},
           {"attention0", path.attention[0]},
           {"attention1", path.attention[1]},
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
} // namespace hybrid
