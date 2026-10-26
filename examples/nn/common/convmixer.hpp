// ConvMixer: patch convolution, residual depthwise and pointwise mixing, BN.
// Eager port of tflow's common/{nn,ml}/convmixer.h.
#pragma once
#include "common/vision_transformer.hpp"
namespace convmixer {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
struct Path {
  Tensor                   logits, stem, hidden;
  std::vector<Tensor>      depthwise, states, means, variances;
  std::vector<std::string> bn_names;
};
inline Path
Forward(nn::ParameterStore& p, nn::ParameterStore& stats, const Tensor& image, bool training) {
  Path out;
  auto bn = [&](const Tensor& x, const std::string& name) {
    auto gamma    = p.get(name + "/gamma", {8}, nn::init::ones()),
         beta     = p.get(name + "/beta", {8}, nn::init::zeros());
    auto mean     = stats.get(name + "/mean", {8}, nn::init::zeros()).detach();
    auto variance = stats.get(name + "/variance", {8}, nn::init::ones()).detach();
    auto mu = nn::mean(x, {0, 1, 2}, true), var = nn::mean(nn::square(x - mu), {0, 1, 2}, true);
    out.means.push_back(nn::reshape(mu, {8}));
    out.variances.push_back(nn::reshape(var, {8}));
    out.bn_names.push_back(name);
    return ((x - (training ? mu : mean)) * nn::rsqrt((training ? var : variance) + 1e-5)) * gamma +
           beta;
  };
  auto conv = [&](const Tensor& x, const std::string& name, int kernel, int stride, bool dw) {
    nn::ConvLayerOptions o;
    o.init    = nn::init::normal(.1);
    o.stride  = stride;
    o.padding = stride == 8 ? nn::Padding::Valid : nn::Padding::Same;
    return dw ? nn::depthwise_conv2d(p, name, x, 1, kernel, o)
              : nn::conv2d(p, name, x, 8, kernel, o);
  };
  auto h   = bn(nn::gelu(conv(image, "patch", 8, 8, false)), "patch_bn");
  out.stem = h;
  for (int i = 0; i < 2; ++i) {
    auto name = "block" + std::to_string(i);
    auto dw   = bn(nn::gelu(conv(h, name + "/depthwise", 3, 1, true)), name + "/depthwise_bn");
    h         = h + dw;
    out.depthwise.push_back(h);
    h = bn(nn::gelu(conv(h, name + "/pointwise", 1, 1, false)), name + "/pointwise_bn");
    out.states.push_back(h);
  }
  out.hidden = h;
  nn::DenseOptions o;
  o.init     = nn::init::normal(.1);
  out.logits = nn::dense(p, "head", nn::mean(h, {1, 2}), 10, o);
  return out;
}
class Model {
 public:
  nn::ParameterStore             parameters{49701}, stats{49702};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(parameters, stats, nn::zeros({1, 32, 32, 3}), false);
    }
    if (training) {
      nn::OptimizerConfig c;
      c.kind    = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, c);
    }
  }
  A Arrays() const {
    auto a = replay::Arrays(parameters);
    for (const auto& [k, v] : replay::Arrays(stats))
      a["stats/" + k] = v;
    return a;
  }
  void Load(const std::filesystem::path& file) {
    auto a = nn::load_npz(file.string());
    check::Require(a.size() == parameters.size() + stats.size(), "all learned and BN tensors");
    for (const auto& [k, v] : a)
      if (k.rfind("stats/", 0) == 0) {
        check::Require(stats.contains(k.substr(6)), "known BN tensor");
        stats.set(k.substr(6), v);
      } else {
        check::Require(parameters.contains(k), "known learned tensor");
        parameters.set(k, v);
      }
  }
  void Save(const std::filesystem::path& file) const {
    nn::save_npz(file.string(), replay::ToNamed(Arrays()));
  }
  A Run(const A& b, bool gradients = false, const A& overrides = {}, bool training = false) {
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
    auto                path = Forward(parameters, stats, x, gradients || training);
    auto                loss = vit::Loss(path.logits, b.at("labels"));
    std::vector<Tensor> inputs{x};
    if (gradients) {
      auto p = parameters.parameters();
      inputs.insert(inputs.end(), p.begin(), p.end());
    }
    auto g = nn::grad(loss, inputs);
    A    out{
           {"logits", path.logits},
           {"stem", path.stem},
           {"hidden", path.hidden},
           {"depthwise0", path.depthwise[0]},
           {"depthwise1", path.depthwise[1]},
           {"state0", path.states[0]},
           {"state1", path.states[1]},
           {"loss", loss},
           {"input_gradient", g[0]}};
    for (int i = 0; i < 5; ++i) {
      out["mean" + std::to_string(i)]     = path.means[i];
      out["variance" + std::to_string(i)] = path.variances[i];
    }
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
    return Forward(parameters, stats, b.at("images"), false).logits;
  }
  void Update(const A& b) {
    vit::Model::Validate(b);
    auto path = Forward(parameters, stats, b.at("images"), true);
    optimizer->minimize(vit::Loss(path.logits, b.at("labels")));
    nn::NoGradGuard guard;
    const double    sample = b.at("images").size(0) * 16;
    for (int i = 0; i < 5; ++i) {
      const auto name = path.bn_names[i];
      stats.set(name + "/mean", stats.at(name + "/mean") * .9 + path.means[i].detach() * .1);
      stats.set(
          name + "/variance",
          stats.at(name + "/variance") * .9 +
              path.variances[i].detach() * (.1 * sample / (sample - 1)));
    }
  }
};
} // namespace convmixer
