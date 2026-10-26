// DeiT hard distillation: separate CLS/DIST tokens and mean-logit inference.
// Port of tflow's common/{nn,ml}/deit.h; the teacher has its own optimizer.
#pragma once
#include "common/vision_transformer.hpp"
namespace deit {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
struct Path {
  Tensor              logits, class_logits, distill_logits, patches, embedded, hidden;
  std::vector<Tensor> attention, states;
};
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  constexpr int D = 8, T = 18;
  const auto    n = images.size(0);
  Path          out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions dense;
  dense.init     = nn::init::normal(.1);
  auto projected = nn::dense(p, "patch", out.patches, D, dense);
  auto cls       = p.get("cls", {1, 1, D}, nn::init::zeros());
  auto dist      = p.get("distillation_token", {1, 1, D}, nn::init::normal(.02));
  auto position  = p.get("position", {1, T, D}, nn::init::normal(.02));
  auto h         = nn::add(
      nn::concat({nn::tile(cls, {n, 1, 1}), nn::tile(dist, {n, 1, 1}), projected}, 1), position);
  out.embedded = h;
  auto linear  = [&](const Tensor& x, const std::string& name, int width) {
    return nn::dense(p, name, x, width, dense);
  };
  for (int l = 0; l < 2; ++l) {
    auto name  = "block" + std::to_string(l);
    auto x     = nn::layer_norm(p, name + "/ln1", h, 1e-6);
    auto split = nn::split(linear(x, name + "/qkv", 3 * D), 3, 2);
    auto heads = [&](const Tensor& z) {
      return nn::transpose(nn::reshape(z, {n, T, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(split[0]), k = heads(split[1]), v = heads(split[2]);
    auto attention = nn::softmax(nn::matmul(q, k, false, true) * .5, -1);
    out.attention.push_back(attention);
    auto mixed = nn::reshape(nn::transpose(nn::matmul(attention, v), {0, 2, 1, 3}), {n, T, D});
    h          = nn::add(h, linear(mixed, name + "/out", D));
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    ff         = linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", D);
    h          = nn::add(h, ff);
    out.states.push_back(h);
  }
  out.hidden         = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features      = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.class_logits   = nn::dense(p, "head", features, 10, dense);
  auto dist_features = nn::squeeze(nn::slice(out.hidden, {0, 1, 0}, {n, 1, D}), {1});
  out.distill_logits = nn::dense(p, "distill_head", dist_features, 10, dense);
  out.logits         = (out.class_logits + out.distill_logits) * .5;
  return out;
}

class Model {
 public:
  bool                           teacher;
  nn::ParameterStore             parameters;
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool cnn = false, bool training = true)
      : teacher(cnn), parameters(cnn ? 48501 : 48503) {
    {
      nn::NoGradGuard guard;
      Fields(nn::zeros({1, 32, 32, 3}));
    }
    if (training) {
      nn::OptimizerConfig c;
      c.kind    = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, c);
    }
  }
  A Fields(const Tensor& images) {
    if (teacher) {
      nn::ConvLayerOptions c;
      c.activation = nn::Activation::Relu;
      auto first   = nn::avg_pool2d(
          nn::conv2d(parameters, "conv1", images, 4, 3, c), 2, 2, nn::Padding::Valid);
      auto second =
          nn::avg_pool2d(nn::conv2d(parameters, "conv2", first, 8, 3, c), 2, 2, nn::Padding::Valid);
      auto             features = nn::global_avg_pool2d(second);
      nn::DenseOptions o;
      o.init      = nn::init::normal(.1);
      auto logits = nn::dense(parameters, "head", features, 10, o);
      return {
          {"logits", logits}, {"features0", first}, {"features1", second}, {"features", features}};
    }
    auto p = Forward(parameters, images);
    return {
        {"logits", p.logits},
        {"class_logits", p.class_logits},
        {"distill_logits", p.distill_logits},
        {"patches", p.patches},
        {"embedded", p.embedded},
        {"hidden", p.hidden},
        {"attention0", p.attention[0]},
        {"attention1", p.attention[1]},
        {"state0", p.states[0]},
        {"state1", p.states[1]}};
  }
  Tensor Objective(A& fields, const A& b, double alpha) {
    check::Require(alpha >= 0 && alpha <= 1, "DeiT teacher weight0..1");
    fields["classification_loss"] =
        vit::Loss(fields.at(teacher ? "logits" : "class_logits"), b.at("labels"));
    auto loss = fields.at("classification_loss");
    if (!teacher) {
      fields["distillation_loss"] = vit::Loss(fields.at("distill_logits"), b.at("teacher_labels"));
      loss                        = loss * (1 - alpha) + fields.at("distillation_loss") * alpha;
    }
    fields["loss"] = loss;
    return loss;
  }
  A Run(const A& b, bool gradients = false, double alpha = .5, const A& overrides = {}) {
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
    auto                out  = Fields(x);
    auto                loss = Objective(out, b, alpha);
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
    return Fields(b.at("images")).at("logits");
  }
  Tensor TeacherLabels(const A& b) { return nn::cast(nn::argmax(Predict(b), 1), nn::DType::Int32); }
  void   Update(const A& b) {
    vit::Model::Validate(b);
    auto fields = Fields(b.at("images"));
    optimizer->minimize(Objective(fields, b, .5));
  }
  A Arrays() const { return replay::Arrays(parameters); }
};
} // namespace deit
