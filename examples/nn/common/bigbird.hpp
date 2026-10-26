// BigBird global/sliding/seeded-random block gathers for bidirectional ViT.
// Port of tflow's common/{nn,ml}/bigbird.h.
#pragma once
#include "common/vision_transformer.hpp"
namespace bigbird {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
using vit::Loss;
struct Pattern {
  std::vector<int>   indices, globals;
  std::vector<float> bias;
  int                first, last, width;
};
inline Pattern Slots(int n, unsigned seed, int block = 2, int random_blocks = 2) {
  check::Require(n > 2 * block && block > 0 && random_blocks >= 0, "valid BigBird blocks");
  int count = (n + block - 1) / block, last = (count - 1) * block,
      width = (5 + random_blocks) * block;
  Pattern p;
  p.first = block;
  p.last  = last;
  p.width = width;
  p.indices.resize(n * width);
  p.bias.resize(n * width);
  p.globals.resize(block + n - last);
  for (int i = 0; i < block; ++i)
    p.globals[i] = i;
  for (int i = last; i < n; ++i)
    p.globals[block + i - last] = i;
  for (int qb = 0; qb < count; ++qb) {
    std::vector<int> chosen{0, count - 1};
    for (int j = std::max(0, qb - 1); j <= std::min(count - 1, qb + 1); ++j)
      if (std::find(chosen.begin(), chosen.end(), j) == chosen.end())
        chosen.push_back(j);
    std::vector<int> candidates;
    for (int j = 0; j < count; ++j)
      if (std::find(chosen.begin(), chosen.end(), j) == chosen.end())
        candidates.push_back(j);
    uint32_t state = seed ^ (0x9e3779b9u * unsigned(qb + 1));
    if (!state)
      state = 1;
    for (int i = int(candidates.size()) - 1; i > 0; --i) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      std::swap(candidates[i], candidates[state % unsigned(i + 1)]);
    }
    for (int i = 0; i < std::min(random_blocks, int(candidates.size())); ++i)
      chosen.push_back(candidates[i]);
    std::vector<int> keys;
    for (auto b : chosen)
      for (int j = b * block; j < std::min(n, (b + 1) * block); ++j)
        keys.push_back(j);
    check::Require(int(keys.size()) <= width, "bounded BigBird slots");
    for (int i = qb * block; i < std::min(n, (qb + 1) * block); ++i)
      for (int j = 0; j < width; ++j) {
        bool valid               = j < int(keys.size());
        p.indices[i * width + j] = valid ? keys[j] : 0;
        p.bias[i * width + j]    = valid ? 0.f : -1e9f;
      }
  }
  return p;
}
struct ContextPath {
  Tensor context, attention, global_attention, indices, mask, global_indices;
};
inline ContextPath
Context(const Tensor& q, const Tensor& k, const Tensor& v, unsigned seed = 50911) {
  auto        n = q.size(2);
  auto        p = Slots(n, seed);
  ContextPath out;
  out.indices        = nn::from_vector(p.indices, {n, p.width});
  out.mask           = nn::from_vector(p.bias, {n, p.width});
  out.global_indices = nn::from_vector(p.globals, {static_cast<long long>(p.globals.size())});
  auto gather        = [&](const Tensor& x) {
    return nn::transpose(
        nn::gather(nn::transpose(x, {2, 0, 1, 3}), out.indices, 0), {2, 3, 0, 1, 4});
  };
  auto keys = gather(k), values = gather(v);
  out.attention = nn::softmax(nn::sum(nn::expand_dims(q, 3) * keys, {4}) * .5 + out.mask, -1);
  auto local    = nn::sum(nn::expand_dims(out.attention, 4) * values, {3});
  auto gq       = nn::transpose(
      nn::gather(nn::transpose(q, {2, 0, 1, 3}), out.global_indices, 0), {1, 2, 0, 3});
  out.global_attention = nn::softmax(nn::matmul(gq, k, false, true) * .5, -1);
  auto global          = nn::matmul(out.global_attention, v);
  auto slice           = [&](const Tensor& x, int start, int length) {
    return nn::slice(x, {0, 0, start, 0}, {q.size(0), q.size(1), length, q.size(3)});
  };
  out.context = nn::concat(
      {slice(global, 0, p.first),
       slice(local, p.first, p.last - p.first),
       slice(global, p.first, n - p.last)},
      2);
  return out;
}
struct Path {
  Tensor              logits, patches, embedded, hidden;
  std::vector<Tensor> attention, states, queries, keys, values, indices, masks, global_attention;
};
inline Path Forward(nn::ParameterStore& p, const Tensor& images) {
  constexpr int D = 8, T = 17;
  const auto    n = images.size(0);
  Path          out;
  out.patches = nn::reshape(
      nn::transpose(nn::reshape(images, {n, 4, 8, 4, 8, 3}), {0, 1, 3, 2, 4, 5}), {n, 16, 192});
  nn::DenseOptions dense;
  dense.init     = nn::init::normal(.1);
  auto projected = nn::dense(p, "patch", out.patches, D, dense);
  auto cls       = p.get("cls", {1, 1, D}, nn::init::zeros());
  auto position  = p.get("position", {1, T, D}, nn::init::normal(.02));
  auto h         = nn::add(nn::concat({nn::tile(cls, {n, 1, 1}), projected}, 1), position);
  out.embedded   = h;
  auto linear    = [&](const Tensor& x, const std::string& name, int width) {
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
    auto attended = Context(q, k, v, 50911 + l);
    out.queries.push_back(q);
    out.keys.push_back(k);
    out.values.push_back(v);
    out.attention.push_back(attended.attention);
    out.global_attention.push_back(attended.global_attention);
    out.indices.push_back(attended.indices);
    out.masks.push_back(attended.mask);
    auto mixed = nn::reshape(nn::transpose(attended.context, {0, 2, 1, 3}), {n, T, D});
    h          = nn::add(h, linear(mixed, name + "/out", D));
    auto ff    = nn::layer_norm(p, name + "/ln2", h, 1e-6);
    ff         = linear(nn::gelu(linear(ff, name + "/ff1", 16), true), name + "/ff2", D);
    h          = nn::add(h, ff);
    out.states.push_back(h);
  }
  out.hidden    = nn::layer_norm(p, "final_ln", h, 1e-6);
  auto features = nn::squeeze(nn::slice(out.hidden, {0, 0, 0}, {n, 1, D}), {1});
  out.logits    = nn::dense(p, "head", features, 10, dense);
  return out;
}
class Model {
 public:
  nn::ParameterStore             parameters{50901};
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
           {"patches", path.patches},
           {"embedded", path.embedded},
           {"hidden", path.hidden},
           {"attention0", path.attention[0]},
           {"attention1", path.attention[1]},
           {"state0", path.states[0]},
           {"state1", path.states[1]},
           {"loss", loss},
           {"input_gradient", g[0]}};
    for (int i = 0; i < 2; ++i) {
      auto suffix                      = std::to_string(i);
      out["q" + suffix]                = path.queries[i];
      out["k" + suffix]                = path.keys[i];
      out["v" + suffix]                = path.values[i];
      out["indices" + suffix]          = path.indices[i];
      out["mask" + suffix]             = path.masks[i];
      out["global_attention" + suffix] = path.global_attention[i];
    }
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
} // namespace bigbird
