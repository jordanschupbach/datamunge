// Reformer single-hash angular buckets, stable sort and bounded chunk attention.
// Port of tflow's common/{nn,ml}/reformer.h.
#pragma once
#include <numeric>
#include <random>

#include "common/vision_transformer.hpp"
namespace reformer {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
using vit::Loss;
inline Tensor Rotation(int d, int half, unsigned seed) {
  std::mt19937                    rng(seed);
  std::normal_distribution<float> normal;
  std::vector<float>              x(d * half);
  for (auto& v : x)
    v = normal(rng);
  return nn::from_vector(x, {d, half});
}
struct ContextPath {
  Tensor context, attention, buckets, order;
};
// Hashing/sorting is intentionally discrete; gathered QK and V retain their
// differentiable paths, matching the source's conditional LSH derivatives.
inline ContextPath Context(
    const Tensor& qk,
    const Tensor& v,
    const Tensor& rotation,
    int           buckets        = 4,
    int           chunk          = 4,
    const Tensor& frozen_buckets = {},
    const Tensor& frozen_order   = {}) {
  int n = qk.size(2), bh = qk.size(0) * qk.size(1), padded = ((n + chunk - 1) / chunk) * chunk,
      groups = padded / chunk;
  check::Require(
      n > chunk && buckets >= 2 && buckets % 2 == 0 && chunk > 0, "valid Reformer dimensions");
  auto q         = nn::pad(qk, {{0, 0}, {0, 0}, {0, padded - n}, {0, 0}}),
       values    = nn::pad(v, {{0, 0}, {0, 0}, {0, padded - n}, {0, 0}});
  auto projected = nn::matmul(q.detach(), rotation.detach());
  auto hash      = nn::argmax(nn::concat({projected, -projected}, 3), 3).to_vector<std::int64_t>();
  std::vector<int> order(bh * padded), h(bh * padded), flat(bh * padded), inverse(bh * padded);
  for (int b = 0; b < bh; ++b) {
    std::vector<int> ids(padded);
    std::iota(ids.begin(), ids.end(), 0);
    for (int i = 0; i < padded; ++i)
      h[b * padded + i] = i < n ? hash[b * padded + i] : buckets;
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int c) {
      return h[b * padded + a] < h[b * padded + c];
    });
    for (int i = 0; i < padded; ++i) {
      order[b * padded + i]        = ids[i];
      flat[b * padded + i]         = b * padded + ids[i];
      inverse[b * padded + ids[i]] = b * padded + i;
    }
  }
  if (frozen_buckets.defined())
    h = frozen_buckets.to_vector<int>();
  if (frozen_order.defined()) {
    order = frozen_order.to_vector<int>();
    for (int b = 0; b < bh; ++b)
      for (int i = 0; i < padded; ++i) {
        flat[b * padded + i]                        = b * padded + order[b * padded + i];
        inverse[b * padded + order[b * padded + i]] = b * padded + i;
      }
  }
  auto gather = [&](const Tensor& x) {
    return nn::reshape(
        nn::gather(nn::reshape(x, {bh * padded, 4}), nn::from_vector(flat, {bh * padded}), 0),
        {bh, groups, chunk, 4});
  };
  auto query = gather(q), val = gather(values);
  auto key =
      query / nn::sqrt(nn::maximum(nn::sum(nn::square(query), {3}, true), nn::scalar(1e-12)));
  auto previous = [&](const Tensor& x) {
    return nn::concat(
        {nn::slice(x, {0, groups - 1, 0, 0}, {bh, 1, chunk, 4}),
         nn::slice(x, {0, 0, 0, 0}, {bh, groups - 1, chunk, 4})},
        1);
  };
  auto keys = nn::concat({previous(key), key}, 2), vals = nn::concat({previous(val), val}, 2);
  std::vector<float> bias(bh * groups * chunk * 2 * chunk), active(bias.size());
  for (int b = 0; b < bh; ++b)
    for (int g = 0; g < groups; ++g)
      for (int i = 0; i < chunk; ++i)
        for (int j = 0; j < 2 * chunk; ++j) {
          int qi        = order[b * padded + g * chunk + i],
              kg        = j < chunk ? (g + groups - 1) % groups : g,
              kj        = order[b * padded + kg * chunk + j % chunk];
          int  index    = ((b * groups + g) * chunk + i) * 2 * chunk + j;
          bool allowed  = h[b * padded + qi] == h[b * padded + kj] && kj < n;
          bool self     = qi == kj;
          active[index] = allowed && !self ? 1.f : 0.f;
          bias[index]   = !allowed ? -1e9f : self ? -1e5f : 0.f;
        }
  auto logits = nn::matmul(query, keys, false, true) * .5;
  auto scores = logits * nn::from_vector(active, {bh, groups, chunk, 2 * chunk}) +
                nn::from_vector(bias, {bh, groups, chunk, 2 * chunk});
  ContextPath out;
  out.attention =
      nn::reshape(nn::softmax(scores, -1), {qk.size(0), qk.size(1), groups, chunk, 2 * chunk});
  auto context  = nn::reshape(nn::matmul(nn::softmax(scores, -1), vals), {bh * padded, 4});
  auto unsorted = nn::reshape(
      nn::gather(context, nn::from_vector(inverse, {bh * padded}), 0),
      {qk.size(0), qk.size(1), padded, 4});
  out.context = nn::slice(unsorted, {0, 0, 0, 0}, {qk.size(0), qk.size(1), n, 4});
  out.buckets = nn::from_vector(h, {qk.size(0), qk.size(1), padded});
  out.order   = nn::from_vector(order, {qk.size(0), qk.size(1), padded});
  return out;
}
struct Path {
  Tensor              logits, patches, embedded, hidden;
  std::vector<Tensor> states, queries, values, attention, buckets, order;
};
inline Path Forward(
    nn::ParameterStore& p,
    nn::ParameterStore& projection_state,
    const Tensor&       images,
    const A&            routing = {}) {
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
    auto split = nn::split(linear(x, name + "/qkv", 2 * D), 2, 2);
    auto heads = [&](const Tensor& z) {
      return nn::transpose(nn::reshape(z, {n, T, 2, 4}), {0, 2, 1, 3});
    };
    auto q = heads(split[0]), v = heads(split[1]);
    auto rotation = projection_state.get(
        name + "/rotation", {4, 2}, [l](const nn::Shape&, nn::DType, nn::Generator&) {
          return Rotation(4, 2, 51111 + l);
        });
    auto suffix  = std::to_string(l);
    auto favored = Context(
        q,
        v,
        rotation,
        4,
        4,
        routing.count("buckets" + suffix) ? routing.at("buckets" + suffix) : Tensor{},
        routing.count("order" + suffix) ? routing.at("order" + suffix) : Tensor{});
    out.queries.push_back(q);
    out.values.push_back(v);
    out.attention.push_back(favored.attention);
    out.buckets.push_back(favored.buckets);
    out.order.push_back(favored.order);
    auto mixed = nn::reshape(nn::transpose(favored.context, {0, 2, 1, 3}), {n, T, D});
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
  nn::ParameterStore             parameters{51101}, projection_state{51103};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(parameters, projection_state, nn::zeros({1, 32, 32, 3}));
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
      if (!parameters.contains(k))
        continue;
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
    auto                path = Forward(parameters, projection_state, x, overrides);
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
           {"state0", path.states[0]},
           {"state1", path.states[1]},
           {"loss", loss},
           {"input_gradient", g[0]}};
    for (int i = 0; i < 2; ++i) {
      auto suffix               = std::to_string(i);
      out["qk" + suffix]        = path.queries[i];
      out["v" + suffix]         = path.values[i];
      out["attention" + suffix] = path.attention[i];
      out["buckets" + suffix]   = path.buckets[i];
      out["order" + suffix]     = path.order[i];
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
    return Forward(parameters, projection_state, b.at("images")).logits;
  }
  void Update(const A& b) {
    Validate(b);
    auto path = Forward(parameters, projection_state, b.at("images"));
    optimizer->minimize(Loss(path.logits, b.at("labels")));
  }
  A Arrays() const {
    auto out = replay::Arrays(parameters);
    for (const auto& [k, v] : replay::Arrays(projection_state))
      check::Require(out.emplace(k, v).second, "distinct projection-state keys");
    return out;
  }
  void Save(const std::filesystem::path& file) const {
    nn::save_npz(file.string(), replay::ToNamed(Arrays()));
  }
  void Load(const std::filesystem::path& file) {
    auto arrays = nn::load_npz(file.string());
    check::Require(
        arrays.size() == parameters.size() + projection_state.size(),
        "all learned/fixed Reformer tensors");
    for (const auto& [k, v] : arrays) {
      check::Require(
          parameters.contains(k) || projection_state.contains(k), "known Reformer checkpoint key");
      (parameters.contains(k) ? parameters : projection_state).set(k, v);
    }
  }
};
} // namespace reformer
