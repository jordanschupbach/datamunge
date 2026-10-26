// FAVOR+ with fixed positive orthogonal random features.
// Port of tflow's common/{nn,ml}/performer.h.
#pragma once
#include <random>

#include "common/vision_transformer.hpp"
namespace performer {
namespace nn = datamunge::nn;
using nn::Tensor;
using vit::A;
using vit::Loss;
inline Tensor Projection(int rows, int dimensions, unsigned seed) {
  std::vector<float>               result(rows * dimensions);
  std::mt19937                     rng(seed);
  std::normal_distribution<double> normal(0., 1.);
  for (int start = 0; start < rows; start += dimensions) {
    std::vector<std::vector<double>> basis;
    for (int i = 0; i < std::min(dimensions, rows - start); ++i) {
      std::vector<double> v(dimensions);
      for (auto& x : v)
        x = normal(rng);
      for (const auto& old : basis) {
        double dot = 0;
        for (int j = 0; j < dimensions; ++j)
          dot += v[j] * old[j];
        for (int j = 0; j < dimensions; ++j)
          v[j] -= dot * old[j];
      }
      double norm = 0;
      for (auto x : v)
        norm += x * x;
      norm = std::sqrt(norm);
      check::Require(norm > 1e-10, "nondegenerate orthogonal random basis");
      for (auto& x : v)
        x /= norm;
      basis.push_back(v);
      double radius = 0;
      for (int j = 0; j < dimensions; ++j) {
        double x = normal(rng);
        radius += x * x;
      }
      radius = std::sqrt(radius);
      for (int j = 0; j < dimensions; ++j)
        result[(start + i) * dimensions + j] = float(v[j] * radius);
    }
  }
  return nn::from_vector(result, {rows, dimensions});
}
inline Tensor Features(const Tensor& x, const Tensor& omega, bool query, int dimensions = 4) {
  auto normalized = x * float(std::pow(double(dimensions), -.25));
  auto projected  = nn::matmul(normalized, nn::transpose(omega, {1, 0}));
  auto logphi     = projected - nn::sum(nn::square(normalized), {3}, true) * .5;
  auto shift      = nn::stop_gradient(nn::max(logphi, query ? nn::Axes{3} : nn::Axes{2, 3}, true));
  return nn::exp(logphi - shift) * (1. / std::sqrt(double(omega.size(0))));
}
struct ContextPath {
  Tensor context, query_features, key_features, summary, denominator;
};
inline ContextPath Context(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& omega,
    int           dimensions = 4) {
  ContextPath out;
  out.query_features = Features(q, omega, true, dimensions);
  out.key_features   = Features(k, omega, false, dimensions);
  out.summary        = nn::matmul(out.key_features, v, true, false);
  out.denominator =
      nn::matmul(out.query_features, nn::expand_dims(nn::sum(out.key_features, {2}), 3));
  out.context =
      nn::matmul(out.query_features, out.summary) / nn::maximum(out.denominator, nn::scalar(1e-12));
  return out;
}
struct Path {
  Tensor              logits, patches, embedded, hidden;
  std::vector<Tensor> states, queries, keys, values, query_features, key_features, summaries,
      denominators;
};
inline Path
Forward(nn::ParameterStore& p, nn::ParameterStore& projection_state, const Tensor& images) {
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
    auto omega = projection_state.get(
        name + "/omega", {16, 4}, [l](const nn::Shape&, nn::DType, nn::Generator&) {
          return Projection(16, 4, 50501 + l);
        });
    auto favored = Context(q, k, v, omega);
    out.queries.push_back(q);
    out.keys.push_back(k);
    out.values.push_back(v);
    out.query_features.push_back(favored.query_features);
    out.key_features.push_back(favored.key_features);
    out.summaries.push_back(favored.summary);
    out.denominators.push_back(favored.denominator);
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
  nn::ParameterStore             parameters{50501}, projection_state{50503};
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
    auto                path = Forward(parameters, projection_state, x);
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
      auto suffix                    = std::to_string(i);
      out["q" + suffix]              = path.queries[i];
      out["k" + suffix]              = path.keys[i];
      out["v" + suffix]              = path.values[i];
      out["query_features" + suffix] = path.query_features[i];
      out["key_features" + suffix]   = path.key_features[i];
      out["summary" + suffix]        = path.summaries[i];
      out["denominator" + suffix]    = path.denominators[i];
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
        "all learned/fixed Performer tensors");
    for (const auto& [k, v] : arrays) {
      check::Require(
          parameters.contains(k) || projection_state.contains(k), "known Performer checkpoint key");
      (parameters.contains(k) ? parameters : projection_state).set(k, v);
    }
  }
};
} // namespace performer
