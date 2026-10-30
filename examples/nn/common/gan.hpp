// GAN support for the 3.1 ports: a port of tflow's common/nn/{vanilla_gan,
// dcgan,wasserstein_gan,kernel_discrepancy}.h, common/ml/{dcgan,
// wasserstein_gan}.h and the parts of common/ml/{saliency,autoencoder}.h and
// common/nn/smoothgrad.h they use.
//
// TF's graph/session models become classes over ParameterStores: G and D
// (or critic) are separate stores with their own optimizers, and the DCGAN
// batch-norm moving statistics live in two more stores (g_state, d_state).
// Checkpoints are .npz files with tflow's array names ("0/<G name>",
// "1/<D name>", "g_state/...", "d_state/..."), so a TF initial_arrays.npz can
// be loaded with init=PATH to follow the TF trajectory.
//
// Random streams that are plain std::mt19937 in tflow (GaussianNoise, the
// WGAN alpha draws) are reproduced exactly; weight initialisation uses
// datamunge's generator unless TF arrays are loaded.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace gan {

namespace nn = datamunge::nn;
using nn::Tensor;
using Arrays = std::map<std::string, Tensor>;

// ---- data and noise ------------------------------------------------------------------

/// tflow nn::GaussianNoise: Box-Muller pairs from std::mt19937(seed), float64 [count].
inline std::vector<double> GaussianNoise(std::int64_t count, std::uint32_t seed) {
  std::vector<double> bank(static_cast<std::size_t>(count));
  std::mt19937        rng(seed);
  constexpr double    tau = 6.283185307179586476925286766559;
  for (std::int64_t i = 0; i < count; i += 2) {
    const double u = (double(rng()) + .5) / 4294967296., v = (double(rng()) + .5) / 4294967296.;
    const double radius = std::sqrt(-2 * std::log(u));
    bank[i] = radius * std::cos(tau * v);
    if (i + 1 < count) bank[i + 1] = radius * std::sin(tau * v);
  }
  return bank;
}

/// Float32 [n, d] standard normal rows (tflow GanGaussian / the WGAN Gaussian).
inline Tensor Gaussian(std::int64_t n, std::int64_t d, std::uint32_t seed) {
  const auto       bank = GaussianNoise(n * d, seed);
  std::vector<float> values(bank.begin(), bank.end());
  return nn::from_vector(values, {n, d});
}

/// One saliency/autoencoder MNIST pool: images [n, 28, 28, 1] in [0, 1].
struct DigitPool {
  Tensor images, labels, source;
};

/// tflow ml::LoadSaliency: fit (1000), validation (200), test (200) balanced
/// MNIST pools from saliency_data.npz.
inline std::vector<DigitPool> LoadSaliency(const std::filesystem::path& root) {
  const auto file = root / "saliency_data.npz";
  dataset_assets::EnsureSaliency(root.string());
  Arrays a;
  for (auto& [name, t] : nn::load_npz(file.string())) a.emplace(name, t);
  std::vector<DigitPool> result;
  std::set<std::int64_t> ids;
  for (const std::string split : {"fit", "validation", "test"}) {
    const Tensor       x = a.at(split + "_images");
    const std::int64_t n = split == "fit" ? 1000 : 200;
    check::Require(x.dtype() == nn::DType::Int32 && x.shape() == nn::Shape({n, 28, 28, 1}),
                   "source 28x28 grayscale digit images");
    DigitPool d{nn::cast(x, nn::DType::Float32) / 255.0, a.at(split + "_labels"), a.at(split + "_source_ids")};
    check::Require(d.labels.shape() == nn::Shape({n}) && d.source.shape() == nn::Shape({n}),
                   "aligned source/label vectors");
    int        counts[10] = {};
    const auto labels = nn::cast(d.labels, nn::DType::Int64).to_vector<std::int64_t>();
    const auto source = nn::cast(d.source, nn::DType::Int64).to_vector<std::int64_t>();
    for (std::int64_t b = 0; b < n; ++b) {
      check::Require(labels[b] >= 0 && labels[b] < 10 && ids.insert(source[b]).second, "source IDs disjoint, labels valid");
      check::Require(split == "test" ? source[b] >= 60000 && source[b] < 70000 : source[b] >= 0 && source[b] < 60000,
                     "original test/train boundary");
      ++counts[labels[b]];
    }
    for (int c : counts) check::Require(c == n / 10, "balanced digit pools");
    result.push_back(d);
  }
  return result;
}

inline Tensor Rows(const Tensor& x, const std::vector<std::int64_t>& rows) { return nn::take_rows(x, rows); }
inline std::vector<std::int64_t> Range(std::int64_t n) {
  std::vector<std::int64_t> r(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) r[i] = i;
  return r;
}

// ---- objectives and diagnostics --------------------------------------------------------

struct LogisticGanLoss {
  Tensor discriminator, generator;
};
/// D: mean softplus(-real) + mean softplus(fake); non-saturating G: mean softplus(-fake).
inline LogisticGanLoss LogisticGanObjectives(const Tensor& real, const Tensor& fake) {
  return {nn::mean(nn::softplus(-real)) + nn::mean(nn::softplus(fake)), nn::mean(nn::softplus(-fake))};
}

/// 4x4 averages of 7x7 pixel blocks divided by 4: [n, 784] -> [n, 16].
inline Tensor GanCoarsePixels(const Tensor& pixels) {
  return nn::reshape(nn::avg_pool2d(nn::reshape(pixels, {-1, 28, 28, 1}), 7, 7), {-1, 16}) / 4.0;
}

/// Mean of inverse multiquadric kernels with c = 1, 4, 16.
inline Tensor ImqKernel(const Tensor& a, const Tensor& b) {
  const Tensor d = nn::sum(nn::square(nn::expand_dims(a, 1) - nn::expand_dims(b, 0)), {2});
  return (1.0 / (d + 1.0) + 4.0 / (d + 4.0) + 16.0 / (d + 16.0)) / 3.0;
}

/// Unbiased two-sample IMQ MMD^2 (diagonals excluded within each sample).
inline Tensor ImqMmd(const Tensor& x, const Tensor& y) {
  const double n = double(x.size(0)), m = double(y.size(0));
  check::Require(n >= 2 && m >= 2, "pixel MMD diagnostic needs >= 2 cases EACH");
  const Tensor within_x = (nn::sum(ImqKernel(x, x)) - n) / (n * (n - 1));
  const Tensor within_y = (nn::sum(ImqKernel(y, y)) - m) / (m * (m - 1));
  return within_x + within_y - 2.0 * nn::mean(ImqKernel(x, y));
}

// ---- parameter groups ------------------------------------------------------------------

/// Copies of every array of `stores`, keyed "<prefix><name>".
inline Arrays Snapshot(const std::vector<std::pair<std::string, const nn::ParameterStore*>>& stores) {
  Arrays out;
  for (const auto& [prefix, store] : stores)
    for (const auto& name : store->names()) out.emplace(prefix + name, store->at(name).detach().clone());
  return out;
}

inline bool BitEqual(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape()) return false;
  const auto x = a.contiguous().to_vector<float>(), y = b.contiguous().to_vector<float>();
  return std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}
inline bool BitEqual(const Arrays& a, const Arrays& b) {
  if (a.size() != b.size()) return false;
  for (const auto& [k, v] : a)
    if (!b.count(k) || !BitEqual(v, b.at(k))) return false;
  return true;
}

/// Overwrites existing parameters from an .npz written by tflow (or by Save):
/// every array must map to an existing parameter of the same shape.
inline void LoadArrays(const std::string& path, const std::vector<std::pair<std::string, nn::ParameterStore*>>& stores) {
  std::size_t loaded = 0, expected = 0;
  for (const auto& s : stores) expected += s.second->size();
  for (const auto& [key, value] : nn::load_npz(path)) {
    bool found = false;
    for (const auto& [prefix, store] : stores)
      if (key.rfind(prefix, 0) == 0 && store->contains(key.substr(prefix.size()))) {
        const auto name = key.substr(prefix.size());
        check::Require(store->at(name).shape() == value.shape(), "array shape matches parameter " + key);
        store->set(name, nn::cast(value, store->dtype()));
        found = true;
        ++loaded;
        break;
      }
    check::Require(found, "unknown array " + key + " in " + path);
  }
  check::Require(loaded == expected, "every parameter restored from " + path);
}

inline void SaveArrays(const std::string& path, const Arrays& arrays) {
  nn::NamedTensors named(arrays.begin(), arrays.end());
  nn::save_npz(path, named);
}

inline void RequireFinite(const Arrays& fields, const std::string& message) {
  for (const auto& [k, v] : fields)
    for (float x : nn::cast(v, nn::DType::Float32).contiguous().to_vector<float>())
      check::Require(std::isfinite(x), message + " (" + k + ")");
}

// ---- DCGAN --------------------------------------------------------------------------------

/// How a DCGAN batch norm runs: batch statistics with or without the
/// moving-average update, or the moving statistics (inference).
enum class BnMode { Train, Probe, Infer };

/// tflow DcganBn: momentum .9, eps .001; records batch moments when not inferring.
inline Tensor DcganBn(nn::ParameterStore& p, nn::ParameterStore& stats, const std::string& name, const Tensor& x,
                      BnMode mode, Arrays* moments) {
  const std::int64_t c     = x.shape().back();
  const Tensor       gamma = p.get(name + "/gamma", {c}, nn::init::ones());
  const Tensor       beta  = p.get(name + "/beta", {c}, nn::init::zeros());
  const Tensor       mu    = stats.get(name + "/moving_mean", {c}, nn::init::zeros());
  const Tensor       var   = stats.get(name + "/moving_var", {c}, nn::init::ones());
  if (mode == BnMode::Infer) return nn::batch_norm_infer(x, gamma, beta, mu.detach(), var.detach(), .001);
  const auto r = nn::batch_norm_train(x, gamma, beta, .001);
  if (moments) {
    moments->emplace(name + "/batch_mean", r.mean);
    moments->emplace(name + "/batch_var", r.variance);
  }
  if (mode == BnMode::Train) {
    nn::NoGradGuard guard;
    stats.set(name + "/moving_mean", mu.detach() * .9 + r.mean * (1 - .9));
    stats.set(name + "/moving_var", var.detach() * .9 + r.variance * (1 - .9));
  }
  return r.y;
}

/// G: z [n, 16] -> 7x7x8 projection/BN/ReLU -> 4x4 stride-2 transposed conv
/// 14x14x4/BN/ReLU -> 4x4 stride-2 transposed conv 28x28x1, tanh.
inline Tensor DcganGenerate(nn::ParameterStore& p, nn::ParameterStore& stats, const Tensor& z, BnMode mode,
                            Arrays* moments = nullptr) {
  const auto   normal     = nn::init::normal(.02);
  const Tensor projection = p.get("projection/W", {16, 392}, normal);
  Tensor       h          = nn::reshape(nn::matmul(z, projection), {-1, 7, 7, 8});
  h                       = nn::relu(DcganBn(p, stats, "stem_bn", h, mode, moments));
  nn::ConvLayerOptions up;
  up.stride = 2;
  up.bias   = false;
  p.get("up14/W", {4, 4, 4, 8}, normal);  // tflow creates the kernels N(0, .02) first
  h = nn::conv2d_transpose(p, "up14", h, 4, 4, up);
  h = nn::relu(DcganBn(p, stats, "up14_bn", h, mode, moments));
  up.bias       = true;
  up.activation = nn::Activation::Tanh;
  p.get("output/W", {4, 4, 1, 4}, normal);
  return nn::conv2d_transpose(p, "output", h, 1, 4, up);
}

/// D: 4x4 stride-2 conv 4/Leaky .2 -> 4x4 stride-2 conv 8/BN/Leaky .2 -> 392 -> 1 raw logit.
inline Tensor DcganDiscriminate(nn::ParameterStore& p, nn::ParameterStore& stats, const Tensor& image, BnMode mode,
                                Arrays* moments = nullptr) {
  const auto           normal = nn::init::normal(.02);
  nn::ConvLayerOptions down;
  down.stride = 2;
  p.get("down14/W", {4, 4, 1, 4}, normal);
  Tensor h  = nn::leaky_relu(nn::conv2d(p, "down14", image, 4, 4, down), .2);
  down.bias = false;
  p.get("down7/W", {4, 4, 4, 8}, normal);
  h              = nn::conv2d(p, "down7", h, 8, 4, down);
  h              = nn::leaky_relu(DcganBn(p, stats, "down7_bn", h, mode, moments), .2);
  const Tensor w = p.get("output/W", {392, 1}, normal), b = p.get("output/b", {1}, nn::init::zeros());
  return nn::bias_add(nn::matmul(nn::reshape(h, {-1, 392}), w), b);
}

/// tflow ml::DcganModel. Group 0 is G, group 1 is D. The D step normalizes
/// concat(real, fake) jointly and updates only D's moving statistics (G runs
/// with batch statistics, no update); the G step uses batch statistics in G
/// (updating G's moving statistics) and inference BN in D.
class DcganModel {
 public:
  nn::ParameterStore generator{5003}, discriminator{5011}, g_state{0}, d_state{0};
  std::vector<std::unique_ptr<nn::Optimizer>> optimizers;

  explicit DcganModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      DcganDiscriminate(discriminator, d_state, DcganGenerate(generator, g_state, nn::zeros({2, 16}), BnMode::Infer),
                        BnMode::Infer);
    }
    if (training)
      for (auto* group : {&generator, &discriminator}) {
        nn::OptimizerConfig c;
        c.kind  = nn::OptimizerKind::Adam;
        c.beta1 = .5;
        optimizers.push_back(std::make_unique<nn::Optimizer>(*group, .0002, c));
      }
  }

  nn::ParameterStore& Group(int k) { return k == 0 ? generator : discriminator; }

  static Tensor Signed(const Tensor& x) { return nn::reshape(x * 2.0 - 1.0, {-1, 28, 28, 1}); }

  /// Phase objective; `update` applies that phase's own moving-statistics update.
  Tensor Objective(int group, const Tensor& x, const Tensor& z, bool update, Arrays* fields = nullptr) {
    check::Require(x.size(0) == z.size(0), "joint real/fake BN requires matching batch sizes");
    Arrays  g_moments, d_moments;
    Arrays* g_m = fields ? &g_moments : nullptr;
    Arrays* d_m = fields ? &d_moments : nullptr;
    Tensor result;
    if (group == 1) {
      Tensor fake;
      {
        nn::NoGradGuard guard;
        fake = DcganGenerate(generator, g_state, z, BnMode::Probe, g_m);
      }
      const Tensor logits = DcganDiscriminate(discriminator, d_state, nn::concat({Signed(x), fake}, 0),
                                              update ? BnMode::Train : BnMode::Probe, d_m);
      const auto   n      = x.size(0);
      result = LogisticGanObjectives(nn::slice_axis(logits, 0, 0, n), nn::slice_axis(logits, 0, n, 2 * n)).discriminator;
      if (fields) fields->emplace("d_train_logits", logits.detach());
    } else {
      const Tensor fake  = DcganGenerate(generator, g_state, z, update ? BnMode::Train : BnMode::Probe, g_m);
      const Tensor logit = DcganDiscriminate(discriminator, d_state, fake, BnMode::Infer);
      result             = nn::mean(nn::softplus(-logit));
      if (fields) fields->emplace("d_for_g_logits", logit.detach()), fields->emplace("g_train_signed", fake.detach());
    }
    if (fields) {
      for (auto& [k, v] : g_moments) fields->emplace("g/" + k, v);
      for (auto& [k, v] : d_moments) fields->emplace("d/" + k, v);
    }
    return result;
  }

  /// Training-mode forward of both phases without any update.
  Arrays TrainingEvaluate(const Tensor& x, const Tensor& z) {
    nn::NoGradGuard guard;
    Arrays          fields;
    fields.emplace("discriminator_loss", Objective(1, x, z, false, &fields));
    fields.emplace("generator_loss", Objective(0, x, z, false, &fields));  // same G batch moments
    RequireFinite(fields, "all DCGAN outputs finite");
    return fields;
  }

  Arrays Gradient(int group, const Tensor& x, const Tensor& z) {
    auto&      store = Group(group);
    const auto g     = nn::grad(Objective(group, x, z, false), store.parameters());
    Arrays     out;
    for (std::size_t i = 0; i < g.size(); ++i) out.emplace(std::to_string(group) + "/" + store.names()[i], g[i]);
    return out;
  }

  void Update(int group, const Tensor& x, const Tensor& z) {
    const Tensor objective = Objective(group, x, z, true);
    optimizers.at(group)->step(nn::grad(objective, Group(group).parameters()));
  }

  /// Inference-BN generation in [0, 1], [n, 784].
  Tensor Generate(const Tensor& z) {
    nn::NoGradGuard guard;
    return nn::reshape((DcganGenerate(generator, g_state, z, BnMode::Infer) + 1.0) * .5, {-1, 784});
  }

  Arrays Evaluate(const Tensor& x, const Tensor& z) {
    check::Require(x.size(0) == z.size(0), "joint real/fake BN requires matching batch sizes");
    nn::NoGradGuard guard;
    const Tensor    image = DcganGenerate(generator, g_state, z, BnMode::Infer);
    const Tensor    fake  = nn::reshape((image + 1.0) * .5, {-1, 784});
    const Tensor    logits =
        DcganDiscriminate(discriminator, d_state, nn::concat({Signed(x), image}, 0), BnMode::Infer);
    const auto   n = x.size(0);
    const Tensor real_logit = nn::slice_axis(logits, 0, 0, n), fake_logit = nn::slice_axis(logits, 0, n, 2 * n);
    const auto   loss       = LogisticGanObjectives(real_logit, fake_logit);
    Arrays       out{{"fake", fake},
                     {"real_logit", real_logit},
                     {"fake_logit", fake_logit},
                     {"real_probability", nn::sigmoid(real_logit)},
                     {"fake_probability", nn::sigmoid(fake_logit)},
                     {"discriminator_loss", loss.discriminator},
                     {"generator_loss", loss.generator},
                     {"pixel_mmd", ImqMmd(GanCoarsePixels(nn::reshape(x, {-1, 784})), GanCoarsePixels(fake))}};
    RequireFinite(out, "all DCGAN outputs finite");
    return out;
  }

  std::vector<std::pair<std::string, nn::ParameterStore*>> Stores() {
    return {{"0/", &generator}, {"1/", &discriminator}, {"g_state/", &g_state}, {"d_state/", &d_state}};
  }
  Arrays AllArrays() {
    std::vector<std::pair<std::string, const nn::ParameterStore*>> s;
    for (auto& [k, v] : Stores()) s.emplace_back(k, v);
    return Snapshot(s);
  }
  void Save(const std::string& path) { SaveArrays(path, AllArrays()); }
  void Load(const std::string& path) { LoadArrays(path, Stores()); }
};

// ---- WGAN ---------------------------------------------------------------------------------

/// tflow ml::WassersteinGanModel on 2-D points. G 2 -> 16 tanh -> 2 linear,
/// critic 2 -> 16 tanh -> 1 raw score. With `gp` the critic loss adds
/// 10 mean(||d critic / d xhat|| - 1)^2 at xhat = alpha real + (1 - alpha)
/// stop_gradient(fake), differentiated through the critic parameters
/// (create_graph); otherwise RMSProp and clipping of every critic parameter
/// to [-.01, .01] after each critic step (and once at initialisation).
class WassersteinGanModel {
 public:
  bool                                        gp;
  nn::ParameterStore                          generator{5501}, critic{5503};
  std::vector<std::unique_ptr<nn::Optimizer>> optimizers;

  static Tensor Generate(nn::ParameterStore& p, const Tensor& z) {
    return nn::dense(p, "output", nn::dense(p, "hidden", z, 16, nn::Activation::Tanh), 2);
  }
  static Tensor Critic(nn::ParameterStore& p, const Tensor& x) {
    return nn::dense(p, "output", nn::dense(p, "hidden", x, 16, nn::Activation::Tanh), 1);
  }

  explicit WassersteinGanModel(bool gradient_penalty, bool training = true) : gp(gradient_penalty) {
    {
      nn::NoGradGuard guard;
      Critic(critic, Generate(generator, nn::zeros({1, 2})));
    }
    if (training)
      for (auto* group : {&generator, &critic}) {
        nn::OptimizerConfig c;
        c.kind  = gp ? nn::OptimizerKind::Adam : nn::OptimizerKind::RMSProp;
        c.beta1 = 0;
        c.beta2 = .9;
        c.rho   = .9;
        optimizers.push_back(std::make_unique<nn::Optimizer>(*group, gp ? .0001 : .00005, c));
      }
    if (!gp) Clip();
  }

  nn::ParameterStore& Group(int k) { return k == 0 ? generator : critic; }

  void Clip() {
    nn::NoGradGuard guard;
    for (const auto& name : critic.names()) critic.set(name, nn::clip(critic.at(name).detach(), -.01, .01));
  }

  /// Every field of tflow's Evaluate; the losses keep their graph when grad mode is on.
  Arrays Forward(const Tensor& x, const Tensor& z, const Tensor& alpha) {
    const Tensor fake = Generate(generator, z);
    const Tensor real_score = Critic(critic, x), fake_score = Critic(critic, fake);
    // xhat is a leaf: alpha and x are data and fake is stopped, as in tflow.
    Tensor interpolation = (alpha * x + (1.0 - alpha) * nn::stop_gradient(fake)).detach();
    interpolation.requires_grad_();
    Tensor symbolic;
    {
      nn::EnableGradGuard enable;  // the input gradient is needed even in evaluation
      nn::GradOptions     options;
      options.create_graph = training_graph_;
      symbolic = nn::grad(nn::sum(Critic(critic, interpolation)), {interpolation}, options)[0];
    }
    if (!training_graph_) symbolic = symbolic.detach();
    // Hand-derived Jacobian: (1 - tanh^2(xhat W1 + b1)) * w2^T, times W1^T.
    const Tensor hidden = nn::dense(critic, "hidden", interpolation.detach(), 16, nn::Activation::Tanh);
    const Tensor slope  = (1.0 - nn::square(hidden)) * nn::reshape(critic.at("output/W"), {1, 16});
    const Tensor input_gradient = nn::matmul(slope, critic.at("hidden/W"), false, true);
    const Tensor norm    = nn::sqrt(nn::sum(nn::square(symbolic), {1}) + 1e-12);
    const Tensor penalty = nn::mean(nn::square(norm - 1.0));
    const Tensor gap     = nn::mean(real_score) - nn::mean(fake_score);
    Tensor       critic_loss = -gap;
    if (gp) critic_loss = critic_loss + 10.0 * penalty;
    return {{"fake", fake},
            {"real_score", real_score},
            {"fake_score", fake_score},
            {"interpolation", interpolation.detach()},
            {"input_gradient", input_gradient},
            {"symbolic_input_gradient", symbolic},
            {"norm", norm},
            {"penalty", penalty},
            {"critic_gap", gap},
            {"generator_loss", -nn::mean(fake_score)},
            {"critic_loss", critic_loss}};
  }

  Arrays Evaluate(const Tensor& x, const Tensor& z, const Tensor& alpha) {
    nn::NoGradGuard guard;
    Arrays          out;
    for (auto& [k, v] : Forward(x, z, alpha)) out.emplace(k, v.detach());
    RequireFinite(out, "finite WGAN fields");
    return out;
  }

  Arrays Gradient(int group, const Tensor& x, const Tensor& z, const Tensor& alpha) {
    training_graph_ = true;
    auto        fields = Forward(x, z, alpha);
    training_graph_    = false;
    auto&       store  = Group(group);
    const auto  g = nn::grad(fields.at(group == 0 ? "generator_loss" : "critic_loss"), store.parameters());
    Arrays      out;
    for (std::size_t i = 0; i < g.size(); ++i) out.emplace(std::to_string(group) + "/" + store.names()[i], g[i]);
    return out;
  }

  void Update(int group, const Tensor& x, const Tensor& z, const Tensor& alpha) {
    const auto g = Gradient(group, x, z, alpha);
    std::vector<Tensor> ordered;
    for (const auto& name : Group(group).names()) ordered.push_back(g.at(std::to_string(group) + "/" + name));
    optimizers.at(group)->step(ordered);
    if (!gp && group == 1) Clip();
  }

  Tensor Generate(const Tensor& z) {
    nn::NoGradGuard guard;
    return Generate(generator, z);
  }

  std::vector<std::pair<std::string, nn::ParameterStore*>> Stores() { return {{"0/", &generator}, {"1/", &critic}}; }
  Arrays AllArrays() { return Snapshot({{"0/", &generator}, {"1/", &critic}}); }
  void   Save(const std::string& path) { SaveArrays(path, AllArrays()); }
  void   Load(const std::string& path) { LoadArrays(path, Stores()); }

 private:
  bool training_graph_{false};
};

// ---- shared pieces of the compact MNIST GANs (job 3.1a: lsgan, hinge_gan,
// conditional_gan, auxiliary_classifier_gan, infogan, ...) -------------------------------

/// tflow LeastSquaresGanObjectives (a = 0, b = c = 1): D .5 mean (real - 1)^2 +
/// .5 mean fake^2; G .5 mean (fake - 1)^2. Raw unbounded scores.
inline LogisticGanLoss LeastSquaresGanObjectives(const Tensor& real, const Tensor& fake) {
  return {.5 * (nn::mean(nn::square(real - 1.0)) + nn::mean(nn::square(fake))), .5 * nn::mean(nn::square(fake - 1.0))};
}

/// tflow HingeGanObjectives: D mean relu(1 - real) + mean relu(1 + fake); G -mean fake
/// (ReLU subgradient 0 at the margin, as TF).
inline LogisticGanLoss HingeGanObjectives(const Tensor& real, const Tensor& fake) {
  return {nn::mean(nn::relu(1.0 - real)) + nn::mean(nn::relu(1.0 + fake)), -nn::mean(fake)};
}

/// [n, 28, 28, 1] (or any [n, ...784]) -> [n, 784] (tflow UndercompleteModel::Flatten).
inline Tensor Flat784(const Tensor& x) { return nn::reshape(x, {-1, 784}); }

/// tflow MlpGanGenerate: z [n, 16] -> 32 tanh -> 784 sigmoid.
inline Tensor MlpGanGenerate(nn::ParameterStore& p, const Tensor& z) {
  return nn::dense(p, "output", nn::dense(p, "hidden", z, 32, nn::Activation::Tanh), 784, nn::Activation::Sigmoid);
}
/// tflow MlpGanDiscriminate: image [n, 784] -> 32 tanh -> 1 raw score.
inline Tensor MlpGanDiscriminate(nn::ParameterStore& p, const Tensor& x) {
  return nn::dense(p, "output", nn::dense(p, "hidden", x, 32, nn::Activation::Tanh), 1);
}
/// One-hot float32 [n, 10] of integer class IDs.
inline Tensor OneHot10(const Tensor& labels) { return nn::one_hot(nn::cast(labels, nn::DType::Int64), 10); }
/// tflow ConditionalGanGenerate: concat(z, onehot) [n, 26] -> 32 tanh -> 784 sigmoid.
inline Tensor ConditionalGanGenerate(nn::ParameterStore& p, const Tensor& z, const Tensor& labels) {
  return MlpGanGenerate(p, nn::concat({z, OneHot10(labels)}, 1));
}
/// tflow ConditionalGanDiscriminate: concat(image, onehot) [n, 794] -> 32 tanh -> 1 logit.
inline Tensor ConditionalGanDiscriminate(nn::ParameterStore& p, const Tensor& image, const Tensor& labels) {
  return MlpGanDiscriminate(p, nn::concat({image, OneHot10(labels)}, 1));
}

/// tflow ConditionalGanModel::ValidateCondition: one int32 class 0..9 per 16-D noise row.
inline void ValidateCondition(const Tensor& z, const Tensor& labels, std::int64_t width = 16) {
  check::Require(z.dtype() == nn::DType::Float32 && z.dim() == 2 && z.size(1) == width &&
                     labels.dtype() == nn::DType::Int32 && labels.shape() == nn::Shape({z.size(0)}),
                 "one valid integer class per " + std::to_string(width) + "-D noise vector");
  for (auto c : labels.contiguous().to_vector<std::int32_t>()) check::Require(c >= 0 && c < 10, "class ID 0..9 before OneHot");
}

/// Separate parameter groups ("<k>/<name>" arrays) with one optimizer each, as
/// tflow's GAN models keep them: gradients of a group's objective w.r.t. that
/// group only, explicit Apply of (possibly pre-captured) gradients.
class GroupedGan {
 public:
  std::vector<std::unique_ptr<nn::ParameterStore>> groups;
  std::vector<std::unique_ptr<nn::Optimizer>>      optimizers;

  explicit GroupedGan(const std::vector<std::uint64_t>& seeds) {
    for (auto s : seeds) groups.push_back(std::make_unique<nn::ParameterStore>(s));
  }
  virtual ~GroupedGan() = default;
  nn::ParameterStore& Group(int k) { return *groups.at(k); }

  /// One Adam per group in `trainable` (default: all groups).
  void MakeAdam(double lr, double beta1, double beta2 = .999, std::vector<int> trainable = {}) {
    if (trainable.empty())
      for (int k = 0; k < int(groups.size()); ++k) trainable.push_back(k);
    optimizers.resize(groups.size());
    for (int k : trainable) {
      nn::OptimizerConfig c;
      c.kind       = nn::OptimizerKind::Adam;
      c.beta1      = beta1;
      c.beta2      = beta2;
      optimizers[k] = std::make_unique<nn::Optimizer>(Group(k), lr, c);
    }
  }

  /// d objective / d group k, keyed "<k>/<name>" (unused parameters get zeros).
  Arrays GradientOf(int k, const Tensor& objective) {
    auto&           store = Group(k);
    nn::GradOptions o;
    o.allow_unused = true;
    const auto g   = nn::grad(objective, store.parameters(), o);
    Arrays     out;
    for (std::size_t i = 0; i < g.size(); ++i)
      out.emplace(std::to_string(k) + "/" + store.names()[i],
                  g[i].defined() ? g[i] : nn::zeros(store.at(store.names()[i]).shape(), store.dtype()));
    return out;
  }
  /// One optimizer step of group k from gradients keyed "<k>/<name>".
  void Apply(int k, const Arrays& gradients) {
    std::vector<Tensor> ordered;
    for (const auto& name : Group(k).names()) ordered.push_back(gradients.at(std::to_string(k) + "/" + name));
    optimizers.at(k)->step(ordered);
  }

  std::vector<std::pair<std::string, nn::ParameterStore*>> Stores() {
    std::vector<std::pair<std::string, nn::ParameterStore*>> s;
    for (int k = 0; k < int(groups.size()); ++k) s.emplace_back(std::to_string(k) + "/", groups[k].get());
    return s;
  }
  Arrays AllArrays() {
    std::vector<std::pair<std::string, const nn::ParameterStore*>> s;
    for (auto& [k, v] : Stores()) s.emplace_back(k, v);
    return Snapshot(s);
  }
  void Save(const std::string& path) { SaveArrays(path, AllArrays()); }
  void Load(const std::string& path) { LoadArrays(path, Stores()); }
  std::int64_t Count(int k) { return Group(k).count(); }
};

/// tflow's Mosaic: 28x28 digits [n, 784] in [0, 1] at 3x with 4-pixel gray-220 gaps.
inline void Mosaic(const std::filesystem::path& path, const Tensor& images, std::int64_t columns) {
  nn::save_image_grid(nn::reshape(images, {-1, 28, 28, 1}), path.string(), columns, 0, 1, 4, 220 / 255., 3);
}

/// The fit-pool mean image repeated n times, [n, 784] (tflow AutoencoderConstant(AutoencoderMean)).
inline Tensor FitMean(const Tensor& fit, std::int64_t n) {
  return nn::tile(nn::mean(Flat784(fit), {0}, true), {n, 1});
}

/// Coarse-pixel MMD between real images and [n, 784] samples.
inline double CoarseMmd(const Tensor& real, const Tensor& fake) {
  return ImqMmd(GanCoarsePixels(Flat784(real)), GanCoarsePixels(Flat784(fake))).item();
}

/// Mean generated pixel and mean per-pixel variance across samples of [n, d].
inline std::pair<double, double> PixelMoments(const Tensor& x) {
  const Tensor mean = nn::mean(nn::cast(x, nn::DType::Float64), {0}, true);
  return {nn::mean(mean).item(), nn::mean(nn::square(nn::cast(x, nn::DType::Float64) - mean)).item()};
}

/// Max |a - b| over all elements.
inline double MaxAbsDiff(const Tensor& a, const Tensor& b) {
  return nn::max(nn::abs(nn::cast(a, nn::DType::Float64) - nn::cast(b, nn::DType::Float64))).item();
}

/// Checks that only arrays with one of `owners` prefixes changed between two
/// snapshots, and (if `require_change`) that at least one of them did.
inline void RequireOnlyOwnersChanged(const Arrays& before, const Arrays& after, const std::vector<std::string>& owners,
                                     const std::string& message, bool require_change = true) {
  bool changed = false;
  for (const auto& [key, value] : before) {
    bool own = false;
    for (const auto& o : owners) own |= key.rfind(o, 0) == 0;
    if (own)
      changed |= !BitEqual(value, after.at(key));
    else
      check::Require(BitEqual(value, after.at(key)), message);
  }
  if (require_change) check::Require(changed, "intended GAN group updates");
}

enum class GanObjective { Logistic, LeastSquares, Hinge };

/// tflow ml::{Vanilla,LeastSquares,Hinge,Conditional}GanModel: the compact MLP
/// GAN (G 16 [+ one-hot 10] -> 32 tanh -> 784 sigmoid, D 784 [+ one-hot 10] ->
/// 32 tanh -> 1) with group 0 = G, group 1 = D and separate Adam (.0002, beta1
/// .5). Logistic models expose real/fake_logit and probabilities, the others
/// raw real/fake_score. `labels` are ignored unless `conditional`.
class MlpGanModel : public GroupedGan {
 public:
  GanObjective objective;
  bool         conditional;

  MlpGanModel(GanObjective kind, bool conditional_model, std::uint64_t g_seed, std::uint64_t d_seed,
              bool training = true)
      : GroupedGan({g_seed, d_seed}), objective(kind), conditional(conditional_model) {
    {
      nn::NoGradGuard guard;
      const Tensor    labels = nn::zeros({2}, nn::DType::Int32);
      Discriminate(Generate(nn::zeros({2, 16}), labels), labels);
    }
    if (training) MakeAdam(.0002, .5);
  }

  Tensor Generate(const Tensor& z, const Tensor& labels) {
    return conditional ? ConditionalGanGenerate(Group(0), z, labels) : MlpGanGenerate(Group(0), z);
  }
  Tensor Discriminate(const Tensor& image, const Tensor& labels) {
    return conditional ? ConditionalGanDiscriminate(Group(1), image, labels) : MlpGanDiscriminate(Group(1), image);
  }

  /// Every Evaluate field; the losses keep their graph when grad mode is on.
  Arrays Forward(const Tensor& x, const Tensor& z, const Tensor& labels) {
    if (conditional) {
      ValidateCondition(z, labels);
      check::Require(x.size(0) == z.size(0), "matching real/fake conditional batch sizes");
    }
    const Tensor fake = Generate(z, labels);
    const Tensor real_score = Discriminate(Flat784(x), labels), fake_score = Discriminate(fake, labels);
    const auto   loss = objective == GanObjective::Logistic       ? LogisticGanObjectives(real_score, fake_score)
                        : objective == GanObjective::LeastSquares ? LeastSquaresGanObjectives(real_score, fake_score)
                                                                  : HingeGanObjectives(real_score, fake_score);
    Arrays out{{"fake", fake}, {"discriminator_loss", loss.discriminator}, {"generator_loss", loss.generator}};
    if (objective == GanObjective::Logistic) {
      out.emplace("real_logit", real_score);
      out.emplace("fake_logit", fake_score);
      out.emplace("real_probability", nn::sigmoid(real_score));
      out.emplace("fake_probability", nn::sigmoid(fake_score));
    } else {
      out.emplace("real_score", real_score);
      out.emplace("fake_score", fake_score);
    }
    return out;
  }

  Arrays Evaluate(const Tensor& x, const Tensor& z, const Tensor& labels = {}) {
    check::Require(x.size(0) >= 2 && z.size(0) >= 2, "pixel MMD diagnostic needs >= 2 cases EACH");
    nn::NoGradGuard guard;
    Arrays          out = Forward(x, z, labels);
    out.emplace("pixel_mmd", ImqMmd(GanCoarsePixels(Flat784(x)), GanCoarsePixels(out.at("fake"))));
    RequireFinite(out, "all GAN fields finite");
    return out;
  }
  Tensor Sample(const Tensor& z, const Tensor& labels = {}) {
    if (conditional) ValidateCondition(z, labels);
    nn::NoGradGuard guard;
    return Generate(z, labels);
  }
  Arrays Gradient(int group, const Tensor& x, const Tensor& z, const Tensor& labels = {}) {
    return GradientOf(group, Forward(x, z, labels).at(group == 0 ? "generator_loss" : "discriminator_loss"));
  }
  void Update(int group, const Tensor& x, const Tensor& z, const Tensor& labels = {}) {
    Apply(group, Gradient(group, x, z, labels));
  }
};

}  // namespace gan
