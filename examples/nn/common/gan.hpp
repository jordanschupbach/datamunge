// Shared GAN example support: source-compatible Gaussian/data readers,
// named-array checkpoints, game diagnostics and remaining grouped model adapters.
// Completed DCGAN and Wasserstein maps now belong to the canonical core API;
// their report programs own every optimizer, constraint and fitting-loop call.
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

// ---- shared MLP and hinge pieces retained by InfoGAN, spectral and related models --------

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

}  // namespace gan
