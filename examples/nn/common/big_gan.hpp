// Port of tflow's common/nn/big_gan.h and common/ml/big_gan.h: a compact
// BigGAN teaching core on 8x8 digits. Hierarchical latent (3 chunks of 4), a
// shared class embedding feeding class-conditional BN, one residual up-block,
// spectral normalization of every kernel with ONE power iteration per phase
// (explicit Advance), a projection discriminator, hinge losses and the
// modified orthogonal regularizer on G.
//
// Groups: 0 = G, 1 = D (trainable), 2 = G's SN vectors, 3 = D's SN vectors,
// 4 = G's conditional-BN moving statistics ("<k>/<name>", as tflow's arrays).
// The D phase advances D's SN, then steps D on the hinge loss with G's
// batch-statistics fake; the G phase advances G's SN, then applies Adam and the
// BN moving-average update from moments CAPTURED in the same forward.
#pragma once

#include <cmath>

#include "common/gan.hpp"

namespace gan {

/// Spectral-normalized kernel: W / max(v^T W u, 1e-6) with stopped u [out, 1], v [rows, 1].
inline Tensor BigKernel(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Shape& shape) {
  const Tensor       w    = p.get(name + "/W", shape, nn::init::normal(.02));
  const std::int64_t out  = shape.back(), rows = w.numel() / out;
  const Tensor       u    = state.get(name + "/u", {out, 1}, nn::init::constant(1.f / std::sqrt(float(out))));
  const Tensor       v    = state.get(name + "/v", {rows, 1}, nn::init::constant(1.f / std::sqrt(float(rows))));
  const Tensor       m    = nn::reshape(w, {rows, out});
  const Tensor       sigma = nn::sum(v.detach() * nn::matmul(m, u.detach()));
  return w / nn::maximum(sigma, nn::scalar_like(double(1e-6f), sigma));
}
inline Tensor BigDense(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const Tensor& x,
                       std::int64_t in, std::int64_t out, bool bias = true) {
  Tensor y = nn::matmul(x, BigKernel(p, state, name, {in, out}));
  if (bias) y = nn::bias_add(y, p.get(name + "/b", {out}, nn::init::zeros()));
  return y;
}
inline Tensor BigConv(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const Tensor& x,
                      std::int64_t in, std::int64_t out, std::int64_t kernel, bool bias = true) {
  Tensor y = nn::conv2d(x, BigKernel(p, state, name, {kernel, kernel, in, out}));
  if (bias) y = nn::bias_add(y, p.get(name + "/b", {out}, nn::init::zeros()));
  return y;
}

struct BigPath {
  Tensor image;
  Arrays moments;  ///< "<bn>/mean", "<bn>/unbiased_var" of the batch-statistics path
};

/// Unparameterized 4-channel BN: batch moments (training) or moving statistics.
inline Tensor BigNorm(nn::ParameterStore& stats, BigPath& path, const std::string& name, const Tensor& x, bool training) {
  Tensor mean = stats.get(name + "/moving_mean", {4}, nn::init::zeros()).detach();
  Tensor var  = stats.get(name + "/moving_var", {4}, nn::init::ones()).detach();
  if (training) {
    mean               = nn::mean(x, {0, 1, 2});
    var                = nn::mean(nn::square(x - mean), {0, 1, 2});
    const double count = double(float(x.numel()) / 4.f);
    path.moments.emplace(name + "/mean", mean);
    path.moments.emplace(name + "/unbiased_var", var * double(float(count) / float(count - 1)));
  }
  return (x - mean) / nn::sqrt(var + double(1e-5f));
}
inline Tensor BigConditionalNorm(nn::ParameterStore& p, nn::ParameterStore& sn, nn::ParameterStore& stats, BigPath& path,
                                 const std::string& name, const Tensor& x, const Tensor& condition, bool training) {
  const Tensor normalized = BigNorm(stats, path, name, x, training);
  const Tensor gain       = BigDense(p, sn, name + "/gain", condition, 8, 4) + 1.0;
  const Tensor bias       = BigDense(p, sn, name + "/bias", condition, 8, 4);
  return normalized * nn::reshape(gain, {-1, 1, 1, 4}) + nn::reshape(bias, {-1, 1, 1, 4});
}

/// z [n, 12] (chunks z0 stem, z1 block BN1, z2 block BN2) + class -> [n, 8, 8, 1] sigmoid.
inline BigPath BigGenerate(nn::ParameterStore& p, nn::ParameterStore& sn, nn::ParameterStore& stats, const Tensor& z,
                           const Tensor& labels, bool training) {
  BigPath      path;
  const Tensor embedding = p.get("class_embedding", {10, 4}, nn::init::normal(.1));
  const Tensor shared    = nn::matmul(OneHot10(labels), embedding);
  const Tensor z0 = nn::slice_axis(z, 1, 0, 4), z1 = nn::slice_axis(z, 1, 4, 8), z2 = nn::slice_axis(z, 1, 8, 12);
  const Tensor x  = nn::reshape(BigDense(p, sn, "stem", z0, 4, 64, false), {-1, 4, 4, 4});
  const Tensor condition1 = nn::concat({shared, z1}, 1), condition2 = nn::concat({shared, z2}, 1);
  Tensor       h = nn::relu(BigConditionalNorm(p, sn, stats, path, "bn1", x, condition1, training));
  h              = nn::resize_nearest(h, {8, 8});
  Tensor hidden  = BigConv(p, sn, "conv1", h, 4, 4, 3, false);
  hidden         = nn::relu(BigConditionalNorm(p, sn, stats, path, "bn2", hidden, condition2, training));
  hidden         = BigConv(p, sn, "conv2", hidden, 4, 4, 3, false);
  const Tensor shortcut   = nn::resize_nearest(BigConv(p, sn, "shortcut", x, 4, 4, 1, false), {8, 8});
  const Tensor normalized = BigNorm(stats, path, "output_bn", hidden + shortcut, training);
  const Tensor gamma = p.get("output_bn/gamma", {4}, nn::init::ones()), beta = p.get("output_bn/beta", {4}, nn::init::zeros());
  const Tensor output = nn::relu(normalized * gamma + beta);
  path.image          = nn::sigmoid(BigConv(p, sn, "rgb", output, 4, 1, 3));
  return path;
}

/// Residual down-block features + projection: score = w^T phi(x) + embed(y)^T phi(x).
inline Tensor BigDiscriminate(nn::ParameterStore& p, nn::ParameterStore& sn, const Tensor& image, const Tensor& labels) {
  Tensor h = BigConv(p, sn, "conv1", image, 1, 4, 3);
  h        = BigConv(p, sn, "conv2", nn::relu(h), 4, 4, 3);
  h        = nn::avg_pool2d(h, 2, 2);
  const Tensor shortcut  = nn::avg_pool2d(BigConv(p, sn, "shortcut", image, 1, 4, 1), 2, 2);
  const Tensor features  = nn::sum(nn::relu(h + shortcut), {1, 2});
  const Tensor source    = BigDense(p, sn, "score", features, 4, 1);
  const Tensor table     = BigKernel(p, sn, "projection", {10, 4});
  const Tensor embedding = nn::matmul(OneHot10(labels), table);
  return source + nn::sum(features * embedding, {1}, true);
}

inline Tensor SpectralUnit(const Tensor& x) { return x / nn::sqrt(nn::sum(nn::square(x)) + double(1e-12f)); }

/// One power iteration for every "<name>/W" of p: v = unit(W u), u = unit(W^T v).
inline void BigAdvance(nn::ParameterStore& p, nn::ParameterStore& state) {
  nn::NoGradGuard                          guard;
  std::vector<std::pair<std::string, Tensor>> next;
  for (const auto& key : p.names()) {
    if (key.size() < 2 || key.substr(key.size() - 2) != "/W") continue;
    const auto         name = key.substr(0, key.size() - 2);
    const Tensor       w    = p.at(key);
    const std::int64_t out = w.shape().back(), rows = w.numel() / out;
    const Tensor       m = nn::reshape(w, {rows, out});
    const Tensor       v = SpectralUnit(nn::matmul(m, state.at(name + "/u")));
    const Tensor       u = SpectralUnit(nn::matmul(m, v, true));
    next.emplace_back(name + "/u", u);
    next.emplace_back(name + "/v", v);
  }
  for (const auto& [k, t] : next) state.set(k, t);
}

/// Modified orthogonal regularizer .5e-4 sum((W^T W) o (1 - I))^2 over G kernels;
/// other G arrays contribute 0 * sum (zero but defined partials, as tflow).
inline Tensor BigOrthogonalPenalty(nn::ParameterStore& p) {
  Tensor total;
  for (const auto& key : p.names()) {
    const Tensor w = p.at(key);
    Tensor       term;
    if (key.size() < 2 || key.substr(key.size() - 2) != "/W") {
      term = 0.0 * nn::sum(w);
    } else {
      const std::int64_t out = w.shape().back(), rows = w.numel() / out;
      const Tensor       m    = nn::reshape(w, {rows, out});
      const Tensor       gram = nn::matmul(m, m, true);
      std::vector<float> mask(out * out, 1.f);
      for (std::int64_t i = 0; i < out; ++i) mask[i * out + i] = 0.f;
      term = double(.5e-4f) * nn::sum(nn::square(gram * nn::from_vector(mask, {out, out})));
    }
    total = total.defined() ? total + term : term;
  }
  return total;
}

class BigGanModel : public GroupedGan {
 public:
  static void Validate(const Tensor& z, const Tensor& labels) { ValidateCondition(z, labels, 12); }

  explicit BigGanModel(bool training = true, std::uint64_t g_seed = 6803, std::uint64_t d_seed = 6823)
      : GroupedGan({g_seed, d_seed, 0, 0, 0}) {
    {
      nn::NoGradGuard guard;
      const Tensor    z = nn::zeros({2, 12}), labels = nn::zeros({2}, nn::DType::Int32);
      BigGenerate(Group(0), Group(2), Group(4), z, labels, false);
      BigDiscriminate(Group(1), Group(3), nn::zeros({2, 8, 8, 1}), labels);
    }
    if (training) {
      MakeAdam(.0002, 0, .999, {0, 1});
      for (int i = 0; i < 16; ++i) Advance(0), Advance(1);
    }
  }

  /// ONE phase-owned SN power iteration (group 0 -> G's vectors, 1 -> D's).
  void Advance(int group) { BigAdvance(Group(group), Group(group + 2)); }

  /// Every Evaluate field (graphs kept when grad mode is on); `batch_statistics`
  /// selects the training-BN fake as "fake"/"fake_score" and adds the moments.
  Arrays Forward(const Tensor& x, const Tensor& z, const Tensor& labels, bool batch_statistics) {
    Validate(z, labels);
    auto&        g = Group(0);
    auto&        d = Group(1);
    const auto   fake_d = BigGenerate(g, Group(2), Group(4), z, labels, false);
    const auto   fake_g = BigGenerate(g, Group(2), Group(4), z, labels, true);
    const Tensor real_score           = BigDiscriminate(d, Group(3), x, labels);
    const Tensor fake_g_score         = BigDiscriminate(d, Group(3), fake_g.image, labels);
    const Tensor fake_inference_score = BigDiscriminate(d, Group(3), fake_d.image, labels);
    const Tensor d_loss               = HingeGanObjectives(real_score, fake_g_score).discriminator;
    const Tensor g_source             = -nn::mean(fake_g_score);
    const Tensor g_ortho              = BigOrthogonalPenalty(g);
    Arrays       out{{"fake", batch_statistics ? fake_g.image : fake_d.image},
                     {"fake_training", fake_g.image},
                     {"fake_inference", fake_d.image},
                     {"fake_score_training", fake_g_score},
                     {"fake_score_inference", fake_inference_score},
                     {"real_score", real_score},
                     {"fake_score", batch_statistics ? fake_g_score : fake_inference_score},
                     {"d_loss", d_loss},
                     {"g_source", g_source},
                     {"g_orthogonal", g_ortho},
                     {"g_loss", g_source + g_ortho}};
    for (const auto& [k, v] : fake_g.moments) out.emplace("moment/" + k, v);
    return out;
  }
  Arrays Evaluate(const Tensor& x, const Tensor& z, const Tensor& labels, bool batch_statistics = false) {
    nn::NoGradGuard guard;
    Arrays          out;
    for (auto& [k, v] : Forward(x, z, labels, batch_statistics))
      if (batch_statistics || k.rfind("moment/", 0) != 0) out.emplace(k, v);
    RequireFinite(out, "finite BigGAN fields");
    return out;
  }
  /// Conditional noise + class ONLY generation with FROZEN BN/SN.
  Tensor Generate(const Tensor& z, const Tensor& labels) {
    Validate(z, labels);
    nn::NoGradGuard guard;
    return BigGenerate(Group(0), Group(2), Group(4), z, labels, false).image;
  }
  /// Projection D from image + class ONLY.
  Tensor Score(const Tensor& x, const Tensor& labels) {
    check::Require(labels.dtype() == nn::DType::Int32 && labels.shape() == nn::Shape({x.size(0)}),
                   "one projection class per image");
    for (auto c : labels.contiguous().to_vector<std::int32_t>()) check::Require(c >= 0 && c < 10, "projection class 0..9");
    nn::NoGradGuard guard;
    return BigDiscriminate(Group(1), Group(3), x, labels);
  }
  /// Group gradient (+ G's captured batch moments "moment/<bn>/...").
  Arrays Capture(int group, const Tensor& x, const Tensor& z, const Tensor& labels) {
    const auto fields = Forward(x, z, labels, true);
    Arrays     out    = GradientOf(group, fields.at(group ? "d_loss" : "g_loss"));
    if (group == 0)
      for (const auto& [k, v] : fields)
        if (k.rfind("moment/", 0) == 0) out.emplace(k, v.detach());
    return out;
  }
  /// Isolated G partials of the adversarial (false) or orthogonal (true) term.
  Arrays Component(bool orthogonal, const Tensor& x, const Tensor& z, const Tensor& labels) {
    return GradientOf(0, Forward(x, z, labels, true).at(orthogonal ? "g_orthogonal" : "g_source"));
  }
  /// Captured Adam step; the G phase also applies the BN moving-average update (momentum .9).
  void ApplyCaptured(int group, const Arrays& captured) {
    Apply(group, captured);
    if (group != 0) return;
    nn::NoGradGuard guard;
    auto&           stats = Group(4);
    for (const auto& [key, value] : captured) {
      if (key.rfind("moment/", 0) != 0) continue;
      const auto rest  = key.substr(7);
      const auto slash = rest.find('/');
      const auto name  = rest.substr(0, slash) + (rest.substr(slash + 1) == "mean" ? "/moving_mean" : "/moving_var");
      stats.set(name, double(.9f) * stats.at(name).detach() + double(1.f - .9f) * value);
    }
  }
};

}  // namespace gan
