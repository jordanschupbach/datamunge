// Port of tflow's common/nn/hierarchical_variational_autoencoder.h and
// common/ml/hierarchical_variational_autoencoder.h: a two-group spatial
// hierarchical VAE (global 2-D top latent, 7x7 lower latent with a learned
// conditional prior) and a miniature NVAE variant (residual posterior heads,
// expansion/depthwise/BN/Swish/projection/SE residual cell, spectral penalty on the
// two rank-one prior kernels). Model weights and BN moving statistics live in two
// stores ("state/<name>" in saved arrays, as tflow).
#pragma once

#include "common/beta_variational_autoencoder.hpp"
#include "common/ladder_variational_autoencoder.hpp"

namespace ae {

inline nn::Tensor HierarchySwish(const nn::Tensor& x) { return x * nn::sigmoid(x); }

/// Batch-norm mode of one forward pass: training statistics (optionally updating the
/// moving averages, as TF's update op alongside a train step) or moving statistics.
struct HierarchyMode {
  bool training{false};
  bool update_stats{false};
};

inline nn::Tensor HierarchyBn(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                              const nn::Tensor& y, HierarchyMode mode) {
  nn::BatchNormOptions o;
  o.momentum     = .9;
  o.eps          = 1e-3;
  o.update_stats = mode.update_stats;
  return nn::batch_norm(p, state, name, y, mode.training, o);
}

inline nn::Tensor HierarchyNvaeCell(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& x,
                                    HierarchyMode mode) {
  nn::ConvLayerOptions no_bias;
  no_bias.bias = false;
  auto expand  = HierarchySwish(HierarchyBn(p, state, "cell/expand_bn", nn::conv2d(p, "cell/expand", x, 8, 1, no_bias), mode));
  auto depth   = HierarchySwish(
      HierarchyBn(p, state, "cell/depth_bn", nn::depthwise_conv2d(p, "cell/depthwise", expand, 1, 3, no_bias), mode));
  auto project = HierarchyBn(p, state, "cell/project_bn", nn::conv2d(p, "cell/project", depth, 4, 1, no_bias), mode);
  auto pooled  = nn::mean(project, {1, 2});
  auto hidden  = HierarchySwish(nn::dense(p, "cell/se_reduce", pooled, 2));
  auto gate    = nn::reshape(nn::sigmoid(nn::dense(p, "cell/se_expand", hidden, 4)), {x.size(0), 1, 1, 4});
  return x + 0.1 * (project * gate);
}

/// top_projection(z2) as a 7x7x4 map, then the ordinary 3x3 Swish cell or the NVAE cell.
inline nn::Tensor HierarchyTopFeature(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& top_z,
                                      bool nvae, HierarchyMode mode) {
  auto projected = nn::reshape(nn::dense(p, "top_projection", top_z, 196), {top_z.size(0), 7, 7, 4});
  return nvae ? HierarchyNvaeCell(p, state, projected, mode) : HierarchySwish(nn::conv2d(p, "cell", projected, 4, 3));
}

/// p(x | z1, z2) logits [n, 784].
inline nn::Tensor HierarchyDecoder(nn::ParameterStore& p, const nn::Tensor& lower, const nn::Tensor& top_feature) {
  auto lower_map = nn::reshape(lower, {lower.size(0), 7, 7, 1});
  auto h         = HierarchySwish(nn::conv2d(p, "combine", nn::concat({top_feature, lower_map}, 3), 4, 1));
  h              = HierarchySwish(nn::conv2d(p, "decoder14", nn::upsample_nearest(h, 2), 4, 3));
  return nn::reshape(nn::conv2d(p, "output", nn::upsample_nearest(h, 2), 1, 3), {lower.size(0), 784});
}

inline LadderGaussian HierarchyPosterior(const nn::Tensor& pm, const nn::Tensor& pl, const nn::Tensor& hm,
                                         const nn::Tensor& hl, bool residual) {
  return residual ? LadderGaussian{pm + hm, pl + hl} : LadderGaussian{hm, hl};
}

struct HierarchicalVaeOutput {
  nn::Tensor encoder_feature, top_feature, top_mu, top_lv, top_z, prior_mu, prior_lv, head_mu, head_lv, lower_mu,
      lower_lv, lower_z, probability, nll, kl_lower, kl_top, loss, regularizer, train_loss;
};

inline void HierarchyPriorHeads(nn::ParameterStore& p, const nn::Tensor& top_feature, nn::Tensor& prior_mu,
                                nn::Tensor& prior_lv) {
  const std::int64_t n = top_feature.size(0);
  prior_mu             = nn::reshape(nn::conv2d(p, "prior_mu", top_feature, 1, 1), {n, 49});
  prior_lv             = nn::reshape(nn::conv2d(p, "prior_lv", top_feature, 1, 1), {n, 49});
}

/// epsilon [n, 51]: columns 0..48 drive the lower 7x7 group, 49..50 the top group.
inline HierarchicalVaeOutput HierarchicalVariationalAutoencoder(nn::ParameterStore& p, nn::ParameterStore& state,
                                                                const nn::Tensor& image, const nn::Tensor& epsilon,
                                                                bool nvae, HierarchyMode mode) {
  HierarchicalVaeOutput o;
  const std::int64_t    n = image.size(0);
  auto                  x = nn::reshape(image, {n, 28, 28, 1});
  auto                  h = HierarchySwish(nn::conv2d(p, "encoder1", x, 4, 3));
  h                       = nn::avg_pool2d(h, 2, 2);
  h                       = HierarchySwish(nn::conv2d(p, "encoder2", h, 4, 3));
  o.encoder_feature       = nn::avg_pool2d(h, 2, 2);
  auto flat               = nn::reshape(o.encoder_feature, {n, 196});
  o.top_mu                = nn::dense(p, "top_mu", flat, 2);
  o.top_lv                = nn::dense(p, "top_lv", flat, 2);
  o.top_z                 = VaeReparameterize(o.top_mu, o.top_lv, nn::slice_axis(epsilon, 1, 49, 51));
  o.top_feature           = HierarchyTopFeature(p, state, o.top_z, nvae, mode);
  HierarchyPriorHeads(p, o.top_feature, o.prior_mu, o.prior_lv);
  auto posterior_h = HierarchySwish(
      nn::conv2d(p, "posterior_hidden", nn::concat({o.encoder_feature, o.top_feature}, 3), 4, 1));
  auto heads = nn::conv2d(p, "posterior_heads", posterior_h, 2, 1);
  o.head_mu  = nn::reshape(nn::slice_axis(heads, 3, 0, 1), {n, 49});
  o.head_lv  = nn::reshape(nn::slice_axis(heads, 3, 1, 2), {n, 49});
  auto q     = HierarchyPosterior(o.prior_mu, o.prior_lv, o.head_mu, o.head_lv, nvae);
  o.lower_mu = q.mu;
  o.lower_lv = q.log_variance;
  o.lower_z  = VaeReparameterize(o.lower_mu, o.lower_lv, nn::slice_axis(epsilon, 1, 0, 49));
  auto logits = HierarchyDecoder(p, o.lower_z, o.top_feature);
  o.probability = nn::sigmoid(logits);
  o.nll         = VaePixelNll(logits, image);
  o.kl_lower    = LadderConditionalKl(o.lower_mu, o.lower_lv, o.prior_mu, o.prior_lv);
  o.kl_top      = VaeKl(o.top_mu, o.top_lv);
  o.loss        = nn::mean(o.nll + (o.kl_lower + o.kl_top), {0});
  o.regularizer = nn::scalar(0.0, nn::DType::Float32);
  // Each prior kernel is a 4x1 matrix: its sole singular value is exactly its L2 norm.
  // Regularize these TWO heads only, not NVAE's all-kernel spectral schedule.
  if (nvae) {
    std::vector<nn::Tensor> norms;
    for (const std::string name : {"prior_mu/W", "prior_lv/W"})
      norms.push_back(nn::sqrt(nn::sum(nn::square(p.at(name))) + 1e-12));
    o.regularizer = 1e-4 * (norms[0] + norms[1]);
  }
  o.train_loss = o.loss + o.regularizer;
  return o;
}

class HierarchicalVaeModel {
 public:
  nn::ParameterStore             parameters{2609}, state{0};
  bool                           nvae;
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit HierarchicalVaeModel(bool miniature_nvae, bool training = true) : nvae(miniature_nvae) {
    {
      nn::NoGradGuard guard;
      HierarchicalVariationalAutoencoder(parameters, state, nn::zeros({2, 784}), nn::zeros({2, 51}), nvae, {});
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  HierarchicalVaeOutput Forward(const nn::Tensor& images, const nn::Tensor& noise, HierarchyMode mode) {
    return HierarchicalVariationalAutoencoder(parameters, state, Flatten(images), noise, nvae, mode);
  }
  /// tflow's Evaluate map (training=true uses batch statistics WITHOUT updating the moving averages).
  nn::NamedTensors Evaluate(const nn::Tensor& images, const nn::Tensor& noise, bool training = false) {
    nn::NoGradGuard  guard;
    auto             o = Forward(images, noise, {training, false});
    nn::NamedTensors out{{"encoder_feature", o.encoder_feature},
                         {"top_feature", o.top_feature},
                         {"top_mu", o.top_mu},
                         {"top_lv", o.top_lv},
                         {"top_z", o.top_z},
                         {"prior_mu", o.prior_mu},
                         {"prior_lv", o.prior_lv},
                         {"head_mu", o.head_mu},
                         {"head_lv", o.head_lv},
                         {"lower_mu", o.lower_mu},
                         {"lower_lv", o.lower_lv},
                         {"lower_z", o.lower_z},
                         {"probability", o.probability},
                         {"nll", o.nll},
                         {"kl_lower", o.kl_lower},
                         {"kl_top", o.kl_top},
                         {"loss", o.loss},
                         {"regularizer", o.regularizer},
                         {"train_loss", o.train_loss}};
    for (auto& [name, t] : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "all hierarchy fields finite");
    }
    return out;
  }
  /// Image-free p(x | z1, z2) decode.
  nn::Tensor Decode(const nn::Tensor& lower, const nn::Tensor& top) {
    nn::NoGradGuard guard;
    auto            feature = HierarchyTopFeature(parameters, state, top, nvae, {});
    return nn::sigmoid(HierarchyDecoder(parameters, lower, feature)).contiguous();
  }
  /// Ancestral sample: z2 given, z1 = prior_mu + exp(prior_lv / 2) lower_noise.
  nn::NamedTensors Prior(const nn::Tensor& top, const nn::Tensor& lower_noise) {
    nn::NoGradGuard guard;
    auto            feature = HierarchyTopFeature(parameters, state, top, nvae, {});
    nn::Tensor      pm, pl;
    HierarchyPriorHeads(parameters, feature, pm, pl);
    auto z = VaeReparameterize(pm, pl, lower_noise);
    return {{"top_z", top},
            {"lower_noise", lower_noise},
            {"prior_mu", pm.contiguous()},
            {"prior_lv", pl.contiguous()},
            {"lower_z", z.contiguous()},
            {"probability", nn::sigmoid(HierarchyDecoder(parameters, z, feature)).contiguous()}};
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& noise) {
    return nn::grad(Forward(images, noise, {true, false}).train_loss, parameters.parameters());
  }
  /// Isolated spectral-penalty and unregularized prior-head gradients (miniature NVAE).
  nn::NamedTensors RegularizerGradients(const nn::Tensor& images, const nn::Tensor& noise) {
    const std::vector<nn::Tensor> heads{parameters.at("prior_mu/W"), parameters.at("prior_lv/W")};
    auto                          o = Forward(images, noise, {true, false});
    nn::GradOptions               keep;
    keep.retain_graph = true;
    auto r            = nn::grad(o.regularizer, heads, keep);
    auto                          l = nn::grad(o.loss, heads);
    return {{"regularizer/prior_mu/W", r[0]},
            {"regularizer/prior_lv/W", r[1]},
            {"without_regularizer/prior_mu/W", l[0]},
            {"without_regularizer/prior_lv/W", l[1]}};
  }
  /// One Adam step and moving-statistics update sharing the pre-update forward.
  double Update(const nn::Tensor& images, const nn::Tensor& noise) {
    auto         loss  = Forward(images, noise, {true, true}).train_loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite hierarchical objective");
    optimizer->minimize(loss);
    return value;
  }
  /// Parameters AND inference statistics ("state/<name>").
  nn::NamedTensors Arrays() const {
    auto out = ae::Arrays(parameters);
    for (const auto& name : state.names()) out.emplace_back("state/" + name, state.at(name).detach().clone());
    return out;
  }
  std::vector<std::string> Snapshot() const {
    std::vector<std::string> out;
    for (const auto& [name, t] : Arrays()) out.push_back(Bytes(t));
    return out;
  }
  void Load(const nn::NamedTensors& arrays) {
    check::Require(arrays.size() == parameters.size() + state.size(), "all hierarchy parameters and statistics supplied");
    for (const auto& [name, value] : arrays) {
      const bool is_state = name.rfind("state/", 0) == 0;
      auto&      store    = is_state ? state : parameters;
      const auto key      = is_state ? name.substr(6) : name;
      check::Require(store.contains(key) && store.at(key).shape() == value.shape(), "matching hierarchy array");
      store.set(key, nn::cast(value, nn::DType::Float32).contiguous().clone());
    }
  }
  void Load(const std::string& path) { Load(nn::load_npz(path)); }
  void Save(const std::string& path) const { nn::save_npz(path, Arrays()); }
};

inline VaeNoiseBank HierarchyNoise(std::int64_t n, std::int64_t samples, std::uint32_t seed) {
  return GaussianBank(n, 51, samples, seed);
}
inline nn::Tensor HierarchyNoiseAt(const VaeNoiseBank& bank, std::int64_t sample) {
  check::Require(sample >= 0 && sample < bank.epsilon.size(0), "available hierarchy draw");
  return nn::select(bank.epsilon, 0, sample).contiguous();
}

/// {NLL, lower KL, top KL, negative ELBO} per image, every term averaged over all draws.
inline std::vector<double> HierarchyScore(HierarchicalVaeModel& model, const nn::Tensor& images,
                                          const VaeNoiseBank& noise) {
  double             nll = 0, lower = 0, top = 0;
  const std::int64_t n = images.size(0), samples = noise.epsilon.size(0);
  for (std::int64_t k = 0; k < samples; ++k) {
    auto       f = model.Evaluate(images, HierarchyNoiseAt(noise, k));
    const auto a = Find(f, "nll").to_vector<float>(), l = Find(f, "kl_lower").to_vector<float>(),
               t = Find(f, "kl_top").to_vector<float>();
    for (std::int64_t b = 0; b < n; ++b) {
      nll += a[b];
      lower += l[b];
      top += t[b];
    }
  }
  const double count = double(n * samples);
  return {nll / count, lower / count, top / count, (nll + lower + top) / count};
}

}  // namespace ae
