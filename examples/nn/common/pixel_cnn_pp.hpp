// Compact PixelCNN++ likelihood core: port of tflow's common/nn/discretized_logistic.h
// (exact, stable discretized logistic RGB mixture with within-pixel channel
// coupling) and common/ml/pixel_cnn_pp.h (whole-pixel causal gated context,
// likelihood, Jacobian and channel-coupled ancestral sampling).
//
// The context network is the gated dual-stream PixelCNN of gated_pixel_cnn.hpp
// with 3 input channels and 30 outputs per pixel (3 components x {mixture
// logit, mean RGB, log-scale RGB, raw coefficients RG/BR/BG}); its first-block
// masks exclude the whole current pixel, so all three channels' parameters
// depend only on earlier pixels. Store seed 8609.
#pragma once

#include "common/gated_pixel_cnn.hpp"

namespace pcpp {

namespace nn = datamunge::nn;
using nn::Tensor;
using Map = replay::Map;

struct LogisticMixturePath {
  Tensor pixel_log_probability, component_log_probability, mixture_log_probability, means, log_scales, coefficients,
      channel_log_mass;
};

/// tflow's DiscretizedLogisticRGB: image [..., 3] in [-1, 1] (8-bit codes / 127.5 - 1),
/// parameters [..., mixtures * 10] (component-major). Interval mass of each
/// 1/127.5-wide bin via hi - softplus(hi) - softplus(lo) + log(-expm1(-delta))
/// (no CDF subtraction); codes 0 / 255 take the infinite lower / upper tails.
inline LogisticMixturePath DiscretizedLogisticRGB(const Tensor& image, const Tensor& parameters, int mixtures = 3) {
  const Tensor raw = nn::reshape(parameters, {-1, mixtures, 10}), rgb = nn::reshape(image, {-1, 3});
  auto         part    = [&](int i) { return nn::reshape(nn::slice_axis(raw, 2, i, i + 1), {-1, mixtures}); };
  auto         channel = [&](int i) { return nn::slice_axis(rgb, 1, i, i + 1); };
  const Tensor logits  = part(0);
  std::vector<Tensor> means, scales, coefficients, log_mass;
  for (int c = 0; c < 3; ++c) {
    means.push_back(part(c + 1));
    scales.push_back(nn::maximum(part(c + 4), nn::scalar_like(-7.0, part(c + 4))));
    coefficients.push_back(nn::tanh(part(c + 7)));
  }
  means[1] = nn::add(means[1], nn::mul(coefficients[0], channel(0)));
  means[2] = nn::add(means[2], nn::add(nn::mul(coefficients[1], channel(0)), nn::mul(coefficients[2], channel(1))));
  const float step = 1.f / 255, width = 2.f / 255;
  for (int c = 0; c < 3; ++c) {
    const Tensor inv      = nn::exp(nn::neg(scales[c]));
    const Tensor centered = nn::sub(channel(c), means[c]);
    const Tensor hi = nn::mul(nn::add(centered, double(step)), inv), lo = nn::mul(nn::sub(centered, double(step)), inv);
    const Tensor delta  = nn::mul(inv, double(width));
    const Tensor middle = nn::add(nn::sub(nn::sub(hi, nn::softplus(hi)), nn::softplus(lo)), nn::log(nn::neg(nn::expm1(nn::neg(delta)))));
    const Tensor lower = nn::neg(nn::softplus(nn::neg(hi))), upper = nn::neg(nn::softplus(lo));
    const Tensor low_mask  = nn::broadcast_to(nn::less_equal(channel(c), -.999f), middle.shape());
    const Tensor high_mask = nn::broadcast_to(nn::greater_equal(channel(c), .999f), middle.shape());
    log_mass.push_back(nn::where(low_mask, lower, nn::where(high_mask, upper, middle)));
  }
  const Tensor components = nn::add(nn::add(log_mass[0], log_mass[1]), log_mass[2]);
  const Tensor mixture    = nn::log_softmax(logits, 1);
  const Tensor weighted   = nn::add(components, mixture);
  const Tensor maximum    = nn::stop_gradient(nn::max(weighted, {1}, true));
  const Tensor log_probability =
      nn::add(nn::reshape(maximum, {-1}), nn::log(nn::sum(nn::exp(nn::sub(weighted, maximum)), {1})));
  return {log_probability,       components, mixture, nn::stack(means, 2), nn::stack(scales, 2), nn::stack(coefficients, 2),
          nn::stack(log_mass, 2)};
}

/// tflow's ml::PixelCNNPPModel.
class PixelCnnPPModel {
 public:
  nn::ParameterStore             store{8609};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit PixelCnnPPModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      Head(nn::zeros({1, 8, 8, 3}));
    }
    if (training) Train();
  }
  void Train() { optimizer = std::make_unique<nn::Optimizer>(store, .003, replay::Adam()); }

  bpix::GatedPath Context(const Tensor& x) { return bpix::GatedPixelConvolutional(store, x, Tensor(), false, 3, 30); }
  static Tensor   Loss(const LogisticMixturePath& l) { return nn::div(nn::neg(nn::mean(l.pixel_log_probability)), 3.0); }

  Map Evaluate(const Tensor& x) {
    nn::NoGradGuard guard;
    const auto      c = Context(x);
    const auto      l = DiscretizedLogisticRGB(x, c.logits);
    Map             out{{"parameters", c.logits},
                        {"vertical", c.vertical},
                        {"horizontal", c.horizontal},
                        {"pixel_log_probability", l.pixel_log_probability},
                        {"component_log_probability", l.component_log_probability},
                        {"mixture_log_probability", l.mixture_log_probability},
                        {"means", l.means},
                        {"log_scales", l.log_scales},
                        {"coefficients", l.coefficients},
                        {"channel_log_mass", l.channel_log_mass},
                        {"per_image_nll", nn::neg(nn::sum(nn::reshape(l.pixel_log_probability, {-1, 64}), {1}))},
                        {"loss", Loss(l)}};
    for (auto& [k, v] : out) v = v.detach().contiguous();
    bpix::RequireFinite(out, "finite RGB mixture fields");
    return out;
  }
  /// The 30 mixture parameters of every pixel, given the WHOLE-PIXEL prefix only.
  Tensor Head(const Tensor& x) {
    nn::NoGradGuard guard;
    return Context(x).logits.contiguous();
  }
  /// d sum_b sum_k parameters[b, site, k] / d image.
  Tensor CausalGradient(const Tensor& x, int site) {
    check::Require(site >= 0 && site < 64, "valid pixel index");
    Tensor input = x.detach().clone();
    input.requires_grad_(true);
    const Tensor head = nn::reshape(Context(input).logits, {-1, 64, 30});
    return nn::grad(nn::sum(nn::slice_axis(head, 1, site, site + 1)), {input})[0].detach().contiguous();
  }
  std::vector<Tensor> RawGradients(const Tensor& x) {
    const auto      c = Context(x);
    nn::GradOptions go;
    go.allow_unused = true;
    auto       g  = nn::grad(Loss(DiscretizedLogisticRGB(x, c.logits)), store.parameters(), go);
    const auto ps = store.parameters();
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = g[i].defined() ? g[i].detach() : nn::zeros_like(ps[i]);
    return g;
  }
  Map  Gradient(const Tensor& x) { return replay::Named(store, RawGradients(x)); }
  void Update(const Tensor& x) { optimizer->step(RawGradients(x)); }
  Map  Arrays() const { return replay::Arrays(store); }

  /// tflow's Sample: per pixel one component (uniform 0), then R, G, B logistic
  /// draws (uniforms 1..3) with G/B means coupled to the SAMPLED quantized R/G,
  /// clipped and rounded to 8-bit codes; MT19937 bank [n, 64, 4] case-major.
  Map Sample(int n, std::uint32_t seed, const Tensor* prefix = nullptr, int prefix_size = 0) {
    check::Require(n > 0 && prefix_size >= 0 && prefix_size <= 64, "valid RGB ancestral sample/prefix");
    std::vector<float>        images(static_cast<std::size_t>(n * 192), 0.f), raw(static_cast<std::size_t>(n * 64 * 30), 0.f);
    std::vector<double>       uniform(static_cast<std::size_t>(n * 256));
    std::vector<std::int32_t> components(static_cast<std::size_t>(n * 64), -1);
    std::mt19937              rng(seed);
    for (auto& u : uniform) u = (double(rng()) + .5) / 4294967296.;
    if (prefix) {
      check::Require(prefix->shape() == nn::Shape({n, 8, 8, 3}), "aligned RGB prefix");
      const auto p = prefix->contiguous().to_vector<float>();
      for (int b = 0; b < n; ++b) std::copy_n(p.data() + b * 192, prefix_size * 3, images.data() + b * 192);
    }
    for (int j = prefix_size; j < 64; ++j) {
      const auto head = Head(bpix::Floats(images, {n, 8, 8, 3})).to_vector<float>();
      for (int b = 0; b < n; ++b) {
        const float* h       = head.data() + (b * 64 + j) * 30;
        double       maximum = -INFINITY, denominator = 0, sum = 0;
        for (int k = 0; k < 3; ++k) maximum = std::max(maximum, double(h[k * 10]));
        for (int k = 0; k < 3; ++k) denominator += std::exp(double(h[k * 10]) - maximum);
        int component = 2;
        for (int k = 0; k < 3; ++k) {
          sum += std::exp(double(h[k * 10]) - maximum) / denominator;
          if (uniform[(b * 64 + j) * 4] < sum) {
            component = k;
            break;
          }
        }
        components[b * 64 + j] = component;
        std::copy_n(h, 30, raw.data() + (b * 64 + j) * 30);
        auto   value = [&](int i) { return double(h[component * 10 + i]); };
        float* pixel = images.data() + (b * 64 + j) * 3;
        for (int c = 0; c < 3; ++c) {
          double mean = value(c + 1);
          if (c == 1) mean += std::tanh(value(7)) * pixel[0];
          if (c == 2) mean += std::tanh(value(8)) * pixel[0] + std::tanh(value(9)) * pixel[1];
          const double scale = std::exp(std::max(-7., value(c + 4))), u = uniform[(b * 64 + j) * 4 + c + 1];
          const double sample = mean + scale * (std::log(u) - std::log1p(-u));
          const int    code   = int(std::floor((std::clamp(sample, -1., 1.) + 1) * 127.5 + .5));
          pixel[c]            = float(double(code) / 127.5 - 1);
        }
      }
    }
    return {{"images", bpix::Floats(images, {n, 8, 8, 3})},
            {"uniform", nn::from_vector(uniform, {n, 64, 4})},
            {"parameters", bpix::Floats(raw, {n, 64, 30})},
            {"components", nn::from_vector(components, {n, 64})}};
  }
};

}  // namespace pcpp
