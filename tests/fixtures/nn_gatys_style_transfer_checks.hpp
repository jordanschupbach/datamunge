// Independent numeric and model-geometry checks; experiment driver omitted.
#pragma once
#include "nn_interpretation_check_support.hpp"
namespace inpaint {
using namespace dm::nn::models;
}
namespace denoise {
using namespace dm::nn::models;
}
namespace flow {
using namespace dm::nn::models;
}
namespace color {
using namespace dm::nn::models;
}
namespace depth {
using namespace dm::nn::models;
}
namespace pose {
using namespace dm::nn::models;
}
namespace style {
using namespace dm::nn::models;
}
namespace crnn {
using namespace dm::nn::models;
}
namespace ocr {
using namespace dm::nn::models;
}
namespace cifar {
using namespace dm::nn::models;
}

#include <array>
#include <chrono>
#include <cmath>
#include <datamunge/datasets/assets.hpp>
#include <fstream>
#include <iostream>
#include <random>
namespace nn = datamunge::nn;
using nn::Tensor;
using namespace style;

namespace {

double RelativeError(const Tensor& actual, const Tensor& expected) {
  check::Require(
      actual.dtype() == nn::DType::Float32 && expected.dtype() == nn::DType::Float32 &&
          actual.shape() == expected.shape(),
      "reference feature/gradient dtype and shape");
  double     error = 0, scale = 1;
  const auto a = actual.to_vector<double>(), b = expected.to_vector<double>();
  for (std::size_t i = 0; i < a.size(); ++i) {
    check::Require(std::isfinite(a[i]) && std::isfinite(b[i]), "finite reference comparison");
    error = std::max(error, std::abs(a[i] - b[i]));
    scale = std::max(scale, std::abs(b[i]));
  }
  return error / scale;
}

void PreprocessChecks() {
  const std::vector<float> input    = {1, 0, .5f, .1f, .2f, .3f},
                           upstream = {.2f, -.3f, .7f, 1, -.5f, .4f};
  Tensor     x                      = nn::from_vector(input, {1, 1, 2, 3}).requires_grad_();
  const auto bgr                    = VggImageNetInput(x);
  const auto g =
      nn::grad(nn::sum(bgr * nn::from_vector(upstream, {1, 1, 2, 3})), {x})[0].to_vector<double>();
  const auto  out    = bgr.to_vector<double>();
  const float mean[] = {103.939f, 116.779f, 123.68f};
  for (int p = 0; p < 2; ++p)
    for (int c = 0; c < 3; ++c) {
      const volatile float scaled = input[3 * p + 2 - c] * 255.f;
      check::Near(out[3 * p + c], scaled - mean[c], 1e-6, "ImageNet BGR scale/mean order");
      check::Near(
          g[3 * p + c],
          255 * upstream[3 * p + 2 - c],
          1e-6,
          "every preprocessing input derivative");
    }
  std::cout << "  ImageNet BGR255/means, signed channel order and every RGB input derivative passed"
            << std::endl;
}

void GramAndContentChecks() {
  constexpr int      batches = 2, positions = 6, channels = 2;
  std::vector<float> f(24), p(24), a(8), upstream(8), permuted(24), changed(24);
  for (int i = 0; i < 24; ++i)
    f[i] = .15f * (i % 9 - 4), p[i] = .12f * (i % 7 - 2), changed[i] = f[i] + (i < 12 ? 1 : 0);
  for (int b = 0; b < batches; ++b)
    for (int j = 0; j < channels; ++j)
      for (int k = 0; k < channels; ++k) {
        double value = 0;
        for (int m = 0; m < positions; ++m)
          value +=
              double(p[(b * positions + m) * channels + j]) * p[(b * positions + m) * channels + k];
        a[(b * 2 + j) * 2 + k]        = float(value / positions);
        upstream[(b * 2 + j) * 2 + k] = .1f * (1 + b * 4 + j * 2 + k);
      }
  for (int b = 0; b < batches; ++b)
    for (int m = 0; m < positions; ++m)
      for (int c = 0; c < channels; ++c)
        permuted[(b * positions + m) * channels + c] =
            f[(b * positions + positions - 1 - m) * channels + c];
  Tensor       x = nn::from_vector(f, {2, 2, 3, 2}).requires_grad_();
  const Tensor A = nn::from_vector(a, {2, 2, 2}), P = nn::from_vector(p, {2, 2, 3, 2});
  const auto   gram       = StyleGram(x);
  const auto   style_loss = GatysStyleLayer(x, A), content = GatysContentLoss(x, P);
  const auto   gs = nn::grad(style_loss, {x})[0].to_vector<double>(),
             gc   = nn::grad(content, {x})[0].to_vector<double>(),
             gg   = nn::grad(nn::sum(gram * nn::from_vector(upstream, {2, 2, 2})), {x})[0]
                      .to_vector<double>();
  const auto zero = GatysStyleLayer(x, StyleGram(x)).item(), self = GatysContentLoss(x, x).item(),
             perm_content   = GatysContentLoss(x, nn::from_vector(permuted, {2, 2, 3, 2})).item();
  const auto gv             = gram.to_vector<double>();
  double     grams[2][2][2] = {}, host_style = 0, host_content = 0, error = 0;
  for (int b = 0; b < batches; ++b)
    for (int j = 0; j < channels; ++j)
      for (int k = 0; k < channels; ++k) {
        for (int m = 0; m < positions; ++m)
          grams[b][j][k] += double(f[(b * positions + m) * channels + j]) *
                            f[(b * positions + m) * channels + k] / positions;
        check::Near(
            gv[(b * 2 + j) * 2 + k], grams[b][j][k], 1e-7, "each batched normalized Gram value");
        host_style += std::pow(grams[b][j][k] - a[(b * 2 + j) * 2 + k], 2) /
                      (4 * batches * channels * channels);
      }
  for (int b = 0; b < batches; ++b)
    for (int m = 0; m < positions; ++m)
      for (int c = 0; c < channels; ++c) {
        const int    i          = (b * positions + m) * channels + c;
        const double difference = f[i] - p[i];
        host_content += .5 * difference * difference;
        double ds = 0, dg = 0;
        for (int j = 0; j < channels; ++j) {
          ds += f[(b * positions + m) * channels + j] * (grams[b][j][c] - a[(b * 2 + j) * 2 + c]) /
                (batches * channels * channels * positions);
          dg += f[(b * positions + m) * channels + j] *
                (upstream[(b * 2 + j) * 2 + c] + upstream[(b * 2 + c) * 2 + j]) / positions;
        }
        error = std::max(error, std::abs(gs[i] - ds));
        error = std::max(error, std::abs(gc[i] - difference));
        error = std::max(error, std::abs(gg[i] - dg));
      }
  check::Near(style_loss.item(), host_style, 1e-8, "exact Gatys 1/(4*N^2*M^2) style normalization");
  check::Near(content.item(), host_content, 2e-7, "half-summed content loss");
  check::Near(error, 0, 1e-7, "every batched style/content/nonsymmetric Gram derivative");
  check::Near(zero, 0, 0, "self style zero");
  check::Near(self, 0, 0, "self content zero");
  check::Require(perm_content > 0, "content retains spatial arrangement");
  check::Near(
      RelativeError(StyleGram(nn::from_vector(permuted, {2, 2, 3, 2})), gram.detach()),
      0,
      1e-7,
      "Gram ignores spatial permutation");
  const auto other = StyleGram(nn::from_vector(changed, {2, 2, 3, 2})).to_vector<float>();
  for (int c = 0; c < 4; ++c)
    check::Near(other[4 + c], gv[4 + c], 0, "Gram never mixes different images in batch");
  std::cout
      << "  host Gram/content/style values and every derivative error=" << error
      << "; nonsymmetric upstream, batch independence, permutation and self-loss checks passed"
      << std::endl;
}

void KerasParity(const TensorMap& weights, const TensorMap& reference) {
  const FrozenVgg19     vgg(weights);
  Tensor                x   = reference.at("input").clone().requires_grad_();
  const auto            net = vgg.Extract(x);
  std::array<Tensor, 5> targets;
  for (int i = 0; i < 5; ++i)
    targets[std::size_t(i)] = reference.at("block" + std::to_string(i + 1) + "_conv1_gram");
  const Tensor content       = GatysContentLoss(net.content, reference.at("content_target"));
  const Tensor style         = GatysStyleLoss(net.style, targets);
  const Tensor objective     = content * double(1e-7f) + style * double(1e-5f);
  const Tensor gradient      = nn::grad(objective, {x})[0];
  double       feature_error = RelativeError(net.content, reference.at("block4_conv2"));
  for (int i = 0; i < 5; ++i)
    feature_error = std::max(
        feature_error,
        RelativeError(
            net.style[std::size_t(i)], reference.at("block" + std::to_string(i + 1) + "_conv1")));
  check::Near(feature_error, 0, 3e-4, "all six full-width pretrained VGG-19 feature arrays");
  const double content_error =
      RelativeError(content.reshape({}), reference.at("content_loss").reshape({}));
  const double style_error =
      RelativeError(style.reshape({}), reference.at("style_loss").reshape({}));
  const double objective_error =
      RelativeError(objective.reshape({}), reference.at("objective").reshape({}));
  const double gradient_error = RelativeError(gradient, reference.at("image_gradient"));
  check::Near(content_error, 0, 3e-4, "independent Keras content objective");
  check::Near(style_error, 0, 3e-4, "independent Keras Gram objective");
  check::Near(objective_error, 0, 3e-4, "independent combined objective");
  check::Near(gradient_error, 0, 5e-4, "every end-to-end image gradient vs GradientTape");
  check::Require(weights.size() == 32, "all16 pretrained layers imported");
  std::cout << "  genuine full-width ImageNet VGG-19 Keras/C++ parity (32x48): feature error="
            << feature_error << ", content/style=" << content_error << '/' << style_error
            << ", objective=" << objective_error << ", every-pixel gradient=" << gradient_error
            << std::endl;
}

/// The pixel optimization itself: frozen VGG-19 feature targets, projected Adam on
/// the image, best-objective checkpoint, frozen-weight proof and PNG exports. The
/// whole loop lives in this example (it used to be a shared style_transfer.hpp harness).
} // namespace
