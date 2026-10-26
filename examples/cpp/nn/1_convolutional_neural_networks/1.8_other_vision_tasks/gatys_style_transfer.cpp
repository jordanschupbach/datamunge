// CMD: just run-nn gatys_style_transfer
// Gatys image optimization through genuine frozen pretrained VGG-19 features.
// https://arxiv.org/abs/1508.06576
// Preparation: just style-assets (verified official Keras ImageNet weights,
// tutorial Labrador/Kandinsky images, independent Keras feature/gradient fixtures).
// RGB[0,1] -> BGR255/ImageNet means; replace max pools with average pools.
// Match conv4_2 content and equally weighted conv1_1..conv5_1 style Grams.
// Style Gram is per image and divided by spatial count, retaining the exact
// 1/(4*N^2*M^2) squared Gram coefficient; content is.5 summed squared error.
// Optimize only generated pixels; all20,024,384 imported weights are constants.
// Default64px,30 projected Adam(.03) updates; content/noise initialization.
// Fixed content/style coefficients balance their initial raw loss magnitudes;
// targets and coefficients never change during optimization. Restore minimum
// combined objective, prove all frozen weights unchanged and export four PNGs.
// This is a short low-resolution example, not the paper's high-res convergence
// or exact original VGG rescaling/optimizer reproduction.
//
// Port of tflow's gatys_style_transfer. Preparation (in ~/projects/tflow_examples):
// just style-assets; vgg19.npz, reference.npz, content.jpg and style.jpg are read
// from $DATAMUNGE_DATA_DIR/style_transfer. Everything here is deterministic and
// shared with TF (pretrained weights, the JPEG decode via datamunge::image::read_jpeg,
// TF ResizeBilinear via nn::resize_bilinear, the std::mt19937(17) noise), so the
// optimization follows TF's trajectory up to float rounding. The image
// checkpoint is best_image.npz. Outputs go to output/nn/models/gatys_style_transfer.
// Options: assets=PATH size=64 steps=30 strength=1 checks_only=0|1 threads=N.

#include <cmath>
#include <iostream>

#include "common/paths.hpp"
#include "common/style_transfer.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using namespace style;

namespace {

double RelativeError(const Tensor& actual, const Tensor& expected) {
  check::Require(actual.dtype() == nn::DType::Float32 && expected.dtype() == nn::DType::Float32 && actual.shape() == expected.shape(),
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
  const std::vector<float> input = {1, 0, .5f, .1f, .2f, .3f}, upstream = {.2f, -.3f, .7f, 1, -.5f, .4f};
  Tensor     x   = nn::from_vector(input, {1, 1, 2, 3}).requires_grad_();
  const auto bgr = VggImageNetInput(x);
  const auto g   = nn::grad(nn::sum(bgr * nn::from_vector(upstream, {1, 1, 2, 3})), {x})[0].to_vector<double>();
  const auto out = bgr.to_vector<double>();
  const float mean[] = {103.939f, 116.779f, 123.68f};
  for (int p = 0; p < 2; ++p)
    for (int c = 0; c < 3; ++c) {
      const volatile float scaled = input[3 * p + 2 - c] * 255.f;
      check::Near(out[3 * p + c], scaled - mean[c], 1e-6, "ImageNet BGR scale/mean order");
      check::Near(g[3 * p + c], 255 * upstream[3 * p + 2 - c], 1e-6, "every preprocessing input derivative");
    }
  std::cout << "  ImageNet BGR255/means, signed channel order and every RGB input derivative passed" << std::endl;
}

void GramAndContentChecks() {
  constexpr int      batches = 2, positions = 6, channels = 2;
  std::vector<float> f(24), p(24), a(8), upstream(8), permuted(24), changed(24);
  for (int i = 0; i < 24; ++i) f[i] = .15f * (i % 9 - 4), p[i] = .12f * (i % 7 - 2), changed[i] = f[i] + (i < 12 ? 1 : 0);
  for (int b = 0; b < batches; ++b)
    for (int j = 0; j < channels; ++j)
      for (int k = 0; k < channels; ++k) {
        double value = 0;
        for (int m = 0; m < positions; ++m) value += double(p[(b * positions + m) * channels + j]) * p[(b * positions + m) * channels + k];
        a[(b * 2 + j) * 2 + k]        = float(value / positions);
        upstream[(b * 2 + j) * 2 + k] = .1f * (1 + b * 4 + j * 2 + k);
      }
  for (int b = 0; b < batches; ++b)
    for (int m = 0; m < positions; ++m)
      for (int c = 0; c < channels; ++c) permuted[(b * positions + m) * channels + c] = f[(b * positions + positions - 1 - m) * channels + c];
  Tensor       x   = nn::from_vector(f, {2, 2, 3, 2}).requires_grad_();
  const Tensor A = nn::from_vector(a, {2, 2, 2}), P = nn::from_vector(p, {2, 2, 3, 2});
  const auto   gram = StyleGram(x);
  const auto   style_loss = GatysStyleLayer(x, A), content = GatysContentLoss(x, P);
  const auto   gs = nn::grad(style_loss, {x})[0].to_vector<double>(), gc = nn::grad(content, {x})[0].to_vector<double>(),
             gg = nn::grad(nn::sum(gram * nn::from_vector(upstream, {2, 2, 2})), {x})[0].to_vector<double>();
  const auto zero = GatysStyleLayer(x, StyleGram(x)).item(), self = GatysContentLoss(x, x).item(),
             perm_content = GatysContentLoss(x, nn::from_vector(permuted, {2, 2, 3, 2})).item();
  const auto gv = gram.to_vector<double>();
  double     grams[2][2][2] = {}, host_style = 0, host_content = 0, error = 0;
  for (int b = 0; b < batches; ++b)
    for (int j = 0; j < channels; ++j)
      for (int k = 0; k < channels; ++k) {
        for (int m = 0; m < positions; ++m)
          grams[b][j][k] += double(f[(b * positions + m) * channels + j]) * f[(b * positions + m) * channels + k] / positions;
        check::Near(gv[(b * 2 + j) * 2 + k], grams[b][j][k], 1e-7, "each batched normalized Gram value");
        host_style += std::pow(grams[b][j][k] - a[(b * 2 + j) * 2 + k], 2) / (4 * batches * channels * channels);
      }
  for (int b = 0; b < batches; ++b)
    for (int m = 0; m < positions; ++m)
      for (int c = 0; c < channels; ++c) {
        const int    i          = (b * positions + m) * channels + c;
        const double difference = f[i] - p[i];
        host_content += .5 * difference * difference;
        double ds = 0, dg = 0;
        for (int j = 0; j < channels; ++j) {
          ds += f[(b * positions + m) * channels + j] * (grams[b][j][c] - a[(b * 2 + j) * 2 + c]) / (batches * channels * channels * positions);
          dg += f[(b * positions + m) * channels + j] * (upstream[(b * 2 + j) * 2 + c] + upstream[(b * 2 + c) * 2 + j]) / positions;
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
  check::Near(RelativeError(StyleGram(nn::from_vector(permuted, {2, 2, 3, 2})), gram.detach()), 0, 1e-7, "Gram ignores spatial permutation");
  const auto other = StyleGram(nn::from_vector(changed, {2, 2, 3, 2})).to_vector<float>();
  for (int c = 0; c < 4; ++c) check::Near(other[4 + c], gv[4 + c], 0, "Gram never mixes different images in batch");
  std::cout << "  host Gram/content/style values and every derivative error=" << error
            << "; nonsymmetric upstream, batch independence, permutation and self-loss checks passed" << std::endl;
}

void KerasParity(const TensorMap& weights, const TensorMap& reference) {
  const FrozenVgg19         vgg(weights);
  Tensor                    x   = reference.at("input").clone().requires_grad_();
  const auto                net = vgg.Extract(x);
  std::array<Tensor, 5>     targets;
  for (int i = 0; i < 5; ++i) targets[std::size_t(i)] = reference.at("block" + std::to_string(i + 1) + "_conv1_gram");
  const Tensor content   = GatysContentLoss(net.content, reference.at("content_target"));
  const Tensor style     = GatysStyleLoss(net.style, targets);
  const Tensor objective = content * double(1e-7f) + style * double(1e-5f);
  const Tensor gradient  = nn::grad(objective, {x})[0];
  double       feature_error = RelativeError(net.content, reference.at("block4_conv2"));
  for (int i = 0; i < 5; ++i)
    feature_error = std::max(feature_error, RelativeError(net.style[std::size_t(i)], reference.at("block" + std::to_string(i + 1) + "_conv1")));
  check::Near(feature_error, 0, 3e-4, "all six full-width pretrained VGG-19 feature arrays");
  const double content_error   = RelativeError(content.reshape({}), reference.at("content_loss").reshape({}));
  const double style_error     = RelativeError(style.reshape({}), reference.at("style_loss").reshape({}));
  const double objective_error = RelativeError(objective.reshape({}), reference.at("objective").reshape({}));
  const double gradient_error  = RelativeError(gradient, reference.at("image_gradient"));
  check::Near(content_error, 0, 3e-4, "independent Keras content objective");
  check::Near(style_error, 0, 3e-4, "independent Keras Gram objective");
  check::Near(objective_error, 0, 3e-4, "independent combined objective");
  check::Near(gradient_error, 0, 5e-4, "every end-to-end image gradient vs GradientTape");
  check::Require(weights.size() == 32, "all16 pretrained layers imported");
  std::cout << "  genuine full-width ImageNet VGG-19 Keras/C++ parity (32x48): feature error=" << feature_error << ", content/style="
            << content_error << '/' << style_error << ", objective=" << objective_error << ", every-pixel gradient=" << gradient_error << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   size = 64, steps = 30;
    float                 strength    = 1;
    bool                  checks_only = false;
    std::filesystem::path assets      = paths::Dataset("style_transfer");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("size=", 0) == 0)
        size = std::stoi(arg.substr(5));
      else if (arg.rfind("steps=", 0) == 0)
        steps = std::stoi(arg.substr(6));
      else if (arg.rfind("strength=", 0) == 0)
        strength = std::stof(arg.substr(9));
      else if (arg.rfind("assets=", 0) == 0)
        assets = arg.substr(7);
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected assets=PATH size=N steps=N strength=F checks_only=0|1 threads=N");
    }
    PreprocessChecks();
    GramAndContentChecks();
    check::Require(std::filesystem::exists(assets / "vgg19.npz") && std::filesystem::exists(assets / "reference.npz"),
                   "prepared VGG-19 assets missing; run just style-assets inside nix develop");
    const auto weights = LoadTensorMap((assets / "vgg19.npz").string()), reference = LoadTensorMap((assets / "reference.npz").string());
    KerasParity(weights, reference);
    if (checks_only) {
      std::cout << "All Gatys fixtures passed (optimization not requested).\n";
      return 0;
    }
    OptimizeGatys(weights, assets, size, steps, strength, paths::OutputDir("models/gatys_style_transfer"));
    std::cout << "All Gatys neural style transfer checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
