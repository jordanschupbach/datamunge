// CMD: just run-nn deeplab_v1_v2
// DeepLab v1/v2: atrous dense features, ASPP and dense Potts CRF refinement.
// https://arxiv.org/abs/1412.7062
// https://arxiv.org/abs/1606.00915
// https://arxiv.org/abs/1210.5644
//
// Port of tflow's deeplab_v1_v2. V1 full graph: VGG LargeFOV, outputstride8,
// conv5rate2,fc6rate12/1024, 20505429parameters. V2: ResNet101 outputstride8
// plus summed class-score ASPP rates6/12/18/24,44048532parameters. Check every
// full derivative, independent host ASPP values/input gradients and all
// compact derivatives. Verify standard ResNet stride32 geometry/counts are
// unchanged by stride16/8 options.
//
// Check fully connected Gaussian/bilateral Potts mean field against an
// independent host iteration and finite-difference unary gradients. Remove
// self edges, normalize kernels symmetrically and retain differentiability
// through five iterations. This exact O(pixels^2) reference is bounded4096
// pixels, not the original fast permutohedral implementation.
//
// Train compact V1(widths/8,fc128/rate4,no dropout) and V2(ResNet18/8,ASPP
// rates1/2/3/4) from scratch on the shared96pixel procedural shape dataset,
// 600fit/120validation/120test.3epochs,balanced CE,Adam(.001),batch16,L2 .0001,
// best-validation mIoU. Apply a fixed CRF to32pixel samples of the first four
// held-out images, report before/after mIoU and export image/truth/raw/CRF PNG.
// Full native-resolution test mIoU is reported separately. No VOC claim.
//
// TF -> datamunge: atrous convolutions use nn::conv2d's native dilation (TF
// needed expanded "portable" kernels for CPU gradients). TF builds the full
// models as graphs only; here they execute forward/backward on a 33x33 image
// (same parameters), so counts and derivatives come from a real run. The
// procedural shapes are bit-identical to TF's; weights come from datamunge's
// generator, so trained numbers match TF statistically.
// Options: model=v1|v2 (default both) epochs=3 accuracy=0|1 (default0) threads=N.
// Short mode verifies learning; accuracy=1 retains the .55 mIoU target.
// Artifacts: output/nn/models/deeplab_v{1,2}_shapes/.

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/deeplab.hpp"
#include "common/dense_crf.hpp"
#include "common/segmentation.hpp"

// Experiment workflow from examples/nn/common/segmentation.hpp.
namespace seg {
using namespace ::seg;
inline SegmentationMetrics TrainSegmentation(const SegmentationConfig& requested_cfg, const SegmentationData& requested_fit,
                                             const SegmentationData& requested_validation, const SegmentationData& requested_test,
                                             const SegmentationBuilder& builder) {
  auto cfg = requested_cfg;
  cfg.epochs = example_runtime::Limit(cfg.epochs, 1);
  cfg.batch = example_runtime::Limit(cfg.batch, 2);
  auto subset = [](const SegmentationData& data, int count) {
    SegmentationData result;
    result.count = std::min(data.count, count);
    result.height = data.height; result.width = data.width; result.classes = data.classes;
    const std::size_t pixels = std::size_t(result.count) * data.height * data.width;
    result.images.assign(data.images.begin(), data.images.begin() + 3 * pixels);
    result.labels.assign(data.labels.begin(), data.labels.begin() + pixels);
    return result;
  };
  SegmentationData quick_fit, quick_validation, quick_test;
  if (!example_runtime::Full()) {
    quick_fit = subset(requested_fit, 8);
    quick_validation = subset(requested_validation, 4);
    quick_test = subset(requested_test, 4);
  }
  const auto& fit = example_runtime::Full() ? requested_fit : quick_fit;
  const auto& validation = example_runtime::Full() ? requested_validation : quick_validation;
  const auto& test = example_runtime::Full() ? requested_test : quick_test;
  example_runtime::Describe();
  check::Require(cfg.epochs > 0 && cfg.batch > 0 && fit.classes == validation.classes && fit.classes == test.classes,
                 "valid segmentation configuration");
  check::Require(!example_runtime::Full() || cfg.accuracy || cfg.epochs >= 2, "short segmentation demonstration needs at least two epochs");
  nn::ParameterStore p(cfg.seed), state(cfg.seed);
  {
    nn::NoGradGuard guard;  // create every parameter (no statistics update) before the optimizer
    const auto      probe = SegmentationBatch(fit, {0});
    builder(p, state, probe.first, false);
  }
  const LogitsFn infer = [&](const nn::Tensor& images) { return builder(p, state, images, false).logits; };
  std::vector<std::int64_t> frequencies(std::size_t(fit.classes), 1);
  for (int label : fit.labels) ++frequencies.at(std::size_t(label));
  std::vector<float> weights(std::size_t(fit.classes));
  for (int c = 0; c < fit.classes; ++c)
    weights[std::size_t(c)] = 1 / std::sqrt(float(frequencies[std::size_t(c)]) / float(fit.labels.size()));
  const auto class_weights = nn::from_vector(weights, {fit.classes});
  std::vector<nn::Tensor> kernels;
  for (const auto& t : p.parameters())
    if (t.dim() > 1) kernels.push_back(t);
  nn::OptimizerConfig oc;
  oc.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(p, cfg.learning_rate, oc);
  const auto    directory = SegmentationDirectory(cfg.name);
  const auto    prefix    = (directory / "best").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\tloss\tvalidation_mean_iou\n";
  std::cout << "== " << cfg.name << ": " << fit.count << " fit / " << validation.count << " validation / " << test.count
            << " test; " << p.count() << " parameters; " << fit.height << 'x' << fit.width << ", classes=" << fit.classes
            << std::endl;
  nn::BatchOrder      batches(std::size_t(fit.count), std::size_t(cfg.batch), 29);
  double              best       = -1;
  int                 best_epoch = -1;
  std::vector<double> epoch_losses;
  if (!cfg.warm_start.empty()) {
    check::Require(std::filesystem::absolute(cfg.warm_start).lexically_normal() !=
                       std::filesystem::absolute(prefix).lexically_normal(),
                   "segmentation warm start requires a separate output tag");
    p.load(cfg.warm_start + "_params.npz");
    state.load(cfg.warm_start + "_state.npz");
    best       = SegmentationScore(infer, validation).mean_iou;
    best_epoch = 0;
    p.save(prefix + "_params.npz");
    state.save(prefix + "_state.npz");
    history << 0 << "\tnan\t" << best << '\n';
    history.flush();
    std::cout << "  warm start: " << cfg.warm_start << "; fresh Adam moments; restored validation mIoU=" << best
              << std::endl;
  }
  double train_seconds = 0;
  for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
    const auto started = std::chrono::steady_clock::now();
    batches.begin_epoch();
    double total = 0;
    int    seen  = 0;
    for (;;) {
      const auto picked = batches.next();
      if (picked.empty()) break;
      const std::vector<int> rows(picked.begin(), picked.end());
      const auto             batch = SegmentationBatch(fit, rows);
      const auto             model = builder(p, state, batch.first, true);
      nn::Tensor loss = SegmentationCrossEntropy(model.logits, batch.second, fit.classes, class_weights);
      if (!model.auxiliary.empty()) {
        nn::Tensor sum = loss;
        for (const auto& output : model.auxiliary)
          sum = sum + double(model.auxiliary_weight) *
                          SegmentationCrossEntropy(output, batch.second, fit.classes, class_weights);
        loss = sum / (1.0 + double(model.auxiliary_weight) * double(model.auxiliary.size()));
      }
      const double value = loss.item<double>();
      check::Require(std::isfinite(value), "finite segmentation training loss");
      nn::Tensor objective = loss;
      if (cfg.l2 > 0 && !kernels.empty()) {
        nn::Tensor penalty = nn::sum(nn::square(kernels[0]));
        for (std::size_t i = 1; i < kernels.size(); ++i) penalty = penalty + nn::sum(nn::square(kernels[i]));
        objective = objective + .5 * double(cfg.l2) * penalty;
      }
      optimizer.minimize(objective);
      total += value * double(rows.size());
      seen += int(rows.size());
    }
    train_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    epoch_losses.push_back(total / seen);
    check::Require(std::isfinite(epoch_losses.back()), "finite segmentation epoch mean");
    const auto score = SegmentationScore(infer, validation);
    check::Require(std::isfinite(score.mean_iou), "finite validation mIoU");
    if (score.mean_iou > best) {
      best       = score.mean_iou;
      best_epoch = epoch;
      p.save(prefix + "_params.npz");
      state.save(prefix + "_state.npz");
    }
    history << epoch << '\t' << total / seen << '\t' << score.mean_iou << '\n';
    history.flush();
    std::cout << "  epoch " << epoch << ": loss=" << total / seen << ", validation mIoU=" << score.mean_iou
              << ", pixel accuracy=" << score.pixel_accuracy << std::endl;
  }
  p.load(prefix + "_params.npz");
  state.load(prefix + "_state.npz");
  check::Near(SegmentationScore(infer, validation).mean_iou, best, 1e-10,
              "restored validation score matches selected checkpoint");
  std::cout << "  restored validation mIoU=" << best << "; checkpoint agreement verified" << std::endl;
  if (cfg.inspect) cfg.inspect(infer, test, directory);
  const auto result = SegmentationScore(infer, test);
  std::cout << "  selected epoch=" << best_epoch << "; test mIoU=" << result.mean_iou
            << ", pixel accuracy=" << result.pixel_accuracy << "\n  test confusion (truth rows, predicted columns):\n";
  for (int c = 0; c < fit.classes; ++c) {
    std::cout << "   ";
    for (int k = 0; k < fit.classes; ++k) std::cout << ' ' << result.confusion[std::size_t(c * fit.classes + k)];
    std::cout << '\n';
  }
  SegmentationPreview(infer, test, directory / "preview.png");
  std::cout << "  checkpoint, history and image/truth/prediction preview: " << directory.string() << std::endl;
  std::printf("  training time %.1f s (%.2f s/epoch, excluding validation)\n", train_seconds, train_seconds / cfg.epochs);
  if (example_runtime::Full()) CheckSegmentationDemonstration(cfg, epoch_losses, result);
  else check::Require(std::isfinite(result.mean_iou) && result.mean_iou >= 0 && result.mean_iou <= 1,
                      "finite quick segmentation evaluation");
  if (example_runtime::Full() && !cfg.accuracy)
    std::cout << "  short learning demonstration passed: mean loss " << epoch_losses.front() << " -> "
              << epoch_losses.back() << "; accuracy=1 enables mIoU target " << cfg.minimum_test_iou << std::endl;
  return result;
}
}  // namespace seg



namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

std::vector<float> Wave(std::size_t n, const std::function<float(int)>& f) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = f(int(i));
  return v;
}

std::pair<double, double> GradientNorms(const nn::ParameterStore& p, const Tensor& objective, bool nonzero,
                                        const std::string& what) {
  const auto params = p.parameters();
  const auto grads  = nn::grad(objective, params);
  double     low = INFINITY, high = 0;
  for (std::size_t k = 0; k < grads.size(); ++k) {
    const double norm = std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
    check::Require(std::isfinite(norm) && (!nonzero || norm > 0), "finite nonzero " + what + " derivative: " + p.names()[k]);
    low  = std::min(low, norm);
    high = std::max(high, norm);
  }
  return {low, high};
}

void AspChecks() {
  nn::ParameterStore       p(17);
  const std::array<int, 4> rates = {1, 2, 3, 4};
  const auto value   = Wave(13 * 15, [](int i) { return std::sin(.13f * i) - .3f; });
  const auto weights = Wave(13 * 15, [](int i) { return std::cos(.21f * i); });
  Tensor     image   = nn::from_vector(value, {1, 13, 15, 1});
  image.requires_grad_(true);
  {
    nn::NoGradGuard guard;
    deeplab::AtrousScorePyramid(p, "aspp", image, 1, rates);
  }
  for (int i = 0; i < 4; ++i) {
    const auto name = "aspp/rate" + std::to_string(rates[std::size_t(i)]);
    p.set(name + "/W", nn::full({3, 3, 1, 1}, .1f * (i + 1)));
    p.set(name + "/b", nn::full({1}, .01f * (i + 1)));
  }
  const auto branches = deeplab::AtrousScorePyramid(p, "aspp", image, 1, rates);
  Tensor     output   = branches[0];
  for (int i = 1; i < 4; ++i) output = output + branches[std::size_t(i)];
  const auto gradient = nn::grad(nn::sum(output * nn::from_vector(weights, {1, 13, 15, 1})), {image})[0].to_vector<float>();
  const auto out      = output.to_vector<float>();
  std::vector<double> expected(13 * 15), derivative(13 * 15);
  for (int r = 0; r < 13; ++r)
    for (int c = 0; c < 15; ++c)
      for (int branch = 0; branch < 4; ++branch) {
        expected[std::size_t(r * 15 + c)] += .01f * (branch + 1);
        const double coefficient = .1f * (branch + 1);
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int y = r + dy * rates[std::size_t(branch)], x = c + dx * rates[std::size_t(branch)];
            if (y < 0 || x < 0 || y >= 13 || x >= 15) continue;
            expected[std::size_t(r * 15 + c)] += coefficient * value[std::size_t(y * 15 + x)];
            derivative[std::size_t(y * 15 + x)] += coefficient * weights[std::size_t(r * 15 + c)];
          }
      }
  double error = 0;
  for (int i = 0; i < 13 * 15; ++i) {
    error = std::max(error, std::abs(out[std::size_t(i)] - expected[std::size_t(i)]));
    error = std::max(error, std::abs(gradient[std::size_t(i)] - derivative[std::size_t(i)]));
  }
  check::Near(error, 0, 3e-6, "independent four-rate atrous score sum and all input derivatives");
  std::cout << "  ASPP rates1/2/3/4 sum and input derivatives match host taps; error=" << error << std::endl;
}

std::vector<double> CrfReference(const std::vector<double>& logits, const std::vector<float>& kernel, int iterations) {
  auto softmax = [](const std::vector<double>& x) {
    std::vector<double> p(x.size());
    for (std::size_t i = 0; i < 3; ++i) {
      const double maximum = std::max(x[2 * i], x[2 * i + 1]), a = std::exp(x[2 * i] - maximum),
                   b = std::exp(x[2 * i + 1] - maximum);
      p[2 * i]     = a / (a + b);
      p[2 * i + 1] = b / (a + b);
    }
    return p;
  };
  auto q = softmax(logits);
  for (int step = 0; step < iterations; ++step) {
    auto next = logits;
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t c = 0; c < 2; ++c) next[2 * i + c] += kernel[i * 3 + j] * q[2 * j + c];
    q = softmax(next);
  }
  return q;
}

void CrfChecks() {
  std::vector<float> pixels(9);
  for (int i = 0; i < 3; ++i)
    for (int c = 0; c < 3; ++c) pixels[std::size_t(i * 3 + c)] = i == 2 ? .9f : .1f;
  seg::DenseCrfConfig cfg;
  cfg.spatial_weight    = .1f;
  cfg.appearance_weight = 4;
  cfg.appearance_sigma  = 2;
  const auto kernel     = seg::DenseCrfKernel(nn::from_vector(pixels, {1, 3, 3}), cfg);
  const auto k          = kernel.to_vector<float>();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const float v = k[std::size_t(i * 3 + j)];
      check::Require(std::isfinite(v) && v >= 0, "nonnegative finite CRF kernel");
      check::Near(v, k[std::size_t(j * 3 + i)], 0, "symmetric normalization");
      if (i == j) check::Near(v, 0, 0, "no self interaction");
    }
  const std::vector<float> unary = {-3, 3, .2f, 0, 3, -3}, weights = {.1f, .9f, .3f, -.4f, .8f, -.2f};
  Tensor                   logits = nn::from_vector(unary, {3, 2});
  logits.requires_grad_(true);
  const Tensor result   = seg::DenseCrfMeanField(logits, kernel, 5);
  const auto   gradient = nn::grad(nn::sum(result * nn::from_vector(weights, {3, 2})), {logits})[0].to_vector<float>();
  const auto   out      = result.to_vector<float>();
  std::vector<double> raw(unary.begin(), unary.end());
  const auto          reference = CrfReference(raw, k, 5);
  auto objective = [&](const std::vector<double>& values) {
    const auto probabilities = CrfReference(values, k, 5);
    double     total         = 0;
    for (std::size_t i = 0; i < 6; ++i) total += probabilities[i] * weights[i];
    return total;
  };
  double error = 0, gradient_error = 0;
  for (std::size_t i = 0; i < 6; ++i) {
    error = std::max(error, std::abs(out[i] - reference[i]));
    const double original = raw[i], step = 1e-4;
    raw[i]                = original + step;
    const double plus     = objective(raw);
    raw[i]                = original - step;
    const double minus    = objective(raw);
    raw[i]                = original;
    gradient_error        = std::max(gradient_error, std::abs(gradient[i] - (plus - minus) / (2 * step)));
  }
  check::Near(error, 0, 2e-7, "independent host CRF updates");
  check::Near(gradient_error, 0, 2e-6, "CRF input central differences");
  check::Require(out[1] > out[0] && out[3] > out[2] && out[4] > out[5],
                 "bilateral affinity corrects ambiguous same-color middle pixel and preserves color boundary");
  std::cout << "  Gaussian/bilateral Potts mean field matches host; forward error=" << error
            << ", unary derivative error=" << gradient_error << std::endl;
}

deeplab::DeepLabOutputs Build(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& images, bool v2,
                              bool compact, bool training) {
  if (v2) {
    deeplab::DeepLabV2Config c;
    if (compact) {
      c.classes        = 4;
      c.backbone_depth = 18;
      c.width_divisor  = 8;
      c.rates          = {1, 2, 3, 4};
    }
    return deeplab::DeepLabV2(p, state, images, c, training);
  }
  deeplab::DeepLabV1Config c;
  if (compact) {
    c.classes          = 4;
    c.width_divisor    = 8;
    c.fc_width         = 128;
    c.fc_rate          = 4;
    c.keep_probability = 1;
  }
  return deeplab::DeepLabV1(p, images, c, training);
}

void BackboneChecks() {
  for (int stride : {32, 16, 8}) {
    nn::ParameterStore   p(17), state(18);
    resnet::ResNetConfig c;
    c.depth              = 18;
    c.width_divisor      = 8;
    c.include_classifier = false;
    const auto net       = deeplab::DenseResNet(p, state, nn::full({1, 96, 96, 3}, .5), c, stride, true);
    check::Require(p.count() == 176712 && state.count() == 1200,
                   "output-stride change preserves basic ResNet parameters and default counts");
    for (int stage = 0; stage < 4; ++stage) {
      const std::int64_t side = 96 / std::min(stride, 4 << stage);
      check::Require(net.stages[std::size_t(stage)].shape()[1] == side && net.stages[std::size_t(stage)].shape()[2] == side,
                     "unchanged standard strides or requested dense stages");
    }
  }
  std::cout << "  default stride32 and atrous stride16/8 backbones preserve parameter/state counts and expected stage geometry"
            << std::endl;
}

void ModelChecks() {
  for (bool v2 : {false, true}) {
    nn::ParameterStore p(17), state(18);
    const auto net = Build(p, state, nn::from_vector(Wave(33 * 33 * 3, [](int i) { return std::sin(.13f * i); }), {1, 33, 33, 3}),
                           v2, false, true);
    check::Require(p.count() == (v2 ? 44048532 : 20505429) && state.count() == (v2 ? 105344 : 0),
                   "full DeepLab trainable/state counts");
    GradientNorms(p, nn::mean(nn::square(net.logits)), false, "full DeepLab");
    check::Require(p.size() == std::size_t(v2 ? 320 : 32), "every full DeepLab parameter connected");
    std::cout << "  full DeepLab v" << (v2 ? 2 : 1) << ": " << p.count() << " parameters, " << p.size()
              << " derivatives (executed on a 33x33 image)" << std::endl;
  }
  for (bool v2 : {false, true}) {
    nn::ParameterStore p(17), state(18);
    const auto net = Build(p, state, nn::from_vector(Wave(2 * 96 * 96 * 3, [](int i) { return std::sin(.13f * i); }), {2, 96, 96, 3}),
                           v2, true, true);
    check::Require(p.count() == (v2 ? 185944 : 321452), "compact DeepLab parameter counts");
    check::Require(net.logits.shape() == nn::Shape({2, 96, 96, 4}) && net.features.shape()[1] == 12 &&
                       net.features.shape()[2] == 12,
                   "stride-eight features and resized logits");
    const auto norms = GradientNorms(p, nn::mean(nn::square(net.logits)), true, "DeepLab");
    std::cout << "  compact DeepLab v" << (v2 ? 2 : 1) << ": " << p.count() << " parameters; all derivative norms ["
              << norms.first << ',' << norms.second << "]" << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int               epochs   = 3;
    bool              accuracy = false;
    std::vector<bool> models   = {false, true};
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        accuracy = arg == "accuracy=1";
      else if (arg == "model=v1")
        models = {false};
      else if (arg == "model=v2")
        models = {true};
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N, model=v1|v2, accuracy=0|1 or threads=N");
    }
    seg::SegmentationTrainingChecks();
    seg::SegmentationLossChecks();
    AspChecks();
    CrfChecks();
    BackboneChecks();
    ModelChecks();
    const auto fit = seg::SegmentationShapes(600, 96, 11), validation = seg::SegmentationShapes(120, 96, 22),
               test = seg::SegmentationShapes(120, 96, 33);
    for (bool v2 : models) {
      seg::SegmentationConfig cfg;
      cfg.name           = v2 ? "deeplab_v2_shapes" : "deeplab_v1_shapes";
      cfg.epochs         = epochs;
      cfg.accuracy       = accuracy;
      cfg.inspect        = seg::DenseCrfPreview;
      const auto builder = [v2](nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& images, bool training) {
        return seg::SegmentationModel{Build(p, state, images, v2, true, training).logits, {}};
      };
      seg::TrainSegmentation(cfg, fit, validation, test, builder);
    }
    std::cout << "All DeepLab v1/v2 and CRF checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
