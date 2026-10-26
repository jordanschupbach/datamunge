// CMD: just run-nn deeplab_v3_plus
// DeepLab v3/v3+: image-pooling ASPP and low-level separable decoding.
// https://arxiv.org/abs/1706.05587
// https://arxiv.org/abs/1802.02611
//
// Port of tflow's deeplab_v3_plus. Use the ResNet50 backbone variant for full
// graphs, outputstride16. V3 has 1x1/three atrous/global-image branches,
// concat projection and .1dropout; v3+ replaces atrous branches with
// separable blocks, projects C2 to48channels and applies two separable decoder
// convolutions at stride4. Count39048533 / 26696133parameters,56192/70720BN
// state and179/203 derivatives. Check the image-level branch's spatially
// constant response and all input derivatives against independent means with
// controlled projection/BN. Check every compact derivative, all five branches
// and decoder stride-four features.
//
// Compact ResNet18/8,ASPP32/rates1/2/3,decoder32/low8,no dropout. Train both
// models on600fit/120validation/120test procedural shape images,96pixels,
// four classes.20epochs,balanced CE,Adam(.001),batch16,L2 .0001,best validation
// mIoU. Report confusion and input/truth/prediction PNGs. No pretrained weights
// or VOC/biomedical benchmark claims.
//
// TF -> datamunge: atrous and separable atrous convolutions use native
// nn::conv2d / nn::depthwise_conv2d dilation (TF expands the kernels). TF
// builds the full models as graphs only; here they execute forward/backward on
// a 33x33 image (same parameters). The shapes are bit-identical to TF's;
// weights come from datamunge's generator, so trained numbers match TF
// statistically. As in TF, training keeps the default .55 test-mIoU gate.
// Options: model=v3|plus (default both) epochs=20 seed=17 (parameter initialization) threads=N.
// Artifacts: output/nn/models/deeplab_v3{,_plus}_shapes/.

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/deeplab_v3.hpp"
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

std::vector<float> Wave(std::size_t n, double a, double b, double scale) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = float(a + b * std::sin(scale * double(i)));
  return v;
}

void ImagePoolingChecks() {
  nn::ParameterStore      p(17), state(18);
  deeplab::ConvNormConfig cfg;
  cfg.scale   = true;
  cfg.epsilon = 1e-5;
  const auto value = Wave(2 * 7 * 9 * 2, .7, .4, .13);
  Tensor     input = nn::from_vector(value, {2, 7, 9, 2});
  {
    nn::NoGradGuard guard;
    deeplab::ImagePoolBranch(p, state, "pool", input, 2, cfg);
  }
  p.set("pool/project/W", nn::from_vector(std::vector<float>{1, 0, 0, 1}, {1, 1, 2, 2}));
  p.set("pool/project/bn/gamma", nn::full({2}, std::sqrt(1.f + float(cfg.epsilon))));
  input.requires_grad_(true);
  const Tensor output   = deeplab::ImagePoolBranch(p, state, "pool", input, 2, cfg);
  const auto   gradient = nn::grad(nn::sum(output), {input})[0].to_vector<float>();
  const auto   out      = output.to_vector<float>();
  double       error    = 0;
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 2; ++c) {
      double mean = 0;
      for (int i = 0; i < 63; ++i) mean += value[std::size_t((n * 63 + i) * 2 + c)] / 63.;
      for (int i = 0; i < 63; ++i) {
        const auto index = std::size_t((n * 63 + i) * 2 + c);
        error            = std::max(error, std::abs(out[index] - mean));
        error            = std::max(error, std::abs(gradient[index] - 1.));
      }
    }
  check::Near(error, 0, 5e-6,
              "image-pooling projection broadcasts independent global means and accumulates all pixel derivatives");
  std::cout << "  image-level ASPP branch mean, constant broadcast and every input derivative passed; error=" << error << std::endl;
}

deeplab::DeepLabV3Config Config(bool plus, bool compact) {
  deeplab::DeepLabV3Config c;
  c.decoder = plus;
  if (compact) {
    c.classes          = 4;
    c.backbone_depth   = 18;
    c.width_divisor    = 8;
    c.head_width       = 32;
    c.decoder_width    = 32;
    c.low_width        = 8;
    c.rates            = {1, 2, 3};
    c.keep_probability = 1;
  }
  return c;
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

void ModelChecks() {
  for (bool plus : {false, true}) {
    nn::ParameterStore p(17), state(18);
    const auto net = deeplab::DeepLabV3(p, state, nn::from_vector(Wave(33 * 33 * 3, 0, 1, .13), {1, 33, 33, 3}),
                                        Config(plus, false), true);
    check::Require(p.count() == (plus ? 26696133 : 39048533) && state.count() == (plus ? 70720 : 56192),
                   "full ResNet50 DeepLab v3/plus parameter/state counts");
    GradientNorms(p, nn::mean(nn::square(net.logits)), false, "full DeepLab v3");
    check::Require(p.size() == std::size_t(plus ? 203 : 179) && net.branches.size() == 5,
                   "all five branches and decoder parameters connected");
    std::cout << "  full DeepLab " << (plus ? "v3+" : "v3") << ": " << p.count() << " parameters, " << p.size()
              << " derivatives (executed on a 33x33 image)" << std::endl;
  }
  for (bool plus : {false, true}) {
    nn::ParameterStore p(17), state(18);
    const auto net = deeplab::DeepLabV3(p, state, nn::from_vector(Wave(3 * 96 * 96 * 3, 0, 1, .13), {3, 96, 96, 3}),
                                        Config(plus, true), true);
    std::cout << "  compact parameter/state counts:" << p.count() << '/' << state.count() << std::endl;
    check::Require(p.count() == (plus ? 198004 : 241740) && state.count() == (plus ? 2256 : 1584),
                   "compact v3/plus parameter/state counts");
    check::Require(p.at("aspp/project/W").shape()[2] == 160, "five ASPP branches concatenate before projection");
    check::Require(net.logits.shape() == nn::Shape({3, 96, 96, 4}) && net.features.shape()[1] == 6 &&
                       (plus ? net.low : net.coarse).shape()[1] == (plus ? 24 : 6),
                   "stride16 features, optional stride4 decoder and full logits");
    const auto norms = GradientNorms(p, nn::mean(nn::square(net.logits)), true, "DeepLab v3");
    std::cout << "  compact " << (plus ? "v3+" : "v3") << ": " << p.count() << " parameters; every derivative norm ["
              << norms.first << ',' << norms.second << "]" << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int               epochs = 20;
    std::uint64_t     seed   = 17;
    std::vector<bool> models = {false, true};
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg == "model=v3")
        models = {false};
      else if (arg == "model=plus")
        models = {true};
      else if (arg.rfind("seed=", 0) == 0)
        seed = std::stoull(arg.substr(5));
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N, model=v3|plus, seed=N or threads=N");
    }
    seg::SegmentationLossChecks();
    ImagePoolingChecks();
    ModelChecks();
    const auto fit = seg::SegmentationShapes(600, 96, 11), validation = seg::SegmentationShapes(120, 96, 22),
               test = seg::SegmentationShapes(120, 96, 33);
    for (bool plus : models) {
      seg::SegmentationConfig cfg;
      cfg.name   = plus ? "deeplab_v3_plus_shapes" : "deeplab_v3_shapes";
      cfg.epochs = epochs;
      cfg.seed   = seed;
      seg::TrainSegmentation(cfg, fit, validation, test,
                             [plus](nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& images, bool training) {
                               return seg::SegmentationModel{
                                   deeplab::DeepLabV3(p, state, images, Config(plus, true), training).logits, {}};
                             });
    }
    std::cout << "All DeepLab v3/v3+ checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
