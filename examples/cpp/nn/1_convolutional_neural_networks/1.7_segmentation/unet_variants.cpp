// CMD: just run-nn unet_variants
// UNet++ dense nested skip paths and additive attention U-Net.
// https://arxiv.org/abs/1807.10165
// https://arxiv.org/abs/1804.03999
//
// Port of tflow's unet_variants. Check additive gate values and analytic
// skip/gate input derivatives with positive/negative projections and
// saturated sigmoid. Build full base 32 UNet++ (fifteen nodes, four
// deep-supervision heads) and base 64 attention U-Net (four additive gates);
// verify all parameter connections. Check every compact derivative and dense
// concatenation widths. Nested upsampling is bilinear with aligned corners and
// actual skip size; attention U-Net uses learned 2x2 transposed convolution
// then skip gating. Convolution pairs/gates omit BN in this adaptation.
//
// Train both base 8, four-level, four-class compact models on procedural shape
// scenes, 600 fit/120 validation/120 test, 96 pixels, occlusion/noise and
// identical photometry across classes. UNet++ averages balanced CE over four
// heads; evaluation uses the deepest. Use three epochs, Adam(.001), batch 16,
// L2 .0001 and best-validation mIoU. Report confusion and colored
// input/truth/prediction PNGs. No biomedical benchmark or pretrained-weight
// claim is made.
//
// TF -> datamunge: TF builds the full 256x256 models as graphs only; here they
// execute forward and backward on 32x32 images (same parameter census and
// derivative count). The shapes data are bit-identical to TF's; weights come
// from datamunge's generator, so trained numbers match TF statistically.
// Options: epochs=3 model=nested|attention (default both) accuracy=0|1 threads=N
// seed=N (parameter initialization, default 17) tag=NAME (output directory suffix).
// Default accuracy=0 checks learning; accuracy=1 retains the .55 mIoU target.
// Artifacts: output/nn/models/{nested,attention}_unet_shapes/.

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/segmentation.hpp"
#include "common/unet_variants.hpp"

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

Tensor Wave(const nn::Shape& shape, double scale) {
  std::int64_t n = 1;
  for (auto d : shape) n *= d;
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(float(scale) * float(i));
  return nn::from_vector(v, shape);
}

std::pair<double, double> GradientNorms(const nn::ParameterStore& p, const Tensor& objective, bool nonzero,
                                        const std::string& what) {
  const auto params = p.parameters();
  const auto grads  = nn::grad(objective, params);
  check::Require(grads.size() == params.size(), what + ": every parameter has a derivative");
  double low = INFINITY, high = 0;
  for (std::size_t k = 0; k < grads.size(); ++k) {
    const double norm = std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
    check::Require(std::isfinite(norm) && (!nonzero || norm > 0), "finite nonzero " + what + " derivative: " + p.names()[k]);
    low  = std::min(low, norm);
    high = std::max(high, norm);
  }
  return {low, high};
}

void GateChecks() {
  nn::ParameterStore p(17);
  p.set("gate/x/W", nn::from_vector(std::vector<float>{.5f, -.2f}, {1, 1, 2, 1}));
  p.set("gate/g/W", nn::from_vector(std::vector<float>{.3f, .4f}, {1, 1, 2, 1}));
  p.set("gate/x/b", nn::from_vector(std::vector<float>{.1f}, {1}));
  p.set("gate/g/b", nn::from_vector(std::vector<float>{-.1f}, {1}));
  p.set("gate/psi/W", nn::from_vector(std::vector<float>{.7f}, {1, 1, 1, 1}));
  p.set("gate/psi/b", nn::from_vector(std::vector<float>{-.2f}, {1}));
  std::vector<float> a(12), b(12);
  for (int i = 0; i < 6; ++i) {
    a[std::size_t(2 * i)]     = float(.2 + .1 * i);
    a[std::size_t(2 * i + 1)] = float(-.3 + .15 * i);
    b[std::size_t(2 * i)]     = float(-.2 + .05 * i);
    b[std::size_t(2 * i + 1)] = i ? float(.4 - .02 * i) : -2.f;
  }
  auto run = [&](std::vector<float>* coefficients, std::vector<float>* output) {
    Tensor x = nn::from_vector(a, {1, 2, 3, 2}), g = nn::from_vector(b, {1, 2, 3, 2});
    x.requires_grad_(true);
    g.requires_grad_(true);
    const auto net       = unet::AttentionGate(p, "gate", x, g, 1);
    const auto gradients = nn::grad(nn::sum(net.output), {x, g});
    if (coefficients) *coefficients = net.coefficients.to_vector<float>();
    if (output) *output = net.output.to_vector<float>();
    return std::make_pair(gradients[0].to_vector<float>(), gradients[1].to_vector<float>());
  };
  std::vector<float> coefficients, output;
  const auto         grads = run(&coefficients, &output);
  double             error = 0;
  const double       wx[] = {.5, -.2}, wg[] = {.3, .4};
  for (int i = 0; i < 6; ++i) {
    const double z = .5 * a[std::size_t(2 * i)] - .2 * a[std::size_t(2 * i + 1)] + .3 * b[std::size_t(2 * i)] +
                     .4 * b[std::size_t(2 * i + 1)];
    const double coefficient = 1 / (1 + std::exp(-(.7 * std::max(z, 0.) - .2)));
    const double sum         = a[std::size_t(2 * i)] + a[std::size_t(2 * i + 1)];
    const double chain       = z > 0 ? sum * coefficient * (1 - coefficient) * .7 : 0;
    error = std::max(error, std::abs(coefficients[std::size_t(i)] - coefficient));
    for (int c = 0; c < 2; ++c) {
      const std::size_t k = std::size_t(2 * i + c);
      error               = std::max(error, std::abs(output[k] - a[k] * coefficient));
      error               = std::max(error, std::abs(grads.first[k] - (coefficient + chain * wx[c])));
      error               = std::max(error, std::abs(grads.second[k] - chain * wg[c]));
    }
  }
  check::Near(error, 0, 2e-7, "independent additive-gate forward and chain rule");
  p.set("gate/psi/b", nn::from_vector(std::vector<float>{100.f}, {1}));
  const auto saturated = run(nullptr, nullptr);
  for (int i = 0; i < 12; ++i) {
    check::Near(saturated.first[std::size_t(i)], 1, 0, "saturated open gate identity derivative");
    check::Near(saturated.second[std::size_t(i)], 0, 0, "saturated gate query derivative");
  }
  std::cout << "  additive attention values/input derivatives, negative ReLU and saturated sigmoid passed; error=" << error
            << std::endl;
}

seg::SegmentationModel Build(nn::ParameterStore& p, const Tensor& images, bool nested, bool compact) {
  if (nested) {
    unet::NestedUnetConfig c;
    c.classes    = compact ? 4 : 2;
    c.base_width = compact ? 8 : 32;
    const auto net = unet::NestedUnet(p, images, c);
    return {net.logits, std::vector<Tensor>(net.heads.begin(), net.heads.end() - 1)};
  }
  unet::AttentionUnetConfig c;
  c.classes    = compact ? 4 : 2;
  c.base_width = compact ? 8 : 64;
  return {unet::AttentionUnet(p, images, c).logits, {}};
}

Tensor Objective(const seg::SegmentationModel& net) {
  Tensor sum = nn::mean(nn::square(net.logits));
  for (const auto& output : net.auxiliary) sum = sum + nn::mean(nn::square(output));
  return sum;
}

void ModelChecks() {
  for (bool nested : {true, false}) {
    nn::ParameterStore p(17);
    const auto         net = Build(p, Wave({1, 32, 32, 3}, .13), nested, false);
    GradientNorms(p, Objective(net), false, "full U-Net variant");
    check::Require(p.count() == (nested ? 9156264 : 31381414), "full no-BN nested/attention U-Net parameter counts");
    check::Require(p.size() == (nested ? 68u : 70u), "full U-Net variant parameter connections (TF: 68/70)");
    if (nested)
      check::Require(p.at("x0_4/conv1/W").shape()[2] == 192 && net.auxiliary.size() == 3,
                     "deepest dense node concatenates all four prior nodes and two-width lower feature");
    std::cout << "  full " << (nested ? "UNet++" : "attention U-Net") << ": " << p.count() << " parameters, all "
              << p.size() << " derivatives (executed on 32x32)" << std::endl;
  }
  for (bool nested : {true, false}) {
    nn::ParameterStore p(17);
    const auto         net = Build(p, Wave({2, 96, 96, 3}, .13), nested, true);
    check::Require(net.logits.shape() == nn::Shape({2, 96, 96, 4}), "full-resolution variant logits");
    const auto norms = GradientNorms(p, Objective(net), true, "variant");
    std::cout << "  compact " << (nested ? "UNet++" : "attention U-Net") << ": " << p.count()
              << " parameters, every derivative norm [" << norms.first << ',' << norms.second << "]" << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int               epochs   = 3;
    bool              accuracy = false;
    std::vector<bool> models   = {true, false};
    std::uint64_t     seed     = 17;
    std::string       tag;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        accuracy = arg == "accuracy=1";
      else if (arg == "model=nested")
        models = {true};
      else if (arg == "model=attention")
        models = {false};
      else if (arg.rfind("seed=", 0) == 0)
        seed = std::stoull(arg.substr(5));
      else if (arg.rfind("tag=", 0) == 0)
        tag = "_" + arg.substr(4);
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N, model=nested|attention, accuracy=0|1, seed=N, tag=NAME or threads=N");
    }
    seg::SegmentationTrainingChecks();
    seg::SegmentationLossChecks();
    GateChecks();
    ModelChecks();
    const auto fit = seg::SegmentationShapes(600, 96, 11), validation = seg::SegmentationShapes(120, 96, 22),
               test = seg::SegmentationShapes(120, 96, 33);
    for (bool nested : models) {
      seg::SegmentationConfig cfg;
      cfg.name           = (nested ? "nested_unet_shapes" : "attention_unet_shapes") + tag;
      cfg.seed           = seed;
      cfg.epochs         = epochs;
      cfg.accuracy       = accuracy;
      const auto builder = [nested](nn::ParameterStore& p, nn::ParameterStore&, const Tensor& images, bool) {
        return Build(p, images, nested, true);
      };
      seg::TrainSegmentation(cfg, fit, validation, test, builder);
    }
    std::cout << "All UNet++ and attention U-Net checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
