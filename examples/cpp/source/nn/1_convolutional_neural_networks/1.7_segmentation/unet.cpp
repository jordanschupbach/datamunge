// CMD: just run-nn unet
// U-Net: crop-and-concatenate skips and learned symmetric upsampling.
// https://arxiv.org/abs/1505.04597
//
// Port of tflow's unet. Construct the original valid-convolution U-Net:
// base 64, four down/up levels, grayscale 572x572 -> 388x388, two classes,
// 31030658 trainable parameters. Check every full derivative, centre-crop
// values/derivatives, canonical intermediate sizes and compact gradients.
// The learned 2x2 transpose convolutions halve channels before concatenating
// skips.
//
// Train a compact base 8, SAME-padding, RGB/four-class network with no dropout
// on the same independent procedural shape segmentation splits as FCN:
// 600 fit/120 validation/120 test, 96 pixels, occlusion/noise, class-independent
// photometry. Preserve all four contracting/expanding levels. Use three
// epochs, balanced pixel CE, Adam(.001), batch 16, L2 .0001, best validation
// mIoU. Report test confusion and export input/truth/prediction PNG. This
// teaching dataset does not imply biomedical or Pascal VOC benchmark
// performance.
//
// TF -> datamunge: TF builds the full 572x572 model as a graph only; here the
// full model executes forward and backward on a 188x188 image (the smallest
// valid input, 4x4 output), so the parameter census and derivative count come
// from a real run; the 572 -> 388 geometry is checked with UnetOutputSide.
// Shapes use tensor RNG and geometry, so fixed-seed pixels differ from TF's
// mt19937 implementation. Weights also use datamunge's generator.
// Options: epochs=3 accuracy=0|1 (default 0) warm_start=PREFIX threads=N.
// Short mode checks learning; accuracy=1 retains the .55 test mIoU target.
// Artifacts: output/nn/models/unet_shapes/.

#include "common/unet.hpp"

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/segmentation.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

Tensor Wave(const nn::Shape& shape, double scale) {
  std::int64_t n = 1;
  for (auto d : shape)
    n *= d;
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = float(std::sin(scale * double(i)));
  return nn::from_vector(v, shape);
}

/// [min, max] gradient norm over every parameter; requires finite (and nonzero when `nonzero`).
std::pair<double, double> GradientNorms(
    const nn::ParameterStore& p,
    const Tensor&             objective,
    bool                      nonzero,
    const std::string&        what) {
  const auto params = p.parameters();
  const auto grads  = nn::grad(objective, params);
  check::Require(grads.size() == params.size(), what + ": every parameter has a derivative");
  double low = INFINITY, high = 0;
  for (std::size_t k = 0; k < grads.size(); ++k) {
    const double norm =
        std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
    check::Require(
        std::isfinite(norm) && (!nonzero || norm > 0),
        "finite nonzero " + what + " derivative: " + p.names()[k]);
    low  = std::min(low, norm);
    high = std::max(high, norm);
  }
  return {low, high};
}

void CropChecks() {
  check::Require(
      unet::UnetOutputSide(572) == 388 && unet::UnetOutputSide(96, 4, false) == 96,
      "canonical and SAME U-Net geometry");
  bool rejected = false;
  try {
    unet::UnetOutputSide(64);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check::Require(rejected, "too-small valid input rejected");
  std::vector<float> values(9 * 12 * 2);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = float(i) - 100;
  Tensor x = nn::from_vector(values, {1, 9, 12, 2});
  x.requires_grad_(true);
  const Tensor cropped  = unet::CenterCropTo(x, nn::zeros({1, 4, 7, 3}));
  const auto   gradient = nn::grad(nn::sum(cropped), {x})[0].to_vector<float>();
  check::Require(
      cropped.shape() == nn::Shape({1, 4, 7, 2}), "centre crop preserves source channels");
  const auto out = cropped.to_vector<float>();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 7; ++c)
      for (int k = 0; k < 2; ++k)
        check::Near(
            out[std::size_t((r * 7 + c) * 2 + k)],
            values[std::size_t(((r + 2) * 12 + c + 2) * 2 + k)],
            0,
            "odd difference uses floor-centred crop");
  for (int r = 0; r < 9; ++r)
    for (int c = 0; c < 12; ++c)
      for (int k = 0; k < 2; ++k)
        check::Near(
            gradient[std::size_t((r * 12 + c) * 2 + k)],
            r >= 2 && r < 6 && c >= 2 && c < 9 ? 1 : 0,
            0,
            "crop Jacobian masks exactly the retained region");
  std::cout << "  572->388 geometry, too-small rejection and signed rectangular "
               "centre-crop/Jacobian checks passed"
            << std::endl;
}

unet::UnetConfig CompactConfig() {
  unet::UnetConfig c;
  c.classes        = 4;
  c.input_channels = 3;
  c.base_width     = 8;
  c.valid          = false;
  c.dropout        = 0;
  return c;
}

void ModelChecks() {
  {
    nn::ParameterStore p(17);
    const int          side = 188;
    check::Require(unet::UnetOutputSide(side) == 4, "smallest valid full U-Net input");
    const auto net = unet::Unet(p, Wave({1, side, side, 1}, .13), {}, false);
    check::Require(p.count() == 31030658, "original grayscale/two-class U-Net parameter count");
    check::Require(net.logits.shape() == nn::Shape({1, 4, 4, 2}), "valid U-Net output geometry");
    check::Require(
        net.skips[0].shape()[1] == 184 && net.decoder.back().shape()[1] == 4,
        "valid skip/decoder sizes");
    GradientNorms(p, nn::mean(nn::square(net.logits)), false, "full U-Net");
    check::Require(p.size() == 46, "full U-Net parameter connections");
    std::cout << "  original U-Net: 31030658 parameters, 46 derivatives (executed on 188x188 -> "
                 "4x4), four cropped skips"
              << std::endl;
  }
  nn::ParameterStore p(17);
  double             low = INFINITY, high = 0;
  for (int side : {96, 112, 128}) {
    const auto net = unet::Unet(p, Wave({2, side, side, 3}, .13), CompactConfig(), true);
    check::Require(
        net.logits.shape() == nn::Shape({2, side, side, 4}), "SAME U-Net output alignment");
    if (side != 96)
      continue;
    const auto norms = GradientNorms(p, nn::mean(nn::square(net.logits)), true, "U-Net");
    low              = norms.first;
    high             = norms.second;
  }
  std::cout << "  compact U-Net: " << p.count()
            << " parameters; 96/112/128 alignment; all gradient norms [" << low << ',' << high
            << "]" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  try {
    seg::SegmentationConfig cfg;
    cfg.name     = "unet_shapes";
    cfg.epochs   = 3;
    cfg.accuracy = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        cfg.epochs = std::stoi(arg.substr(7));
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        cfg.accuracy = arg == "accuracy=1";
      else if (arg.rfind("warm_start=", 0) == 0)
        cfg.warm_start = arg.substr(11);
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument(
            "expected epochs=N, accuracy=0|1, warm_start=PREFIX or threads=N");
    }
    seg::SegmentationTrainingChecks();
    seg::SegmentationLossChecks();
    CropChecks();
    ModelChecks();
    const auto fit        = seg::SegmentationShapes(600, 96, 11),
               validation = seg::SegmentationShapes(120, 96, 22),
               test       = seg::SegmentationShapes(120, 96, 33);
    const auto builder =
        [](nn::ParameterStore& p, nn::ParameterStore&, const Tensor& images, bool training) {
          return seg::SegmentationModel{
              unet::Unet(p, images, CompactConfig(), training).logits, {}};
        };
    {
      using namespace seg;
      const auto& source_fit        = fit;
      const auto& source_validation = validation;
      const auto& source_test       = test;
      auto        training_cfg      = cfg;
      bool        explicit_epochs   = false;
      for (int option = 1; option < argc; ++option)
        explicit_epochs = explicit_epochs || std::string(argv[option]).rfind("epochs=", 0) == 0;
      if (!explicit_epochs)
        training_cfg.epochs = example_runtime::Limit(training_cfg.epochs, 1);
      training_cfg.batch = example_runtime::Limit(training_cfg.batch, 2);
      auto subset        = [](const SegmentationData& data, int count) {
        SegmentationData result;
        result.count             = std::min(data.count, count);
        result.height            = data.height;
        result.width             = data.width;
        result.classes           = data.classes;
        const std::size_t pixels = std::size_t(result.count) * data.height * data.width;
        result.images.assign(data.images.begin(), data.images.begin() + 3 * pixels);
        result.labels.assign(data.labels.begin(), data.labels.begin() + pixels);
        return result;
      };
      SegmentationData quick_fit, quick_validation, quick_test;
      if (!example_runtime::Full()) {
        quick_fit        = subset(source_fit, 8);
        quick_validation = subset(source_validation, 4);
        quick_test       = subset(source_test, 4);
      }
      const auto& fit        = example_runtime::Full() ? source_fit : quick_fit;
      const auto& validation = example_runtime::Full() ? source_validation : quick_validation;
      const auto& test       = example_runtime::Full() ? source_test : quick_test;
      example_runtime::Describe();
      check::Require(
          training_cfg.epochs > 0 && training_cfg.batch > 0 && fit.classes == validation.classes &&
              fit.classes == test.classes,
          "valid segmentation configuration");
      check::Require(
          !example_runtime::Full() || training_cfg.accuracy || training_cfg.epochs >= 2,
          "short segmentation demonstration needs at least two epochs");
      nn::ParameterStore p(training_cfg.seed), state(training_cfg.seed);
      {
        nn::NoGradGuard guard; // create every parameter (no statistics update) before the optimizer
        const auto      probe = SegmentationBatch(fit, {0});
        builder(p, state, probe.first, false);
      }
      const LogitsFn infer = [&](const nn::Tensor& images) {
        return builder(p, state, images, false).logits;
      };
      std::vector<std::int64_t> frequencies(std::size_t(fit.classes), 1);
      for (int label : fit.labels)
        ++frequencies.at(std::size_t(label));
      std::vector<float> weights(std::size_t(fit.classes));
      for (int c = 0; c < fit.classes; ++c)
        weights[std::size_t(c)] =
            1 / std::sqrt(float(frequencies[std::size_t(c)]) / float(fit.labels.size()));
      const auto              class_weights = nn::from_vector(weights, {fit.classes});
      std::vector<nn::Tensor> kernels;
      for (const auto& t : p.parameters())
        if (t.dim() > 1)
          kernels.push_back(t);
      nn::OptimizerConfig oc;
      oc.kind = nn::OptimizerKind::Adam;
      nn::Optimizer optimizer(p, training_cfg.learning_rate, oc);
      const auto    directory = SegmentationDirectory(training_cfg.name);
      const auto    prefix    = (directory / "best").string();
      std::ofstream history(directory / "history.tsv");
      history << "epoch\tloss\tvalidation_mean_iou\n";
      std::cout << "== " << training_cfg.name << ": " << fit.count << " fit / " << validation.count
                << " validation / " << test.count << " test; " << p.count() << " parameters; "
                << fit.height << 'x' << fit.width << ", classes=" << fit.classes << std::endl;
      nn::BatchOrder      batches(std::size_t(fit.count), std::size_t(training_cfg.batch), 29);
      double              best       = -1;
      int                 best_epoch = -1;
      std::vector<double> epoch_losses;
      if (!training_cfg.warm_start.empty()) {
        check::Require(
            std::filesystem::absolute(training_cfg.warm_start).lexically_normal() !=
                std::filesystem::absolute(prefix).lexically_normal(),
            "segmentation warm start requires a separate output tag");
        p.load(training_cfg.warm_start + "_params.npz");
        state.load(training_cfg.warm_start + "_state.npz");
        best       = SegmentationScore(infer, validation).mean_iou;
        best_epoch = 0;
        p.save(prefix + "_params.npz");
        state.save(prefix + "_state.npz");
        history << 0 << "\tnan\t" << best << '\n';
        history.flush();
        std::cout << "  warm start: " << training_cfg.warm_start
                  << "; fresh Adam moments; restored validation mIoU=" << best << std::endl;
      }
      double train_seconds = 0;
      for (int epoch = 1; epoch <= training_cfg.epochs; ++epoch) {
        const auto started = std::chrono::steady_clock::now();
        batches.begin_epoch();
        double total = 0;
        int    seen  = 0;
        for (;;) {
          const auto picked = batches.next();
          if (picked.empty())
            break;
          const std::vector<int> rows(picked.begin(), picked.end());
          const auto             batch = SegmentationBatch(fit, rows);
          const auto             model = builder(p, state, batch.first, true);
          nn::Tensor             loss =
              SegmentationCrossEntropy(model.logits, batch.second, fit.classes, class_weights);
          if (!model.auxiliary.empty()) {
            nn::Tensor sum = loss;
            for (const auto& output : model.auxiliary)
              sum = sum +
                    double(model.auxiliary_weight) *
                        SegmentationCrossEntropy(output, batch.second, fit.classes, class_weights);
            loss = sum / (1.0 + double(model.auxiliary_weight) * double(model.auxiliary.size()));
          }
          const double value = loss.item<double>();
          check::Require(std::isfinite(value), "finite segmentation training loss");
          nn::Tensor objective = loss;
          if (training_cfg.l2 > 0 && !kernels.empty()) {
            nn::Tensor penalty = nn::sum(nn::square(kernels[0]));
            for (std::size_t i = 1; i < kernels.size(); ++i)
              penalty = penalty + nn::sum(nn::square(kernels[i]));
            objective = objective + .5 * double(training_cfg.l2) * penalty;
          }
          optimizer.minimize(objective);
          total += value * double(rows.size());
          seen += int(rows.size());
        }
        train_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
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
        std::cout << "  epoch " << epoch << ": loss=" << total / seen
                  << ", validation mIoU=" << score.mean_iou
                  << ", pixel accuracy=" << score.pixel_accuracy << std::endl;
      }
      p.load(prefix + "_params.npz");
      state.load(prefix + "_state.npz");
      check::Near(
          SegmentationScore(infer, validation).mean_iou,
          best,
          1e-10,
          "restored validation score matches selected checkpoint");
      std::cout << "  restored validation mIoU=" << best << "; checkpoint agreement verified"
                << std::endl;
      if (training_cfg.inspect)
        training_cfg.inspect(infer, test, directory);
      const auto result = SegmentationScore(infer, test);
      std::cout << "  selected epoch=" << best_epoch << "; test mIoU=" << result.mean_iou
                << ", pixel accuracy=" << result.pixel_accuracy
                << "\n  test confusion (truth rows, predicted columns):\n";
      for (int c = 0; c < fit.classes; ++c) {
        std::cout << "   ";
        for (int k = 0; k < fit.classes; ++k)
          std::cout << ' ' << result.confusion[std::size_t(c * fit.classes + k)];
        std::cout << '\n';
      }
      SegmentationPreview(infer, test, directory / "preview.png");
      std::cout << "  checkpoint, history and image/truth/prediction preview: "
                << directory.string() << std::endl;
      std::printf(
          "  training time %.1f s (%.2f s/epoch, excluding validation)\n",
          train_seconds,
          train_seconds / training_cfg.epochs);
      if (example_runtime::Full())
        CheckSegmentationDemonstration(training_cfg, epoch_losses, result);
      else
        check::Require(
            std::isfinite(result.mean_iou) && result.mean_iou >= 0 && result.mean_iou <= 1,
            "finite quick segmentation evaluation");
      if (example_runtime::Full() && !training_cfg.accuracy)
        std::cout << "  short learning demonstration passed: mean loss " << epoch_losses.front()
                  << " -> " << epoch_losses.back() << "; accuracy=1 enables mIoU target "
                  << training_cfg.minimum_test_iou << std::endl;
    }
    std::cout << "All U-Net checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
