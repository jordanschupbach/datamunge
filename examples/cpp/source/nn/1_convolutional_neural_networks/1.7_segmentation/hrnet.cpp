// CMD: just run-nn hrnet
// HRNetV2: persistent high-resolution branches and repeated exchange fusion.
// https://arxiv.org/abs/1908.07919
// https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py
//
// Port of tflow's hrnet. W32: two stride2 stem convolutions,4 bottlenecks/64,
// then2/3/4 parallel streams of width32/64/128/256. Stage module counts1/4/3,
// each4basic blocks per branch,all-to-all sum/ReLU fusion. Upsampling projects
// with1x1 and aligned bilinear resize; downsampling uses3x3/stride2 chains.
// Final streams concatenate at stride4,1x1/480 BN/ReLU and linear classes21,
// then resize. Full-model derivatives, controlled identity-kernel odd-grid
// exchange against independent host interpolation/subsampling and all pixel
// gradients; compact forward/backward on96/97px checks every branch and
// parameter derivative.
//
// Compact W8,stem16,stage1width16/two bottlenecks,two basic blocks per branch,
// one module per stage. Train600fit/120validation/120test procedural96px shape
// scenes for3epochs,Adam(.001),batch16,balanced CE,L2 .0001. Select validation
// mIoU,restore checkpoint,report test confusion and image/truth/prediction PNG.
// Scratch teaching data; no pretrained medical/Cityscapes claims.
//
// TF -> datamunge: TF builds the full W32 model as a graph only; here it
// executes forward/backward on a 33x33 image (same parameters). The shapes
// use tensor RNG/geometry,so pixels differ from TF's fixed-seed mt19937
// implementation. Weights also use datamunge's generator.
// Options: epochs=3 accuracy=0|1 (default0; accuracy=1 retains .55 mIoU) threads=N.
// Artifacts: output/nn/models/hrnet_shapes/.

#include "common/hrnet.hpp"

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

void FusionChecks() {
  nn::ParameterStore     p(17), state(18);
  resnet::ConvNormConfig norm;
  norm.scale   = true;
  norm.epsilon = 1e-5;
  std::vector<float> h(35), l(12);
  for (int i = 0; i < 35; ++i)
    h[std::size_t(i)] = 1 + .03f * i;
  for (int i = 0; i < 12; ++i)
    l[std::size_t(i)] = .5f + .07f * i;
  Tensor high = nn::from_vector(h, {1, 5, 7, 1}), low = nn::from_vector(l, {1, 3, 4, 1});
  {
    nn::NoGradGuard guard;
    hrnet::HRFuse(p, state, "fixture", {high, low}, {1, 1}, norm);
  }
  for (const auto& name : p.names()) {
    const auto         shape = p.at(name).shape();
    std::vector<float> value(std::size_t(nn::shape_numel(shape)), 0.f);
    if (name.size() >= 2 && name.substr(name.size() - 2) == "/W")
      value[value.size() / 2] = 1;
    else if (name.find("/gamma") != std::string::npos)
      std::fill(value.begin(), value.end(), std::sqrt(1.f + float(norm.epsilon)));
    p.set(name, nn::from_vector(value, shape));
  }
  high.requires_grad_(true);
  low.requires_grad_(true);
  const auto          fused    = hrnet::HRFuse(p, state, "fixture", {high, low}, {1, 1}, norm);
  const auto          gradient = nn::grad(nn::sum(fused[0]) + nn::sum(fused[1]), {high, low});
  const auto          o0 = fused[0].to_vector<float>(), o1 = fused[1].to_vector<float>();
  const auto          g0 = gradient[0].to_vector<float>(), g1 = gradient[1].to_vector<float>();
  std::vector<double> gh(35, 1), gl(12, 1);
  double              error = 0;
  for (int y = 0; y < 5; ++y)
    for (int x = 0; x < 7; ++x) {
      const double sy = y * .5, sx = x * .5;
      const int    y0 = int(sy), x0 = int(sx), y1 = std::min(y0 + 1, 2), x1 = std::min(x0 + 1, 3);
      double       up = 0;
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
          const int    index  = (dy ? y1 : y0) * 4 + (dx ? x1 : x0);
          const double weight = (dy ? sy - y0 : 1 - (sy - y0)) * (dx ? sx - x0 : 1 - (sx - x0));
          up += weight * l[std::size_t(index)];
          gl[std::size_t(index)] += weight;
        }
      error =
          std::max(error, std::abs(o0[std::size_t(y * 7 + x)] - h[std::size_t(y * 7 + x)] - up));
    }
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 4; ++x) {
      const int index = y * 14 + x * 2;
      gh[std::size_t(index)] += 1;
      error = std::max(
          error,
          std::abs(
              double(
                  o1[std::size_t(y * 4 + x)] - l[std::size_t(y * 4 + x)] - h[std::size_t(index)])));
    }
  for (int i = 0; i < 35; ++i)
    error = std::max(error, std::abs(gh[std::size_t(i)] - g0[std::size_t(i)]));
  for (int i = 0; i < 12; ++i)
    error = std::max(error, std::abs(gl[std::size_t(i)] - g1[std::size_t(i)]));
  check::Near(error, 0, 2e-6, "every high/low exchange value and input gradient");
  std::cout << "  odd-grid high/low fusion independent bilinear/subsample and gradient error="
            << error << std::endl;
}

hrnet::HRNetConfig Config(bool compact) {
  hrnet::HRNetConfig c;
  if (compact) {
    c.classes       = 4;
    c.width         = 8;
    c.stem_width    = 16;
    c.stage1_width  = 16;
    c.stage1_blocks = 2;
    c.blocks        = 2;
    c.modules       = {1, 1, 1};
  }
  return c;
}

std::vector<float> Sine(std::size_t n) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = std::sin(.13f * float(i));
  return v;
}

void ModelChecks() {
  {
    nn::ParameterStore p(17), state(18);
    const auto         net = hrnet::HRNet(
        p, state, nn::from_vector(Sine(33 * 33 * 3), {1, 33, 33, 3}), Config(false), true);
    const auto params = p.parameters();
    const auto grads  = nn::grad(nn::mean(nn::square(net.logits)), params);
    for (std::size_t k = 0; k < grads.size(); ++k)
      check::Require(
          std::isfinite(nn::sum(nn::square(grads[k])).item<double>()),
          "finite full HRNet derivative: " + p.names()[k]);
    check::Require(
        net.fusion_modules == 8 && net.branches.size() == 4 && grads.size() == p.size(),
        "full W32 eight exchange modules/four branches/all parameters");
    check::Require(
        p.at("head/project/W").shape() == nn::Shape({1, 1, 480, 480}),
        "full four streams concatenate480channels");
    std::cout << "  full HRNetV2 W32: " << p.count() << " parameters, " << state.count()
              << " BN state, " << grads.size() << " gradients (executed on a 33x33 image)"
              << std::endl;
  }
  for (int size : {96, 97}) {
    nn::ParameterStore p(17), state(18);
    const auto         net = hrnet::HRNet(
        p,
        state,
        nn::from_vector(Sine(std::size_t(2 * size * size * 3)), {2, size, size, 3}),
        Config(true),
        true);
    check::Require(
        net.logits.shape() == nn::Shape({2, size, size, 4}),
        "HRNet logits restore odd/even input grid");
    std::int64_t resolution = (size + 3) / 4;
    for (int k = 0; k < 4; ++k) {
      const auto& shape = net.branches[std::size_t(k)].shape();
      check::Require(
          shape[1] == resolution && shape[2] == resolution && shape[3] == 8 * (1 << k),
          "HRNet persistent branch dimensions");
      resolution = (resolution + 1) / 2;
    }
    const auto params = p.parameters();
    const auto grads  = nn::grad(nn::mean(nn::square(net.logits)), params);
    double     low = INFINITY, high = 0;
    for (std::size_t k = 0; k < grads.size(); ++k) {
      const double norm =
          std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
      check::Require(
          std::isfinite(norm) && norm > 0, "finite nonzero HRNet derivative: " + p.names()[k]);
      low  = std::min(low, norm);
      high = std::max(high, norm);
    }
    std::cout << "  compact HRNet " << size << "px: " << p.count()
              << " parameters; all gradient norms [" << low << ',' << high << "]" << std::endl;
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    int  epochs   = 3;
    bool accuracy = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        accuracy = arg == "accuracy=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N, accuracy=0|1 or threads=N");
    }
    seg::SegmentationTrainingChecks();
    seg::SegmentationLossChecks();
    FusionChecks();
    ModelChecks();
    const auto fit        = seg::SegmentationShapes(600, 96, 11),
               validation = seg::SegmentationShapes(120, 96, 22),
               test       = seg::SegmentationShapes(120, 96, 33);
    seg::SegmentationConfig cfg;
    cfg.name     = "hrnet_shapes";
    cfg.epochs   = epochs;
    cfg.accuracy = accuracy;
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
        [](nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& images, bool training) {
          return seg::SegmentationModel{
              hrnet::HRNet(p, state, images, Config(true), training).logits, {}};
        }(p, state, probe.first, false);
      }
      const LogitsFn infer = [&](const nn::Tensor& images) {
        return [](nn::ParameterStore& p,
                  nn::ParameterStore& state,
                  const Tensor&       images,
                  bool                training) {
          return seg::SegmentationModel{
              hrnet::HRNet(p, state, images, Config(true), training).logits, {}};
        }(p, state, images, false)
                   .logits;
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
          const auto             model = [](nn::ParameterStore& p,
                                nn::ParameterStore& state,
                                const Tensor&       images,
                                bool                training) {
            return seg::SegmentationModel{
                hrnet::HRNet(p, state, images, Config(true), training).logits, {}};
          }(p, state, batch.first, true);
          nn::Tensor loss =
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
    std::cout << "All HRNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
