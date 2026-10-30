// CMD: just run-nn centernet
// CenterNet: Objects as Points with Gaussian heatmaps and learned upsampling.
//
// Port of tflow's centernet. Preparation (in ~/projects/tflow_examples):
// dm::datasets::fetch("mnist").
// Short default: three training epochs; use epochs=N for longer accuracy experiments.
// https://arxiv.org/abs/1904.07850
//
// Check the authors' Gaussian radius and clipped, maximum-combined kernels;
// analytic modified focal/L1 values and derivatives; empty targets and extreme
// logits; centre-cell collision handling; and classwise peak decoding without
// box NMS. Construct the full ResNet-18/256-128-64 upsampling model and all
// parameter derivatives. Check every compact derivative is finite and nonzero.
//
// Train a compact ResNet-18 (1/8 widths), learned 64/32/32 upsampling stages
// and 32-channel prediction heads on 128-pixel grayscale digit scenes. Predict
// stride-four heatmaps, height/width and subpixel dy/dx. Use Gaussian focal
// heatmaps, .1*size L1 and offset L1, normalized by retained object centres.
// Cell collisions retain the larger object. Decode the best 100 classwise
// 3x3 local maxima, without box NMS. Use 5500 fit / 500 validation / 1000 test
// scenes, Adam(.001), batch32, kernel L2 .0001, three epochs and best
// validation AP50. This is training from scratch, without pretrained weights.
// Options: data=PATH size=128 epochs=3 warm_start=PREFIX lr=.001 accuracy=0|1 threads=N.
//
// Digit cells scale with input size (32..64 at the 128-pixel default).
// size=64 reproduces the initial smaller-input geometry. Optional
// warm_start=PREFIX restores parameters/BN state (.npz) with fresh Adam moments.
// Structural/numerical fixtures retain their original input sizes, except the
// full ResNet-18 model: tflow builds its 512-pixel graph without running it;
// this eager port runs it forward and backward at 128 pixels (parameter
// counts and gradient connectivity do not depend on the input size).
//
// Default accuracy=0 verifies finite, decreasing loss; accuracy=1 enforces
// the original test AP50 target. Report measured metrics in either mode.
// Weights come from nn's generator, so trained numbers match TF statistically.
// Checkpoint (.npz) and history: output/nn/models/centernet_digits/.

#include "common/centernet.hpp"

#include <iostream>

namespace nn = datamunge::nn;
using centernet::Grid;

namespace {
void GaussianChecks() {
  check::Near(
      centernet::CenterGaussianRadius(10, 10),
      (std::sqrt(1120.) - 28) / 2,
      1e-12,
      "authors' Gaussian radius formula");
  Grid heat(1, 7, 7, 1);
  centernet::DrawCenterGaussian(heat, 0, 3, 3, 0, 2);
  check::Near(heat.at(0, 3, 3, 0), 1, 0, "Gaussian centre");
  check::Near(heat.at(0, 3, 4, 0), std::exp(-.72), 1e-7, "Gaussian sigma diameter/6");
  check::Near(heat.at(0, 1, 1, 0), std::exp(-5.76), 1e-8, "Gaussian corner");
  check::Near(heat.at(0, 0, 3, 0), 0, 0, "finite kernel extent");
  centernet::DrawCenterGaussian(heat, 0, 0, 0, 0, 2);
  centernet::DrawCenterGaussian(heat, 0, 3, 4, 0, 2);
  check::Near(heat.at(0, 0, 0, 0), 1, 0, "border-clipped Gaussian");
  check::Near(heat.at(0, 3, 3, 0), 1, 0, "overlaps combine by maximum");
  std::cout
      << "  Gaussian radius, sigma, finite support, border clipping and max combination passed"
      << std::endl;
}

void LossChecks() {
  std::vector<float> logits(3, 0.f), heat = {1, .5f, 0}, mask = {1, 0, 0};
  std::vector<float> size = {.2f, 4, 2, 3, 4, 5}, offset = {.1f, .8f, 2, 3, 4, 5},
                     target_size(6, 0.f), target_offset(6, 0.f);
  target_size[0] = 1, target_size[1] = 3, target_offset[0] = .3f, target_offset[1] = .5f;
  std::vector<std::vector<float>> out;
  auto                            run = [&] {
    auto       l = nn::from_vector(logits, {1, 1, 3, 1}).requires_grad_();
    auto       s = nn::from_vector(size, {1, 1, 3, 2}).requires_grad_();
    auto       o = nn::from_vector(offset, {1, 1, 3, 2}).requires_grad_();
    const auto loss = centernet::CenterLoss(
        l,
        s,
        o,
        nn::from_vector(heat, {1, 1, 3, 1}),
        nn::from_vector(target_size, {1, 1, 3, 2}),
        nn::from_vector(target_offset, {1, 1, 3, 2}),
        nn::from_vector(mask, {1, 1, 3, 1}));
    const auto g = nn::grad(loss, {l, s, o});
    out          = {
        loss.to_vector<float>(),
        g[0].to_vector<float>(),
        g[1].to_vector<float>(),
        g[2].to_vector<float>()};
  };
  run();
  check::Near(
      out[0][0],
      .515625 * std::log(2.) + .68,
      2e-6,
      "modified focal plus weighted size and offset L1");
  const double a          = .125 + .25 * std::log(2.);
  const double expected[] = {-a, a / 16, a};
  for (int i = 0; i < 3; ++i)
    check::Near(
        out[1][std::size_t(i)], expected[i], 2e-6, "Gaussian shoulder scales negative derivative");
  for (int i = 0; i < 6; ++i) {
    check::Near(
        out[2][std::size_t(i)],
        i == 0   ? -.1
        : i == 1 ? .1
                 : 0,
        2e-6,
        "size gradient only at centres");
    check::Near(
        out[3][std::size_t(i)],
        i == 0   ? -1
        : i == 1 ? 1
                 : 0,
        2e-6,
        "offset gradient only at centres");
  }
  std::fill(heat.begin(), heat.end(), 0.f);
  std::fill(mask.begin(), mask.end(), 0.f);
  run();
  check::Near(out[0][0], .75 * std::log(2.), 2e-6, "empty images retain background heatmap loss");
  for (int k : {2, 3})
    for (int i = 0; i < 6; ++i)
      check::Near(out[std::size_t(k)][std::size_t(i)], 0, 0, "empty image regression gradient");
  logits = {80, -80, -80};
  heat   = {1, 0, 0};
  run();
  check::Require(
      std::isfinite(out[0][0]) && out[0][0] < 1e-10, "extreme correct logits remain finite");
  for (int i = 0; i < 3; ++i)
    check::Require(std::isfinite(out[1][std::size_t(i)]), "extreme logits have finite derivatives");
  std::cout << "  modified focal/size/offset analytic values and gradients, Gaussian shoulders, "
               "empty targets and extreme "
               "logits passed"
            << std::endl;
}

void TargetAndDecodeChecks() {
  detector::DetectionScenes scene;
  scene.height = 64;
  scene.width  = 64;
  scene.boxes  = {{{8, 12, 24, 28}, {9, 13, 25, 29}, {7, 11, 27, 31}}};
  scene.labels = {{1, 2, 3}};
  scene.pixels.resize(4096);
  auto t = centernet::MakeCenterTargets(scene, {0}, 4);
  check::Require(t.collisions == 2, "colliding centre cells counted");
  check::Near(t.heatmap.at(0, 4, 5, 3), 1, 0, "largest colliding object's class retained");
  check::Near(t.heatmap.at(0, 4, 5, 1), 0, 0, "discarded target draws no Gaussian");
  double retained = 0;
  for (float v : t.mask.v)
    retained += v;
  check::Near(retained, 1, 0, "one retained centre");
  for (int k = 0; k < 2; ++k) {
    check::Near(t.size.at(0, 4, 5, k), 5, 0, "size in stride-four cells");
    check::Near(t.offset.at(0, 4, 5, k), .25, 0, "fractional centre offsets");
  }
  Grid logits(1, 16, 16, 4), sizes(1, 16, 16, 2), offsets(1, 16, 16, 2);
  std::fill(logits.v.begin(), logits.v.end(), -20.f);
  logits.at(0, 4, 5, 3) = std::log(9.f);
  logits.at(0, 3, 5, 3) = std::log(4.f);
  logits.at(0, 4, 5, 1) = std::log(.7f / .3f);
  for (int k = 0; k < 2; ++k) {
    sizes.at(0, 4, 5, k)   = 5;
    offsets.at(0, 4, 5, k) = .25;
  }
  const auto detections = centernet::DecodeCenter(
      centernet::CenterPeaks(logits.tensor()), sizes.tensor(), offsets.tensor(), scene, {0});
  check::Require(
      detections.size() == 2 && detections[0].label == 3 && detections[1].label == 1,
      "classwise local maxima suppress neighbour and preserve overlapping classes");
  check::Near(detections[0].score, .9, 1e-6, "decoded peak score");
  for (int k = 0; k < 4; ++k)
    check::Near(
        detections[0].box[std::size_t(k)],
        scene.boxes[0][2][std::size_t(k)],
        1e-6,
        "centre-size-offset box round trip");
  std::cout << "  centre collision retention and classwise peak/box decoding without box NMS passed"
            << std::endl;
}

centernet::CenterNetConfig CompactConfig() {
  centernet::CenterNetConfig c;
  c.classes         = 10;
  c.width_divisor   = 8;
  c.head_width      = 32;
  c.upsample_widths = {64, 32, 32};
  c.grayscale       = true;
  return c;
}
nn::Tensor Objective(const centernet::CenterNetOutputs& n) {
  return nn::mean(nn::square(n.heatmap)) +
         (nn::mean(nn::square(n.size)) + nn::mean(nn::square(n.offset)));
}
nn::Tensor SinImages(std::int64_t n, std::int64_t size, std::int64_t channels) {
  std::vector<float> v(std::size_t(n * size * size * channels));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = std::sin(.13f * float(i));
  return nn::from_vector(v, {n, size, size, channels});
}

void ModelChecks() {
  {
    nn::ParameterStore         p(17), state(17);
    centernet::CenterNetConfig full;
    full.update_stats = false;
    const auto net    = centernet::CenterNet(p, state, SinImages(1, 128, 3), full, true);
    check::Require(
        p.count() == 14046164 && state.count() == 10496,
        "full ResNet-18 CenterNet parameter/state counts");
    const auto gradients = nn::grad(Objective(net), p.parameters());
    check::Require(gradients.size() == 81, "full CenterNet parameter tensors connected");
    std::cout << "  full ResNet-18 CenterNet: 14046164 parameters, all 81 parameter derivatives "
                 "computed (128-pixel input)"
              << std::endl;
  }
  nn::ParameterStore p(17), state(17);
  auto               cfg = CompactConfig();
  cfg.update_stats       = false;
  const auto net         = centernet::CenterNet(p, state, SinImages(2, 64, 1), cfg, true);
  check::Require(
      p.count() == 319862 && state.count() == 1456, "compact CenterNet parameter/state counts");
  const auto       gradients  = nn::grad(Objective(net), p.parameters());
  const int        channels[] = {32, 10, 2, 2};
  const nn::Tensor outputs[]  = {net.features, net.heatmap, net.size, net.offset};
  for (int k = 0; k < 4; ++k)
    check::Require(
        outputs[k].shape()[1] == 16 && outputs[k].shape()[2] == 16 &&
            outputs[k].shape()[3] == channels[k],
        "stride-four learned upsampling/head shapes");
  const auto names   = p.names();
  double     minimum = INFINITY, maximum = 0;
  for (std::size_t k = 0; k < gradients.size(); ++k) {
    const double norm = std::sqrt(nn::sum(nn::square(gradients[k])).item<float>());
    check::Require(
        std::isfinite(norm) && norm > 0, "finite nonzero CenterNet derivative: " + names[k]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  std::cout << "  compact CenterNet: 319862 parameters; stride-four heads, all derivative norms ["
            << minimum << ',' << maximum << "]" << std::endl;
}
} // namespace

int main(int argc, char** argv) {
  try {
    int                              scene_size = 128;
    std::filesystem::path            directory  = paths::Dataset("mnist");
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "centernet_digits";
    cfg.epochs            = 3;
    cfg.accuracy          = false;
    cfg.minimum_test_ap50 = .7;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        eq  = arg.find('=');
      if (eq == std::string::npos)
        throw std::invalid_argument("expected key=value option");
      const auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
      if (key == "size") {
        scene_size = std::stoi(value);
        if (scene_size < 64 || scene_size % 32)
          throw std::invalid_argument("size must be at least64 and divisible by32");
      } else if (key == "data")
        directory = value;
      else if (key == "epochs")
        cfg.epochs = std::stoi(value);
      else if (key == "accuracy" && (value == "0" || value == "1"))
        cfg.accuracy = value == "1";
      else if (key == "warm_start")
        cfg.warm_start = value;
      else if (key == "lr")
        cfg.learning_rate = std::stof(value);
      else if (key == "threads")
        nn::set_num_threads(std::stoi(value));
      else
        throw std::invalid_argument("unknown option " + key);
    }
    GaussianChecks();
    LossChecks();
    TargetAndDecodeChecks();
    ModelChecks();
    const auto [train, test]      = detector::StandardScenes(directory, scene_size);
    const auto              model = CompactConfig();
    detector::DetectorHooks hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore& state) {
      centernet::CenterNet(p, state, nn::zeros({1, scene_size, scene_size, 1}), model, false);
    };
    hooks.loss = [&](nn::ParameterStore&              p,
                     nn::ParameterStore&              state,
                     const detector::DetectionScenes& scenes,
                     const std::vector<int>&          rows) {
      const auto net =
          centernet::CenterNet(p, state, detector::SceneImages(scenes, rows), model, true);
      return centernet::CenterLoss(net, centernet::MakeCenterTargets(scenes, rows, 10));
    };
    hooks.detect = [&](nn::ParameterStore&              p,
                       nn::ParameterStore&              state,
                       const detector::DetectionScenes& scenes,
                       const std::vector<int>&          rows) {
      return centernet::DetectCenter(p, state, model, scenes, rows);
    };
    {
      using namespace detector;
      auto training_cfg    = cfg;
      bool explicit_epochs = false;
      for (int option = 1; option < argc; ++option)
        explicit_epochs = explicit_epochs || std::string(argv[option]).rfind("epochs=", 0) == 0;
      if (!explicit_epochs)
        training_cfg.epochs = example_runtime::Limit(training_cfg.epochs, 1);
      training_cfg.batch      = example_runtime::Limit(training_cfg.batch, 2);
      training_cfg.validation = example_runtime::Limit(training_cfg.validation, 8);
      if (!example_runtime::Full())
        training_cfg.fit = training_cfg.fit ? std::min(training_cfg.fit, 8) : 8;
      example_runtime::Describe();
      check::Require(
          training_cfg.validation > 0 && training_cfg.validation < train.size() &&
              training_cfg.epochs > 0 && training_cfg.batch > 0,
          "valid detector training configuration");
      check::Require(
          !example_runtime::Full() || training_cfg.accuracy || training_cfg.epochs >= 2,
          "short learning demonstration needs at least two epochs");
      const int        fit_count = training_cfg.fit > 0
                                       ? std::min(training_cfg.fit, train.size() - training_cfg.validation)
                                       : train.size() - training_cfg.validation;
      std::vector<int> fit(static_cast<std::size_t>(fit_count)),
          validation(static_cast<std::size_t>(training_cfg.validation)),
          test_rows(example_runtime::Limit<std::size_t>(test.size(), 8));
      std::iota(fit.begin(), fit.end(), 0);
      std::iota(validation.begin(), validation.end(), train.size() - training_cfg.validation);
      std::iota(test_rows.begin(), test_rows.end(), 0);
      nn::ParameterStore params(17), state(17);
      {
        nn::NoGradGuard guard;
        hooks.build(params, state);
      }
      std::vector<nn::Tensor> kernels;
      for (const auto& p : params.parameters())
        if (p.dim() >= 2)
          kernels.push_back(p);
      const auto directory = TrainingDirectory(training_cfg.name);
      std::filesystem::create_directories(directory);
      const auto prefix = (directory / "best").string();
      if (!training_cfg.warm_start.empty()) {
        check::Require(
            std::filesystem::absolute(training_cfg.warm_start).lexically_normal() !=
                std::filesystem::absolute(prefix).lexically_normal(),
            "warm start requires a separate output directory/run tag");
        RestoreCheckpoint(params, state, training_cfg.warm_start);
        std::cout << "  warm start: " << training_cfg.warm_start << "; fresh Adam moments"
                  << std::endl;
      }
      nn::OptimizerConfig optimizer_config;
      optimizer_config.kind = nn::OptimizerKind::Adam;
      nn::Optimizer optimizer(params, training_cfg.learning_rate, optimizer_config);
      std::ofstream history(directory / "history.tsv");
      history << "epoch\ttraining_loss\tvalidation_ap50\tvalidation_coco_ap\n";
      std::cout << "== " << training_cfg.name << ": " << fit_count << " fit / "
                << training_cfg.validation << " validation / " << test_rows.size()
                << " test scenes (" << train.height << "x" << train.width
                << ")\n  parameters=" << params.count() << "; batch=" << training_cfg.batch
                << "; lr=" << training_cfg.learning_rate << std::endl;
      nn::BatchOrder      batches(std::size_t(fit_count), std::size_t(training_cfg.batch), 29);
      double              best       = -1;
      int                 best_epoch = -1;
      std::vector<double> epoch_losses;
      if (!training_cfg.warm_start.empty()) {
        const auto baseline = EvaluateDetections(
            hooks.detect(params, state, train, validation),
            train,
            validation,
            training_cfg.classes);
        best       = baseline.ap50;
        best_epoch = 0;
        SaveCheckpoint(params, state, prefix);
        history << 0 << "\tnan\t" << baseline.ap50 << '\t' << baseline.coco << '\n';
        history.flush();
        std::cout << "  restored baseline validation " << ApSummary(baseline) << std::endl;
      }
      for (int epoch = 1; epoch <= training_cfg.epochs; ++epoch) {
        batches.begin_epoch();
        double      total = 0;
        std::size_t seen  = 0;
        for (;;) {
          const auto picked = batches.next();
          if (picked.empty())
            break;
          std::vector<int> rows;
          for (auto k : picked)
            rows.push_back(fit[std::size_t(k)]);
          nn::Tensor   objective  = hooks.loss(params, state, train, rows);
          const double batch_loss = objective.item();
          check::Require(std::isfinite(batch_loss), "finite detector batch loss");
          if (training_cfg.l2 > 0 && !kernels.empty()) {
            nn::Tensor penalty = nn::sum(nn::square(kernels[0]));
            for (std::size_t i = 1; i < kernels.size(); ++i)
              penalty = penalty + nn::sum(nn::square(kernels[i]));
            objective = objective + 0.5 * training_cfg.l2 * penalty;
          }
          optimizer.minimize(objective);
          total += batch_loss * double(rows.size());
          seen += rows.size();
        }
        epoch_losses.push_back(total / double(seen));
        check::Require(std::isfinite(epoch_losses.back()), "finite detector epoch mean");
        const auto result = EvaluateDetections(
            hooks.detect(params, state, train, validation),
            train,
            validation,
            training_cfg.classes);
        check::Require(std::isfinite(result.ap50), "finite validation AP50");
        if (result.ap50 > best) {
          best       = result.ap50;
          best_epoch = epoch;
          SaveCheckpoint(params, state, prefix);
        }
        history << epoch << '\t' << total / double(seen) << '\t' << result.ap50 << '\t'
                << result.coco << '\n';
        history.flush();
        std::cout << "  epoch " << epoch << ": training loss=" << total / double(seen)
                  << ", validation " << ApSummary(result) << std::endl;
      }
      RestoreCheckpoint(params, state, prefix);
      const auto restored = EvaluateDetections(
          hooks.detect(params, state, train, validation), train, validation, training_cfg.classes);
      check::Near(
          restored.ap50, best, 1e-6, "restored checkpoint reproduces selected validation AP50");
      std::cout << "  restored validation AP50=" << restored.ap50
                << "; checkpoint agreement verified" << std::endl;
      const auto result = EvaluateDetections(
          hooks.detect(params, state, test, test_rows), test, test_rows, training_cfg.classes);
      std::cout << "  selected epoch=" << best_epoch << "; test " << ApSummary(result)
                << "\n  per-class test AP50:";
      for (double ap : result.per_class50)
        std::cout << ' ' << std::fixed << std::setprecision(3) << ap;
      std::cout << std::defaultfloat << std::setprecision(6)
                << "\n  best checkpoint and history: " << directory.string() << std::endl;
      if (hooks.inspect)
        hooks.inspect(params, state, test, test_rows, directory);
      if (example_runtime::Full())
        CheckDetectorDemonstration(training_cfg, epoch_losses, result);
      else
        check::Require(std::isfinite(result.ap50), "finite quick detector evaluation");
      if (example_runtime::Full() && !training_cfg.accuracy)
        std::cout << "  short learning demonstration passed: mean loss " << epoch_losses.front()
                  << " -> " << epoch_losses.back()
                  << "; accuracy benchmark not requested (accuracy=1 enables AP50 gate "
                  << training_cfg.minimum_test_ap50 << ")" << std::endl;
    }
    std::cout << "All CenterNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
