// CMD: just run-nn yolo_v3
// YOLO v3: Darknet-53, three prediction scales and independent logistic classes.
//
// Port of tflow's yolo_v3. Preparation (in ~/projects/tflow_examples):
// dm::datasets::fetch("mnist").
// Short default: three training epochs; use epochs=N for longer accuracy experiments.
//
// Check independent multi-label BCE and coordinate derivatives, background
// and ignored anchors, including zero and extreme logits. Verify global
// prior assignment across all three heads assigns each target exactly once,
// while overlapping non-responsible predictions are ignored. Construct the
// full 75-convolution COCO model (61,949,149 parameters) and VOC model
// (61,626,049), and all their parameter derivatives. Numerically check every
// compact parameter gradient and dynamic 64/96/128-pixel grids.
// https://arxiv.org/abs/1804.02767
// https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
//
// Train the complete topology with 1/8 widths on grayscale digit scenes.
// Cluster nine priors from fitting boxes only; use three per scale, select
// one prior globally per object, logistic objectness/classes and relative
// area-weighted coordinate SSE. Cycle 128/160/192-pixel training resolutions
// every ten batches. Use 5500 fit / 500 validation / 1000 test scenes,
// three epochs of Adam(.001), batch 32 and kernel L2 .0001. Restore the
// validation AP50 checkpoint before evaluating separate test scenes.
// No pretrained weights are used. Digit cells scale with input size (32..64
// at the 128-pixel default); size=64 reproduces the initial smaller-input
// geometry. Optional warm_start=PREFIX restores parameters/BN state (.npz)
// with fresh Adam moments. Default accuracy=0 verifies finite, decreasing
// loss; accuracy=1 enforces the original test AP50 target (.7).
// Options: data=PATH size=128 epochs=3 accuracy=0|1 warm_start=PREFIX lr=.001 threads=N.
//
// Eager port (common/yolo_v3.hpp). tflow builds the 416-pixel full models as
// graphs only; here their forward and backward passes run at 128 pixels.
// Each training step runs one training-mode forward pass and takes the
// matching targets from its detached outputs (tflow runs a separate detached
// pass with the same weights and batch statistics). Weights come from nn's
// generator, so trained numbers match TF statistically.
// Output: output/nn/models/yolo_v3_digits/.

#include "common/yolo_v3.hpp"

#include <iostream>

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;
using yolo::Dimension;

namespace {
nn::Tensor SinImages(std::int64_t n, std::int64_t size, std::int64_t channels) {
  std::vector<float> v(std::size_t(n * size * size * channels));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = std::sin(.13f * float(i));
  return nn::from_vector(v, {n, size, size, channels});
}
yolo::YoloV3Targets EmptyTargets(int classes) {
  yolo::YoloV3Targets t;
  for (auto& g : t.levels) {
    g.geometry.assign(12, 0);
    g.classes.assign(std::size_t(3 * classes), 0);
    g.positive.assign(3, 0);
    g.objectness_weight.assign(3, 0);
    g.geometry_weight.assign(3, 0);
  }
  return t;
}

void LossChecks() {
  auto  t       = EmptyTargets(2);
  auto& v       = t.levels[0];
  v.geometry[0] = v.geometry[1] = .5f;
  v.classes[0] = v.classes[1] = v.positive[0] = v.objectness_weight[0] = v.geometry_weight[0] = 1;
  auto run = [&](double expected_loss, const std::vector<double>& expected_gradient) {
    std::array<nn::Tensor, 3> raw;
    for (auto& r : raw)
      r = nn::zeros({1, 1, 1, 3, 7}).requires_grad_();
    const auto loss      = yolo::YoloV3Loss(raw, t, 2);
    const auto gradients = nn::grad(loss, {raw[0], raw[1], raw[2]});
    check::Near(loss.item(), expected_loss, 2e-6, "independent YOLO v3 loss");
    for (std::size_t level = 0; level < 3; ++level) {
      const auto g = gradients[level].to_vector<float>();
      for (std::size_t i = 0; i < 21; ++i)
        check::Near(
            g[i],
            level == 0 && i < 7 ? expected_gradient[i] : 0,
            2e-6,
            "analytic YOLO v3 gradient");
    }
  };
  run(3 * std::log(2.), {0, 0, 0, 0, -.5, -.5, -.5});
  v.geometry[0]        = .25f;
  v.geometry[1]        = .75f;
  v.geometry[2]        = .2f;
  v.geometry[3]        = -.3f;
  v.geometry_weight[0] = 2;
  run(3 * std::log(2.) + .255, {.125, -.125, -.4, .6, -.5, -.5, -.5});
  v.positive[0] = v.geometry_weight[0] = 0;
  run(std::log(2.), {0, 0, 0, 0, .5, 0, 0});
  v.objectness_weight[0] = 0;
  run(0, {0, 0, 0, 0, 0, 0, 0});
  {
    auto       logits = nn::from_vector(std::vector<float>{-80, 80}, {2}).requires_grad_();
    const auto total =
        nn::sum(yolo::LogisticCrossEntropy(logits, nn::from_vector(std::vector<float>{0, 1}, {2})));
    const auto g = nn::grad(total, {logits})[0].to_vector<float>();
    check::Near(total.item(), 0, 1e-6, "extreme correct classifications have near-zero BCE");
    for (float d : g)
      check::Near(d, 0, 1e-6, "extreme BCE derivatives finite and near zero");
  }
  std::cout << "  multi-label, coordinate, background and ignored losses/derivatives passed; "
               "zero/extreme logits stable"
            << std::endl;
}

void AssignmentChecks() {
  std::array<std::vector<float>, 3> raw;
  std::array<nn::Shape, 3>          shapes;
  for (int level = 0; level < 3; ++level) {
    const int cells            = 2 << level;
    shapes[std::size_t(level)] = {1, cells, cells, 3, 8};
    raw[std::size_t(level)].assign(std::size_t(cells * cells * 3 * 8), 0);
  }
  const std::vector<Dimension> priors = {
      {4, 4}, {6, 6}, {8, 8}, {12, 12}, {16, 16}, {20, 20}, {24, 24}, {32, 32}, {48, 48}};
  DetectionScenes scenes;
  scenes.boxes  = {{{24, 24, 32, 32}, {16, 16, 32, 32}, {0, 0, 32, 32}}};
  scenes.labels = {{0, 1, 2}};
  const auto t  = yolo::MakeYoloV3Targets(raw, shapes, priors, scenes, {0});
  check::Require(t.collisions == 0, "distinct cross-scale target assignments");
  int ignored = 0;
  for (const auto& g : t.levels) {
    int count = 0;
    for (std::size_t i = 0; i < g.positive.size(); ++i) {
      count += g.positive[i] != 0;
      ignored += g.positive[i] == 0 && g.objectness_weight[i] == 0;
    }
    check::Require(count == 1, "each of three targets selects one prior on exactly one scale");
  }
  check::Require(ignored >= 3, "overlapping non-responsible anchors ignored across scales");
  std::cout << "  8/16/32-pixel targets assigned exactly once to fine/middle/coarse heads; "
            << ignored << " non-responsible overlaps ignored" << std::endl;
}

nn::Tensor Objective(const yolo::YoloV3Outputs& net) {
  return nn::mean(nn::square(net.raw[0])) + nn::mean(nn::square(net.raw[1])) +
         nn::mean(nn::square(net.raw[2]));
}

void ModelChecks() {
  for (int classes : {80, 20}) {
    nn::ParameterStore p(17), state(17);
    yolo::YoloV3Config cfg;
    cfg.classes      = classes;
    cfg.update_stats = false;
    const auto net   = yolo::YoloV3(p, state, SinImages(1, 128, 3), cfg, true);
    check::Require(
        p.count() == (classes == 80 ? 61949149 : 61626049) && state.count() == 52608,
        "full YOLO v3 parameter/state counts");
    const auto gradients = nn::grad(Objective(net), p.parameters());
    for (const auto& g : gradients)
      check::Require(g.defined(), "every full YOLO v3 parameter connected");
    check::Require(
        gradients.size() == 222, "72 BN feature convolutions and 3 biased prediction convolutions");
    std::cout << "  " << classes << "-class full YOLO v3: " << p.count()
              << " parameters, all 222 parameter derivatives computed (128-pixel input)"
              << std::endl;
  }
  nn::ParameterStore p(17), state(17);
  yolo::YoloV3Config cfg;
  cfg.classes       = 10;
  cfg.width_divisor = 8;
  cfg.grayscale     = true;
  cfg.update_stats  = false;
  double minimum = INFINITY, maximum = 0;
  for (int resolution : {64, 96, 128}) {
    const auto net = yolo::YoloV3(p, state, SinImages(2, resolution, 1), cfg, true);
    for (int level = 0; level < 3; ++level)
      check::Require(
          net.raw[std::size_t(level)].shape() ==
              nn::Shape{2, resolution / (32 >> level), resolution / (32 >> level), 3, 15},
          "YOLO v3 prediction grid");
    if (resolution != 64)
      continue;
    const auto gradients = nn::grad(Objective(net), p.parameters());
    for (std::size_t i = 0; i < gradients.size(); ++i) {
      const double norm = std::sqrt(nn::sum(nn::square(gradients[i])).item());
      check::Require(
          std::isfinite(norm) && norm > 0, "finite nonzero YOLO v3 derivative: " + p.names()[i]);
      minimum = std::min(minimum, norm);
      maximum = std::max(maximum, norm);
    }
  }
  std::cout << "  compact 75-convolution model: " << p.count()
            << " parameters; three-scale 64/96/128 grids checked; derivative norms [" << minimum
            << ',' << maximum << "]" << std::endl;
}
} // namespace

int main(int argc, char** argv) {
  try {
    int                              scene_size = 128;
    std::filesystem::path            directory  = paths::Dataset("mnist");
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "yolo_v3_digits";
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
          throw std::invalid_argument("size must be at least 64 and divisible by 32");
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
    LossChecks();
    AssignmentChecks();
    ModelChecks();
    const auto [train, test] = detector::StandardScenes(directory, scene_size);
    std::vector<Dimension> dimensions;
    for (std::size_t row = 0; row < 5500; ++row)
      for (const auto& b : train.boxes[row])
        dimensions.push_back({b[2] - b[0], b[3] - b[1]});
    const auto priors = yolo::ClusterDimensions(dimensions, 9);
    std::cout << "  fit-only priors (height,width at scene input size):";
    for (const auto& prior : priors)
      std::cout << " (" << prior[0] << ',' << prior[1] << ')';
    std::cout << std::endl;
    yolo::YoloV3Config model;
    model.classes                = 10;
    model.width_divisor          = 8;
    model.grayscale              = true;
    int                     step = 0, collisions = 0;
    detector::DetectorHooks hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore& state) {
      auto frozen         = model;
      frozen.update_stats = false;
      yolo::YoloV3(p, state, nn::zeros({1, scene_size, scene_size, 1}), frozen, true);
    };
    hooks.loss = [&](nn::ParameterStore&     p,
                     nn::ParameterStore&     state,
                     const DetectionScenes&  scenes,
                     const std::vector<int>& rows) {
      const int  size = scene_size + 32 * ((step++ / 10) % 3);
      const auto net  = yolo::YoloV3(
          p,
          state,
          nn::resize_bilinear(detector::SceneImages(scenes, rows), {size, size}),
          model,
          true);
      std::array<std::vector<float>, 3> values;
      std::array<nn::Shape, 3>          shapes;
      for (std::size_t level = 0; level < 3; ++level) {
        values[level] = net.raw[level].detach().to_vector<float>();
        shapes[level] = net.raw[level].shape();
      }
      const auto targets = yolo::MakeYoloV3Targets(values, shapes, priors, scenes, rows);
      collisions += targets.collisions;
      return yolo::YoloV3Loss(net.raw, targets, 10);
    };
    hooks.detect = [&](nn::ParameterStore&     p,
                       nn::ParameterStore&     state,
                       const DetectionScenes&  scenes,
                       const std::vector<int>& rows) {
      return yolo::DetectYoloV3(
          [&](const nn::Tensor& images) {
            return yolo::YoloV3(p, state, images, model, false).raw;
          },
          priors,
          scenes,
          rows);
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
    std::cout << "  " << collisions
              << " prior/cell collisions discarded during training\nAll YOLO v3 checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
