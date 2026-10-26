// CMD: just run-nn retinanet
// RetinaNet and focal loss: dense detection without hard negative mining.
//
// Port of tflow's retinanet. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
// Short default: three training epochs; use epochs=N for longer accuracy experiments.
//
// Check focal loss values and derivatives for alpha .25 and gamma 0/.5/2,
// including logits -80/0/80. Show how focal weighting suppresses the summed
// contribution of 10000 easy negatives relative to ten hard positives.
// Check positive/background/ignored anchors, beta=1/9 Smooth-L1 gradients,
// nine anchors per cell, prior probability .01 and head sharing. Construct
// the full 37,968,692-parameter ResNet-50/P3-P7 model (P6 from C5), including
// every parameter derivative. Numerically check all compact model
// gradients. A feature-only ResNet option is checked against unchanged
// classifier counts and avoids disconnected classifier parameters.
// https://arxiv.org/abs/1708.02002
//
// Train a compact ResNet-18 backbone (1/8 widths), 32-channel FPN and shared
// subnets on 128-pixel grayscale digit scenes. P3-P7 grids are 16/8/4/2/1;
// base anchor sides are 16/32/64/128/256, with three scales and three ratios
// per cell. Use 5500 fit / 500 validation / 1000 test scenes, three epochs
// of Adam(.001), batch 32, kernel L2 .0001 and the best validation AP50
// checkpoint. No pretrained weights or COCO performance are claimed.
//
// Digit cells scale with input size (32..64 at the 128-pixel default).
// size=64 reproduces the initial smaller-input geometry. Optional
// warm_start=PREFIX restores parameters/BN state (.npz prefix) with fresh
// Adam moments. Default accuracy=0 verifies finite, decreasing loss;
// accuracy=1 enforces the original test AP50 target (.7).
//
// Eager port (common/retinanet.hpp from job 1.6a): tflow builds the full
// 640-pixel model as a graph only; here its forward and backward passes run
// at 128 pixels (counts and gradient connectivity do not depend on the input
// size) and the 640-pixel anchor count comes from the pyramid grids. Weights
// come from nn's generator, so trained numbers match TF statistically.
// Checkpoint (.npz) and history: output/nn/models/retinanet_digits/.
// Options: data=PATH size=128 epochs=3 accuracy=0|1 warm_start=PREFIX lr=.001 threads=N.

#include <iostream>

#include "common/retinanet.hpp"

// Experiment workflow from examples/nn/common/detection.hpp.
namespace detector {
using namespace ::detector;
inline ApResult TrainDetector(const DetectorTrainingConfig& requested_cfg, const DetectionScenes& train,
                              const DetectionScenes& test, const DetectorHooks& hooks) {
  auto cfg = requested_cfg;
  cfg.epochs = example_runtime::Limit(cfg.epochs, 1);
  cfg.batch = example_runtime::Limit(cfg.batch, 2);
  cfg.validation = example_runtime::Limit(cfg.validation, 8);
  if (!example_runtime::Full()) cfg.fit = cfg.fit ? std::min(cfg.fit, 8) : 8;
  example_runtime::Describe();
  check::Require(cfg.validation > 0 && cfg.validation < train.size() && cfg.epochs > 0 && cfg.batch > 0,
                 "valid detector training configuration");
  check::Require(!example_runtime::Full() || cfg.accuracy || cfg.epochs >= 2, "short learning demonstration needs at least two epochs");
  const int fit_count =
      cfg.fit > 0 ? std::min(cfg.fit, train.size() - cfg.validation) : train.size() - cfg.validation;
  std::vector<int> fit(static_cast<std::size_t>(fit_count)), validation(static_cast<std::size_t>(cfg.validation)),
      test_rows(example_runtime::Limit<std::size_t>(test.size(), 8));
  std::iota(fit.begin(), fit.end(), 0);
  std::iota(validation.begin(), validation.end(), train.size() - cfg.validation);
  std::iota(test_rows.begin(), test_rows.end(), 0);
  nn::ParameterStore params(17), state(17);
  {
    nn::NoGradGuard guard;
    hooks.build(params, state);
  }
  std::vector<nn::Tensor> kernels;
  for (const auto& p : params.parameters())
    if (p.dim() >= 2) kernels.push_back(p);
  const auto directory = TrainingDirectory(cfg.name);
  std::filesystem::create_directories(directory);
  const auto prefix = (directory / "best").string();
  if (!cfg.warm_start.empty()) {
    check::Require(std::filesystem::absolute(cfg.warm_start).lexically_normal() !=
                       std::filesystem::absolute(prefix).lexically_normal(),
                   "warm start requires a separate output directory/run tag");
    RestoreCheckpoint(params, state, cfg.warm_start);
    std::cout << "  warm start: " << cfg.warm_start << "; fresh Adam moments" << std::endl;
  }
  nn::OptimizerConfig optimizer_config;
  optimizer_config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(params, cfg.learning_rate, optimizer_config);
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttraining_loss\tvalidation_ap50\tvalidation_coco_ap\n";
  std::cout << "== " << cfg.name << ": " << fit_count << " fit / " << cfg.validation << " validation / " << test_rows.size()
            << " test scenes (" << train.height << "x" << train.width << ")\n  parameters=" << params.count()
            << "; batch=" << cfg.batch << "; lr=" << cfg.learning_rate << std::endl;
  nn::BatchOrder      batches(std::size_t(fit_count), std::size_t(cfg.batch), 29);
  double              best = -1;
  int                 best_epoch = -1;
  std::vector<double> epoch_losses;
  if (!cfg.warm_start.empty()) {
    const auto baseline = EvaluateDetections(hooks.detect(params, state, train, validation), train, validation, cfg.classes);
    best       = baseline.ap50;
    best_epoch = 0;
    SaveCheckpoint(params, state, prefix);
    history << 0 << "\tnan\t" << baseline.ap50 << '\t' << baseline.coco << '\n';
    history.flush();
    std::cout << "  restored baseline validation " << ApSummary(baseline) << std::endl;
  }
  for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
    batches.begin_epoch();
    double      total = 0;
    std::size_t seen  = 0;
    for (;;) {
      const auto picked = batches.next();
      if (picked.empty()) break;
      std::vector<int> rows;
      for (auto k : picked) rows.push_back(fit[std::size_t(k)]);
      nn::Tensor objective = hooks.loss(params, state, train, rows);
      const double batch_loss = objective.item();
      check::Require(std::isfinite(batch_loss), "finite detector batch loss");
      if (cfg.l2 > 0 && !kernels.empty()) {
        nn::Tensor penalty = nn::sum(nn::square(kernels[0]));
        for (std::size_t i = 1; i < kernels.size(); ++i) penalty = penalty + nn::sum(nn::square(kernels[i]));
        objective = objective + 0.5 * cfg.l2 * penalty;
      }
      optimizer.minimize(objective);
      total += batch_loss * double(rows.size());
      seen += rows.size();
    }
    epoch_losses.push_back(total / double(seen));
    check::Require(std::isfinite(epoch_losses.back()), "finite detector epoch mean");
    const auto result = EvaluateDetections(hooks.detect(params, state, train, validation), train, validation, cfg.classes);
    check::Require(std::isfinite(result.ap50), "finite validation AP50");
    if (result.ap50 > best) {
      best       = result.ap50;
      best_epoch = epoch;
      SaveCheckpoint(params, state, prefix);
    }
    history << epoch << '\t' << total / double(seen) << '\t' << result.ap50 << '\t' << result.coco << '\n';
    history.flush();
    std::cout << "  epoch " << epoch << ": training loss=" << total / double(seen) << ", validation " << ApSummary(result)
              << std::endl;
  }
  RestoreCheckpoint(params, state, prefix);
  const auto restored = EvaluateDetections(hooks.detect(params, state, train, validation), train, validation, cfg.classes);
  check::Near(restored.ap50, best, 1e-6, "restored checkpoint reproduces selected validation AP50");
  std::cout << "  restored validation AP50=" << restored.ap50 << "; checkpoint agreement verified" << std::endl;
  const auto result = EvaluateDetections(hooks.detect(params, state, test, test_rows), test, test_rows, cfg.classes);
  std::cout << "  selected epoch=" << best_epoch << "; test " << ApSummary(result) << "\n  per-class test AP50:";
  for (double ap : result.per_class50) std::cout << ' ' << std::fixed << std::setprecision(3) << ap;
  std::cout << std::defaultfloat << std::setprecision(6) << "\n  best checkpoint and history: " << directory.string()
            << std::endl;
  if (hooks.inspect) hooks.inspect(params, state, test, test_rows, directory);
  if (example_runtime::Full()) CheckDetectorDemonstration(cfg, epoch_losses, result);
  else check::Require(std::isfinite(result.ap50), "finite quick detector evaluation");
  if (example_runtime::Full() && !cfg.accuracy)
    std::cout << "  short learning demonstration passed: mean loss " << epoch_losses.front() << " -> "
              << epoch_losses.back() << "; accuracy benchmark not requested (accuracy=1 enables AP50 gate "
              << cfg.minimum_test_ap50 << ")" << std::endl;
  return result;
}
}  // namespace detector



namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

namespace {
double Focal(double x, double target, double gamma, double alpha = .25) {
  const double ce       = std::max(x, 0.) - x * target + std::log1p(std::exp(-std::abs(x)));
  const double signed_x = (2 * target - 1) * x, softplus = std::max(signed_x, 0.) + std::log1p(std::exp(-std::abs(signed_x)));
  return (target ? alpha : 1 - alpha) * std::exp(-gamma * softplus) * ce;
}
nn::Tensor SinImages(std::int64_t n, std::int64_t size, std::int64_t channels) {
  std::vector<float> v(std::size_t(n * size * size * channels));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(.13f * float(i));
  return nn::from_vector(v, {n, size, size, channels});
}
nn::Tensor Objective(const retinanet::RetinaNetOutputs& n) { return nn::mean(nn::square(n.logits)) + nn::mean(nn::square(n.deltas)); }

void FocalChecks() {
  std::vector<float> value(14), labels(14);
  const float        cases[] = {-80, -10, -2, 0, 2, 10, 80};
  for (int i = 0; i < 7; ++i)
    for (int y = 0; y < 2; ++y) {
      value[std::size_t(2 * i + y)]  = cases[i];
      labels[std::size_t(2 * i + y)] = float(y);
    }
  double       maximum  = 0;
  const double gammas[] = {0, .5, 2};
  for (double gamma : gammas) {
    auto       logits   = nn::from_vector(value, {14}).requires_grad_();
    const auto loss     = retinanet::SigmoidFocalLoss(logits, nn::from_vector(labels, {14}), .25, gamma);
    const auto gradient = nn::grad(nn::sum(loss), {logits})[0].to_vector<float>();
    const auto values   = loss.to_vector<float>();
    for (std::size_t i = 0; i < 14; ++i) {
      const double x = value[i], y = labels[i], step = 1e-4;
      check::Near(values[i], Focal(x, y, gamma), 5e-6, "independent focal value, including extreme logits");
      const double derivative = (Focal(x + step, y, gamma) - Focal(x - step, y, gamma)) / (2 * step);
      maximum                 = std::max(maximum, std::abs(gradient[i] - derivative));
    }
  }
  check::Near(maximum, 0, 3e-6, "independent focal central differences");
  const double negative = std::log(.01 / .99), positive = std::log(.1 / .9);
  const double ce_ratio    = 10000 * Focal(negative, 0, 0) / (10 * Focal(positive, 1, 0));
  const double focal_ratio = 10000 * Focal(negative, 0, 2) / (10 * Focal(positive, 1, 2));
  check::Require(ce_ratio > 10 && focal_ratio < .01, "focal modulation prevents easy negatives overwhelming positives");
  std::cout << "  42 focal value/gradient cases including gamma zero and +/-80 passed; derivative error=" << maximum
            << "; easy-negative/hard-positive loss ratio " << ce_ratio << " -> " << focal_ratio << std::endl;
}

void LossAndMatchingChecks() {
  std::vector<float> deltas = {.05f, .2f, 0, -.2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f, 1};
  std::vector<float> labels(6, 0), positive = {1, 0, 0}, weights = {1, 1, 0};
  labels[0]          = 1;
  std::vector<float> out_loss;
  auto run = [&](std::vector<float>& gl, std::vector<float>& gd) {
    auto                     logits = nn::zeros({1, 3, 2}).requires_grad_();
    auto                     d      = nn::from_vector(deltas, {1, 3, 4}).requires_grad_();
    retinanet::RetinaTargets t{nn::from_vector(labels, {1, 3, 2}), nn::from_vector(positive, {1, 3}),
                               nn::from_vector(weights, {1, 3}), nn::zeros({1, 3, 4})};
    const auto               loss = retinanet::RetinaLoss(logits, d, t);
    const auto               g    = nn::grad(loss, {logits, d});
    gl                            = g[0].to_vector<float>();
    gd                            = g[1].to_vector<float>();
    return loss.item();
  };
  std::vector<float> gl, gd;
  const double       regression = .5 * .05 * .05 / (1. / 9) + 2 * (.2 - .5 / 9);
  check::Near(run(gl, gd), .625 * std::log(2.) + regression, 2e-6, "independent focal plus Smooth-L1 loss");
  const double positive_gradient = -.25 * (.125 + .25 * std::log(2.)), negative_gradient = .75 * (.125 + .25 * std::log(2.));
  for (std::size_t i = 0; i < 6; ++i)
    check::Near(gl[i], i == 0 ? positive_gradient : i < 4 ? negative_gradient : 0, 2e-6, "analytic positive/background/ignored focal gradient");
  const double expected[] = {.45, 1, 0, -1};
  for (std::size_t i = 0; i < 12; ++i) check::Near(gd[i], i < 4 ? expected[i] : 0, 2e-6, "beta=1/9 Smooth-L1 derivatives on positives only");
  std::fill(labels.begin(), labels.end(), 0.f);
  std::fill(positive.begin(), positive.end(), 0.f);
  check::Near(run(gl, gd), .75 * std::log(2.), 2e-6, "no-positive batch retains focal background learning");
  std::fill(weights.begin(), weights.end(), 0.f);
  check::Near(run(gl, gd), 0, 0, "all ignored anchors have zero loss");
  for (float g : gl) check::Near(g, 0, 0, "all ignored anchors have zero gradients");
  for (float g : gd) check::Near(g, 0, 0, "all ignored anchors have zero gradients");
  DetectionScenes scene;
  scene.boxes  = {{{0, 0, 10, 10}}};
  scene.labels = {{1}};
  const std::vector<Box> anchors = {{0, 0, 10, 10}, {0, 0, 10, 22}, {30, 30, 40, 40}};
  const auto             t       = retinanet::MakeRetinaTargets(anchors, scene, {0}, 2);
  const auto             p = t.positive.to_vector<float>(), w = t.weights.to_vector<float>();
  check::Require(p[0] == 1 && w[0] == 1 && w[1] == 0 && w[2] == 1, "RetinaNet positive/ignored/background IoU bands");
  std::cout << "  positive/background/ignored focal and Smooth-L1 derivatives passed; IoU bands and empty/all-ignored batches checked" << std::endl;
}

void ModelChecks() {
  for (bool classifier : {true, false}) {
    nn::ParameterStore   p(17), state(17);
    resnet::ResNetConfig cfg;
    cfg.depth              = 50;
    cfg.include_classifier = classifier;
    nn::NoGradGuard guard;
    resnet::ResNet(p, state, nn::zeros({1, 64, 64, 3}), cfg);
    check::Require(p.count() == (classifier ? 25557032 : 23508032), "default and feature-only ResNet-50 parameter counts");
  }
  {
    nn::ParameterStore         p(17), state(17);
    retinanet::RetinaNetConfig full;
    full.image_size   = 128;
    full.update_stats = false;
    const auto net    = retinanet::RetinaNet(p, state, SinImages(1, 128, 3), full, true);
    int        anchors640 = 0;
    for (int level = 0; level < 5; ++level) {
      const int cells = (640 + (8 << level) - 1) / (8 << level);
      anchors640 += cells * cells * 9;
    }
    check::Require(p.count() == 37968692 && state.count() == 53120 && anchors640 == 76725,
                   "full original-C5-P6 RetinaNet parameter/state/anchor counts");
    const auto gradients = nn::grad(Objective(net), p.parameters());
    check::Require(gradients.size() == 195, "full RetinaNet parameter connections, shared subnet variables counted once");
    for (const auto& g : gradients) check::Require(g.defined(), "every full RetinaNet parameter receives a derivative");
    std::cout << "  full RetinaNet (P6 from C5): 37968692 parameters, 76725 anchors at 640 pixels, all 195 parameter derivatives "
                 "computed (128-pixel input)"
              << std::endl;
  }
  nn::ParameterStore         p(17), state(17);
  retinanet::RetinaNetConfig cfg;
  cfg.classes        = 10;
  cfg.image_size     = 64;
  cfg.backbone_depth = 18;
  cfg.width_divisor  = 8;
  cfg.pyramid_width  = 32;
  cfg.grayscale      = true;
  cfg.update_stats   = false;
  const auto net     = retinanet::RetinaNet(p, state, SinImages(2, 64, 1), cfg, true);
  check::Require(p.count() == 346246 && state.count() == 1200 && net.anchors == 774, "compact RetinaNet counts and head sharing");
  const auto anchors = retinanet::RetinaAnchors(net.cells, 8);
  check::Require(anchors.size() == 774, "nine priors per compact pyramid cell");
  const int cells[] = {8, 4, 2, 1, 1};
  for (std::size_t level = 0; level < 5; ++level) {
    const auto& s = net.features[level].shape();
    check::Require(s[1] == cells[level] && s[2] == cells[level] && s[3] == 32, "compact five-level pyramid dimensions");
  }
  double prior_error = 0;
  for (float v : net.probabilities.to_vector<float>()) prior_error = std::max(prior_error, std::abs(double(v) - .01));
  check::Require(prior_error < .001, "classification starts close to the .01 foreground prior");
  const auto gradients = nn::grad(Objective(net), p.parameters());
  double     minimum = INFINITY, maximum = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double norm = std::sqrt(nn::sum(nn::square(gradients[i])).item());
    check::Require(std::isfinite(norm) && norm > 0, "finite nonzero RetinaNet derivative: " + p.names()[i]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  std::cout << "  compact ResNet-18/FPN: 346246 parameters, grids 8/4/2/1/1, shared heads, prior error=" << prior_error
            << "; derivative norms [" << minimum << ',' << maximum << "]" << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int                              scene_size = 128;
    std::filesystem::path            directory  = paths::Dataset("mnist");
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "retinanet_digits";
    cfg.epochs            = 3;
    cfg.accuracy          = false;
    cfg.minimum_test_ap50 = .7;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        eq  = arg.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("expected key=value option");
      const auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
      if (key == "size") {
        scene_size = std::stoi(value);
        if (scene_size < 64 || scene_size % 32) throw std::invalid_argument("size must be at least 64 and divisible by 32");
      } else if (key == "data") directory = value;
      else if (key == "epochs") cfg.epochs = std::stoi(value);
      else if (key == "accuracy" && (value == "0" || value == "1")) cfg.accuracy = value == "1";
      else if (key == "warm_start") cfg.warm_start = value;
      else if (key == "lr") cfg.learning_rate = std::stof(value);
      else if (key == "threads") nn::set_num_threads(std::stoi(value));
      else throw std::invalid_argument("unknown option " + key);
    }
    FocalChecks();
    LossAndMatchingChecks();
    ModelChecks();
    const auto [train, test] = detector::StandardScenes(directory, scene_size);
    retinanet::RetinaNetConfig model;
    model.classes        = 10;
    model.image_size     = scene_size;
    model.backbone_depth = 18;
    model.width_divisor  = 8;
    model.pyramid_width  = 32;
    model.grayscale      = true;
    std::vector<Box>        anchors;
    detector::DetectorHooks hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore& state) {
      auto frozen         = model;
      frozen.update_stats = false;
      const auto net      = retinanet::RetinaNet(p, state, nn::zeros({1, scene_size, scene_size, 1}), frozen, true);
      anchors             = retinanet::RetinaAnchors(net.cells, float(scene_size) / 8);
    };
    hooks.loss = [&](nn::ParameterStore& p, nn::ParameterStore& state, const DetectionScenes& scenes, const std::vector<int>& rows) {
      const auto net = retinanet::RetinaNet(p, state, detector::SceneImages(scenes, rows), model, true);
      return retinanet::RetinaLoss(net.logits, net.deltas, retinanet::MakeRetinaTargets(anchors, scenes, rows, 10));
    };
    hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore& state, const DetectionScenes& scenes, const std::vector<int>& rows) {
      return retinanet::DetectRetina([&](const nn::Tensor& images) { return retinanet::RetinaNet(p, state, images, model, false); },
                                     anchors, scenes, rows, 10);
    };
    detector::TrainDetector(cfg, train, test, hooks);
    std::cout << "All RetinaNet and focal-loss checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
