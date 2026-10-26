// CMD: just run-nn yolact
// YOLACT: learned shared prototypes and per-anchor instance coefficients.
// https://arxiv.org/abs/1904.02689
// https://github.com/dbolya/yolact/blob/master/data/config.py
//
// Port of tflow's yolact. Full ResNet101/FPN, shared prediction heads, 32 ReLU
// prototypes and tanh coefficients, semantic auxiliary head; every full
// parameter derivative. Independent linear-mask/cropped-BCE derivatives,
// ignored-band/hard-negative fixtures and the Fast NMS suppression-chain
// distinction from greedy NMS. Compact ResNet18/8, FPN 32, 8 prototypes
// retains every branch and five scales. Teaching adaptations are explicit in
// common/yolact.hpp: background last, true aspect ratios, pixel-centre
// targets/discrete crop normalization.
// Train all branches from scratch on nonoverlapping MNIST instance scenes:
// default 128 fit/32 validation/64 test, 64px, 3 epochs, Adam(.001), batch 4,
// L2 .0001. Loss: softmax/3:1 mining + 1.5 SmoothL1 + 6.125 cropped mask BCE +
// semantic BCE. At most 32 randomly selected positive masks per image
// (compact adaptation). Restore best validation box AP50, report box/mask AP
// and export mask preview. No COCO accuracy, pretrained weights or real-time
// CPU claims are made.
//
// TF -> datamunge: TF builds the full 550x550 model as a graph only; here the
// full model executes forward and backward on a 64x64 image (the parameter
// census and the 348 derivatives do not depend on the input size; the
// 550-pixel prior/prototype geometry is checked from the level grid formula,
// which the executed 64-pixel model also satisfies). Training is eager; the
// fixed fitting-batch mask-loss probe uses training-mode BN without updating
// the moving averages, as TF's separate session run does. Scenes are the
// MNIST instance scenes bit-identical to TF's (output/datasets/mnist);
// weights come from datamunge's generator, so trained numbers match TF
// statistically.
// Options: data=PATH epochs=3 accuracy=0|1 checks_only=0|1 threads=N.
// accuracy=1 retains box AP50 .7/mask AP50 .6 benchmark targets; short mode
// checks actual learning.
// Artifacts: output/nn/models/yolact_digits/.

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/detection_scenes.hpp"
#include "common/instance_segmentation.hpp"
#include "common/paths.hpp"
#include "common/yolact.hpp"

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
using detector::Detection;
using nn::Tensor;

namespace {

yolact::YolactConfig Compact() {
  yolact::YolactConfig c;
  c.size       = 64;
  c.classes    = 10;
  c.depth      = 18;
  c.divisor    = 8;
  c.width      = 32;
  c.prototypes = 8;
  c.grayscale  = true;
  return c;
}

std::int64_t ParameterReference(const yolact::YolactConfig& c) {
  // Independently enumerate canonical ResNet blocks and convolution shapes.
  const bool   bottleneck = c.depth == 101;
  const int    blocks101[] = {3, 4, 23, 3}, blocks18[] = {2, 2, 2, 2};
  int          in = 64 / c.divisor;
  std::int64_t n  = 49 * 3 * in + 2 * in;
  for (int stage = 0; stage < 4; ++stage) {
    const int w = (64 << stage) / c.divisor, out = w * (bottleneck ? 4 : 1);
    for (int block = 0; block < (bottleneck ? blocks101[stage] : blocks18[stage]); ++block) {
      if (bottleneck)
        n += std::int64_t(in) * w + 2 * w + 9LL * w * w + 2 * w + std::int64_t(w) * out + 2 * out;
      else
        n += 9LL * in * w + 2 * w + 9LL * w * out + 2 * out;
      if (in != out) n += std::int64_t(in) * out + 2 * out;
      in = out;
    }
  }
  if (c.depth == 101 && c.divisor == 1) check::Require(n == 42500160, "canonical ResNet101 without classifier");
  const int w = c.width, k = c.prototypes;
  for (int i = 0; i < 3; ++i) n += std::int64_t((128 << i) * (bottleneck ? 4 : 1) / c.divisor) * w + w;
  n += 10LL * (9LL * w * w + w);  // three smooth, two down, one prediction, four protonet.
  n += (9LL * w + 1) * (3 * (c.classes + 1) + 12 + 3 * k) + (std::int64_t(w) + 1) * (k + c.classes);
  return n;
}

void LinearMaskChecks() {
  const std::vector<float> p_values = {1, 2, -1, 3, 2, -2, 4, 1, 0, 2, -3, 1}, c_values = {.5f, -.25f},
                           t_values = {0, 1, 0, 1, 0, 1};
  std::vector<float>       m_values = {1, 1, 0, 0, 1, 0};
  struct Result {
    std::vector<float> logits, dp, dc;
    double             loss;
  };
  auto run = [&] {
    Tensor proto = nn::from_vector(p_values, {1, 2, 3, 2}), coeff = nn::from_vector(c_values, {1, 2});
    proto.requires_grad_(true);
    coeff.requires_grad_(true);
    const Tensor logits = yolact::LinearMasks(proto, coeff);
    const Tensor loss   = yolact::CroppedMaskLoss(logits, nn::from_vector(t_values, {1, 2, 3}), nn::from_vector(m_values, {1, 2, 3}));
    const auto   g      = nn::grad(loss, {proto, coeff});
    return Result{logits.to_vector<float>(), g[0].to_vector<float>(), g[1].to_vector<float>(), loss.item<double>()};
  };
  auto   out      = run();
  double expected = 0, dc[2] = {0, 0};
  for (int i = 0; i < 6; ++i) {
    const double z = p_values[std::size_t(2 * i)] * .5 - p_values[std::size_t(2 * i + 1)] * .25;
    check::Near(out.logits[std::size_t(i)], z, 1e-7, "signed prototype linear combination");
    const double g = (1 / (1 + std::exp(-z)) - t_values[std::size_t(i)]) * m_values[std::size_t(i)] / 3;
    expected += (std::log1p(std::exp(z)) - z * t_values[std::size_t(i)]) * m_values[std::size_t(i)] / 3;
    for (int k = 0; k < 2; ++k) {
      check::Near(out.dp[std::size_t(2 * i + k)], g * c_values[std::size_t(k)], 1e-7,
                  "every prototype derivative and zero outside crop");
      dc[k] += g * p_values[std::size_t(2 * i + k)];
    }
  }
  check::Near(out.loss, expected, 1e-7, "area-normalized cropped BCE");
  for (int k = 0; k < 2; ++k) check::Near(out.dc[std::size_t(k)], dc[k], 1e-7, "every coefficient derivative");
  m_values.assign(6, 0.f);
  out = run();
  check::Near(out.loss, 0, 0, "empty crop has zero mask loss");
  for (float v : out.dp) check::Near(v, 0, 0, "empty crop masks prototype gradients");
  const Tensor empty = yolact::CroppedMaskLoss(nn::zeros({0, 2, 3}), nn::zeros({0, 2, 3}), nn::zeros({0, 2, 3}));
  check::Near(empty.item<double>(), 0, 0, "zero positive masks produce finite zero loss");
  std::cout << "  signed prototype assembly, cropped BCE and every analytic prototype/coefficient derivative; empty "
               "crop/positive set passed"
            << std::endl;
}

void DetectionChecks() {
  const std::vector<Box> truth = {{0, 0, 10, 10}, {20, 20, 30, 30}}, anchors = {truth[0], truth[1], {0, 0, 10, 22}, {50, 50, 60, 60}};
  check::Require(yolact::YolactMatching(anchors, truth) == std::vector<int>({0, 1, -2, -1}),
                 "positive/ignored/background matching");
  check::Require(yolact::YolactMatching({truth[0], truth[0]}, {truth[0], truth[0]}) == std::vector<int>({0, 1}),
                 "forced matches use distinct anchors");
  check::Require(yolact::YolactMatching(anchors, {}) == std::vector<int>(4, -1), "empty scene matching");
  const std::vector<Detection> chain = {{0, 0, .9f, {0, 0, 10, 10}}, {0, 0, .8f, {0, 4, 10, 14}}, {0, 0, .7f, {0, 8, 10, 18}},
                                        {1, 0, .8f, {0, 0, 10, 10}}, {0, 1, .6f, {0, 0, 10, 10}}};
  check::Require(yolact::FastNms(chain, .3f) == std::vector<int>({0, 4, 3}),
                 "Fast NMS includes suppressed predecessors, isolates class/image");
  check::Require(detector::NonMaxSuppression({chain[0].box, chain[1].box, chain[2].box}, {.9f, .8f, .7f}, .3f) ==
                     std::vector<int>({0, 2}),
                 "greedy NMS differs on suppression chain");
  check::Require(yolact::FastNms(chain, .3f, 200, 1) == std::vector<int>({0, 3}) && yolact::FastNms(chain, .3f, 0).empty(),
                 "Fast NMS global cap and empty top-k");
  // Combined loss fixture: 4 anchors, 2 classes (+ background), 2x2 prototypes.
  Tensor logits = nn::zeros({1, 4, 3}), coefficients = nn::zeros({1, 4, 2});
  logits.requires_grad_(true);
  coefficients.requires_grad_(true);
  yolact::YolactOutputs net;
  net.anchors      = 4;
  net.proto_side   = 2;
  net.cells        = {1};
  net.logits       = logits;
  net.deltas       = nn::zeros({1, 4, 4});
  net.coefficients = coefficients;
  net.prototypes   = nn::ones({1, 2, 2, 2});
  net.semantic     = nn::zeros({1, 1, 1, 2});
  yolact::YolactTargets t;
  t.batch = 1;
  t.anchors = 4;
  t.side = 2;
  t.cells = 1;
  t.classes = 2;
  t.labels       = {0, 2, 2, -1};
  t.deltas.assign(16, 0.f);
  t.pairs        = {{0, 0}};
  t.mask_targets = {1, 0, 0, 0};
  t.crop         = {1, 0, 0, 0};
  t.semantic     = {0, 0};
  auto       loss = yolact::BuildYolactLoss(net, t);
  const auto g    = nn::grad(loss.loss, {logits, coefficients});
  const auto gl = g[0].to_vector<float>(), gc = g[1].to_vector<float>();
  check::Near(loss.loss.item<double>(), 3 * std::log(3.) + 7.125 * std::log(2.), 2e-6,
              "independent classification/mask/semantic weighted loss");
  for (int a = 0; a < 4; ++a)
    for (int k = 0; k < 3; ++k)
      check::Near(gl[std::size_t(3 * a + k)], a == 3 ? 0 : 1. / 3 - (k == (a == 0 ? 0 : 2)), 1e-7,
                  "positive/background/ignored class derivative");
  for (int a = 0; a < 4; ++a)
    for (int k = 0; k < 2; ++k)
      check::Near(gc[std::size_t(2 * a + k)], a == 0 ? -3.0625 : 0, 1e-7, "only selected anchor coefficients get mask derivative");
  check::Require(!loss.negatives[0] && loss.negatives[1] && loss.negatives[2] && !loss.negatives[3],
                 "ignore excluded from hard negatives");
  t.labels.assign(4, 2);
  t.pairs.clear();
  t.mask_targets.clear();
  t.crop.clear();
  loss = yolact::BuildYolactLoss(net, t);
  check::Near(loss.loss.item<double>(), std::log(2.), 1e-7, "empty detection image leaves semantic background BCE only");
  // Target construction on two reordered 4x4 scenes.
  instance::InstanceScenes scene;
  scene.height = scene.width = 4;
  scene.pixels.assign(32, 0);
  scene.boxes          = {{{0, 0, 2, 2}}, {{2, 2, 4, 4}}};
  scene.labels         = {{0}, {1}};
  scene.instance_masks = {{{1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}, {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1}}};
  std::mt19937           random(41);
  const std::vector<Box> priors = {{0, 0, 2, 2}, {2, 2, 4, 4}, {8, 8, 9, 9}, {10, 10, 11, 11}};
  const auto             fed    = yolact::MakeYolactTargets(net, priors, scene, {1, 0}, 2, random);
  check::Require(fed.pairs.size() == 2, "one selected anchor mask per scene");
  const int   expected_labels[] = {2, 1, 2, 2, 0, 2, 2, 2}, expected_pairs[] = {0, 1, 1, 0};
  const float expected_masks[] = {0, 0, 0, 1, 1, 0, 0, 0}, expected_semantic[] = {0, 1, 0, 0};
  for (int i = 0; i < 8; ++i) {
    check::Require(fed.labels[std::size_t(i)] == expected_labels[i], "reordered scene labels preserve matched anchor");
    check::Near(fed.mask_targets[std::size_t(i)], expected_masks[i], 0, "instance supports paired with selected coefficients");
    check::Near(fed.crop[std::size_t(i)], expected_masks[i], 0, "pixel-centre rectangular crop follows target box");
  }
  for (int i = 0; i < 2; ++i)
    check::Require(fed.pairs[std::size_t(i)].first == expected_pairs[2 * i] &&
                       fed.pairs[std::size_t(i)].second == expected_pairs[2 * i + 1],
                   "selected batch/anchor indices follow row order");
  for (int i = 0; i < 4; ++i)
    check::Near(fed.semantic[std::size_t(i)], expected_semantic[i], 0, "class-specific semantic union follows reordered image");
  for (float d : fed.deltas) check::Near(d, 0, 0, "exact prior geometry encodes zero deltas");
  std::cout << "  distinct/ignored/empty assignment, 3:1 mining, weighted loss/analytic derivatives and Fast NMS "
               "chain/class/image/cap and reordered-instance target fixtures passed"
            << std::endl;
}

void ModelChecks() {
  {
    nn::ParameterStore   p(17), state(17);
    yolact::YolactConfig cfg;  // full ResNet101/FPN256/32 prototypes/80 classes
    check::Require(yolact::YolactCells(550) == std::vector<int>({69, 35, 18, 9, 5}), "full 550px level grids");
    int anchors = 0;
    for (int c : yolact::YolactCells(550)) anchors += 3 * c * c;
    check::Require(anchors == 19248 && 2 * yolact::YolactCells(550)[0] == 138, "full YOLACT prior/prototype geometry");
    cfg.size       = 64;
    std::vector<float> v(64 * 64 * 3);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(.13f * float(i));
    const auto net = yolact::Yolact(p, state, nn::from_vector(v, {1, 64, 64, 3}), cfg, true);
    check::Require(p.count() == ParameterReference(cfg), "full YOLACT parameter count");
    Tensor objective = nn::sum(nn::square(net.logits));
    for (const auto& x : {net.deltas, net.coefficients, net.prototypes, net.semantic}) objective = objective + nn::sum(nn::square(x));
    const auto grads = nn::grad(objective, p.parameters());
    for (std::size_t k = 0; k < grads.size(); ++k)
      check::Require(std::isfinite(nn::sum(nn::square(grads[k])).item<double>()), "finite full YOLACT derivative");
    check::Require(grads.size() == 348 && p.size() == 348, "full ResNet101 and all YOLACT branches connected");
    std::cout << "  full ResNet101 YOLACT: " << p.count()
              << " parameters, 19248 priors, 138px/32 prototypes at 550px, all 348 derivatives (executed on 64x64)"
              << std::endl;
  }
  nn::ParameterStore p(17), state(17);
  const auto         cfg = Compact();
  std::vector<float> v(2 * 64 * 64);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(.13f * float(i));
  const auto net = yolact::Yolact(p, state, nn::from_vector(v, {2, 64, 64, 1}), cfg, true);
  check::Require(p.count() == ParameterReference(cfg) && net.cells == std::vector<int>({8, 4, 2, 1, 1}) && net.anchors == 258,
                 "compact YOLACT independent parameter/anchor count");
  Tensor objective = nn::mean(nn::square(net.logits));
  for (const auto& x : {net.deltas, net.coefficients, net.prototypes, net.semantic}) objective = objective + nn::mean(nn::square(x));
  const std::vector<nn::Shape> shapes = {{2, 258, 11}, {2, 258, 4}, {2, 258, 8}, {2, 16, 16, 8}, {2, 8, 8, 10}};
  const std::vector<Tensor>    outs   = {net.logits, net.deltas, net.coefficients, net.prototypes, net.semantic};
  for (int i = 0; i < 5; ++i) check::Require(outs[std::size_t(i)].shape() == shapes[std::size_t(i)], "YOLACT branch output shape");
  const auto grads = nn::grad(objective, p.parameters());
  double     low = INFINITY, high = 0;
  for (std::size_t k = 0; k < grads.size(); ++k) {
    const double norm = std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
    check::Require(std::isfinite(norm) && norm > 0, "finite nonzero YOLACT derivative: " + p.names()[k]);
    low  = std::min(low, norm);
    high = std::max(high, norm);
  }
  for (float c : net.coefficients.to_vector<float>()) check::Require(std::abs(c) <= 1, "bounded tanh coefficients");
  for (float x : net.prototypes.to_vector<float>()) check::Require(x >= 0, "nonnegative ReLU prototypes");
  std::cout << "  compact YOLACT: " << p.count() << " parameters, 258 priors, every branch/gradient passed, norms [" << low
            << ',' << high << ']' << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs   = 3;
    bool                  accuracy = false, checks_only = false;
    std::filesystem::path directory = paths::Dataset("mnist");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        directory = arg.substr(5);
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        accuracy = arg == "accuracy=1";
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH accuracy=0|1 checks_only=0|1 threads=N");
    }
    LinearMaskChecks();
    DetectionChecks();
    ModelChecks();
    if (checks_only) {
      std::cout << "All YOLACT fixtures passed (training not requested).\n";
      return 0;
    }
    detector::SceneConfig train_cfg;
    train_cfg.scenes      = 160;
    train_cfg.seed        = 11;
    train_cfg.max_overlap = 0;
    auto test_cfg         = train_cfg;
    test_cfg.scenes       = 64;
    test_cfg.seed         = 22;
    const auto train = instance::MakeInstanceScenes(detector::LoadDigits(directory, "train"), train_cfg);
    const auto test  = instance::MakeInstanceScenes(detector::LoadDigits(directory, "t10k"), test_cfg);
    const auto instances = [](const detector::DetectionScenes& scenes) -> const instance::InstanceScenes& {
      return static_cast<const instance::InstanceScenes&>(scenes);
    };
    std::vector<Box>      anchors;
    std::mt19937          random(41);
    int                   positive_masks    = 0;
    double                initial_mask_loss = NAN;
    std::vector<int>      probe_rows;
    yolact::YolactTargets probe_targets;
    const auto            infer = [](nn::ParameterStore& p, nn::ParameterStore& state) {
      return [&p, &state](const Tensor& images) { return yolact::Yolact(p, state, images, Compact(), false); };
    };
    detector::DetectorHooks hooks;
    hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore& state) {
      const auto net = yolact::Yolact(p, state, detector::SceneImages(train, {0}), Compact(), false);
      anchors        = yolact::YolactAnchors(net, 64, true);
      check::Require(int(anchors.size()) == net.anchors, "prior/head ordering count");
    };
    hooks.loss = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                     const std::vector<int>& rows) {
      const auto net     = yolact::Yolact(p, state, detector::SceneImages(scenes, rows), Compact(), true);
      const auto targets = yolact::MakeYolactTargets(net, anchors, instances(scenes), rows, 10, random);
      const auto loss    = yolact::BuildYolactLoss(net, targets);
      positive_masks += int(targets.pairs.size());
      if (probe_rows.empty() && !targets.pairs.empty()) {
        // Training-mode BN uses batch statistics, so this equals TF's separate probe run before the step.
        probe_rows        = rows;
        probe_targets     = targets;
        initial_mask_loss = loss.mask_loss.item<double>();
      }
      return loss.loss;
    };
    hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                       const std::vector<int>& rows) {
      std::vector<Detection> boxes;
      for (const auto& d : yolact::DetectYolact(infer(p, state), anchors, instances(scenes), rows, 10)) boxes.push_back(d.box);
      return boxes;
    };
    hooks.inspect = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                        const std::vector<int>& rows, const std::filesystem::path& outdir) {
      check::Require(!probe_rows.empty(), "positive fitting-batch mask probe recorded");
      double restored_mask_loss = NAN;
      {
        nn::NoGradGuard guard;
        const auto      net = yolact::Yolact(p, state, detector::SceneImages(train, probe_rows), Compact(), true, false);
        restored_mask_loss  = yolact::BuildYolactLoss(net, probe_targets).mask_loss.item<double>();
      }
      check::Require(std::isfinite(initial_mask_loss) && std::isfinite(restored_mask_loss) &&
                         restored_mask_loss < initial_mask_loss - 1e-4,
                     "learned prototype branch must reduce fixed fitting-batch mask BCE");
      std::cout << "  fixed fitting-batch prototype mask BCE " << initial_mask_loss << " -> " << restored_mask_loss
                << " (no optimizer/BN updates; not held-out accuracy)" << std::endl;
      const auto&   inst  = instances(scenes);
      const auto    found = yolact::DetectYolact(infer(p, state), anchors, inst, rows, 10);
      double        mean  = 0;
      std::ofstream table(outdir / "mask_ap.tsv");
      table << "iou\tmask_ap\n";
      for (int i = 0; i < 10; ++i) {
        const double threshold = .5 + .05 * i, ap = instance::MaskAveragePrecision(found, inst, 10, threshold);
        check::Require(std::isfinite(ap) && ap >= 0 && ap <= 1, "finite YOLACT mask AP");
        mean += ap / 10;
        table << threshold << '\t' << ap << '\n';
      }
      const double ap50 = instance::MaskAveragePrecision(found, inst, 10, .5),
                   ap75 = instance::MaskAveragePrecision(found, inst, 10, .75);
      instance::ExportInstanceMasks(inst, found, outdir / "mask_preview.png");
      std::cout << "  predicted-box prototype masks: AP50=" << ap50 << ", AP75=" << ap75 << ", AP50:95=" << mean << "; "
                << found.size() << " masks, all " << scenes.size() << " held-out scenes" << std::endl;
      if (accuracy) check::Require(ap50 >= .6, "YOLACT optional mask AP50 benchmark");
    };
    detector::DetectorTrainingConfig cfg;
    cfg.name              = "yolact_digits";
    cfg.epochs            = epochs;
    cfg.validation        = 32;
    cfg.batch             = 4;
    cfg.accuracy          = accuracy;
    cfg.minimum_test_ap50 = .7;
    detector::TrainDetector(cfg, train, test, hooks);
    check::Require(positive_masks > 0, "prototype coefficients trained on positive instance targets");
    std::cout << "  sampled positive mask presentations=" << positive_masks << "\nAll YOLACT checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
