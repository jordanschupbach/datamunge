// CMD: just run-nn nms_matching
// Detection utilities: non-maximum suppression, anchor matching and box coding.
//
// Port of tflow's nms_matching, on the library's box utilities
// (datamunge/nn/boxes.hpp) through examples/nn/common/detection.hpp.
// Compare greedy NMS to TensorFlow NonMaxSuppressionV3, including duplicate
// boxes, IoU exactly equal to the threshold, tied scores, zero-area boxes,
// empty inputs and output limits (the TF selections are recorded below:
// tflow runs the TF op live, which this port cannot). Check that class-wise
// NMS separates both images and classes before applying the per-image
// detection budget. Independently exercise RPN positive/ignored/negative
// assignments and low-quality forced matches, and SSD's distinct bipartite
// assignments when two targets prefer the same anchor. Check anchor ordering
// and centres, known box deltas and 1000 randomized encode/decode round trips.
// Reject malformed NMS inputs before they can index mismatched score arrays.
// Check RPN objectness derivatives at exactly zero logits: stable Softplus
// must produce sigmoid(x)-label, including background's positive gradient.
// Finally exercise the detector trainer's warm start (parameters and model
// state restored, fresh Adam moments, tied AP keeps the restored baseline,
// overwriting the source checkpoint rejected). The trainer evaluates the
// validation rows once per epoch, once more after restoring the best
// checkpoint, then the test rows; the call counts below follow that (tflow's
// fixture expects one call fewer, and its AP sums can exceed 1 by an ulp, so
// its build stops at "finite detector AP in [0,1]").
// https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/non-max-suppression-v3
// https://arxiv.org/abs/1506.01497
// https://arxiv.org/abs/1512.02325

#include <array>
#include <cmath>
#include <iostream>
#include <random>

#include "common/checks.hpp"
#include "common/detection.hpp"

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

namespace {

/// TensorFlow NonMaxSuppressionV3 selections for the five-box fixture,
/// printed by a TF 2.20 C++ program (score_threshold = -inf).
struct TfSelection {
  float            threshold;
  int              limit;
  std::vector<int> indices;
};
const std::vector<TfSelection> kTfNms = {
    {0, 0, {}},     {0, 1, {0}},     {0, 3, {0, 3, 4}},   {0, 10, {0, 3, 4}},     {.5f, 0, {}},  {.5f, 1, {0}},
    {.5f, 3, {0, 2, 3}}, {.5f, 10, {0, 2, 3, 4}}, {1, 0, {}}, {1, 1, {0}}, {1, 3, {0, 1, 2}}, {1, 10, {0, 1, 2, 3, 4}},
};
const std::vector<int> kTfTied = {0, 2, 3, 4};  // all scores .8, threshold .5, limit 100

void NmsChecks() {
  const std::vector<Box>   boxes  = {{0, 0, 10, 10}, {0, 0, 10, 10}, {0, 0, 10, 5}, {20, 20, 30, 30}, {40, 40, 40, 40}};
  const std::vector<float> scores = {.9f, .8f, .7f, .6f, .5f};
  check::Require(detector::NonMaxSuppression(boxes, scores, .5) == std::vector<int>({0, 2, 3, 4}),
                 "NMS suppresses duplicates and keeps equality at the threshold");
  int cases = 0;
  for (const auto& tf : kTfNms) {
    check::Require(detector::NonMaxSuppression(boxes, scores, tf.threshold, tf.limit) == tf.indices, "NMS matches TensorFlow");
    // the tensor form of the library op agrees
    const auto r = nn::non_max_suppression(nn::boxes_from_vector(boxes), nn::from_vector(scores), tf.limit, tf.threshold);
    check::Require(r.indices.to_vector<int>() == tf.indices, "tensor NMS matches TensorFlow");
    ++cases;
  }
  const std::vector<float> tied(boxes.size(), .8f);
  check::Require(detector::NonMaxSuppression(boxes, tied, .5) == kTfTied, "stable index ordering for tied scores");
  ++cases;
  check::Require(detector::NonMaxSuppression({}, {}, .5).empty(), "empty NMS");
  ++cases;
  const Box                             a          = {0, 0, 10, 10};
  const std::vector<detector::Detection> detections = {{0, 0, .9f, a}, {0, 0, .8f, a}, {0, 1, .7f, a}, {1, 0, .6f, a}};
  const auto                            kept       = detector::ClassNms(detections, .5);
  check::Require(kept.size() == 3 && kept[0].image == 0 && kept[0].label == 0 && kept[1].image == 0 && kept[1].label == 1 &&
                     kept[2].image == 1,
                 "class NMS separates images and classes");
  const auto capped = detector::ClassNms(detections, .5, 1);
  check::Require(capped.size() == 2 && capped[0].score == .9f && capped[1].score == .6f, "per-image NMS budget");
  auto rejects = [](const auto& operation) {
    try {
      operation();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  check::Require(rejects([&] { detector::NonMaxSuppression(boxes, {.9f}, .5); }), "reject mismatched NMS scores");
  check::Require(rejects([&] { detector::NonMaxSuppression(boxes, scores, NAN); }), "reject invalid NMS threshold");
  check::Require(rejects([&] { detector::NonMaxSuppression({a}, {NAN}, .5); }), "reject nonfinite NMS score");
  check::Require(rejects([&] { detector::NonMaxSuppression({{10, 0, 0, 10}}, {1}, .5); }), "reject reversed NMS corners");
  check::Require(rejects([&] { detector::NonMaxSuppression({a}, {1}, .5, -1); }), "reject negative NMS limit");
  std::cout << "  " << cases
            << " NMS fixtures match TensorFlow, including ties/equality/empty/limits; image/class separation and five "
               "malformed inputs checked"
            << std::endl;
}

void MatchingChecks() {
  const auto grid = detector::AnchorGrid(2, 3, 4, {8, 16}, {.5f, 1, 2});
  check::Require(grid.size() == 36, "anchor grid count");
  for (int i = 0; i < 36; ++i) {
    const int   cell = i / 6, ratio = i % 3;
    const float side = i % 6 < 3 ? 8 : 16, r = ratio == 0 ? .5f : ratio == 1 ? 1 : 2;
    const auto& g    = grid[std::size_t(i)];
    check::Near((g[0] + g[2]) / 2, (cell / 3 + .5f) * 4, 1e-6, "anchor row centre");
    check::Near((g[1] + g[3]) / 2, (cell % 3 + .5f) * 4, 1e-6, "anchor column centre");
    check::Near((g[2] - g[0]) * (g[3] - g[1]), side * side, 5e-5, "anchor area");
    check::Near((g[2] - g[0]) / (g[3] - g[1]), r, 1e-6, "anchor height/width ratio");
  }
  const std::vector<Box> anchors = {{0, 0, 10, 10}, {0, 0, 10, 20}, {30, 30, 40, 40}, {60, 60, 70, 70}};
  const std::vector<Box> truth   = {{0, 0, 10, 10}, {30, 30, 45, 45}};
  check::Require(detector::MatchAnchors(anchors, truth, .7, .3) == std::vector<int>({0, -2, 1, -1}),
                 "RPN threshold and forced best-anchor assignments");
  check::Require(detector::MatchAnchors(anchors, {}, .7, .3) == std::vector<int>(4, -1), "RPN no-target negatives");
  const auto collision = detector::MatchSsd(anchors, {{0, 0, 10, 10}, {0, 0, 10, 10}});
  check::Require(collision[0] == 0 && collision[1] == 1, "SSD bipartite matching preserves both colliding targets");
  std::cout << "  36 grid anchors checked; RPN positive/ignored/forced/negative = 0/-2/1/-1; SSD collisions preserve both targets"
            << std::endl;
}

void CodingChecks() {
  const Box  anchor = {10, 20, 30, 60}, target = {12, 16, 36, 64};
  const auto d      = detector::EncodeBox(anchor, target);
  check::Near(d[0], 2, 1e-6, "normalized vertical centre offset");
  check::Near(d[1], 0, 0, "zero horizontal centre offset");
  check::Near(d[2], std::log(1.2) / .2, 1e-6, "normalized log height");
  check::Near(d[3], std::log(1.2) / .2, 1e-6, "normalized log width");
  std::mt19937                          random(17);
  std::uniform_real_distribution<float> position(0, 200), size(2, 80), delta(-1, 1);
  double                                maximum = 0;
  for (int example = 0; example < 1000; ++example) {
    const float          y = position(random), x = position(random), h = size(random), w = size(random);
    const Box            a = {y, x, y + h, x + w};
    std::array<float, 4> offsets;
    for (float& v : offsets) v = delta(random);
    const auto b = detector::DecodeBox(a, offsets.data()), encoded = detector::EncodeBox(a, b),
               recovered = detector::DecodeBox(a, encoded.data());
    for (int k = 0; k < 4; ++k) maximum = std::max(maximum, std::abs(double(b[std::size_t(k)]) - recovered[std::size_t(k)]));
    for (int k = 0; k < 4; ++k)
      check::Near(encoded[std::size_t(k)], offsets[std::size_t(k)], 1e-4, "box coding recovers each normalized offset");
  }
  check::Near(maximum, 0, 1e-4, "randomized box round trips");
  std::cout << "  known deltas and 1000 randomized box round trips passed; maximum coordinate error=" << maximum << std::endl;
}

void RpnGradientChecks() {
  nn::ParameterStore p(17);
  const auto         features = nn::zeros({1, 1, 1, 4});
  const auto         rpn      = detector::BuildRpn(p, {{features, 4, 1, 2}});
  const auto labels = nn::tensor({1, 0}, {1, 2}), weights = nn::ones({1, 2}), targets = nn::zeros({1, 2, 4}),
             positive = nn::zeros({1, 2});
  const auto loss     = detector::RpnLoss(rpn, labels, weights, targets, positive);
  const auto gradient = nn::grad(loss, {p.at("rpn/objectness/b")})[0].to_vector<double>();
  check::Near(loss.item(), std::log(2.), 1e-6, "RPN zero-logit binary cross entropy");
  check::Near(gradient[0], -.25, 1e-6, "foreground RPN derivative at zero");
  check::Near(gradient[1], .25, 1e-6, "background RPN derivative at zero");
  std::cout << "  zero-logit RPN BCE derivatives: foreground -0.25, background +0.25 (two-anchor mean)" << std::endl;
}

void WarmStartChecks() {
  // Exercise the complete trainer with one scalar and a saved moving-state
  // value: restoring a baseline must preserve it when later AP merely ties.
  detector::DetectionScenes train, test;
  train.boxes  = {{{1, 1, 5, 5}}, {{1, 1, 5, 5}}};
  train.labels = {{0}, {0}};
  test.boxes   = {{{1, 1, 5, 5}}};
  test.labels  = {{0}};
  detector::DetectorTrainingConfig cfg;
  cfg.name              = "detector_warm_start_fixture_initial";
  cfg.epochs            = 1;
  cfg.batch             = 1;
  cfg.validation        = 1;
  cfg.classes           = 1;
  cfg.l2                = 0;
  cfg.minimum_test_ap50 = 1;
  std::vector<std::array<float, 2>> observed;
  detector::DetectorHooks           hooks;
  hooks.build = [](nn::ParameterStore& p, nn::ParameterStore& state) {
    p.get("weight", {}, nn::init::constant(.3));
    state.get("moving", {}, nn::init::constant(7));
  };
  hooks.loss = [](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes&, const std::vector<int>&) {
    {
      nn::NoGradGuard guard;
      state.at("moving").add_(nn::scalar(1.0));
    }
    return nn::square(p.at("weight"));
  };
  hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes&,
                     const std::vector<int>& rows) {
    observed.push_back({p.at("weight").item<float>(), state.at("moving").item<float>()});
    std::vector<detector::Detection> result;
    for (int row : rows) result.push_back({row, 0, .9f, {1, 1, 5, 5}});
    return result;
  };
  detector::TrainDetector(cfg, train, test, hooks);
  check::Require(observed.size() == 3, "initial validation/restored validation/test evaluations");
  const auto initial = observed.back();
  check::Near(initial[0], .299, 2e-6, "initial scalar Adam step");
  check::Near(initial[1], 8, 0, "initial moving state update");
  cfg.warm_start = (detector::TrainingDirectory(cfg.name) / "best").string();
  cfg.name       = "detector_warm_start_fixture_continued";
  observed.clear();
  detector::TrainDetector(cfg, train, test, hooks);
  check::Require(observed.size() == 4, "warm-start baseline/validation/restored validation/test evaluations");
  for (int k = 0; k < 2; ++k) {
    check::Near(observed[0][std::size_t(k)], initial[std::size_t(k)], 0, "warm start restores parameters and model state");
    check::Near(observed[2][std::size_t(k)], initial[std::size_t(k)], 0, "tied AP preserves restored baseline as best checkpoint");
    check::Near(observed[3][std::size_t(k)], initial[std::size_t(k)], 0, "test evaluation uses the restored baseline");
  }
  check::Near(observed[1][1], 9, 0, "continued training updates restored moving state");
  check::Near(observed[1][0], initial[0] - .001, 2e-6, "warm start intentionally uses fresh Adam moments");
  cfg.warm_start = (detector::TrainingDirectory(cfg.name) / "best").string();
  bool rejected  = false;
  try {
    detector::TrainDetector(cfg, train, test, hooks);
  } catch (const std::exception& e) {
    rejected = std::string(e.what()).find("separate output directory") != std::string::npos;
  }
  check::Require(rejected, "warm start rejects overwriting its source checkpoint");
  std::cout << "  detector warm start restores parameters/BN state, resets Adam, preserves best baseline and rejects source overwrite"
            << std::endl;
}

}  // namespace

int main() {
  try {
    NmsChecks();
    MatchingChecks();
    CodingChecks();
    RpnGradientChecks();
    WarmStartChecks();
    std::cout << "All NMS, matching and box-coding checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
