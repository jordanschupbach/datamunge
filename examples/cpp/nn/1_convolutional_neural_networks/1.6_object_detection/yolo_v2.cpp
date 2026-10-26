// CMD: just run-nn yolo_v2
// YOLO v2 and YOLO9000: dimension priors, feature passthrough and WordTree supervision.
//
// Port of tflow's yolo_v2. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
//
// Independently check space-to-depth ordering, its inverse and derivative,
// IoU dimension clustering, sigmoid/exp box decoding, the detection loss and
// its analytic derivatives, and WordTree's sibling normalization, marginal
// products and partial-label gradients. Check classification-only loss does
// not supervise box coordinates. Target fixtures check rectangular geometry,
// best priors, collision order, detached IoU confidence, overlapping unmatched
// predictions and empty images. Decoder fixtures check reordered image batches,
// flat/hierarchical class scores, per-class NMS, clipping and extreme logits.
// A 9418-node wide hierarchy checks scaling.
// Construct full YOLO v2 (50,655,389 parameters, 5 VOC anchors/20 classes)
// and the 3-anchor/9418-node YOLO9000 head (79,502,989 parameters), including
// every parameter derivative. Numerically check the compact model and all
// its derivatives, with grids 8/10/12 at 64/80/96 pixels.
// https://arxiv.org/abs/1612.08242
//
// Train two compact Darknet-19-based models, retaining every convolution
// with 1/8 widths and grayscale inputs, omitting two pools (stride 8).
// Flat YOLO v2 sees digit boxes/classes and five fit-only clustered priors.
// The joint YOLO9000 adaptation sees only even/odd labels for boxed digits,
// and separate single-digit classification images labelled 0..9. Its
// 12-node digit hierarchy replaces the paper's WordNet hierarchy, and uses
// three priors. Mix classification/detection batches 4:1; classification
// selects objectness times the marginal class probability, then updates
// that prediction's ancestor path and objectness below .3, without box
// labels. Change training resolution every ten batches among 64/80/96 pixels.
// Use 5500 fit / 500 validation / 1000 test scenes, batch 32, Adam(.001),
// kernel L2 .0001 and best validation AP50 checkpoints.
// Default epochs: 3 flat, 3 joint. Options: mode=flat|joint (default both),
// data=PATH epochs=N warm_start=PREFIX (requires a single mode) accuracy=0|1
// threads=N. checks_only=1 runs all fixtures without loading datasets or training.
// Default accuracy=0 verifies finite, decreasing loss; accuracy=1 enforces
// the original flat/joint test AP50 targets (.7/.5).
//
// Eager port (common/yolo_v2.hpp, common/word_tree.hpp). tflow builds the
// 416-pixel full models as graphs only; here their forward and backward
// passes run at 128 pixels (counts and connectivity do not depend on the
// input size). Each training step runs one training-mode forward pass and
// takes the matching targets from its detached outputs (tflow runs a separate
// detached pass with the same weights and batch statistics). WordTree
// marginals are exp(sum of log-conditionals along the path) instead of a
// running product (equal up to rounding). Weights come from nn's generator,
// so trained numbers match TF statistically.
// Output: output/nn/models/yolo_v2_digits/ and yolo9000_digits_joint/.

#include <iostream>

#include "common/yolo_v2.hpp"

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
using yolo::Dimension;

namespace {
wordtree::WordTree DigitTree() { return wordtree::WordTree({-1, -1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}); }
nn::Tensor         SinImages(std::int64_t n, std::int64_t size, std::int64_t channels) {
  std::vector<float> v(std::size_t(n * size * size * channels));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(.13f * float(i));
  return nn::from_vector(v, {n, size, size, channels});
}

void ReorgChecks() {
  std::vector<float> x(48);
  std::iota(x.begin(), x.end(), 0.f);
  auto       input      = nn::from_vector(x, {1, 4, 6, 2}).requires_grad_();
  const auto rearranged = nn::space_to_depth(input, 2);
  const auto restored   = nn::depth_to_space(rearranged, 2).to_vector<float>();
  const auto gradient   = nn::grad(nn::sum(nn::square(rearranged)) * .5, {input})[0].to_vector<float>();
  check::Require(rearranged.shape() == nn::Shape{1, 2, 3, 8}, "passthrough packs four offsets into channels");
  const auto packed_values = rearranged.to_vector<float>();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 6; ++c)
      for (int channel = 0; channel < 2; ++channel) {
        const int i = (r * 6 + c) * 2 + channel, packed = ((r / 2) * 3 + c / 2) * 8 + ((r % 2) * 2 + c % 2) * 2 + channel;
        check::Near(packed_values[std::size_t(packed)], i, 0, "reorg channel order");
        check::Near(restored[std::size_t(i)], i, 0, "reorg inverse");
        check::Near(gradient[std::size_t(i)], i, 0, "reorg derivative");
      }
  std::cout << "  4x6x2 -> 2x3x8 passthrough: channel order, inverse and derivatives exact" << std::endl;
}

void PriorChecks() {
  const std::vector<Dimension> examples = {{2, 4}, {8, 4}, {2, 4}, {8, 4}, {2, 4}, {8, 4}};
  const auto                   centres  = yolo::ClusterDimensions(examples, 2);
  check::Require(centres == std::vector<Dimension>({{2, 4}, {8, 4}}), "IoU clustering recovers the two dimension populations");
  check::Near(yolo::DimensionIoU({2, 4}, {8, 4}), .25, 0, "shape IoU is scale-aware but position independent");
  const float raw[] = {0, 0, float(std::log(2.)), float(std::log(.5)), 0};
  const auto  box   = yolo::DecodeYoloV2(raw, 2, 3, 8, {2, 4}, 64, 64);
  const float expected[] = {4, 20, 36, 36};
  for (std::size_t k = 0; k < 4; ++k) check::Near(box[k], expected[k], 2e-6, "sigmoid centre and exponential size decoding");
  std::cout << "  IoU clustering recovers [2,4]/[8,4]; sigmoid/exp decoding matches hand calculation" << std::endl;
}

void AssignmentChecks() {
  const std::vector<Dimension> priors = {{1, 1}, {2, 1}};
  DetectionScenes              scenes;
  scenes.height = 40;
  scenes.width  = 80;
  scenes.boxes  = {{}, {}, {{7, 42, 23, 58}, {5, 40, 25, 60}}};
  scenes.labels = {{}, {}, {3, 4}};
  const nn::Shape    shape = {2, 4, 4, 2, 15};
  std::vector<float> raw(2 * 4 * 4 * 2 * 15, 0);
  auto r5 = [&](int i, int r, int c, int a, int k) -> float& { return raw[std::size_t((((i * 4 + r) * 4 + c) * 2 + a) * 15 + k)]; };
  auto i4 = [](int i, int r, int c, int a) { return std::size_t(((i * 4 + r) * 4 + c) * 2 + a); };
  // The non-responsible prior predicts the same box as the responsible one.
  // It must be ignored rather than penalized as background.
  r5(0, 1, 2, 0, 2) = std::log(2.f);
  for (bool coarse : {false, true})
    for (bool reverse : {false, true}) {
      if (reverse) {
        std::reverse(scenes.boxes[2].begin(), scenes.boxes[2].end());
        std::reverse(scenes.labels[2].begin(), scenes.labels[2].end());
      }
      const auto t = yolo::MakeYoloV2Targets(raw, shape, priors, scenes, {2, 0}, coarse);
      check::Require(t.collisions == 1, "one anchor/cell collision reported");
      const float expected[] = {.5f, .5f, 0, 0};
      for (std::size_t d = 0; d < 4; ++d) check::Near(t.geometry[i4(0, 1, 2, 1) * 4 + d], expected[d], 1e-6, "independent rectangular-image target encoding");
      check::Require(t.labels[i4(0, 1, 2, 1)] == (coarse ? 0 : 4), "larger collision target retained with fine/coarse label");
      check::Near(std::accumulate(t.positive.begin(), t.positive.end(), 0.f), 1, 0, "exactly one responsible anchor across reordered batch");
      check::Near(t.positive[i4(0, 1, 2, 1)], 1, 0, "best shape prior receives target");
      check::Near(t.confidence[i4(0, 1, 2, 1)], 1, 1e-6, "positive confidence is current decoded-box IoU");
      check::Near(t.weights[i4(0, 1, 2, 1)], 1, 0, "positive weight overrides overlap ignore");
      check::Near(t.weights[i4(0, 1, 2, 0)], 0, 0, "overlapping unassigned prediction ignored");
      check::Near(t.weights[i4(0, 0, 0, 0)], .5, 0, "distant background keeps no-object weight");
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          for (int a = 0; a < 2; ++a) {
            check::Near(t.positive[i4(1, r, c, a)], 0, 0, "empty image has no positive anchors");
            check::Near(t.confidence[i4(1, r, c, a)], 0, 0, "empty image has zero confidence targets");
            check::Near(t.weights[i4(1, r, c, a)], .5, 0, "empty image retains background penalty");
          }
      if (reverse) {
        std::reverse(scenes.boxes[2].begin(), scenes.boxes[2].end());
        std::reverse(scenes.labels[2].begin(), scenes.labels[2].end());
      }
    }
  r5(0, 1, 2, 1, 3)    = std::log(.5f);
  const auto imperfect = yolo::MakeYoloV2Targets(raw, shape, priors, scenes, {2, 0}, false);
  check::Near(imperfect.confidence[i4(0, 1, 2, 1)], .5, 1e-6, "halving predicted width gives IoU confidence .5 rather than a constant positive target");
  check::Near(imperfect.geometry[i4(0, 1, 2, 1) * 4 + 3], 0, 0, "prediction changes do not change target geometry");
  std::cout << "  target encoding, best-prior selection, order-independent collisions, IoU confidence, overlap-ignore and empty images passed"
            << std::endl;
}

void DecodingChecks() {
  DetectionScenes scenes;
  scenes.height = 16;
  scenes.width  = 24;
  scenes.pixels.assign(2 * 16 * 24, 0);
  scenes.boxes.resize(2);
  scenes.labels.resize(2);
  const std::vector<Dimension> priors = {{.5f, .5f}, {.5f, .5f}};
  std::vector<float>           raw(2 * 2 * 2 * 2 * 15, 0);
  auto v = [&](int i, int r, int c, int a, int k) -> float& { return raw[std::size_t((((i * 2 + r) * 2 + c) * 2 + a) * 15 + k)]; };
  auto reset = [&] {
    std::fill(raw.begin(), raw.end(), 0.f);
    for (int i = 0; i < 2; ++i)
      for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
          for (int a = 0; a < 2; ++a) {
            v(i, r, c, a, 4) = -1000;
            for (int k = 5; k < 15; ++k) v(i, r, c, a, k) = -100;
          }
  };
  auto decode = [&] {
    return yolo::DetectYoloV2([&](const nn::Tensor&) { return nn::from_vector(raw, {2, 2, 2, 2, 15}); }, priors, scenes, {1, 0}, 10, nullptr);
  };
  reset();
  for (int a = 0; a < 2; ++a) {
    v(0, 0, 1, a, 4)     = std::log(a ? .6f / .4f : .8f / .2f);
    v(0, 0, 1, a, 5 + 3) = std::log(8.f);
    v(0, 0, 1, a, 5 + 7) = std::log(2.f);
  }
  v(1, 1, 0, 0, 4)     = std::log(.9f / .1f);
  v(1, 1, 0, 0, 5 + 7) = 0;
  const auto detections = decode();
  check::Require(detections.size() == 3, "classwise NMS removes duplicate priors without suppressing another class or image");
  bool seen[2][10] = {};
  for (const auto& d : detections) {
    check::Require(d.image == 0 || d.image == 1, "decoding maps reordered rows to original scene index");
    check::Require(d.label == 3 || d.label == 7, "flat class index retained");
    check::Require(!seen[d.image][d.label], "one detection per prescribed image/class");
    seen[d.image][d.label] = true;
    const Box expected     = d.image == 1 ? Box{2, 15, 6, 21} : Box{10, 3, 14, 9};
    if (d.image == 0) check::Require(d.label == 7, "second batch image class");
    for (std::size_t k = 0; k < 4; ++k) check::Near(d.box[k], expected[k], 1e-6, "rectangular-image decoded coordinates");
    check::Near(d.score, d.image == 0 ? .9 : d.label == 3 ? .64 : .16, 1e-6, "objectness times flat class probability");
  }
  reset();
  v(0, 0, 0, 0, 0) = v(0, 0, 0, 0, 1) = -1000;
  v(0, 0, 0, 0, 2) = v(0, 0, 0, 0, 3) = 1000;
  v(0, 0, 0, 0, 4)                    = 1000;
  v(0, 0, 0, 0, 5 + 4)                = 0;
  const auto clipped                  = decode();
  check::Require(clipped.size() == 1 && clipped[0].image == 1 && clipped[0].label == 4, "extreme logits remain finite and filter background");
  const Box expected = {0, 0, 16, 24};
  for (std::size_t d = 0; d < 4; ++d) check::Near(clipped[0].box[d], expected[d], 0, "large decoded sizes clipped to image");
  check::Near(clipped[0].score, 1, 0, "saturated objectness and class score");
  {
    std::vector<float> hierarchy_raw(2 * 17, -100);
    for (int d = 0; d < 4; ++d) hierarchy_raw[std::size_t(d)] = 0;
    hierarchy_raw[4]     = std::log(.8f / .2f);
    hierarchy_raw[5]     = std::log(99.f);
    hierarchy_raw[6]     = 0;
    hierarchy_raw[5 + 6] = 100;
    const auto tree = DigitTree();
    const auto hierarchical = yolo::DetectYoloV2([&](const nn::Tensor&) { return nn::from_vector(hierarchy_raw, {2, 1, 1, 1, 17}); },
                                                 {{.5f, .5f}}, scenes, {1, 0}, 12, &tree);
    check::Require(hierarchical.size() == 1 && hierarchical[0].label == 4 && hierarchical[0].image == 1,
                   "hierarchical decoder emits fine digit leaves rather than parent classes");
    check::Near(hierarchical[0].score, .792, 1e-6, "objectness times WordTree marginal leaf probability includes the .99 ancestor factor");
  }
  std::cout << "  rectangular decoding, reordered batches, flat/hierarchical scores, classwise NMS, clipping and extreme logits passed" << std::endl;
}

void DetectionLossChecks() {
  std::vector<float> x(7, 0);
  yolo::YoloV2Targets t;
  t.geometry   = {.5f, .5f, 0, 0};
  t.labels     = {1};
  t.positive   = {1};
  t.confidence = {1};
  t.weights    = {1};
  auto run = [&](double expected_loss, const std::vector<double>& expected_gradient) {
    auto       raw      = nn::from_vector(x, {1, 1, 1, 1, 7}).requires_grad_();
    const auto loss     = yolo::YoloV2Loss(raw, t, 2, {});
    const auto gradient = nn::grad(loss, {raw})[0].to_vector<float>();
    check::Near(loss.item(), expected_loss, 2e-6, "independent YOLO v2 loss");
    for (std::size_t k = 0; k < 7; ++k) check::Near(gradient[k], expected_gradient[k], 2e-6, "analytic YOLO v2 gradient");
  };
  run(.25 + std::log(2.), {0, 0, 0, 0, -.25, .5, -.5});
  t.geometry = {.25f, .75f, .2f, -.3f};
  run(1.525 + std::log(2.), {.625, -.625, -2, 3, -.25, .5, -.5});
  t.positive = t.confidence = {0};
  t.weights                 = {.5};
  run(.125, {0, 0, 0, 0, .125, 0, 0});
  t.weights = {0};
  run(0, {0, 0, 0, 0, 0, 0, 0});
  std::cout << "  positive/offset/background/ignored losses and all seven analytic derivatives passed" << std::endl;
}

void TreeChecks() {
  const auto         tree = DigitTree();
  std::vector<float> zero(12, 0);
  auto run = [&](int label, std::vector<float>& conditional, std::vector<float>& marginal, double& loss, std::vector<float>& gradient) {
    auto       logits = nn::from_vector(zero, {1, 12}).requires_grad_();
    const auto output = wordtree::WordTreeProbabilities(logits, tree);
    const auto l      = nn::mean(wordtree::WordTreeLoss(output, {label}, tree));
    gradient          = nn::grad(l, {logits})[0].to_vector<float>();
    conditional       = output.conditional.to_vector<float>();
    marginal          = output.marginal.to_vector<float>();
    loss              = l.item();
  };
  std::vector<float> conditional, marginal, gradient;
  double             loss = 0, leaf_total = 0;
  run(0, conditional, marginal, loss, gradient);
  for (std::size_t node = 0; node < 12; ++node) {
    check::Near(conditional[node], node < 2 ? .5 : .2, 1e-6, "sibling softmax");
    check::Near(marginal[node], node < 2 ? .5 : .1, 1e-6, "ancestor product");
    check::Near(gradient[node], node == 0 ? -.5 : node == 1 ? .5 : 0, 1e-6, "coarse labels do not supervise descendants");
    if (node >= 2) leaf_total += marginal[node];
  }
  check::Near(leaf_total, 1, 1e-6, "leaf probabilities partition the root");
  check::Near(loss, std::log(2.), 1e-6, "coarse path loss");
  run(2, conditional, marginal, loss, gradient);
  check::Near(loss, std::log(10.), 1e-6, "fine path loss");
  for (std::size_t node = 0; node < 12; ++node) {
    const double expected = node == 0 ? -.5 : node == 1 ? .5 : node == 2 ? -.8 : node % 2 == 0 ? .2 : 0;
    check::Near(gradient[node], expected, 1e-6, "fine label supervises precisely its ancestor sibling groups");
  }
  {
    const wordtree::WordTree wide(std::vector<int>(9418, -1));
    const auto probabilities = wordtree::WordTreeProbabilities(nn::zeros({1, 9418}), wide);
    check::Near(wordtree::WordTreeLoss(probabilities, {9417}, wide).item(), std::log(9418.), 2e-6, "9418-node path loss");
  }
  std::cout << "  WordTree sibling sums/marginal products, coarse/fine path gradients and 9418-node synthetic capacity passed" << std::endl;
}

void ClassificationOnlyChecks() {
  const auto          tree = DigitTree();
  std::vector<float>  x(34, 0);
  yolo::YoloV2Targets t;
  t.geometry.assign(8, 0);
  t.labels.assign(2, 0);
  t.positive.assign(2, 0);
  t.confidence.assign(2, 0);
  t.weights.assign(2, 0);
  yolo::YoloV2Weights weights;
  weights.detection             = 0;
  weights.classification        = 1;
  weights.classification_labels = {2};
  x[17 + 5]                     = std::log(2.f);
  auto run = [&](std::vector<float>& gradient) {
    auto       raw  = nn::from_vector(x, {1, 1, 2, 1, 17}).requires_grad_();
    const auto loss = yolo::YoloV2Loss(raw, t, 12, weights, &tree);
    gradient        = nn::grad(loss, {raw})[0].to_vector<float>();
    return loss.item();
  };
  std::vector<float> gradient;
  check::Near(run(gradient), std::log(7.5), 2e-6, "classification-only selects the highest marginal fine-label probability");
  for (int anchor = 0; anchor < 2; ++anchor)
    for (int channel = 0; channel < 17; ++channel) {
      double expected = 0;
      if (anchor == 1 && channel >= 5) {
        const int node = channel - 5;
        expected       = node == 0 ? -1. / 3 : node == 1 ? 1. / 3 : node == 2 ? -.8 : node % 2 == 0 ? .2 : 0;
      }
      check::Near(gradient[std::size_t(anchor * 17 + channel)], expected, 2e-6,
                  "classification-only leaves geometry, adequate objectness and other predictions unsupervised");
    }
  x[4]      = std::log(.9f / .1f);
  x[17 + 4] = std::log(.1f / .9f);
  check::Near(run(gradient), std::log(10.), 2e-6, "objectness reverses the preferred conditional-class prediction");
  for (int anchor = 0; anchor < 2; ++anchor)
    for (int channel = 0; channel < 17; ++channel) {
      double expected = 0;
      if (anchor == 0 && channel >= 5) {
        const int node = channel - 5;
        expected       = node == 0 ? -.5 : node == 1 ? .5 : node == 2 ? -.8 : node % 2 == 0 ? .2 : 0;
      }
      check::Near(gradient[std::size_t(anchor * 17 + channel)], expected, 2e-6,
                  "classification-only selects objectness times marginal, without differentiating selection");
    }
  x[4]      = std::log(.1f / .9f);
  x[17 + 4] = std::log(.2f / .8f);
  check::Near(run(gradient), std::log(7.5) + .01, 2e-6, "classification-only weak objectness floor");
  check::Near(gradient[17 + 4], -.032, 2e-6, "weak objectness .3 floor analytic sigmoid derivative");
  std::cout << "  classification-only joint-score selection and weak-objectness gradients passed; geometry/unselected derivatives zero" << std::endl;
}

void ModelChecks() {
  for (bool large : {false, true}) {
    nn::ParameterStore p(17), state(17);
    yolo::YoloV2Config cfg;
    cfg.update_stats = false;
    if (large) {
      cfg.classes = 9418;
      cfg.anchors = 3;
    }
    const auto net = yolo::YoloV2(p, state, SinImages(1, 128, 3), cfg, true);
    check::Require(p.count() == (large ? 79502989 : 50655389) && state.count() == 20672, "full YOLO v2/9000 parameter counts");
    const auto gradients = nn::grad(nn::mean(nn::square(net.raw)), p.parameters());
    for (const auto& g : gradients) check::Require(g.defined(), "every full YOLO parameter connected");
    std::cout << "  " << (large ? "9418-node/3-prior head" : "20-class/5-prior YOLO v2") << ": " << p.count() << " parameters, "
              << gradients.size() << " parameter derivatives computed (128-pixel input)" << std::endl;
  }
  nn::ParameterStore p(17), state(17);
  yolo::YoloV2Config cfg;
  cfg.classes       = 10;
  cfg.compact       = true;
  cfg.width_divisor = 8;
  cfg.update_stats  = false;
  double minimum = INFINITY, maximum = 0;
  for (int resolution : {64, 80, 96}) {
    const auto net = yolo::YoloV2(p, state, SinImages(2, resolution, 1), cfg, true);
    check::Require(net.fine.shape()[1] == resolution / 4 && net.passthrough.shape()[1] == resolution / 8 && net.passthrough.shape()[3] == 32 &&
                       net.raw.shape() == nn::Shape{2, resolution / 8, resolution / 8, 5, 15},
                   "multi-scale grids and passthrough dimensions");
    if (resolution != 64) continue;
    const auto gradients = nn::grad(nn::mean(nn::square(net.raw)), p.parameters());
    for (std::size_t i = 0; i < gradients.size(); ++i) {
      const double norm = std::sqrt(nn::sum(nn::square(gradients[i])).item());
      check::Require(std::isfinite(norm) && norm > 0, "finite nonzero compact YOLO v2 derivative: " + p.names()[i]);
      minimum = std::min(minimum, norm);
      maximum = std::max(maximum, norm);
    }
  }
  std::cout << "  compact model: grids 8/10/12 at 64/80/96 px; " << p.count() << " parameters; derivative norms [" << minimum << ',' << maximum << "]"
            << std::endl;
}

void Train(bool joint, int epochs, const std::filesystem::path& directory, const std::string& warm_start, bool accuracy) {
  detector::SceneConfig train_cfg, test_cfg;
  train_cfg.scenes = 6000;
  train_cfg.seed   = 11;
  test_cfg.scenes  = 1000;
  test_cfg.seed    = 22;
  const auto digits = detector::LoadDigits(directory, "train");
  const auto train  = detector::MakeDetectionScenes(digits, train_cfg);
  const auto test   = detector::MakeDetectionScenes(detector::LoadDigits(directory, "t10k"), test_cfg);
  auto       classification_cfg = train_cfg;
  classification_cfg.seed        = 33;
  classification_cfg.max_objects = 1;
  const auto             classification = detector::MakeDetectionScenes(digits, classification_cfg);
  std::vector<Dimension> dimensions;
  for (std::size_t row = 0; row < 5500; ++row)
    for (const auto& b : train.boxes[row]) dimensions.push_back({(b[2] - b[0]) / 8, (b[3] - b[1]) / 8});
  const auto priors       = yolo::ClusterDimensions(dimensions, joint ? 3 : 5);
  const auto tree         = DigitTree();
  double     average_iou  = 0;
  for (const auto& sample : dimensions) {
    float best = 0;
    for (const auto& prior : priors) best = std::max(best, yolo::DimensionIoU(sample, prior));
    average_iou += best / double(dimensions.size());
  }
  std::cout << "  fit-only dimension priors (height,width in stride-8 cells):";
  for (const auto& prior : priors) std::cout << " (" << prior[0] << ',' << prior[1] << ')';
  std::cout << "; mean best shape IoU=" << average_iou << std::endl;
  detector::DetectorTrainingConfig cfg;
  cfg.name              = joint ? "yolo9000_digits_joint" : "yolo_v2_digits";
  cfg.warm_start        = warm_start;
  cfg.epochs            = epochs;
  cfg.accuracy          = accuracy;
  cfg.minimum_test_ap50 = joint ? .5 : .7;
  yolo::YoloV2Config model;
  model.compact       = true;
  model.width_divisor = 8;
  model.anchors       = int(priors.size());
  model.classes       = joint ? 12 : 10;
  const wordtree::WordTree* hierarchy = joint ? &tree : nullptr;
  int                       step = 0, collisions = 0, classification_images = 0, detection_images = 0;
  detector::DetectorHooks   hooks;
  hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore& state) {
    auto frozen         = model;
    frozen.update_stats = false;
    yolo::YoloV2(p, state, nn::zeros({1, 64, 64, 1}), frozen, true);
  };
  hooks.loss = [&](nn::ParameterStore& p, nn::ParameterStore& state, const DetectionScenes& scenes, const std::vector<int>& rows) {
    const bool  classify = joint && step % 5 != 4;
    const auto& source   = classify ? classification : scenes;
    const int   size     = 64 + 16 * ((step / 10) % 3);
    ++step;
    const auto raw = yolo::YoloV2(p, state, nn::resize_bilinear(detector::SceneImages(source, rows), {size, size}), model, true).raw;
    yolo::YoloV2Targets t;
    yolo::YoloV2Weights weights;
    if (classify) {
      DetectionScenes empty;
      empty.height = empty.width = 64;
      empty.boxes.resize(rows.size());
      empty.labels.resize(rows.size());
      std::vector<int> dummy(rows.size());
      std::iota(dummy.begin(), dummy.end(), 0);
      t = yolo::MakeYoloV2Targets(raw.detach().to_vector<float>(), raw.shape(), priors, empty, dummy, true);
      classification_images += int(rows.size());
      weights.detection      = 0;
      weights.classification = 1;
      for (int row : rows) weights.classification_labels.push_back(classification.labels[std::size_t(row)][0] + 2);
    } else {
      t = yolo::MakeYoloV2Targets(raw.detach().to_vector<float>(), raw.shape(), priors, scenes, rows, joint);
      collisions += t.collisions;
      detection_images += int(rows.size());
    }
    return yolo::YoloV2Loss(raw, t, model.classes, weights, hierarchy);
  };
  hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore& state, const DetectionScenes& scenes, const std::vector<int>& rows) {
    return yolo::DetectYoloV2([&](const nn::Tensor& images) { return yolo::YoloV2(p, state, images, model, false).raw; }, priors, scenes, rows,
                              model.classes, hierarchy);
  };
  detector::TrainDetector(cfg, train, test, hooks);
  std::cout << "  image presentations: " << detection_images << " boxed (" << (joint ? "even/odd labels only" : "fine digit labels") << "), "
            << classification_images << " classification-only; " << collisions << " anchor/cell collisions discarded" << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path directory = paths::Dataset("mnist");
    int                   epochs    = 0;
    std::string           warm_start;
    std::vector<bool>     modes       = {false, true};
    bool                  checks_only = false, accuracy = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        eq  = arg.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("expected key=value option");
      const auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
      if (key == "warm_start") warm_start = value;
      else if (key == "data") directory = value;
      else if (key == "epochs") {
        epochs = std::stoi(value);
        if (epochs < 1) throw std::invalid_argument("epochs must be positive");
      } else if (key == "accuracy" && (value == "0" || value == "1")) accuracy = value == "1";
      else if (key == "checks_only" && (value == "0" || value == "1")) checks_only = value == "1";
      else if (key == "mode" && (value == "flat" || value == "joint")) modes = {value == "joint"};
      else if (key == "threads") nn::set_num_threads(std::stoi(value));
      else throw std::invalid_argument("unknown option " + key);
    }
    ReorgChecks();
    PriorChecks();
    AssignmentChecks();
    DecodingChecks();
    DetectionLossChecks();
    TreeChecks();
    ClassificationOnlyChecks();
    ModelChecks();
    if (checks_only) {
      std::cout << "All YOLO v2 and WordTree fixtures passed (training not requested).\n";
      return 0;
    }
    if (!warm_start.empty() && modes.size() != 1) throw std::invalid_argument("warm_start requires mode=flat or mode=joint");
    for (bool joint : modes) Train(joint, epochs ? epochs : 3, directory, warm_start, accuracy);
    std::cout << "All YOLO v2 and WordTree checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
