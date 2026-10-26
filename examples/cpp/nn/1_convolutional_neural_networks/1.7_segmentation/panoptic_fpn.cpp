// CMD: just run-nn panoptic_fpn
// Panoptic FPN: jointly learned Mask R-CNN and GN semantic pyramid branches.
// https://arxiv.org/abs/1901.02446
// https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/panoptic_fpn.py
//
// Port of tflow's panoptic_fpn. Full ResNet50/FPN256, semantic128/GN32,
// RPN256, box1024 and mask256 heads are checked. Compact ResNet18/8,FPN32,
// semantic16/GN8,RPN32,box128,mask16 retain all five RPN levels, four aligned
// RoI levels and both tasks. Independent merge/PQ/void/crowd and multilevel
// RoI/Jacobian fixtures. Train3 epochs,128fit/32validation/64test
// MNIST-instance scenes,64px,batch2. Two explicitly textured background
// materials are stuff; ten digits are things. Semantic class0 combines things,
// classes1/2 are stuff. All test instances/semantics are predicted, never
// replaced with ground truth. Joint RPN+box+mask+.5 semantic CE,Adam(.001),
// L2 .0001; best boxAP50 restore. Report box/mask AP and dataset/
// category-averaged PQ/SQ/RQ, things/stuff separately. Export
// input/semantic/true-panoptic/predicted-panoptic PNG. Default verifies early
// learning, not benchmark accuracy. accuracy=1 additionally gates boxAP50 .7,
// maskAP50 .6 and thingPQ .25. No real-dataset/pretrained claims.
//
// TF -> datamunge: eager training (a step's RPN proposals come from the same
// forward pass that is differentiated, with the same weights TF uses for its
// separate proposal run). TF builds the full model as a graph only; here it
// executes forward/backward on one 256x256 image. MNIST comes from tflow's
// dataset cache (data=PATH overrides); scenes, instance supports and panoptic
// maps are identical to TF's. Weights come from datamunge's generator, so
// trained numbers match TF statistically.
// Options: epochs=3 data=PATH checks_only=0|1 accuracy=0|1 threads=N.
// Artifacts: output/nn/models/panoptic_fpn_digits/.

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/instance_segmentation.hpp"
#include "common/mask_rcnn.hpp"
#include "common/panoptic.hpp"
#include "common/panoptic_fpn.hpp"
#include "common/panoptic_scenes.hpp"
#include "common/paths.hpp"
#include "common/segmentation.hpp"
#include "common/two_stage.hpp"

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
namespace pan = panoptic;
using detector::Box;
using detector::Detection;
using nn::Tensor;

namespace {

pan::PanopticConfig Compact() {
  pan::PanopticConfig c;
  c.size             = 64;
  c.depth            = 18;
  c.divisor          = 8;
  c.width            = 32;
  c.semantic_width   = 16;
  c.groups           = 8;
  c.semantic_classes = 3;
  c.grayscale        = true;
  return c;
}

void MergeChecks() {
  const std::vector<int> semantic = {1, 1, 1, 2, 1, 0, 0, 2, 1, 0, 2, 2};
  auto instance = [&](float score, int label, std::initializer_list<int> pixels) {
    instance::MaskDetection d;
    d.box = {0, label, score, {0, 0, 3, 4}};
    d.mask.assign(12, 0);
    for (int i : pixels) d.mask[std::size_t(i)] = 1;
    return d;
  };
  std::vector<instance::MaskDetection> masks = {instance(.9f, 3, {0, 1, 4, 5}), instance(.8f, 4, {1, 2, 5, 6}),
                                                instance(.7f, 5, {0, 1, 4}), instance(.49f, 6, {11})};
  const auto result = pan::MergePanoptic(semantic, masks, {-1, 10, 11}, .5, .5, 2);
  check::Require(result.ids == std::vector<int>({1, 1, 2, 3, 1, 1, 2, 3, 0, 0, 3, 3}),
                 "panoptic overlap=.5 trims accepted mask, heavy overlap rejected, small stuff/other-things void");
  check::Require(result.segments.size() == 3 && result.segments.at(1).label == 3 && result.segments.at(2).label == 4 &&
                     result.segments.at(3).label == 11 && !result.segments.at(3).thing,
                 "panoptic category/instance metadata");
  std::reverse(masks.begin(), masks.end());
  check::Require(pan::MergePanoptic(semantic, masks, {-1, 10, 11}, .5, .5, 2).ids == result.ids,
                 "merge follows scores rather than input ordering");
  masks[0].box.score = .5;
  const auto equal   = pan::MergePanoptic(semantic, masks, {-1, 10, 11}, .5, .5, 2);
  check::Require(equal.segments.at(equal.ids[11]).thing, "instance score equality retained");
  const auto all = pan::MergePanoptic(semantic, {instance(.9f, 3, {0, 1}), instance(.8f, 4, {0, 1})}, {-1, 10, 11}, .5, 1, 2);
  check::Require(all.segments.at(all.ids[0]).label == 3, "fully consumed mask creates no empty segment at overlap1");
  pan::PanopticAreas(all);
  std::cout << "  score ordering/equality, overlap rejection/trimming, stuff area, thing void and nonempty panoptic metadata passed"
            << std::endl;
}

void PqChecks() {
  const std::vector<bool> categories = {true, true, false};
  pan::PanopticMap        truth{{1, 1, 1, 2, 2, 2}, {{1, {0, true, false}}, {2, {0, true, false}}}};
  pan::PanopticMap        prediction{{3, 3, 3, 4, 0, 0}, {{3, {0, true, false}}, {4, {0, true, false}}}};
  auto                    stats = pan::PanopticStats(truth, prediction);
  auto                    score = pan::PanopticQuality(stats, categories);
  check::Require(stats[0].tp == 1 && stats[0].fp == 1 && stats[0].fn == 1, "PQ known TP/FP/FN counts");
  check::Near(score.pq, .5, 0, "PQ IoU over TP+.5FP+.5FN");
  check::Near(score.sq, 1, 0, "PQ segmentation quality");
  check::Near(score.rq, .5, 0, "PQ recognition quality");
  truth      = {{1, 1}, {{1, {0, true, false}}}};
  prediction = {{2, 0}, {{2, {0, true, false}}}};
  stats      = pan::PanopticStats(truth, prediction);
  check::Require(stats[0].tp == 0 && stats[0].fp == 1 && stats[0].fn == 1, "PQ requires strict IoU greater than .5");
  truth      = {{1, 1, 0}, {{1, {0, true, false}}}};
  prediction = {{7, 7, 7}, {{7, {0, true, false}}}};
  check::Near(pan::PanopticQuality(pan::PanopticStats(truth, prediction), categories).pq, 1, 0,
              "ground-truth void excluded from match union");
  truth      = {{1, 0, 0}, {{1, {0, true, false}}}};
  prediction = {{0, 7, 7}, {{7, {1, true, false}}}};
  stats      = pan::PanopticStats(truth, prediction);
  check::Require(stats[0].fn == 1 && stats[1].fp == 0, "predictions mostly in void ignored");
  truth      = {{1, 1, 2, 2}, {{1, {0, true, true}}, {2, {2, false, false}}}};
  prediction = {{7, 7, 8, 8}, {{7, {0, true, false}}, {8, {2, false, false}}}};
  stats      = pan::PanopticStats(truth, prediction);
  score      = pan::PanopticQuality(stats, categories);
  check::Near(score.pq, 1, 0, "crowd produces neither FN nor covered FP");
  check::Require(score.classes == 1, "only categories with TP/FP/FN contribute");
  check::Near(pan::PanopticQuality(stats, categories, 0).pq, 1, 0, "stuff-specific PQ");
  check::Require(pan::PanopticQuality(stats, categories, 1).classes == 0, "crowd-only thing category absent");
  truth      = {{1, 1, 2, 2}, {{1, {0, true, false}}, {2, {2, false, false}}}};
  prediction = {{7, 7, 0, 0}, {{7, {0, true, false}}}};
  stats      = pan::PanopticStats(truth, prediction);
  check::Near(pan::PanopticQuality(stats, categories).pq, .5, 0, "category macro-average includes missed stuff");
  pan::PqStats accumulated;
  pan::AddPanopticStats(accumulated, stats);
  pan::AddPanopticStats(accumulated, stats);
  check::Near(pan::PanopticQuality(accumulated, categories).pq, .5, 0, "dataset counts aggregate before category averaging");
  prediction = {{9, 9, 0, 0}, {{9, {1, true, false}}}};
  check::Near(pan::PanopticQuality(pan::PanopticStats(truth, prediction), categories).pq, 0, 0,
              "wrong-category overlap never matches");
  bool rejected = false;
  try {
    pan::PanopticAreas({{9}, {}});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check::Require(rejected, "missing segment metadata rejected");
  std::cout << "  hand-counted PQ/SQ/RQ, strict IoU, category macro-average, void/crowd and malformed metadata fixtures passed"
            << std::endl;
}

void SceneChecks() {
  detector::Digits digits;
  digits.pixels.assign(10 * 784, 0);
  for (int n = 0; n < 10; ++n) {
    digits.labels.push_back(n);
    for (int y = 6; y < 22; ++y)
      for (int x = 6; x < 22; ++x) digits.pixels[std::size_t(n * 784 + y * 28 + x)] = 200;
  }
  const auto            scene = pan::MakePanopticScenes(digits, 4, 11);
  detector::SceneConfig cfg;
  cfg.scenes         = 4;
  cfg.seed           = 11;
  cfg.max_overlap    = 0;
  const auto original = instance::MakeInstanceScenes(digits, cfg);
  const auto plain    = detector::MakeDetectionScenes(digits, cfg);
  check::Require(original.pixels == plain.pixels && original.boxes == plain.boxes && original.labels == plain.labels,
                 "instance-mask scenes reproduce the shared scene generator");
  int stuff[2] = {0, 0}, things = 0;
  for (int row = 0; row < 4; ++row)
    for (int pixel = 0; pixel < 4096; ++pixel) {
      int instance = -1;
      for (std::size_t k = 0; k < original.instance_masks[std::size_t(row)].size(); ++k)
        if (original.instance_masks[std::size_t(row)][k][std::size_t(pixel)]) instance = int(k);
      const auto& segment = scene.panoptic[std::size_t(row)].segments.at(scene.panoptic[std::size_t(row)].ids[std::size_t(pixel)]);
      const int   label   = scene.semantic[std::size_t(row)][std::size_t(pixel)];
      if (instance >= 0) {
        ++things;
        check::Require(label == 0 && segment.thing &&
                           segment.label == original.labels[std::size_t(row)][std::size_t(instance)] &&
                           scene.instances.image(row)[pixel] == original.image(row)[pixel],
                       "thing support/class/intensity survive stuff synthesis");
      } else {
        check::Require((label == 1 || label == 2) && !segment.thing && segment.label == label + 9,
                       "stuff semantic/panoptic identity");
        ++stuff[label - 1];
        const int value = scene.instances.image(row)[pixel], base = label == 1 ? 40 : 100;
        check::Require(value >= base && value <= base + 10, "stuff texture photometry matches its target");
      }
    }
  check::Require(things > 0 && stuff[0] > 0 && stuff[1] > 0, "both tasks and both stuff classes represented");
  const auto targets = pan::PanopticSemanticTargets(scene, {3, 1}).to_vector<std::int32_t>();
  for (int i = 0; i < 2; ++i)
    for (int pixel = 0; pixel < 4096; ++pixel)
      check::Require(targets[std::size_t(i * 4096 + pixel)] == scene.semantic[i ? 1 : 3][std::size_t(pixel)],
                     "semantic batch row pairing");
  std::cout << "  original MNIST supports/intensities preserved, two textured stuff categories, semantic/panoptic targets and "
               "row pairing passed"
            << std::endl;
}

void RoutingChecks() {
  std::vector<Tensor> features;
  for (int i = 0; i < 4; ++i) {
    const int          cells = 16 >> i;
    std::vector<float> values(std::size_t(2 * cells * cells));
    for (int n = 0; n < 2; ++n)
      for (int p = 0; p < cells * cells; ++p) values[std::size_t(n * cells * cells + p)] = float(10 * n + i + 1);
    features.push_back(nn::from_vector(values, {2, cells, cells, 1}));
    features.back().requires_grad_(true);
  }
  const std::vector<Box> boxes  = {{0, 0, 8, 8}, {0, 0, 16, 16}, {0, 0, 32, 32}, {0, 0, 64, 64}};
  const Tensor           pooled = pan::PanopticRoiAlign(features, boxes, {0, 0, 1, 1}, 64, 3, 32);
  const auto             grads  = nn::grad(nn::sum(pooled), features);
  const auto             out    = pooled.to_vector<float>();
  const double           expected[] = {1, 2, 13, 14};
  for (int i = 0; i < 4; ++i)
    for (int pixel = 0; pixel < 9; ++pixel)
      check::Near(out[std::size_t(i * 9 + pixel)], expected[i], 2e-6, "hand-picked scales route to P2/P3/P4/P5");
  for (int i = 0; i < 4; ++i) {
    const int  cells = 16 >> i;
    const auto g     = grads[std::size_t(i)].to_vector<float>();
    for (int n = 0; n < 2; ++n) {
      double total = 0;
      for (int pixel = 0; pixel < cells * cells; ++pixel) total += g[std::size_t(n * cells * cells + pixel)];
      check::Near(total, n == (i < 2 ? 0 : 1) ? 9 : 0, 3e-5, "aligned Jacobian conserves pooled weight only for selected image/level");
    }
  }
  std::cout << "  four pyramid RoI levels, image indices, border-aligned values and conserved/zero feature Jacobians passed"
            << std::endl;
}

/// One forward pass: pyramid/semantic features and the RPN; box and mask heads on demand.
struct Model {
  nn::ParameterStore&   p;
  pan::PanopticConfig   cfg;
  bool                  compact;
  pan::PanopticFeatures features;
  detector::Rpn         rpn;
  Model(nn::ParameterStore& params, nn::ParameterStore& state, const Tensor& images, bool compact_model, bool training,
        bool update_stats = true)
      : p(params), cfg(compact_model ? Compact() : pan::PanopticConfig{}), compact(compact_model) {
    features = pan::PanopticBackbone(p, state, images, cfg, training, update_stats);
    std::vector<detector::RpnLevel> levels;
    for (int i = 0; i < 5; ++i) levels.push_back({features.pyramid[std::size_t(i)], cfg.width, cfg.size / (4 << i), 3});
    rpn = detector::BuildRpn(p, levels, compact ? 32 : 256);
  }
  [[nodiscard]] int Classes() const { return compact ? 10 : 80; }
  detector::RoiHeadOutputs Head(const std::vector<Box>& rois, const std::vector<int>& indices) const {
    return maskrcnn::PooledRoiHead(p, pan::PanopticRoiAlign(features.pyramid, rois, indices, cfg.size, 7, compact ? 32.f : 224.f),
                                   Classes() + 1, compact ? 128 : 1024);
  }
  Tensor MaskLogits(const std::vector<Box>& boxes, const std::vector<int>& indices) const {
    return maskrcnn::MaskHead(p, pan::PanopticRoiAlign(features.pyramid, boxes, indices, cfg.size, 14, compact ? 32.f : 224.f),
                              compact ? 16 : 256, Classes());
  }
};

void ModelChecks() {
  for (bool compact : {false, true}) {
    nn::ParameterStore p(17), state(18);
    const int          side = compact ? 64 : 256, batch = compact ? 2 : 1;
    std::vector<float> pixels(std::size_t(batch * side * side * (compact ? 1 : 3)));
    for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = std::sin(.13f * float(i));
    const Model net(p, state, nn::from_vector(pixels, {batch, side, side, compact ? 1 : 3}), compact, true);
    const float s = compact ? 1.f : 4.f;
    const std::vector<Box> boxes   = {{0, 0, 8 * s, 8 * s}, {0, 0, 16 * s, 16 * s}, {0, 0, 32 * s, 32 * s}, {0, 0, 64 * s, 64 * s}};
    const std::vector<int> indices = compact ? std::vector<int>{0, 0, 1, 1} : std::vector<int>{0, 0, 0, 0};
    const auto             head    = net.Head(boxes, indices);
    const Tensor           masks   = net.MaskLogits(boxes, indices);
    check::Require(p.count() == (compact ? 489751 : 46027110), "independent Panoptic FPN backbone/semantic/RPN/box/mask parameter census");
    Tensor objective = nn::scalar(0.0);
    for (const auto& output : {net.features.semantic, net.rpn.objectness, net.rpn.deltas, head.probabilities, head.deltas, masks})
      objective = objective + nn::mean(nn::square(output));
    const auto grads = nn::grad(objective, p.parameters());
    check::Require(grads.size() == std::size_t(compact ? 125 : 224), "Panoptic FPN independent parameter tensor count");
    double low = INFINITY, high = 0;
    for (std::size_t k = 0; k < grads.size(); ++k) {
      const double norm = std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
      check::Require(std::isfinite(norm) && (!compact || norm > 0), "finite nonzero panoptic derivative: " + p.names()[k]);
      low  = std::min(low, norm);
      high = std::max(high, norm);
    }
    if (!compact) {
      std::cout << "  full ResNet50/FPN/GN32/RPN/box/mask:" << p.count() << " parameters, all" << grads.size()
                << " derivatives (executed on one 256x256 image)" << std::endl;
      continue;
    }
    check::Require(net.features.semantic.shape() == nn::Shape({2, 64, 64, 3}) && net.rpn.objectness.shape() == nn::Shape({2, 1023}) &&
                       masks.shape() == nn::Shape({4, 28, 28, 10}),
                   "compact semantic/RPN/instance shape alignment");
    std::cout << "  compact Panoptic FPN:" << p.count() << " parameters, all" << grads.size() << " derivatives evaluated; norms ["
              << low << ',' << high << ']' << std::endl;
  }
}

std::vector<Box> Anchors() {
  std::vector<Box> anchors;
  const float      scales[] = {8, 16, 24, 40, 56};
  for (int i = 0; i < 5; ++i) {
    auto boxes = detector::AnchorGrid(16 >> i, 16 >> i, float(4 << i), {scales[i]}, {.5f, 1, 2});
    anchors.insert(anchors.end(), boxes.begin(), boxes.end());
  }
  return anchors;
}

std::vector<int> Chunk(const std::vector<int>& rows, std::size_t start, std::size_t size) {
  return std::vector<int>(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + size)));
}

std::vector<Detection> PredictBoxes(nn::ParameterStore& p, nn::ParameterStore& state, const std::vector<Box>& anchors,
                                    const detector::DetectionScenes& scenes, const std::vector<int>& rows) {
  nn::NoGradGuard        guard;
  std::vector<Detection> boxes;
  for (std::size_t start = 0; start < rows.size(); start += 8) {
    const auto  chunk = Chunk(rows, start, 8);
    const Model net(p, state, detector::SceneImages(scenes, chunk), true, false);
    const auto  proposed = detector::ComputeProposals(net.rpn, anchors, scenes.height, scenes.width);
    std::vector<std::pair<int, Box>> candidates;
    for (std::size_t i = 0; i < chunk.size(); ++i)
      for (const auto& box : proposed[i]) candidates.push_back({int(i), box});
    const auto found = maskrcnn::DetectPixelRois(
        [&](const std::vector<Box>& r, const std::vector<int>& ix) { return net.Head(r, ix); }, chunk, candidates, 11);
    boxes.insert(boxes.end(), found.begin(), found.end());
  }
  return detector::ClassNms(boxes, .3f);
}

std::vector<instance::MaskDetection> PredictMasks(nn::ParameterStore& p, nn::ParameterStore& state, const std::vector<Box>& anchors,
                                                  const detector::DetectionScenes& scenes, const std::vector<int>& rows) {
  nn::NoGradGuard                      guard;
  const auto                           boxes = PredictBoxes(p, state, anchors, scenes, rows);
  std::vector<instance::MaskDetection> found;
  for (std::size_t start = 0; start < rows.size(); start += 8) {
    const auto             chunk = Chunk(rows, start, 8);
    std::vector<Detection> selected;
    for (const auto& box : boxes)
      if (box.score > .05f && std::find(chunk.begin(), chunk.end(), box.image) != chunk.end()) selected.push_back(box);
    if (selected.empty()) continue;
    std::vector<Box> rois;
    std::vector<int> indices;
    for (const auto& d : selected) {
      rois.push_back(d.box);
      indices.push_back(int(std::find(chunk.begin(), chunk.end(), d.image) - chunk.begin()));
    }
    const Model net(p, state, detector::SceneImages(scenes, chunk), true, false);
    const auto  probabilities = nn::sigmoid(net.MaskLogits(rois, indices)).to_vector<float>();
    for (std::size_t i = 0; i < selected.size(); ++i)
      found.push_back({selected[i], instance::PasteMask(probabilities.data() + i * 28 * 28 * 10, 28, 10, selected[i].label,
                                                        selected[i].box, 64, 64)});
  }
  return found;
}

/// Mask BCE and semantic CE of a fixed fitting batch (training-mode BN, no statistic updates).
struct Probe {
  std::vector<int>      rows;
  maskrcnn::MaskTargets masks;
  Tensor                semantic_targets;
};
std::pair<double, double> ProbeLosses(nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                                      const Probe& probe, const Tensor& weights) {
  nn::NoGradGuard guard;
  const Model     net(p, state, detector::SceneImages(scenes, probe.rows), true, true, false);
  const double mask = maskrcnn::SelectedMaskLoss(net.MaskLogits(probe.masks.boxes, probe.masks.indices), probe.masks.labels,
                                                 probe.masks.targets, 10)
                          .item<double>();
  const double semantic = seg::SegmentationCrossEntropy(net.features.semantic, probe.semantic_targets, 3, weights).item<double>();
  return {mask, semantic};
}

void Train(const std::filesystem::path& directory, int epochs, bool accuracy) {
  const auto fit     = pan::MakePanopticScenes(detector::LoadDigits(directory, "train"), 160, 11);
  const auto test    = pan::MakePanopticScenes(detector::LoadDigits(directory, "t10k"), 64, 22);
  const auto anchors = Anchors();
  const auto weights = nn::ones({3});
  std::mt19937_64 random(7);
  int             positive_rois = 0;
  Probe           probe;
  double          initial_mask = NAN, initial_semantic = NAN;
  detector::DetectorHooks hooks;
  hooks.build = [&](nn::ParameterStore& p, nn::ParameterStore& state) {
    const Model net(p, state, detector::SceneImages(fit.instances, {0}), true, false);
    net.Head({{0, 0, 16, 16}}, {0});
    net.MaskLogits({{0, 0, 16, 16}}, {0});
  };
  hooks.loss = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                   const std::vector<int>& rows) {
    const Model net(p, state, detector::SceneImages(scenes, rows), true, true);
    const auto  proposed = detector::ComputeProposals(net.rpn, anchors, scenes.height, scenes.width);
    detector::RoiBatch batch;
    for (std::size_t i = 0; i < rows.size(); ++i)
      detector::SampleRois(scenes, rows[i], int(i), proposed[i], 32, 10, 4, 16, random, batch);
    const auto semantic_targets = pan::PanopticSemanticTargets(fit, rows);
    const auto targets          = detector::MakeRpnTargets(anchors, scenes, rows, random);
    const auto rpn_loss = detector::RpnLoss(net.rpn, targets.labels, targets.weights, targets.targets, targets.positives);
    const auto head     = net.Head(batch.pixels, batch.indices);
    const auto masks    = maskrcnn::PositiveMaskTargets(static_cast<const instance::InstanceScenes&>(scenes), rows, batch);
    check::Require(!masks.boxes.empty(), "positive instance support for panoptic mask branch");
    positive_rois += int(masks.boxes.size());
    const auto mask_loss = maskrcnn::SelectedMaskLoss(net.MaskLogits(masks.boxes, masks.indices), masks.labels, masks.targets, 10);
    const auto semantic_loss = seg::SegmentationCrossEntropy(net.features.semantic, semantic_targets, 3, weights);
    if (probe.rows.empty()) {
      probe            = {rows, masks, semantic_targets};
      initial_mask     = mask_loss.item<double>();
      initial_semantic = semantic_loss.item<double>();
    }
    return rpn_loss + detector::RoiLoss(head, batch, 11) + mask_loss + .5 * semantic_loss;
  };
  hooks.detect = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                     const std::vector<int>& rows) { return PredictBoxes(p, state, anchors, scenes, rows); };
  hooks.inspect = [&](nn::ParameterStore& p, nn::ParameterStore& state, const detector::DetectionScenes& scenes,
                      const std::vector<int>& rows, const std::filesystem::path& outdir) {
    const auto [learned_mask, learned_semantic] = ProbeLosses(p, state, fit.instances, probe, weights);
    check::Require(std::isfinite(learned_mask) && std::isfinite(learned_semantic) && learned_mask < initial_mask - 1e-4 &&
                       learned_semantic < initial_semantic - 1e-4,
                   "both panoptic tasks must reduce fixed fitting-batch loss");
    std::cout << "  fixed fitting-batch mask BCE " << initial_mask << " -> " << learned_mask << "; semantic CE " << initial_semantic
              << " -> " << learned_semantic << " (no optimizer/BN updates; not held-out metrics)" << std::endl;
    const auto                    instances = PredictMasks(p, state, anchors, scenes, rows);
    std::vector<std::vector<int>> semantic(std::size_t(scenes.size()));
    {
      nn::NoGradGuard guard;
      for (std::size_t start = 0; start < rows.size(); start += 8) {
        const auto  chunk = Chunk(rows, start, 8);
        const Model net(p, state, detector::SceneImages(scenes, chunk), true, false);
        const auto  predicted = nn::argmax(net.features.semantic, 3).to_vector<std::int64_t>();
        for (std::size_t i = 0; i < chunk.size(); ++i)
          semantic[std::size_t(chunk[i])].assign(predicted.begin() + std::ptrdiff_t(i * 4096),
                                                 predicted.begin() + std::ptrdiff_t((i + 1) * 4096));
      }
    }
    std::vector<pan::PanopticMap> combined(std::size_t(scenes.size()));
    pan::PqStats                  stats;
    std::vector<bool>             things(12, true);
    things[10] = things[11] = false;
    for (int row : rows) {
      std::vector<instance::MaskDetection> local;
      for (const auto& inst : instances)
        if (inst.box.image == row) local.push_back(inst);
      combined[std::size_t(row)] = pan::MergePanoptic(semantic[std::size_t(row)], local, {-1, 10, 11}, .5, .5, 32);
      pan::AddPanopticStats(stats, pan::PanopticStats(test.panoptic[std::size_t(row)], combined[std::size_t(row)]));
    }
    const auto all = pan::PanopticQuality(stats, things), object = pan::PanopticQuality(stats, things, 1),
               stuff = pan::PanopticQuality(stats, things, 0);
    for (double value : {all.pq, all.sq, all.rq, object.pq, stuff.pq})
      check::Require(std::isfinite(value) && value >= 0 && value <= 1, "finite panoptic quality in [0,1]");
    const double mask_ap = instance::MaskAveragePrecision(instances, test.instances, 10, .5);
    check::Require(std::isfinite(mask_ap) && mask_ap >= 0 && mask_ap <= 1, "finite panoptic instance mask AP");
    std::ofstream report(outdir / "panoptic_metrics.tsv");
    report << "category\tthing\ttp\tfp\tfn\tmatched_iou\n";
    for (const auto& [label, value] : stats)
      report << label << '\t' << things[std::size_t(label)] << '\t' << value.tp << '\t' << value.fp << '\t' << value.fn << '\t'
             << value.iou << '\n';
    report << "all_pq\t" << all.pq << "\nall_sq\t" << all.sq << "\nall_rq\t" << all.rq << "\nthings_pq\t" << object.pq
           << "\nstuff_pq\t" << stuff.pq << "\nmask_ap50\t" << mask_ap << '\n';
    pan::PanopticPreview(test, semantic, combined, outdir / "panoptic_preview.png");
    std::cout << "  all64 held-out scenes: mask AP50=" << mask_ap << "; PQ=" << all.pq << ", SQ=" << all.sq << ", RQ=" << all.rq
              << "; things PQ=" << object.pq << ", stuff PQ=" << stuff.pq << std::endl;
    std::cout << "  category counts/quality and input/semantic/GT/merged preview: " << outdir.string() << std::endl;
    if (accuracy) check::Require(mask_ap >= .6 && object.pq >= .25, "optional panoptic mask/thing-quality benchmarks");
  };
  detector::DetectorTrainingConfig cfg;
  cfg.name              = "panoptic_fpn_digits";
  cfg.epochs            = epochs;
  cfg.batch             = 2;
  cfg.validation        = 32;
  cfg.accuracy          = accuracy;
  cfg.minimum_test_ap50 = .7;
  detector::TrainDetector(cfg, fit.instances, test.instances, hooks);
  std::cout << "  positive instance mask training RoIs=" << positive_rois << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool                  checks_only = false, accuracy = false;
    int                   epochs    = 3;
    std::filesystem::path directory = paths::Dataset("mnist");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg == "accuracy=0" || arg == "accuracy=1")
        accuracy = arg == "accuracy=1";
      else if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        directory = arg.substr(5);
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected checks_only=0|1 accuracy=0|1 epochs=N data=PATH threads=N");
    }
    MergeChecks();
    PqChecks();
    SceneChecks();
    RoutingChecks();
    ModelChecks();
    if (checks_only) {
      std::cout << "All Panoptic FPN core fixtures passed (training not requested).\n";
      return 0;
    }
    Train(directory, epochs, accuracy);
    std::cout << "All Panoptic FPN checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
